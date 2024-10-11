# -*- coding: utf-8 -*-
import dataclasses
from typing import Literal, Optional

import torch
import torch.nn as nn
import transformers as tf

from ..data.commitment_bank import load_opensmile


PoolerType = Literal["max", "mean", "sum"]
FusionStrategy = Literal["early", "late"]
@dataclasses.dataclass
class ModelArguments:
    text_model_name_or_path: Optional[str] = dataclasses.field(default=None)
    audio_model_name_or_path: Optional[str] = dataclasses.field(default=None)
    use_opensmile_features: bool = dataclasses.field(default=False)
    num_labels: int = dataclasses.field(default=None)
    text_pooler_type: Optional[PoolerType] = dataclasses.field(default="max")
    audio_pooler_type: Optional[PoolerType] = dataclasses.field(default="max")
    freeze_text_model: bool = dataclasses.field(default=False)
    freeze_audio_model: bool = dataclasses.field(default=False)
    fusion_strategy: FusionStrategy = dataclasses.field(default="early")


def freeze_params(module: torch.nn.Module) -> None:
    for param in module.parameters():
        param.requires_grad = False


def pooler(features: torch.Tensor, dim: int, pooler_type: PoolerType) -> torch.Tensor:
    if not features.numel():
        return features  # Nothing to pool.
    if pooler_type == "max":
        pool_fn = lambda t, dim: torch.max(t, dim=dim).values
    elif pooler_type == "mean":
        pool_fn = torch.mean  # TODO: version that drops padded columns.
    elif pooler_type == "sum":
        pool_fn = torch.sum
    else:
        raise ValueError(f"unknown pooler_type: {pooler_type}")
    return pool_fn(features, dim=dim)


def classification_head(
    input_size: int, proj_size: int, output_size: int
) -> torch.nn.Sequential:
    return torch.nn.Sequential(
        torch.nn.Linear(input_size, proj_size),  # Dense projection layer.
        #  torch.nn.LayerNorm(proj_size),        # XXX: Make optional?
        torch.nn.ReLU(),                         # Activation. TODO: Dropout?
        torch.nn.Linear(proj_size, output_size)  # Classifier.
    )


class MultiheadCrossAttention(nn.Module):
    def __init__(self, d_model_q, d_model_kv, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.d_model_q = d_model_q
        self.d_model_kv = d_model_kv
        self.head_dim = d_model_q // num_heads

        self.q_linear = nn.Linear(d_model_q, d_model_q)
        self.k_linear = nn.Linear(d_model_kv, d_model_q)
        self.v_linear = nn.Linear(d_model_kv, d_model_q)
        
        self.out_proj = nn.Linear(d_model_q, d_model_q)
        
    def forward(self, query, key, value):
        batch_size = query.size(0)
        
        # Linear projections
        q = self.q_linear(query)
        k = self.k_linear(key)
        v = self.v_linear(value)
        
        # Reshape for multi-head attention
        q = q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_weights = torch.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, v)
        
        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model_q)
        attn_output = self.out_proj(attn_output)
        
        return attn_output

class MultimodalClassifier(nn.Module):
    def __init__(self, config: ModelArguments):
        super().__init__()
        self.config = config
        
        # Load the text model
        self.text_model = (
            tf.AutoModel.from_pretrained(config.text_model_name_or_path)
            if config.text_model_name_or_path
            else None
        )
        if self.text_model and self.config.freeze_text_model:
            freeze_params(self.text_model)
        
        # Load the audio model
        self.audio_model = (
            tf.AutoModel.from_pretrained(config.audio_model_name_or_path)
            if config.audio_model_name_or_path
            else None
        )
        if self.audio_model and self.config.freeze_audio_model:
            freeze_params(self.audio_model)
        
        # Throw if neither is given
        if not self.text_model and not self.audio_model and not self.config.use_opensmile_features:
            raise ValueError("No text or audio model(s) specified.")
        
        # Initialize hidden sizes
        self.text_hidden_size = self.text_model.config.hidden_size if self.text_model else 0
        self.audio_hidden_size = self.audio_model.config.hidden_size if self.audio_model else 0
        self.opensmile_hidden_size = (
            load_opensmile().opensmile_features[0].shape[0]
            if self.config.use_opensmile_features
            else 0
        )
        
        # Initialize cross-attention modules
        self.text_to_audio_attention = MultiheadCrossAttention(self.text_hidden_size, self.audio_hidden_size, 8)
        self.audio_to_text_attention = MultiheadCrossAttention(self.audio_hidden_size, self.text_hidden_size, 8)
        
        # Initialize classification head
        self.classifier_proj_size = self.text_hidden_size + self.audio_hidden_size + self.opensmile_hidden_size
        self.classification_head = nn.Sequential(
            nn.Linear(self.classifier_proj_size, self.classifier_proj_size),
            nn.ReLU(),
            nn.Linear(self.classifier_proj_size, config.num_labels)
        )

    def forward(
        self,
        text_input_ids,
        text_attention_mask,
        audio_input_values,
        opensmile_features,
        labels,
        **kwargs
    ):
        device = self.classification_head[0].weight.device
        
        # Process text input
        text_features = torch.tensor([]).to(device)
        if self.text_model:
            text_features = self.text_model(
                input_ids=text_input_ids,
                attention_mask=text_attention_mask
            ).last_hidden_state
        
        # Process audio input
        audio_features = torch.tensor([]).to(device)
        if self.audio_model:
            if self.audio_model.__class__.__name__ == "WhisperModel":
                audio_features = self.audio_model.encoder(audio_input_values).last_hidden_state
            else:
                audio_features = self.audio_model(audio_input_values).last_hidden_state
        
        # Apply cross-attention
        if self.text_model and self.audio_model:
            text_attended = self.text_to_audio_attention(text_features, audio_features, audio_features)
            audio_attended = self.audio_to_text_attention(audio_features, text_features, text_features)
        else:
            text_attended = text_features
            audio_attended = audio_features
        
        # Pooling
        text_pooled = pooler(text_attended, dim=1, pooler_type=self.config.text_pooler_type)
        audio_pooled = pooler(audio_attended, dim=1, pooler_type=self.config.audio_pooler_type)
        
        # OpenSmile features
        if not self.config.use_opensmile_features:
            opensmile_features = torch.tensor([]).to(device)
        # Pooling.
        text_pooled = pooler(
            text_features, dim=1, pooler_type=self.config.text_pooler_type
        )
        audio_pooled = pooler(
            audio_features, dim=1, pooler_type=self.config.audio_pooler_type
        )
        # Classification logits.
        if self.config.fusion_strategy == "early":
            fusion_features = torch.cat([
                text_pooled, audio_pooled, opensmile_features
            ], dim=1)
            logits = self.classification_head(fusion_features)
        elif self.config.fusion_strategy == "late":
            text_logits = torch.tensor([]).to(device)
            if self.text_model:
                text_logits = self.text_classification_head(text_pooled)
            audio_logits = torch.tensor([]).to(device)
            if self.audio_model:
                audio_logits = self.audio_classification_head(audio_pooled)
            opensmile_logits = torch.tensor([]).to(device)
            if self.config.use_opensmile_features:
                opensmile_logits = self.opensmile_classification_head(opensmile_features)
            logits = (text_logits, audio_logits, opensmile_logits)
            logits = torch.stack([lgts for lgts in logits if lgts.numel()]).mean(dim=0)
        else:
            raise ValueError
        # Compute loss.
        
        # Fusion and classification
        fusion_features = torch.cat([text_pooled, audio_pooled, opensmile_features], dim=1)
        logits = self.classification_head(fusion_features)
        
        # Compute loss
        loss = None
        if labels is not None:
            if self.config.num_labels == 1:
                loss_fct = nn.MSELoss()
                loss = loss_fct(logits.squeeze(), labels.squeeze())
            else:
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.config.num_labels), labels.view(-1))
        
        return tf.modeling_outputs.SequenceClassifierOutput(loss=loss, logits=logits)
