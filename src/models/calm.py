# -*- coding: utf-8 -*-
import dataclasses
from typing import Optional

import torch
import transformers as tf


@dataclasses.dataclass
class ModelArguments:
    model_name_or_path: str
    fusion_model_name_or_path: str


class CALM(torch.nn.Module):
    def __init__(self, config: ModelArguments):
        super().__init__()
        self.config = config
        # Load the text models.
        self.text_model = tf.AutoModel.from_pretrained(config.model_name_or_path)
        self.text_fusion_model = tf.AutoModel.from_pretrained(
            config.fusion_model_name_or_path
        )
        # Load the state model.
        self.state_model = tf.AutoModel.from_pretrained(config.model_name_or_path)
        self.state_fusion_model = tf.AutoModel.from_pretrained(
            config.fusion_model_name_or_path
        )


    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        state_ids: Optional[torch.LongTensor] = None,
        state_mask: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        **kwargs
    ) -> tf.modeling_outputs.CausalLMOutputWithCrossAttentions:
        raise NotImplementedError
        device = self.text_model.device
        # Concatenate text and state inputs.
        input_ids = torch.cat([text_input_ids, state_input_ids], dim=1)
        attention_mask = torch.cat([text_attention_mask, state_attention_mask], dim=1)
        # First pass on creating t+1 features.
        text_features = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask
        ).last_hidden_state
        state_features = self.state_model(
            input_ids=input_ids,
            attention_mask=attention_mask
        ).last_hidden_state
        fusion_features = torch.cat([text_features, state_features], dim=1)
        # Second pass.
        state_fusion_features = self.text_fusion_model(fusion_features)
        text_fusion_features = self.text_fusion_model(fusion_features)
        # Compute loss.
        loss = None
