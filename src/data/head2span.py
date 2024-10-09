# -*- coding: utf-8 -*-
import re
from functools import lru_cache

import spacy


@lru_cache
def get_spacy(model: str = "en_core_web_sm") -> spacy.Language:
    try:
        return spacy.load(model)
    except:
        spacy.cli.download(model)
        return spacy.load(model)


def get_final_span(syntactic_head_token, fb_head_token):
    # mention subtree vs children distinction in meeting!
    syntactic_head_subtree = list(syntactic_head_token.subtree)

    relevant_tokens = []

    for token in syntactic_head_subtree:
        if token.dep_ in ['cc', 'conj'] and token.i > fb_head_token.i:
            break
        relevant_tokens.append(token)

    left_edge = relevant_tokens[0].idx
    right_edge = relevant_tokens[-1].idx + len(relevant_tokens[-1].text)

    return left_edge, right_edge


def get_head_span(head_token_offset_start, head_token_offset_end, doc):
    fb_head_token = doc.char_span(head_token_offset_start, head_token_offset_end,
                                  alignment_mode='expand')[0]

    # when above target, eliminate CC or CONJ arcs
    # if on non-FB-target verb mid-traversal, DO take CC or CONJ arcs
    # if hit AUX, don't take CC or CONJ - don't worry for now
    if fb_head_token.dep_ == 'ROOT':
        syntactic_head_token = fb_head_token
    else:
        syntactic_head_token = None
        ancestors = list(fb_head_token.ancestors)
        ancestors.insert(0, fb_head_token)

        if len(ancestors) == 1:
            syntactic_head_token = ancestors[0]
        else:
            for token in ancestors:
                if token.pos_ in ['PRON', 'PROPN', 'NOUN']:
                    syntactic_head_token = token
                    break
                elif token.pos_ in ['VERB', 'AUX']:
                    syntactic_head_token = token
                    break

            if syntactic_head_token is None:
                for token in ancestors:
                    if token.pos_ == 'NUM':
                        syntactic_head_token = token
                        break

    # postprocessing for CC and CONJ -- exclude child arcs with CC or CONJ
    span_start, span_end = get_final_span(syntactic_head_token, fb_head_token)
    return span_start, span_end


def get_span_indices(word, text):
    try:
        return re.search(word, text, flags=re.IGNORECASE).span()
    except:
        return False


def get_text_span(target, text):
    doc = get_spacy("en_core_web_sm")(text)
    span_ixs = get_span_indices(target, text)
    try:
        s, e = get_head_span(*span_ixs, doc)
        span = text[s:e]
        return span
    except:
        return False
