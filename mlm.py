"""This is a minor modification of huggingface's toking masking:"""
"""original source:
https://github.com/huggingface/transformers/blob/130b987880a9b1ade5c76dc1413c12c8924fda50/src/transformers/data/data_collator.py#L748
at commit f00f22a3e290fd377b979124dcf9800b3d73eb11

based on code in https://github.com/JonasGeiping/cramming/blob/main/cramming/utils.py
"""

import torch


def mask_tokens(
    inputs,
    mlm_probability=0.15,
    mask_token=50254, # <unk>
    endoftext_token=50256, # <|endoftext|>
    max_token=50257 # <pad>
):
    """
    Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
    """
    labels = inputs.clone()

    # We sample a few tokens in each sequence for MLM training (with probability `mlm_probability`)
    probability_matrix = torch.full(labels.shape, mlm_probability)
    probability_matrix.masked_fill_(labels == endoftext_token, value=0.0)
    masked_indices = torch.bernoulli(probability_matrix).bool()
    labels[~masked_indices] = 64444  # We only compute loss on masked tokens

    # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
    indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
    inputs[indices_replaced] = mask_token

    # 10% of the time, we replace masked input tokens with random word
    indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
    random_words = torch.randint(max_token, labels.shape, dtype=inputs.dtype)
    inputs[indices_random] = random_words[indices_random]

    # The rest of the time (10% of the time) we keep the masked input tokens unchanged

    return inputs, labels
