"""
Constrained decoding for NER
"""
import argparse
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from model import GPTConfig, GPT
import sentencepiece as spm
import sys
from termcolor import colored
import itertools
from typing import List
import re

from convert2vulyk import reconstruct_tokenized


parser = argparse.ArgumentParser('sample')
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--seed', type=int, default=1337)
parser.add_argument('--steps', type=int, default=256)
parser.add_argument('--lora', action='store_true')
parser.add_argument('--peft', action='store_true')
parser.add_argument('--spm', type=str, default='wiki.model', help='sentencepiece tokenizer')
parser.add_argument('--no_eot', action='store_true')
parser.add_argument('ckpt_path')
parser.add_argument('infile', type=argparse.FileType("r"))
args = parser.parse_args()

device = args.device
torch.manual_seed(args.seed)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn

checkpoint = torch.load(args.ckpt_path, map_location=device)

# model
gptconf = GPTConfig(**checkpoint['model_args'])

if args.peft:
    model = GPT(gptconf)

    from peft import get_peft_model, LoraConfig, TaskType
    model.prepare_inputs_for_generation = lambda *x: None
 
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM, inference_mode=True, r=4, lora_alpha=32, lora_dropout=0.1,
        target_modules=["c_attn"]
    )

    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    model.load_state_dict(checkpoint['model'])
    #print(model)

    # model = nn.ModuleDict({'base_model': nn.ModuleDict({'model': GPT(gptconf)})})

    # from lora import gpt2_peft_config, lora_find_and_replace
    # lora_find_and_replace(model, gpt2_peft_config)
    
    # model.load_state_dict(checkpoint['model'])
    # model = model.base_model.model
elif args.lora:
    model = GPT(gptconf)

    from lora import gpt2_peft_config, lora_find_and_replace
    lora_find_and_replace(model, gpt2_peft_config)
    
    model.load_state_dict(checkpoint['model'])

else:
    model = nn.ModuleDict({'_orig_mod': GPT(gptconf)})
    model = model._orig_mod
    model = torch.compile(model) # requires PyTorch 2.0

    model.load_state_dict(checkpoint['model'])

model.eval()
model.to(device)


sp = spm.SentencePieceProcessor(model_file=args.spm)


def generate_step(self, idx, temperature=1.0, top_k=None):
    """
    Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
    the sequence 1 time
    """
    # if the sequence context is growing too long we must crop it at block_size
    idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
    # forward the model to get the logits for the index in the sequence
    logits, _ = self(idx_cond)
    # pluck the logits at the final step and scale by desired temperature
    logits = logits[:, -1, :] / temperature
    # optionally crop the logits to only the top k options
    if top_k is not None:
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        logits[logits < v[:, [-1]]] = -float('Inf')
    # apply softmax to convert logits to (normalized) probabilities
    probs = F.softmax(logits, dim=-1)
    # sample from the distribution
    idx_next = torch.multinomial(probs, num_samples=1)
    # append sampled index to the running sequence and continue
    idx = torch.cat((idx, idx_next), dim=1)

    return idx


def encode_idx(text, prefix=[], suffix=[]):
    start = prefix + sp.encode(text)
    idx = (torch.tensor(start, dtype=torch.long, device=device)[None, ...])
    return idx


@torch.inference_mode()
def ner_decode(self, constraint_tokens, prompt, temperature=1.0, top_k=1):
    if args.no_eot:
        idx = start = encode_idx(prompt)
    else:
        idx = start = encode_idx(prompt, prefix=[50256])

    with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
        idx = generate_step(self, idx, top_k=1)

    for token in constraint_tokens[1:]:
        #print(sp.decode(idx[0].tolist()))
        idx = torch.cat((idx, encode_idx(token)), dim=1)
        #idx = generate_step(self, idx, top_k=1) # /  # [1]
        idx = generate_step(self, idx, top_k=1) # tok

    idx = idx[0].tolist()
    prefix, gen = idx[:len(start)], idx[len(start):]
    try:
        eot = gen.index(50256)
        gen = gen[:eot]
    except ValueError:
        pass
    
    #print([sp.id_to_piece(i) for i in start], file=sys.stderr)
    prefix, gen = sp.decode(prefix), sp.decode(gen)
    print(prefix, colored(gen, "magenta"), sep='')
    print()


def convert_sentence_inline(sentence: List[str],
                            prefix_text: str = "",
                            annotation: str = "анотація:",
                            test: bool = False) -> str:
    tokens: List[str] = []
    constraint_tokens: List[str] = []
    ner_tokens: List[str] = []

    mapping = {
        "B-PERS": "P",
        "B-ORG": "O",
        "B-MISC": "M",
        "B-LOC": "L",
        "I-PERS": "p",
        "I-ORG": "o",
        "I-MISC": "m",
        "I-LOC": "l",
        "O": "X",
    }

    for line in sentence:
        w, tag = line.split(" ")
        tokens.append(w)
        constraint_tokens.append(w + " /") # [1]

        
    for line in sentence:
        w, tag = line.split(" ")
        ner_tokens.append(w)
        ner_tokens.append("/")
        break

    final_sentence: str = "".join(map(str, reconstruct_tokenized([tokens])))
    final_tagged_sentence: str = " ".join(map(str, reconstruct_tokenized([ner_tokens])))
    final_tagged_sentence = re.sub(r'\s+', ' ', final_tagged_sentence)

    prompt = prefix_text + final_sentence + "\n" + annotation + " " + final_tagged_sentence
    return constraint_tokens, prompt
        

accum = []
for line in map(str.strip, args.infile):
    if not line.strip():
        if accum:
            constraint_tokens, prompt = convert_sentence_inline(accum, test=True)
            ner_decode(model, constraint_tokens, prompt)
            accum = []
    else:
        accum.append(line)


