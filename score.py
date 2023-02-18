"""
Score samples under from a trained model
"""
import argparse
import itertools
from pathlib import Path
import sys
from hashlib import sha1


import torch
import torch.nn as nn
import sentencepiece as spm
from termcolor import colored

from model import GPTConfig, GPT


parser = argparse.ArgumentParser('sample')
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--seed', type=int, default=1337)
parser.add_argument('--steps', type=int, default=256)
parser.add_argument('--lora', action='store_true')
parser.add_argument('--peft', action='store_true')
parser.add_argument('--tsv', action='store_true', help='output tsv')
parser.add_argument('--spm', type=str, default='wiki.model', help='sentencepiece tokenizer')
parser.add_argument('--no_eot', action='store_true')
parser.add_argument('--seq_len', type=int, default=1024)
parser.add_argument('--paragraphs', nargs='*', help='files with paragraphs to score', type=Path)
parser.add_argument('--sentences', nargs='*', help='files with sentences to score', type=Path)
parser.add_argument('--ids', action='store_true', help='output integer ids')
parser.add_argument('--pieces', action='store_true', help='output pieces')
parser.add_argument('--unblank', action='store_true', help='replace blank tokens with the token with the highest probability')
parser.add_argument('-v', '--verbose', action='store_true', help='verbose output')
parser.add_argument('ckpt_path')
parser.add_argument('prompts', nargs='*')
args = parser.parse_args()

device = args.device
torch.manual_seed(args.seed)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn

checkpoint = torch.load(args.ckpt_path, map_location=device)

# model
if not 'vocab_size' in checkpoint['model_args']:
    print('WARNING: vocab_size not found in checkpoint, assuming 50257', file=sys.stderr)
    vocab_size = 50257
    checkpoint['model_args']['vocab_size'] = 50257
else:
    vocab_size = checkpoint['model_args']['vocab_size']
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
    model.eval()
    model.to(device)
else:
    print(gptconf, file=sys.stderr)
    model = nn.ModuleDict({'_orig_mod': GPT(gptconf)})
    model = model._orig_mod
    print('compiling model', file=sys.stderr)
    #model = torch.compile(model, mode='reduce-overhead') # requires PyTorch 2.0
    model = torch.compile(model) # requires PyTorch 2.0
    print('done compiling model', file=sys.stderr)

    model.load_state_dict(checkpoint['model'], strict=False)

model.eval()
model.to(device)


sp = spm.SentencePieceProcessor(model_file=args.spm)

if args.tsv:
    print('id', 'sentence' , 'ppl', 'sentence_len', sep='\t')

for prompt in itertools.chain(args.prompts,
                              *(f.read_text().split("\n\n") for f in args.paragraphs or []),
                              *(f.read_text().split("\n") for f in args.sentences or [])):
    prompt = prompt.strip()
    print('prompt:', prompt, file=sys.stderr)
    if args.no_eot:
        start = sp.encode(prompt)
    else:
        start = [50256] + sp.encode(prompt)

    length = len(start)

    if length > args.seq_len:
        start = start[-args.seq_len:] # truncate very long sequences from the beginning
        length = args.seq_len
        print(colored('truncated', 'red'), prompt, file=sys.stderr)

    x = torch.tensor(start, dtype=torch.long, device=device)[None, ...]

    with torch.inference_mode():
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            y, _ = model(x, decode_full=True)

    output_length = length - 1 # output eats the first token
    y = y[:, :output_length, :] 
    sequence = x[0, 1:]
    
    if args.tsv:
        id = sha1(prompt.encode("utf-8")).hexdigest()
        print(id, prompt, sep='\t', end='\t')

    if args.verbose:
        print(colored(output_length, 'green'), end=' ')

    if args.unblank:
        blank = 50229 # _    
        best_tok_pointwise = y.argmax(-1)
        sequence = torch.where(sequence == blank, best_tok_pointwise, sequence)

    sequence = sequence.view(-1)
    log_prob = nn.functional.cross_entropy(y.view(-1, vocab_size), sequence, reduction='none')[:output_length].sum()

    if args.verbose:
        print(colored(log_prob.item(), 'red'), end=' ')

    sequence = sequence.tolist()
    
    if args.tsv:
        print((log_prob / output_length).exp().item(), output_length, sep='\t', flush=True)
    elif args.ids:
        print(*sequence)
    elif args.pieces:
        print(' '.join(sp.id_to_piece(sequence)))
    else:
        print(sp.decode(sequence))

    if args.paragraphs:
        print(flush=True)
