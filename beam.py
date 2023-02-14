#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import argparse
from pathlib import Path
import time
from typing import Tuple

import torch
from torch import Tensor
from torch.nn import functional as F
from torch.utils.data import DataLoader
import torch.nn.functional as F
#torch.set_printoptions(threshold=100000, linewidth=400)

import sentencepiece as spm
from model import GPTConfig, GPT
from lora import gpt2_peft_config, lora_find_and_replace
from torch.nn.utils.rnn import pad_sequence


parser = argparse.ArgumentParser(description='PyTorch GPT2 beam decoding')

parser.add_argument('--batch_size', type=int, default=4,
                    help='batch size')

parser.add_argument('--eval_len', type=int, default=256,
                    help='max tokens to generate')

parser.add_argument('--min_length', type=int, default=0,
                    help='min tokens to generate')

parser.add_argument('--beam', type=int, default=4, help='beam search size')

parser.add_argument('--length_penalty', type=float, default=0, help='length penalty')

parser.add_argument('--no_repeat_ngram_size', type=int, default=6, help='no_repeat_ngram_size')

parser.add_argument('--repetition_penalty', type=float, default=1.0, help='repetition_penalty')

parser.add_argument('--spm', type=str, default='wiki.model', help='sentencepiece tokenizer')

parser.add_argument('--device', type=str, default='cuda:0')

parser.add_argument('--seq_len', type=int, default=1024, help='input sequence length (including context)')

parser.add_argument('ckpt_path', type=Path)
parser.add_argument('context', help='data to use as padding context', type=Path)
parser.add_argument('data', help='paragraphs', type=Path)


def _reorder_cache(past: Tuple, beam_idx: Tensor) -> Tuple[Tensor]:
    if past is None:
        return None
    return tuple(layer_past.index_select(1, beam_idx).contiguous().detach() for layer_past in past)


def _calc_banned_ngram_tokens(
    prev_input_ids: Tensor, 
    num_hypos: int, 
    no_repeat_ngram_size: int, 
    cur_len: int
) -> None:
    """Copied from fairseq for no_repeat_ngram in beam_search"""
    if cur_len + 1 < no_repeat_ngram_size:
        # return no banned tokens if we haven't generated no_repeat_ngram_size tokens yet
        return [[] for _ in range(num_hypos)]
    
    generated_ngrams = [{} for _ in range(num_hypos)]
    for idx in range(num_hypos):
        gen_tokens = prev_input_ids[idx].tolist()
        generated_ngram = generated_ngrams[idx]
        for ngram in zip(*[gen_tokens[i:] for i in range(no_repeat_ngram_size)]):
            prev_ngram_tuple = tuple(ngram[:-1])
            generated_ngram[prev_ngram_tuple] = generated_ngram.get(prev_ngram_tuple, []) + [ngram[-1]]

    def _get_generated_ngrams(hypo_idx):
        # Before decoding the next token, prevent decoding of ngrams that have already appeared
        start_idx = cur_len + 1 - no_repeat_ngram_size
        ngram_idx = tuple(prev_input_ids[hypo_idx, start_idx:cur_len].tolist())
        return generated_ngrams[hypo_idx].get(ngram_idx, [])

    banned_tokens = [_get_generated_ngrams(hypo_idx) for hypo_idx in range(num_hypos)]
    return banned_tokens


def _enforce_repetition_penalty_(
    lprobs, 
    batch_size, 
    num_beams, 
    prev_output_tokens, 
    repetition_penalty
):
    """repetition penalty (from CTRL paper https://arxiv.org/abs/1909.05858). """

    for i in range(batch_size * num_beams):
        for previous_token in set(prev_output_tokens[i].tolist()):
            # if score < 0 then repetition penalty has to multiplied to reduce the previous token probability
            if lprobs[i, previous_token] < 0:
                lprobs[i, previous_token] *= repetition_penalty
            else:
                lprobs[i, previous_token] /= repetition_penalty

def _postprocess_next_token_scores(
    scores,
    history,
    cur_len,
    batch_size,
    num_beams,
    repetition_penalty=1.0,                                
    no_repeat_ngram_size=4,
    bad_words_ids=None,
    min_length=0,
    max_length=100,
    eos_token_id=None,
):
    # repetition penalty (from CTRL paper https://arxiv.org/abs/1909.05858)
    if repetition_penalty != 1.0 and history is not None:
        _enforce_repetition_penalty_(scores, batch_size, num_beams, history, repetition_penalty)

    # score: batch_size * beam, vocab
    # set eos token prob to zero if min_length is not reached
    if eos_token_id is not None and cur_len < min_length:
        for eos in eos_token_id:
            scores[:, eos] = -float("inf")

    if no_repeat_ngram_size > 0 and history is not None:
        # calculate a list of banned tokens to prevent repetitively generating the same ngrams
        num_batch_hypotheses = batch_size * num_beams
        # from fairseq: https://github.com/pytorch/fairseq/blob/a07cb6f40480928c9e0548b737aadd36ee66ac76/fairseq/sequence_generator.py#L345
        banned_batch_tokens = _calc_banned_ngram_tokens(
                history, num_batch_hypotheses, no_repeat_ngram_size, cur_len
        )

        for i, banned_tokens in enumerate(banned_batch_tokens):
            scores[i, banned_tokens] = -float("inf")

    return scores


def _add_beam_candidate(
    best_score, 
    best_sequence, 
    batch_size, 
    num_beams, 
    beam_scores, 
    history, 
    eos_token_id=None
):
    last_tokens = history[:, -1]
    for _i in range(batch_size * num_beams):
        if eos_token_id is None or last_tokens[_i] in eos_token_id:
            cur_len = history.shape[-1]
            _score = beam_scores.view(-1)[_i] / cur_len ** args.length_penalty

            batch_id = _i // num_beams

            if not batch_id in best_score or best_score[batch_id] < _score:
                best_score[batch_id] = _score
                best_sequence[batch_id][:cur_len] = history[_i]

            beam_scores.view(-1)[_i] = -float("inf")


@torch.inference_mode()
def beam(model, data_iter, args, eos_token_id=[50256]):
    model.eval()
    total_loss = 0.
    start_time = time.time()

    all_predictions = {}
    for idx, data in enumerate(data_iter):
        data = {key: value for key, value in data.items()}

        _id = data['id'].to(args.device)
        _query = data['query'].to(args.device)
        _query_len = data['query_len'].to(args.device)

        ## local adaptation start.

        ## local adaptation end.


        output = None
        score = None

        batch_size = _id.size(0)
        num_beams = args.beam
        length_penalty = args.length_penalty

        _batch = torch.arange(0, _id.size(0), device=args.device, dtype=torch.long)
        
        kv_cache = None
        len_kv_cache = None

        _query = _query.repeat(1, num_beams).view(batch_size * num_beams, -1)
        _query_len = _query_len.unsqueeze(-1).repeat(1, num_beams).view(-1)

        _bbatch = _batch.unsqueeze(-1).repeat(1, num_beams).view(-1)
        
        # scores for each sentence in the beam
        beam_scores = torch.zeros(
            (batch_size, num_beams), dtype=torch.float, device=_query.device
        )

        best_sequence = torch.zeros(
            (batch_size, args.eval_len), dtype=torch.long, device=_query.device
        )
        best_score = {}

        history = None
        for i in range(0, args.eval_len):
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                if i == 0:
                    logits, kv_cache = model(_query, attention_mask=make_padded_causal_masks(_query_len + i))[0], None
                    logits = logits[:, -1, :] # batch_size * beam, vocab
                else:
                    #print('token_id.shape', token_id.shape, token_id)
                    #print('past.shape', past[0].shape)
                    #print('len_past.shape', len_past.shape, len_past)

                    #logits, kv_cache = model(token_id, past=kv_cache, len_past=len_kv_cache), None
                    logits, kv_cache = model(_query, attention_mask=make_padded_causal_masks(_query_len + i))[0], None
                    logits = logits[:, -1, :]    # batch_size * beam, vocab

            logits = _postprocess_next_token_scores(           
                logits,
                history,
                i,
                batch_size,
                num_beams,
                repetition_penalty=args.repetition_penalty,                                
                no_repeat_ngram_size=args.no_repeat_ngram_size,
                min_length=args.min_length,
                eos_token_id=eos_token_id,
            )

            softmax_probs = F.softmax(logits, dim=-1)
            ##_prob, _w_idx = torch.topk(softmax_probs, num_beams) # batch_size, beam

            vocab_size = softmax_probs.shape[-1] 
            

            _logprob = torch.log(softmax_probs) # batch_size * beam, vocab
            if i == 0:
                next_scores = _logprob.view(batch_size, num_beams, -1)[:, 0, :] # batch_size, vocab
            else:
                next_scores = beam_scores.unsqueeze(-1) + _logprob.view(batch_size, num_beams, -1)
                next_scores = next_scores.view(batch_size, -1) # batch_size, beam * vocab

            next_scores, next_tokens = torch.topk(
                next_scores, num_beams, dim=1, largest=True, sorted=True
            )     # batch_size, num_beams
            
            beam_id = (next_tokens // vocab_size).view(-1)    # batch_size * num_beams
            token_id = (next_tokens % vocab_size).view(-1).unsqueeze(-1) # batch_size, num_beams

            beam_idx = beam_id.view(batch_size, num_beams) + (_batch * num_beams).unsqueeze(-1)
            kv_cache = _reorder_cache(kv_cache, beam_idx.view(-1))                
            beam_scores = next_scores # batch_size, num_beams
            len_kv_cache = (_query_len + i).long()

            if history is None:
                history = token_id.detach()
            else:
                history = torch.cat((history[beam_idx.view(-1)], token_id), dim=1)

            _query = torch.cat((_query[beam_idx.view(-1)], token_id), dim=1)[:, -args.seq_len:]

            _add_beam_candidate(
                best_score, best_sequence, batch_size, num_beams, beam_scores, history, 
                eos_token_id=eos_token_id
            )
        
        _add_beam_candidate(
            best_score, best_sequence, batch_size, num_beams, beam_scores, history
        )

        output = best_sequence

        _id = _id.view(-1).cpu()
        output = output.view(-1, output.shape[-1]).cpu()
        #score = score.view(-1, score.shape[-1]).cpu()

        for _b in range(0, _id.shape[-1]):
            _i = int(_id[_b].item())
            all_predictions[_i] = {}
            all_predictions[_i]['id'] = _i
            all_predictions[_i]['predict'] = output[_b].tolist()
            #all_predictions[_i]['score'] = score[_b].tolist()

            print(sp.decode(data['query'][_b].tolist() + output[_b].tolist()))
            print(flush=True)


def make_padded_causal_masks(query_len, _enabled=False):
    if not _enabled: # global kludge
        return None

    max_len = max(query_len)
    masks = []
    for l in query_len:
        l = l.item()
        mask = ~torch.nn.Transformer.generate_square_subsequent_mask(l).bool()
        bigger_mask = torch.zeros((max_len, max_len), dtype=torch.bool, device=query_len.device)
        bigger_mask[max_len-l:, max_len-l:] = mask
        masks.append(bigger_mask)

    megamask = torch.stack(masks)
    return megamask.unsqueeze(1) # batch_size, 1, max_len, max_len


if __name__ == '__main__':
    args = parser.parse_args()
    
    sp = spm.SentencePieceProcessor(model_file=args.spm)
    
    def tokenize_full(paragraph):
        query = [50256] + sp.encode(paragraph)
        return query
    
    def tokenize(i, paragraph):
        text = paragraph.split("\n")[0]
        query = [50256] + sp.encode(text + "\n")
        return {'query': query, 'query_len': len(query), 'id': i}

    # prefix actual data with this context information instead of padding tokens
    context = filter(None, args.context.read_text().split("\n\n"))
    context_data = torch.cat([torch.LongTensor(tokenize_full(paragraph)) for paragraph in context])
    
    paragraphs = args.data.read_text().split("\n\n")
    valid_data = [tokenize(i, paragraph) for i, paragraph in enumerate(paragraphs)]
    
    def collate(batch):
        x = {
            'query': torch.stack([
                torch.cat((context_data, torch.LongTensor(b['query'])))[-args.seq_len:]
                for b in batch
            ]),
            'query_len': torch.LongTensor([args.seq_len for b in batch]),
            'id': torch.LongTensor([b['id'] for b in batch]),
        }
        return x

    valid_loader = DataLoader(
        valid_data,
        batch_size=args.batch_size, collate_fn=collate,
        num_workers=0, shuffle=False, 
        pin_memory=False, drop_last=False,
    )

    checkpoint = torch.load(args.ckpt_path, map_location=args.device)

    # model
    gptconf = GPTConfig(**checkpoint['model_args'])
    model = GPT(gptconf)

    lora_find_and_replace(model, gpt2_peft_config)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    model.to(args.device)
    #model = torch.compile(model) # CUDA error: an illegal memory access was encountered

    beam(model, valid_loader, args)
