# saves the dataset to a binary file for training. following was helpful:
# https://github.com/HazyResearch/flash-attention/blob/main/training/src/datamodules/language_modeling_hf.py
# prepare.py in karpathy/nanoGPT

import numpy as np
import sentencepiece as spm
from datasets import load_dataset, Value, Features
from tqdm import tqdm

num_proc = 24
split_dataset = load_dataset(
    "text",
    data_files={
        "train": ["gec-only.train.txt", "gec-fluency.train.txt"],
        "val": ["gec-only.valid.txt", "gec-fluency.valid.txt"]
    },
    sample_by="paragraph")

sp = spm.SentencePieceProcessor(model_file='wiki.model')

class Tok:
    endoftext = 50256
    endofprompt = 1

def process(example):
    text = example['text']
    ids = sp.encode(text)
    ids = ids + [Tok.endoftext]
    out = {'ids': ids, 'len': len(ids), 'text_len': len(text)}
    return out

# tokenize the dataset
tokenized = split_dataset.map(
    process,
    desc="tokenizing",
    remove_columns=['text'],
    num_proc=num_proc,
)

# shuffle documents to avoid back to back paragraphs with different edits
shuffled = tokenized.shuffle(seed=1337)

# concatenate all the ids in each dataset into one large file we can use for training
for split, dset in tokenized.items():
    arr_len = np.sum(dset['len'])
    # preallocate space in a temporary file to store the concatenated ids
    filename = f'gec_{split}_wiki.bin'
    arr = np.memmap(filename, dtype=np.uint16, mode='w+', shape=(arr_len,))
    total_batches = 8
    idx = 0
    
    for batch_idx in tqdm(range(total_batches), desc=f'writing {filename}'):
         # Batch together samples for faster write
         batch = dset.shard(num_shards=total_batches, index=batch_idx, contiguous=True).with_format('numpy')
         arr_batch = np.concatenate(batch['ids'])
         # Write into mmap
         arr[idx : idx + len(arr_batch)] = arr_batch
         idx += len(arr_batch)
    arr.flush()
