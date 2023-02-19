import argparse
import pandas as pd
import numpy as np

parser = argparse.ArgumentParser('evaluate average NLL and bpc excluding some overlapping sentences')
parser.add_argument('--intersect')
parser.add_argument('contamination')
parser.add_argument('results_tsv')
args = parser.parse_args()

contamination = pd.read_csv(args.contamination, index_col=0)

df = pd.read_csv(args.results_tsv, sep='\t', header=0, names=('id', 'text', 'ppl', 'sentence_len'), index_col=0)
df = df.dropna()
df = df.drop(index=contamination.index, errors='ignore')
df = df.drop_duplicates()

if args.intersect:
    idf = pd.read_csv(args.intersect, sep='\t', header=0, names=('id', 'text', 'ppl', 'sentence_len'), index_col=0)
    idf = idf.dropna()
    idf = idf.drop_duplicates()
    idf = idf.drop(index=contamination.index, errors='ignore')

    df = df.loc[df.index.intersection(idf.index)]

nll = np.log(df.ppl.to_numpy()) * df.sentence_len.to_numpy()
nll2 = nll / np.log2(np.e)

char_len = df.text.str.len().to_numpy()
N = np.sum(char_len)

print(args.results_tsv, 'sentences', len(df))
print(args.results_tsv, 'nll_mean', nll.mean())

bpc = nll2.sum() / N
print(args.results_tsv, 'bpc', bpc)

#ppc = np.exp(nll.sum()/N)
#print('perplexity_per_character', ppc)
