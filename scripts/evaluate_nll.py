import argparse
import pandas as pd
import numpy as np

parser = argparse.ArgumentParser('evaluate average NLL and bpc excluding some overlapping sentences')
parser.add_argument('contamination')
parser.add_argument('results_tsv')
args = parser.parse_args()

contamination = pd.read_csv(args.contamination, index_col=0)

df = pd.read_csv(args.results_tsv, sep='\t', header=None, names=('id', 'text', 'ppl', 'sentence_len'), index_col=0)
df = df.drop(index=contamination.index, errors='ignore')

nll = np.log(df.ppl.to_numpy()) * df.sentence_len.to_numpy()
nll2 = nll / np.log2(np.e)

char_len = df.text.str.len().to_numpy()
N = np.sum(char_len)

print('nll_mean', nll.mean())

bpc = nll2.sum() / N
print('bpc', bpc)

#ppc = np.exp(nll.sum()/N)
#print('perplexity_per_character', ppc)
