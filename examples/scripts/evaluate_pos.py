import sys

from seqeval.metrics import accuracy_score
from seqeval.metrics import classification_report
from seqeval.metrics import f1_score


forward_mapping = dict([
     ("NOUN", "N"),  # 4130481
     ("VERB", "V"),  # 3345193
     ("ADP", "a"),  # 1851693
     ("ADV", "A"),  # 1651200
     ("PRON", "P"),  # 1525969
     ("ADJ", "J"),  # 1427357
     ("PART", "p"),  # 1147072
     ("CCONJ", "C"),  # 1101499
     ("DET", "D"),  # 873070
     ("PROPN", "O"),  # 684675
     ("SCONJ", "S"),  # 484188
     ("X", "X"),  # 175188
     ("NUM", "n"),  # 96248
     ("PUNCT", "t"),  # 88265
     ("INTJ", "I"),  # 61924
     ("SYM", "s"),  # 415
     ("AUX", "x"),  # 275
 ])

inverse_mapping = {v: k for k, v in forward_mapping.items()}

y_pred = {}
bad = set()

with open('exp/pos/decode-test4.ark') as f:
    for line in f:
        sentence_id, seq = line.strip().split("\t")
        try:
            x = [inverse_mapping[tag[1:]] for tag in seq.split()]
        except KeyError:
            print('faulty prediction:', sentence_id, seq, file=sys.stderr)
            x = [inverse_mapping.get(tag[1:], 'X') for tag in seq.split()]
            bad.add(sentence_id)
        y_pred[sentence_id] = x

y_true = {}

with open('data/udpos/test.inline.gpt2.ark') as f:
    for line in f:
        sentence_id, seq = line.strip().split("\t")
        
        try:
            x = [inverse_mapping[tag[1:]] for tag in seq.split()]
        except KeyError as e:
            print('faulty test:', sentence_id, seq, e, file=sys.stderr)
            x = [inverse_mapping.get(tag[1:], 'X') for tag in seq.split()]
            bad.add(sentence_id)
        y_true[sentence_id] = x
        
for sentence_id in bad:
    del y_pred[sentence_id]
    del y_true[sentence_id]
        
assert len(y_true) == len(y_pred)

y_true = list(y_true.values())
y_pred = list(y_pred.values())

print('f1', f1_score(y_true, y_pred))
print('accuracy', accuracy_score(y_true, y_pred))
print(classification_report(y_true, y_pred))
