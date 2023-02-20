import sys

from seqeval.metrics import accuracy_score
from seqeval.metrics import classification_report
from seqeval.metrics import f1_score


forward_mapping = {
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
inverse_mapping = {v: k for k, v in forward_mapping.items()}

y_pred = []

with open('exp/ner/decode-test.ark') as f:
    for line in f:
        sentence_id, seq = line.strip().split("\t")
        try:
            x = [inverse_mapping[tag[1:]] for tag in seq.split()]
        except KeyError:
            print('faulty prediction:', sentence_id, seq, file=sys.stderr)
            x = [inverse_mapping.get(tag[1:], 'X') for tag in seq.split()]
        y_pred.append(x)

y_true = []

with open('data/ner/test.gpt2.ark') as f:
    for line in f:
        sentence_id, seq = line.strip().split("\t")
        
        try:
            x = [inverse_mapping[tag[1:]] for tag in seq.split()]
        except KeyError:
            print('faulty test:', sentence_id, seq, file=sys.stderr)
            x = [inverse_mapping.get(tag[1:], 'X') for tag in seq.split()]
        y_true.append(x)
        
assert len(y_true) == len(y_pred)

print('f1', f1_score(y_true, y_pred))
print('accuracy', accuracy_score(y_true, y_pred))
print(classification_report(y_true, y_pred))
