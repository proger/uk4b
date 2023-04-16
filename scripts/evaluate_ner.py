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

y_true = {}
y_pred = {}
bad = set()

#with open('exp/ner/constrained-test2-gpu1.ark') as f:
#with open('exp/ner/large-constrained-test-gpu1.ark') as f:
with open('exp/ner-newlines/decode-constrained.arknl') as f:
    for line in f:
        sentence_id, seq = line.strip().split("\t")
        try:
            x = [inverse_mapping[tag[1:]] for tag in seq.split()]
        except KeyError:
            print('faulty prediction:', sentence_id, seq, file=sys.stderr)
            x = [inverse_mapping.get(tag[1:], 'X') for tag in seq.split()]
            bad.add(sentence_id)
        y_pred[sentence_id] = x


with open('data/ner/test.gt.ark') as f:
    for line in f:
        sentence_id, seq = line.strip().split("\t")
        
        try:
            x = [inverse_mapping[tag[1:]] for tag in seq.split()]
        except KeyError:
            print('faulty test:', sentence_id, seq, file=sys.stderr)
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
