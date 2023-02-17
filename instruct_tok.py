import argparse
import stanza
from collections import defaultdict
import sys

parser = argparse.ArgumentParser("convert m2 into instruction format")
parser.add_argument('m2')
parser.add_argument('src')
parser.add_argument('tgt')
args = parser.parse_args()

#      11 G/Aspect
#     221 G/Case
#       7 G/Comparison
#      39 G/Conjunction
#      37 G/Gender
#      56 G/Number
#       7 G/Other
#       2 G/Participle
#       7 G/Particle
#       8 G/PartVoice
#      94 G/Prep
#      50 G/Tense
#     152 G/UngrammaticalStructure
#      15 G/VerbAForm
#      35 G/VerbVoice
#    1549 noop
#      76 Other
#    1483 Punctuation
#     923 Spelling
# replace each key with a single-letter code:
error_codes = {
    'G/Aspect': 'A',
    'G/Case': 'C',
    'G/Comparison': 'O',
    'G/Conjunction': 'J',
    'G/Gender': 'G',
    'G/Number': 'N',
    'G/Other': 'o',
    'G/Participle': 'P',
    'G/Particle': 'i',
    'G/PartVoice': 'p',
    'G/Prep': 'r',
    'G/Tense': 'T',
    'G/UngrammaticalStructure': 'U',
    'G/VerbAForm': 'V',
    'G/VerbVoice': 'v',
    'Other': '0',
    'Punctuation': ',',
    'Spelling': 'S',
}

srcfile = iter(open(args.src))
tgtfile = iter(open(args.tgt))

current_version = "0"
ops = []
skip = False
tokens = defaultdict(list)
insertions = {}


def tokenize(text: str) -> list[str]:
    if not hasattr(tokenize, "nlp"):
        #tokenize.nlp = stanza.Pipeline(lang="uk")
        tokenize.nlp = stanza.Pipeline(lang="uk", processors="tokenize")
    nlp = tokenize.nlp
    return [t for t in nlp(text).iter_tokens()]

def reset_tokens():
    global tokens, insertions
    tokens = defaultdict(list)
    for i, t in enumerate(srctok):
        tokens[i].append(t)
    insertions = {}


def flush():
    global tokens
    print("речення:", src)
    for i in tokens:
        if i in insertions:
            (loc, kind, target) = insertions[i]
            print("\t", kind, target)
        match tokens[i]:
            case [tok, (loc, kind, target)]:
                #pos = '|'.join(word.pos for word in tok.words)
                #print(tok.text, pos, kind, target or "#")
                print(tok.text, '/'+kind, target or "\t")
            case [tok]:
                #pos = '|'.join(word.pos for word in tok.words)
                #print(tok.text, pos, '_', '_')
                print(tok.text, '/+', "\t")

    print(chr(12) + ":", tgt) # form feed
    reset_tokens()

    print()
    global current_version
    current_version = "0"


for command in open(args.m2):
    command = command.rstrip()
    match command[:1]:
        case "":
            current_version = "0"
            if not skip:
                flush()
            else:
                skip = False

        case "S":
            src = next(srcfile).strip()
            srctok = tokenize(src)
            tgt = next(tgtfile).strip()
            if len(srctok) != len(command[1:].strip().split()):
                print("skipping sentence due to tokenizer mismatch", src, file=sys.stderr)

            if src[:1] == "#":
                skip = True

            reset_tokens()


        case "A":
            if skip:
                continue
            
            op, error, t, _, _, version = command.split("|||")
            
            _, start, end = op.split()
            start, end = int(start), int(end)
            if start == -1:
                assert error == "noop"
                continue
            
            if version != current_version:
                flush()
                current_version = version

            error = error_codes[error]

            if start == end:
                # insert
                insertions[start] = (op, error, t)
            else:
                # replace or delete
                for i in range(start, end):
                    if i == end-1:
                        tokens[i].append((op, error, t))
                    else:
                        tokens[i].append((op, error, "@"))