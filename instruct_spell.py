import argparse
import stanza

parser = argparse.ArgumentParser("convert m2 into instruction format")
parser.add_argument('m2')
parser.add_argument('src')
parser.add_argument('tgt')
args = parser.parse_args()

m2 = open(args.m2)
srcfile = iter(open(args.src))
tgtfile = iter(open(args.tgt))

current_version = "0"
ops = []
skip = False


def tokenize(text: str) -> list[str]:
    if not hasattr(tokenize, "nlp"):
        tokenize.nlp = stanza.Pipeline(lang="uk", processors="tokenize")
    nlp = tokenize.nlp
    return [t for t in nlp(text).iter_tokens()]


class Replace:
    def __init__(self, error, hyp, ref):
        self.error = error
        self.hyp = hyp
        self.ref = ref

    def __str__(self):
        return f"{self.error}: {self.hyp} → {self.ref}"


class Insert:
    def __init__(self, error, ctx, ref):
        self.error = error
        self.ref = ref
        self.ctx = ctx

    def __str__(self):
        return f"- додати {self.error} після {self.ctx}: {self.ref}"


class Delete:
    def __init__(self, error, ctx, hyp):
        self.error = error
        self.hyp = hyp
        self.ctx = ctx

    def __str__(self):
        return f"- забрати {self.error} після {self.ctx}: {self.hyp}"


def flush():
    if ops:
        print("речення:", src)
        for op in ops:
            print(op)
        ops.clear()
        print()
    global current_version
    current_version = "0"


for command in m2:
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
            srctok1 = command[1:].strip().split()
            tgt = next(tgtfile).strip()
            if src[:1] == "#":
                skip = True
        case "A":
            if skip:
                continue
            op, error, t, _, _, version = command.split("|||")
            
            _, start, end = op.split()
            start, end = int(start), int(end)
            if start == -1:
                continue

            if version != current_version:
                flush()
                current_version = version

            if error != "Spelling":
                continue

            toks = srctok[start:end]
            pre = srctok[start-1:start]
            if toks:
                s = src[toks[0].start_char:toks[-1].end_char]
            else:
                s = ""

            if pre:
                ctx = src[pre[0].start_char:pre[-1].end_char]
            else:
                ctx = ""

            if not s:
                ops.append(Insert(error, ctx, t))
            elif not t:
                ops.append(Delete(error, ctx, s))
            else:
                ops.append(Replace(error, s, t))