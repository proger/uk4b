import argparse

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

class Replace:
    def __init__(self, error, hyp, ref):
        self.error = error
        self.hyp = hyp
        self.ref = ref

    def __str__(self):
        return f"- корегувати {self.error}: {self.hyp} → {self.ref}"

class Insert:
    def __init__(self, error, ref):
        self.error = error
        self.ref = ref
        
    def __str__(self):
        return f"- додати {self.error}: {self.ref}"

class Delete:
    def __init__(self, error, hyp):
        self.error = error
        self.hyp = hyp
        
    def __str__(self):
        return f"- забрати {self.error}: {self.hyp}"

def flush():
    print("речення:", src)
    print("проаналізуй:")
    if ops:
        for op in ops:
            print(op)
    else:
        print("- чудово")
    ops.clear()
    print("перепиши:", tgt)
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
            srctok  = command[1:].split()
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

            s = srctok[start:end]
            if not s:
                ops.append(Insert(error, t))
            elif not t:
                ops.append(Delete(error, ' '.join(s)))
            else:
                ops.append(Replace(error, ' '.join(s), t))