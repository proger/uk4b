"""
Get a list of m2 entries from a small file on the left,
align with a large file on the right
to extract matching lines from tgt
"""
import sys

left, right, tgt = sys.argv[1:4]

left = open(left)
right = iter(open(right))
tgt = iter(open(tgt))

ri, rline, tline = 1, next(right).strip(), next(tgt).strip()

for line in left:
    if line[:1] != "S":
        continue
    line = line[2:].strip()
    while line != rline:
        ri, rline, tline = ri+1, next(right).strip(), next(tgt).strip()
    print(tline)