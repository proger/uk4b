#!/usr/bin/env python3
"""Evaluate the model output.

Usage:
    evaluate.py <corrected> [--no-tokenize] --m2 <path_to_m2>
    evaluate.py (-h | --help)

Options:
    -h --help           Show this screen.
    --no-tokenize       Do not tokenize the input.
    --layer <layer>     Annotation layer to evaluate: `gec-only` or `gec-fluency`.

<corrected> is the path to the model output. If --no-tokenize is not specified,
the input will be tokenized before evaluation.

"""

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import stanza


def tokenize(text: str) -> [str]:
    if not hasattr(tokenize, "nlp"):
        tokenize.nlp = stanza.Pipeline(lang="uk", processors="tokenize")
    nlp = tokenize.nlp

    tokenized = " ".join([t.text for t in nlp(text).iter_tokens()])
    return tokenized


def tokenize_file(input_file: Path, output_file: Path):
    with open(input_file) as f, open(output_file, "w") as out:
        for line in f:
            line = line.rstrip("\n")
            tokenized = tokenize(line)
            out.write(tokenized + "\n")


def main():
    parser = argparse.ArgumentParser(description="Evaluate the model output.")
    parser.add_argument("corrected", type=str, help="Path to the model output")
    parser.add_argument("--m2", type=str, help="Path to the golden annotated data (.m2 file)",
            required=True)
    parser.add_argument("--no-tokenize", action="store_true", help="Do not tokenize the input")
    args = parser.parse_args()
    tmp = Path(tempfile.gettempdir())

    # Tokenize input
    if args.no_tokenize:
        tokenized_path = args.corrected
    else:
        tokenized_path = tmp / f"{args.corrected}.tok"
        tokenize_file(args.corrected, tokenized_path)
    print(f"Tokenized input: {tokenized_path}", file=sys.stderr)

    # Get the source text out of m2
    source_path = tmp / f"{args.corrected}.src"
    with open(args.m2) as f, open(source_path, "w") as out:
        for line in f:
            if line.startswith("S "):
                out.write(line[2:])

    # Align tokenized input with the original text with Errant
    m2_input = tmp / f"{tokenized_path}.m2"
    subprocess.run(["errant_parallel", "-orig", source_path, "-cor", tokenized_path, "-out", m2_input], check=True)
    print(f"Aligned input: {m2_input}", file=sys.stderr)

    # Evaluate
    subprocess.run(["errant_compare", "-hyp", m2_input, "-ref", args.m2])
    subprocess.run(["errant_compare", "-hyp", m2_input, "-ref", args.m2, "-ds"])


if __name__ == "__main__":
    main()
