#!/usr/bin/env python3
"""Evaluate the model output.

Usage:
    evaluate.py <corrected> [--no-tokenize] --m2 <path_to_m2>
    evaluate.py (-h | --help)

Options:
    -h --help           Show this screen.
    --no-tokenize       Do not tokenize the submission
    --layer <layer>     Annotation layer to evaluate: `gec-only` or `gec-fluency`.

<corrected> is the path to the model output. If --no-tokenize is not specified,
the input will be tokenized before evaluation.

"""

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import spacy
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
    parser.add_argument("--no-tokenize", action="store_true", help="Do not tokenize the submission")
    args = parser.parse_args()
    tmp = Path(tempfile.gettempdir())

    # Make sure we have spacy resources downlaoded
    try:
        spacy.load("en")
    except OSError:
        print("Downloading spacy resources...", file=sys.stderr)
        subprocess.run(["python", "-m", "spacy", "download", "en"], check=True)

    # Tokenize corrected file if needed
    if args.no_tokenize:
        tokenized_path = args.corrected
    else:
        print("Tokenizing submission...", file=sys.stderr)
        tokenized_path = tmp / f"unlp.target.tok"
        tokenize_file(args.corrected, tokenized_path)
    print(f"Tokenized: {tokenized_path}", file=sys.stderr)

    # Get the source text out of m2
    source_path = tmp / f"unlp.source.tok"
    with open(args.m2) as f, open(source_path, "w") as out:
        for line in f:
            if line.startswith("S "):
                out.write(line[2:])

    # Align tokenized submission with the original text with Errant
    m2_target = tmp / "unlp.target.m2"
    subprocess.run(["errant_parallel", "-orig", source_path, "-cor", tokenized_path, "-out", m2_target], check=True)
    print(f"Aligned submission: {m2_target}", file=sys.stderr)

    # Evaluate
    subprocess.run(["errant_compare", "-hyp", m2_target, "-ref", args.m2])
    subprocess.run(["errant_compare", "-hyp", m2_target, "-ref", args.m2, "-ds"])


if __name__ == "__main__":
    main()
