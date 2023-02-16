import sys
from typing import List
import argparse
import re

from convert2vulyk import reconstruct_tokenized


def convert_sentence(sentence: List[str], prefix_text: str = "речення: ", no_tags_text: str = "Ніц нема") -> str:
    tokens: List[str] = []
    ner_tokens: List[str] = []

    ner_token_accum: List[str] = []
    ner_token_type: str = ""

    for line in sentence:
        w, tag = line.split(" ")
        tokens.append(w)

        if tag == "O" or tag.startswith("B-"):
            if ner_token_accum:
                ner_tokens.append(f"{ner_token_type}: {''.join(map(str, reconstruct_tokenized([ner_token_accum])))}")
                ner_token_accum = []

        if tag.startswith("B-"):
            ner_token_accum.append(w)
            ner_token_type = tag.replace("B-", "")

        if tag.startswith("I-"):
            ner_token_accum.append(w)

    final_sentence: str = "".join(map(str, reconstruct_tokenized([tokens])))
    if ner_tokens:
        return prefix_text + final_sentence + "\n" + '\n'.join(ner_tokens)
    else:
        return prefix_text + final_sentence + "\n" + no_tags_text


def convert_sentence_inline(sentence: List[str], prefix_text: str = "", annotation: str = "анотація:") -> str:
    tokens: List[str] = []
    ner_tokens: List[str] = []

    mapping = {
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

    for line in sentence:
        w, tag = line.split(" ")
        tokens.append(w)
        ner_tokens.append(w)
        ner_tokens.append("/" + mapping[tag])

    final_sentence: str = "".join(map(str, reconstruct_tokenized([tokens])))
    final_tagged_sentence: str = " ".join(map(str, reconstruct_tokenized([ner_tokens])))
    final_tagged_sentence = re.sub(r'\s+', ' ', final_tagged_sentence)
    return prefix_text + final_sentence + "\n" + annotation + " " + final_tagged_sentence


if __name__ == "__main__":
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="Convert fixed-split dataset (IOB) prepared for the training of classifiers to"
        "the prompt format, suitable for the GPT2 eval. Output goes to stdout by default"
    )

    parser.add_argument("infile", nargs="?", type=argparse.FileType("r"), default=sys.stdin)
    parser.add_argument("--format", default="inline", choices=["inline", "post"])
    parser.add_argument("outfile", nargs="?", type=argparse.FileType("w"), default=sys.stdout)

    args: argparse.Namespace = parser.parse_args()

    accum: List[str] = []

    for line in map(str.strip, args.infile):
        if not line.strip():
            if accum:
                if args.format == "inline":
                    args.outfile.write(convert_sentence_inline(accum) + "\n\n")
                else:
                    args.outfile.write(convert_sentence(accum) + "\n\n")
                accum = []
        else:
            accum.append(line)
