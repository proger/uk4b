import json
import argparse
from typing import Dict
from pathlib import Path

import smart_open
import ftfy
from tqdm import tqdm
import html2text
from datasets import load_dataset

h = html2text.HTML2Text()
h.ignore_links = True
h.ignore_images = True
h.used = 0


def remove_tags(s: str) -> str:
    """
    Turn html into markdown format
    """
    global h

    if h.used > 1000:
        h = html2text.HTML2Text()
        h.ignore_links = True
        h.ignore_images = True
        h.used = 0
    else:
        h.used += 1

    return h.handle(s).strip()


def process_doc(doc: Dict) -> str:
    """
    Render doc with into the jsonl format suitable for Volodymyr
    :param doc: doc dict from the dataset
    :return:
    """

    return {
        "_id": str(doc.get("id")),
        "text": ftfy.fix_text(remove_tags(doc.get("text", "") or "")),
        "title": ftfy.fix_text(doc.get("title", "") or ""),
        "date_of_publish": doc.get("datetime", ""),
        "tags": [ftfy.fix_text(doc.get("owner", "") or "")],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Export news dataset in the format requested by Volodymyr"
    )
    parser.add_argument("output_file", help="path to input JSONL file", type=Path)
    args = parser.parse_args()

    dataset = load_dataset("zeusfsx/ukrainian-news", split="train", streaming=True)
    with smart_open.open(args.output_file, "wt", encoding="utf-8") as writer:
        for doc in tqdm(dataset, total=10_569_428):
            writer.write(
                json.dumps(process_doc(doc), ensure_ascii=False, sort_keys=True) + "\n"
            )
