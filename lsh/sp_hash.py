from typing import List, Set, Tuple, Iterator, TypeVar, Dict, Optional
from pathlib import Path
from collections import namedtuple
import pickle
import argparse
import json
import multiprocessing
from hashlib import sha256
from functools import partial
from itertools import islice

import smart_open
from tqdm import tqdm
import sentencepiece as spm
from datasketch import MinHash, MinHashLSH

T = TypeVar("T")
LSHParam = namedtuple("LSHParam", ["threshold", "num_perm", "shingle_length"])

sp_model: Optional[spm.SentencePieceProcessor] = None


def _handle_xz(file_obj, mode):
    return lzma.LZMAFile(filename=file_obj, mode=mode, format=lzma.FORMAT_XZ)


smart_open.register_compressor(".xz", _handle_xz)


def batch_iterator(iterator: Iterator[T], batch_size: int = 50) -> Iterator[List[T]]:
    """
    Iterates over the given iterator in batches.
    iterator: the iterator to iterate over
    batch_size: the size of the batch
    returns an iterator over batches
    """
    iterator = iter(iterator)
    while True:
        batch = list(islice(iterator, batch_size))
        if not batch:
            return
        yield batch


def tokenize_text(text: str, sp_model) -> List[int]:
    """
    Tokenizes the given text using SentencePiece.
    text: the text to tokenize
    sp_model: the SentencePiece model
    returns a list of tokens

    >>> tokenize_text("привіт, як справи?", sp_model)
    [395, 627, 50096, 524, 5833, 50219]
    """
    return sp_model.encode(text)


def get_shingles(tokens: List[int], shingle_length: int) -> Set[Tuple[int, ...]]:
    """
    Computes a set of shingles from the given list of tokens.
    tokens: the list of tokens
    shingle_length: the length of the shingle
    returns a set of shingles
    >>> get_shingles([1, 2, 3, 4, 5], 2)
    {(1, 2), (2, 3), (3, 4), (4, 5)}

    >>> get_shingles(tokenize_text("привіт, як справи?", sp_model), 3)
    {(395, 627, 50096), (524, 5833, 50219), (627, 50096, 524), (50096, 524, 5833)}
    """
    shingles = set()
    for i in range(len(tokens) - shingle_length + 1):
        shingle = tuple(tokens[i : i + shingle_length])
        shingles.add(shingle)
    return shingles


def create_minhash(shingles: Set[Tuple[int, ...]], num_perm: int) -> MinHash:
    """
    Creates a MinHash of the given set of shingles using the specified number of permutations.
    shingles: the set of shingles
    num_perm: the number of permutations
    returns a MinHash
    """
    m = MinHash(num_perm=num_perm)
    for shingle in shingles:
        m.update(str(shingle).encode("utf8"))
    return m


def process_records(
    record_str: str, shingle_length: int, num_perm: int
) -> Dict[str, str]:
    """
    Computes the MinHash of the text in the given record, and adds it to the LSH index.
    """
    global sp_model

    record = json.loads(record_str)

    text = record["text"]
    id_ = record.get("id", record.get("_id", record.get("id_", None)))

    if id_ is None:
        id_ = sha256(record_str.encode("utf-8")).hexdigest()

    tokens = tokenize_text(text, sp_model)
    minhash = create_minhash(get_shingles(tokens, shingle_length), num_perm)

    return {"id": id_, "minhash": minhash, "tokens": len(tokens)}


def worker_init(
    sp_model_name: str,
) -> None:
    """
    Initializes the worker process.
    sp_model_name: the path to the SentencePiece model
    """
    global sp_model
    sp_model = spm.SentencePieceProcessor(model_file=sp_model_name)


def main(cli_args: argparse.Namespace) -> None:
    """
    Creates an LSH index of shingles from a JSONL file.
    """

    cli_args.output_dir.mkdir(parents=True, exist_ok=True)

    indexes: Dict[LSHParam, MinHashLSH] = {}

    for threshold in cli_args.threshold:
        indexes[
            LSHParam(
                threshold=threshold,
                num_perm=cli_args.num_perm,
                shingle_length=cli_args.shingle_length,
            )
        ] = MinHashLSH(threshold=threshold, num_perm=cli_args.num_perm)

    documents: Dict[str, Dict] = {}

    for input_file in cli_args.input_files:
        print(f"Processing {input_file}, sit tight...")

        with smart_open.open(input_file, "rt", encoding="utf-8") as reader:
            with multiprocessing.Pool(
                processes=cli_args.num_processes,
                initializer=worker_init,
                initargs=(cli_args.sp_model,),
            ) as pool:
                for chunk in batch_iterator(
                    tqdm(reader), batch_size=cli_args.chunk_size
                ):
                    if not chunk:
                        break

                    for record in pool.imap(
                        partial(
                            process_records,
                            shingle_length=cli_args.shingle_length,
                            num_perm=cli_args.num_perm,
                        ),
                        chunk,
                    ):
                        for index in indexes.values():
                            index.insert(record["id"], record["minhash"])
                        documents[record["id"]] = record

    # Write index to output file
    # print("Writing index to output file...")
    # with smart_open.open(cli_args.output_dir, "wb") as fh_out:
    #     pickle.dump(index, fh_out)

    print("Estimating number of unique documents...")
    for params, index in indexes.items():
        total_tokens: int = 0
        filtered_tokens: int = 0
        deduped_docs: List[str] = []
        for id_, doc in documents.items():
            total_tokens += doc["tokens"]
            duplicates = index.query(doc["minhash"])
            first_duplicate = min(duplicates)
            if id_ == first_duplicate:
                filtered_tokens += doc["tokens"]
                deduped_docs.append(id_)

        print(f"Threshold: {params}:")
        print(
            f"Total number of documents {len(documents)}, total number of unique documents " + 
            f"{len(deduped_docs)}, ratio {len(deduped_docs) / len(documents)}"
        )
        print(
            f"Total number of tokens {total_tokens}, total number of unique tokens {filtered_tokens}, " +
            f"ratio {filtered_tokens / total_tokens}"
        )

        with cli_args.output_dir.joinpath(
            f"deduped_threshold-{params.threshold}.num_perm-{params.num_perm}."
            + f"shingle_length-{params.shingle_length}.tokens_left-{filtered_tokens}.txt"
        ).open("w", encoding="utf-8") as fh:
            for doc_id in deduped_docs:
                fh.write(doc_id + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create LSH index of shingles from a JSONL file"
    )
    parser.add_argument(
        "input_files", nargs="+", help="input JSONL files (archives are welcome)"
    )
    parser.add_argument(
        "output_dir",
        help="path to output dir to store ids of deduplicated texts",
        type=Path,
    )
    parser.add_argument("sp_model", help="path to SentencePiece model file")
    parser.add_argument(
        "--shingle_length", type=int, default=3, help="length of shingles"
    )
    parser.add_argument(
        "--num_perm", type=int, default=128, help="number of permutations for MinHash"
    )
    parser.add_argument(
        "--threshold", type=float, default=[0.5], help="threshold for LSH", nargs="*"
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=1000,
        help="number of records to process in each chunk",
    )
    parser.add_argument(
        "--num_processes",
        type=int,
        default=multiprocessing.cpu_count(),
        help="number of processes to use for parallel processing",
    )
    args = parser.parse_args()

    main(args)
