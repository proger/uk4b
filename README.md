# unlp-2023-shared-task
UNLP 2023 Shared Task in Grammatical Error Correction for Ukrainian


## Installation

Install the requirements:

```bash
pip install -r requirements.txt
```


## Evaluation

Example:

```bash
scripts/evaluate.py model-output.txt --m2 ./data/gec-only/valid.m2
```

If your output is already tokenized, add the `--no-tokenize` switch to the
command above.


