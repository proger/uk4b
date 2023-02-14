python data/gec-only/align_fewshot.py data/gec-only/fewshot.m2 data/gec-only/train.src.tok data/gec-only/train.tgt.txt > data/gec-only/fewshot.tgt.txt
python data/gec-only/align_fewshot.py data/gec-only/fewshot.m2 data/gec-only/train.src.tok data/gec-only/train.tgt.tok > data/gec-only/fewshot.tgt.tok
python data/gec-only/align_fewshot.py data/gec-only/fewshot.m2 data/gec-only/train.src.tok data/gec-only/train.src.tok > data/gec-only/fewshot.src.tok
python data/gec-only/align_fewshot.py data/gec-only/fewshot.m2 data/gec-only/train.src.tok data/gec-only/train.src.txt > data/gec-only/fewshot.src.txt
mkdir -p exp/fewshot
python -m instruct_tok data/gec-only/fewshot.m2 data/gec-only/fewshot.src.txt data/gec-only/fewshot.tgt.txt > exp/fewshot/fewshot.txt
python -m prepare1 exp/fewshot/fewshot.txt exp/fewshot/fewshot.bin
stat exp/fewshot/fewshot.bin # tokens is bytes / 2