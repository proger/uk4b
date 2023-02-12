all: pos_train_wiki.bin pos_valid_wiki.bin gec_train_wiki.bin gec_valid_wiki.bin

gec_train_wiki.bin gec_valid_wiki.bin: gec-fluency.train.txt gec-only.train.txt gec-fluency.valid.txt gec-only.valid.txt
	python -m prepare --name gec --train gec-fluency.train.txt gec-only.train.txt --valid gec-fluency.valid.txt gec-only.valid.txt

gec-only.%.txt: data/gec-only/%.m2 data/gec-only/%.src.txt data/gec-only/%.tgt.txt
	python -m instruct $^ > $@

gec-fluency.%.txt: data/gec-fluency/%.m2 data/gec-fluency/%.src.txt data/gec-fluency/%.tgt.txt
	python -m instruct $^ > $@

pos_train_wiki.bin pos_valid_wiki.bin:
	python -m prepare --name pos --train data/udpos/train.gpt2.txt --valid data/udpos/dev.gpt2.txt

INIT=$(HOME)/gpt/exp/uk4b_medium/ckpt.pt

exp/gec_medium/ckpt.pt: gec_train_wiki.bin gec_valid_wiki.bin
	python -m train --compile=False --train_bin=gec_train_wiki.bin --valid_bin=gec_valid_wiki.bin --wand_run_name=gec --ckpt_path=$@ --init=$(INIT)

exp/pos_medium/ckpt.pt: pos_train_wiki.bin pos_valid_wiki.bin
	python -m train --compile=False --train_bin=pos_train_wiki.bin --valid_bin=pos_valid_wiki.bin --wandb_run_name=pos --ckpt_path=$@ --init=$(INIT)
