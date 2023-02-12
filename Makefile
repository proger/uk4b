# INIT=$(HOME)/gpt/exp/uk4b_medium/ckpt.pt
INIT=/data/gpt2/uk4b/exp/uk4b_small/ckpt.pt

all: pos_train_wiki.bin pos_valid_wiki.bin gec_train_wiki.bin gec_valid_wiki.bin

#
# gec
#

gec_train_wiki.bin gec_valid_wiki.bin: gec-fluency.train.txt gec-only.train.txt gec-fluency.valid.txt gec-only.valid.txt
	python -m prepare --name gec --train gec-fluency.train.txt gec-only.train.txt --valid gec-fluency.valid.txt gec-only.valid.txt

gec-only_train_wiki.bin gec-only_valid_wiki.bin: gec-only.train.txt gec-only.valid.txt
	python -m prepare --name gec-only --train gec-only.train.txt --valid gec-only.valid.txt

gec-only.%.txt: data/gec-only/%.m2 data/gec-only/%.src.txt data/gec-only/%.tgt.txt
	python -m instruct $^ > $@

gec-fluency.%.txt: data/gec-fluency/%.m2 data/gec-fluency/%.src.txt data/gec-fluency/%.tgt.txt
	python -m instruct $^ > $@

exp/gec_medium/ckpt.pt: gec_train_wiki.bin gec_valid_wiki.bin
	python -m train --compile=False --train_bin=gec_train_wiki.bin --valid_bin=gec_valid_wiki.bin --wand_run_name=gec --ckpt_path=$@ --init=$(INIT)

exp/gec-only_medium/ckpt.pt: gec-only_train_wiki.bin gec-only_valid_wiki.bin
	python -m train --compile=False --train_bin=gec-only_train_wiki.bin --valid_bin=gec-only_valid_wiki.bin --wand_run_name=gec --ckpt_path=$@ --init=$(INIT)

#
# pos
#

pos_train_wiki.bin pos_valid_wiki.bin:
	python -m prepare --name pos --train data/udpos/train.gpt2.txt --valid data/udpos/dev.gpt2.txt

exp/pos_medium/ckpt.pt: pos_train_wiki.bin pos_valid_wiki.bin
	python -m train --compile=False --train_bin=pos_train_wiki.bin --valid_bin=pos_valid_wiki.bin --wandb_run_name=pos --ckpt_path=$@ --init=$(INIT)

#
# ner
# 

exp/ner/train.bin: data/ner/train.gpt2.txt
	python -m prepare1 $^ $@

exp/ner/valid.bin: data/ner/test.gpt2.txt
	python -m prepare1 $^ $@

exp/ner/ckpt.pt: exp/ner/train.bin exp/ner/valid.bin
	python -m train --compile=False --train_bin=exp/ner/train.bin --valid_bin=exp/ner/valid.bin --wandb_run_name=ner_small --ckpt_path=$@ --init=$(INIT)

#
#
# spelling
#

exp/spell/train.bin: exp/spell/train.txt exp/spell/fluency-train.txt
	python -m prepare1 $^ $@

exp/spell/valid.bin: exp/spell/valid.txt exp/spell/fluency-valid.txt
	python -m prepare1 $^ $@

exp/spell/fluency-%.txt: data/gec-fluency/%.m2 data/gec-fluency/%.src.txt data/gec-fluency/%.tgt.txt
	python -m instruct_spell $^ > $@

exp/spell/%.txt: data/gec-only/%.m2 data/gec-only/%.src.txt data/gec-only/%.tgt.txt
	python -m instruct_spell $^ > $@

exp/spell/ckpt.pt: exp/spell/train.bin exp/spell/valid.bin
	python -m train --compile=False --train_bin=exp/spell/train.bin --valid_bin=exp/spell/valid.bin --wandb_run_name=spell --ckpt_path=$@ --init=$(INIT)

exp/spell/decode-valid.txt: exp/spell/ckpt.pt exp/spell/valid.txt
	python -m beam $^ | tee $@

spell: exp/spell/ckpt.pt