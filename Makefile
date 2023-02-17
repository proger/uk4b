ifeq ($(shell hostname),rt)
INIT=$(HOME)/gpt/exp/uk4b_medium/ckpt2.096.pt
else ifeq ($(shell hostname),dima-farm)
INIT=/data/gpt2/uk4b/exp/uk4b_small/ckpt.pt
else
INIT=exp/uk4b_medium/ckpt2.096.pt
endif

all:

#
# pretrained
#

exp/uk4b_medium/ckpt2.096.pt:
	mkdir -p exp/uk4b_medium
	curl -o $@ https://a.wilab.org.ua/gpt/uk4b_medium/ckpt2.096.pt

#
# perplexity
#

exp/ppl/scores.tsv:
	python -m score --tsv $(INIT) --sentences data/flair-ppl/bruk.sentences.combined.txt > $@

#
# gec
#

exp/fewshot/fewshot.txt:
	bash data/gec-only/compile_fewshot.sh

exp/gec/%.txt: data/gec-only/%.m2 data/gec-only/%.src.txt data/gec-only/%.tgt.txt
	python -m instruct_tok $^ > $@

exp/gec/%.bin: exp/gec/%.txt
	python -m prepare1 $^ $@

exp/gec/ckpt.pt: exp/gec/train.bin exp/gec/valid.bin
	python -m train --compile=False --train_bin=exp/gec/train.bin --valid_bin=exp/gec/valid.bin --wandb_run_name=gec --ckpt_path=$@ --init=$(INIT)

exp/gec/decode-valid.txt: exp/fewshot/fewshot.txt exp/gec/valid.txt
	python -m beam --beam 4 --batch_size 64 exp/gec/ckpt.pt exp/fewshot/fewshot.txt exp/gec/valid.txt | tee $@

exp/gec/score-valid.txt: exp/gec/valid.txt exp/gec/ckpt.pt
	cat exp/gec/valid.txt | sed 's,/[A-Za-z0+],/_,g' > exp/gec/valid.txt.blank
	python -m score --unblank --seq_len 1024 --lora exp/gec/*.pt --paragraphs exp/gec/valid.txt.blank  > $@


#
# pos
#

%.ark: %.txt
	< $< awk -v OFS='\t' -v c=0 '/анотація:/{printf "%d\t", c; q=substr($$0, 0); while(t = index(q, "/")) {printf "%s ", substr(q, t, 2); q=substr(q,t+2); s=s+t;}; print ""; c += 1;}' > $@

exp/pos/train.bin: data/udpos/train.inline.gpt2.txt
	python -m prepare1 $^ $@

exp/pos/test.bin: data/udpos/test.inline.gpt2.txt
	python -m prepare1 $^ $@

exp/pos/dev.bin: data/udpos/dev.inline.gpt2.txt
	python -m prepare1 $^ $@

exp/pos/ckpt.pt: exp/pos/train.bin exp/pos/dev.bin
	python -m train --compile=False --block_size=512 --batch_size=8 --gradient_accumulation_steps=1 --train_bin=exp/pos/train.bin --valid_bin=exp/pos/dev.bin --wandb_run_name=pos --ckpt_path=$@ --init=$(INIT)

# exp/pos/decode-test.txt: exp/pos/ckpt.pt
# 	python -m beam --beam 2 --batch_size 768 --eval_len 64 --seq_len 128 exp/pos/ckpt.pt data/udpos/train.inline.gpt2.txt data/udpos/test.inline.gpt2.txt | tee $@

exp/pos/decode-test.txt:
	cat data/udpos/test.inline.gpt2.txt | sed 's,/[A-Za-z],/_,g' > data/udpos/test.inline.gpt2.txt.blank
	python -m score --seq_len 512 --unblank --lora exp/pos/ckpt.pt --paragraphs data/udpos/test.inline.gpt2.txt.blank  > $@

exp/pos/WER: data/udpos/test.gpt2.ark exp/pos/decode-test.ark
	compute-wer --mode=strict ark:data/udpos/test.gpt2.ark ark:exp/pos/decode-test.ark > $@

#
# ner
# 

data/ner/train.gpt2.txt: data/flair-ner/fixed-split/train.iob
	PYTHONPATH=data/vulyk-ner/bin python data/ner/convert2gpt2.py $^ $@

# XXX: filtered out some sentences that do not fit into 512 tokens
# data/ner/test.gpt2.txt: data/flair-ner/fixed-split/test.iob
# 	PYTHONPATH=data/vulyk-ner/bin python data/ner/convert2gpt2.py $^ $@

exp/ner/train.bin: data/ner/train.gpt2.txt
	python -m prepare1 $^ $@

exp/ner/valid.bin: data/ner/test.gpt2.txt
	python -m prepare1 $^ $@

exp/ner/ckpt.pt: exp/ner/train.bin exp/ner/valid.bin
	python -m train --compile=False --train_bin=exp/ner/train.bin --valid_bin=exp/ner/valid.bin --wandb_run_name=ner_small --ckpt_path=$@ --init=$(INIT)

exp/ner/decode-test.txt:
	cat data/ner/test.gpt2.txt | sed 's,/[A-Za-z],/_,g' > data/ner/test.gpt2.txt.blank
	python -m score --seq_len 512 --unblank --lora exp/ner/*.pt --paragraphs data/ner/test.gpt2.txt.blank  > $@

exp/ner/score-test.txt:
	cat data/ner/test.gpt2.txt | sed 's,/[A-Za-z],/_,g' > data/ner/test.gpt2.txt.blank
	python -m score --seq_len 512 --lora exp/ner/*.pt --paragraphs data/ner/test.gpt2.txt.blank  > $@


exp/ner/WER: data/ner/test.gpt2.ark exp/ner/decode-test.ark
	compute-wer --mode=strict ark:data/ner/test.gpt2.ark ark:exp/ner/decode-test.ark > $@

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


#
# squad
#

