all: gec-fluency.train.txt gec-only.train.txt gec-fluency.valid.txt gec-only.valid.txt
	python -m prepare_gec

gec-only.%.txt: data/gec-only/%.m2 data/gec-only/%.src.txt data/gec-only/%.tgt.txt
	python -m instruct $^ > $@

gec-fluency.%.txt: data/gec-fluency/%.m2 data/gec-fluency/%.src.txt data/gec-fluency/%.tgt.txt
	python -m instruct $^ > $@
