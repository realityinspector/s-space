PY ?= python3

.PHONY: data train eval loop fake-loop overnight status explorer clean-runs

data:
	$(PY) data/generate.py

train:
	$(PY) -m sspace.train

eval:
	$(PY) -m sspace.eval patches/default.toml

loop:
	$(PY) -m sspace.loop --minutes 30

fake-loop:
	$(PY) -m sspace.loop --fake --iters 60

# all-night autoresearch; caffeinate keeps the Mac awake (safe no-op elsewhere)
overnight:
	caffeinate -i $(PY) -m sspace.overnight --until 07:00 2>&1 | tee -a runs/overnight/run.log

status:
	@cat runs/overnight/STATUS.md

explorer:
	$(PY) explorer/server.py

# only synthetic (--fake) outputs; real runs are never touched
clean-runs:
	rm -f runs/*.fake.*
