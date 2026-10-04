# Train each model over its four splits -> that network's figs/.
# Override the interpreter with:  make all PY=/path/to/python
PY ?= /home/wfarmilo/ai-for-science/.venv/bin/python
NETS = direct-network transformer-network squeeze-boost-network boltz-network

.PHONY: all compare clean $(NETS)

all: $(NETS)

$(NETS):
	cd $@ && $(PY) compare.py templates/*.yaml

# Cross-network comparison on the test set (see figs/README.md).
compare:
	cd figs && $(PY) plots.py templates/*.yaml --summary-prefix summary_full

clean:
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
