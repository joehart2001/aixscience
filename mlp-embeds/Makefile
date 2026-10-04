# Train each model on the all75 Boltz2-embedded rows over its four splits
# -> <net>/figs/all75_split_{A,B,C,C2}/ (one run-data dir per split).
# Override the interpreter with:  make all PY=/path/to/python
#
#   make all                                         # every network
#   make make_model direct-network transformer-network   # just those
#   make compare                                     # cross-network test plots
PY ?= /home/wfarmilo/ai-for-science/.venv/bin/python
NETS = direct-network transformer-network squeeze-boost-network boltz-network

# squeeze-boost trains via boost.py so both heads (SE+MLP and SE+XGB) are saved.
TRAIN = $(if $(filter squeeze-boost-network,$@),boost.py,train.py)

.PHONY: all make_model compare clean $(NETS)

all: $(NETS)

# The networks named alongside make_model are goals themselves, so they get
# trained; this target only checks that at least one valid network was named.
make_model:
	@$(if $(filter $(NETS),$(MAKECMDGOALS)),:,echo "usage: make make_model <net> [<net> ...]  (nets: $(NETS))"; exit 1)

$(NETS):
	cd $@ && for t in templates/*.yaml; do $(PY) $(TRAIN) $$t || exit 1; done

# Cross-network comparison on the test set (see figs/README.md).
compare:
	cd figs && $(PY) plots.py templates/all75_split_*.yaml --summary-prefix summary_all75

clean:
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
