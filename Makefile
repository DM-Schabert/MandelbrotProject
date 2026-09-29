# Mandelbrot MPI project - DTU HPC
#
# First time:   make setup
# Then:         make submit          (batch job, 4 processes)
#               make submit NP=8     (batch job, 8 processes)
#
# Run `make` with no arguments to see all commands.

SHELL := /bin/bash
.ONESHELL:
.SHELLFLAGS := -e -o pipefail -c

NP     ?= 4
SCRIPT ?= Mandelbrot.py
VENV   := .venv

# Loads the modules and activates the project's virtual environment
ACTIVATE := source Env.sh && source $(VENV)/bin/activate && unset PYTHONPATH

.PHONY: help setup test run submit status clean distclean

help:
	@echo "Mandelbrot MPI project - available commands:"
	@echo "  make setup      Create .venv and install requirements (run once)"
	@echo "  make test       Check that MPI + mpi4py work (NP processes)"
	@echo "  make run        Run $(SCRIPT) directly, e.g. on a linuxsh node"
	@echo "  make submit     Submit batch.sh to the queue with NP cores"
	@echo "  make status     Show your queued/running jobs"
	@echo "  make clean      Remove job output/error files"
	@echo "  make distclean  Also remove the virtual environment"
	@echo ""
	@echo "Change the number of processes with NP, e.g. make submit NP=8 (default $(NP))"

setup: $(VENV)/.installed

# Rebuilt automatically if requirements.txt changes
$(VENV)/.installed: requirements.txt Env.sh
	source Env.sh
	python3 -m venv $(VENV)
	source $(VENV)/bin/activate
	unset PYTHONPATH
	pip install --upgrade pip
	# Build mpi4py from source so it links against DTU's MPI module
	pip install --no-binary=mpi4py -r requirements.txt
	touch $@

test: setup
	$(ACTIVATE)
	mpirun -n $(NP) python3 -c "from mpi4py import MPI; c = MPI.COMM_WORLD; print(f'rank {c.Get_rank()} of {c.Get_size()}')"

run: setup
	$(ACTIVATE)
	mpirun -n $(NP) python3 $(SCRIPT)

submit: setup
	bsub -n $(NP) < batch.sh

status:
	bstat

clean:
	rm -f mandelbrot_*.out mandelbrot_*.err

distclean: clean
	rm -rf $(VENV)