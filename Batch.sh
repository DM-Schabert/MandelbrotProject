#!/bin/bash
# ---- LSF job options (DTU HPC) ----
# Submit with `make submit` (or `bsub < batch.sh`) from the project folder.
#BSUB -J mandelbrot              # job name
#BSUB -q hpc                     # queue
#BSUB -n 4                       # cores (overridden by `make submit NP=...`)
#BSUB -R "span[hosts=1]"         # keep all cores on one node
#BSUB -R "rusage[mem=2GB]"       # memory per core
#BSUB -W 00:30                   # wall-time limit (hh:mm)
#BSUB -o mandelbrot_%J.out       # stdout (%J = job ID)
#BSUB -e mandelbrot_%J.err       # stderr

# Run from the folder the job was submitted from
cd "${LS_SUBCWD:-.}"

# ---- Environment (module versions live in env.sh) ----
source env.sh
source .venv/bin/activate
unset PYTHONPATH

# ---- Run ----
mpirun -n $LSB_DJOB_NUMPROC python3 Mandelbrot.py