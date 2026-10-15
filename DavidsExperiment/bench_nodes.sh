#!/bin/bash
# ---- LSF job: image-size sweep at a fixed number of ranks (16) ----
# 1 node vs. multiple nodes: submit the SAME script once per placement,
# the placement is given on the command line:
#   bsub -R "rusage[mem=3GB] span[ptile=16]" < DavidsExperiment/bench_nodes.sh   # 1 node  x 16 ranks
#   bsub -R "rusage[mem=3GB] span[ptile=8]"  < DavidsExperiment/bench_nodes.sh   # 2 nodes x  8 ranks
#   bsub -R "rusage[mem=3GB] span[ptile=4]"  < DavidsExperiment/bench_nodes.sh   # 4 nodes x  4 ranks
# (from the project root; submitting from inside DavidsExperiment/ works too)
#BSUB -J mandel_nodes             # job name
#BSUB -q hpcintro                 # course queue
#BSUB -n 16                       # 16 ranks in every placement
#BSUB -W 01:00                    # wall-time limit, ~20 min expected
#BSUB -o nodes_%J.out             # stdout (%J = job ID)
#BSUB -e nodes_%J.err             # stderr

# Run from the project root, also when submitted from inside DavidsExperiment/
cd "${LS_SUBCWD:-.}"
if [ ! -f Env.sh ] && [ -f ../Env.sh ]; then cd ..; fi
OUTDIR=DavidsExperiment/results
mkdir -p "$OUTDIR"

# ---- Environment (module versions live in Env.sh) ----
source Env.sh
source .venv/bin/activate
unset PYTHONPATH
export MPLBACKEND=Agg            # no display on the compute nodes

# ---- Record where the job ran ----
NODES=$(echo $LSB_HOSTS | tr ' ' '\n' | sort -u | wc -l)
TAG="${NODES}nodes_${LSB_JOBID:-local}"
echo "LSF hosts: $LSB_HOSTS"
echo "Nodes:     $NODES"
echo "CPU model: $(lscpu | grep 'Model name' | sed 's/.*: *//')"
echo "Where the 16 ranks actually run (ranks per host):"
mpirun -n 16 hostname | sort | uniq -c

# ---- Sweep: image size, blocking vs. non-blocking ----
for SIZE in 500x500 1000x1000 2000x2000 4000x4000 6000x6000; do
  python3 benchmark.py --ranks 16 --schedules chunk --modes blocking nonblocking \
      --chunk-sizes 10 --size "$SIZE" --repeats 3 --timeout 900 \
      --out "$OUTDIR/nodes_${SIZE}_${TAG}.csv"
done
