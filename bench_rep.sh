#!/bin/bash
# ---- LSF job options (DTU HPC) ----
# Benchmark of all four communication modes, everything on ONE node so that
# every run sees the same hardware.
# Submit from the project folder with:   bsub < bench_rep.sh
#BSUB -J mandelbrot_bench        # job name
#BSUB -q hpcintro                # course queue
#BSUB -n 16                      # reserve 16 cores, runs use 1..16 of them
#BSUB -R "span[hosts=1]"         # all cores on one node
#BSUB -R "rusage[mem=2GB]"       # memory per core
#BSUB -W 01:00                   # wall-time limit (hh:mm), ~25 min expected
#BSUB -o bench_%J.out            # stdout (%J = job ID)
#BSUB -e bench_%J.err            # stderr
# Optional: pin a CPU model so repeated jobs are comparable (pick one from `nodestat`)
##BSUB -R "select[model == <CPU model>]"

# Run from the folder the job was submitted from
cd "${LS_SUBCWD:-.}"

# ---- Environment (module versions live in Env.sh) ----
source Env.sh
source .venv/bin/activate
unset PYTHONPATH
export MPLBACKEND=Agg            # no display on the compute nodes

# ---- Benchmark parameters ----
REPS=3
RANKS="1 2 4 8 16"
MODES="blocking nonblocking blocking_perchunk nonblocking_perchunk"
ARGS="10 5000x5000 -2.2:0.75 -1.3:1.3 chunk"   # chunk-size size xlim ylim schedule
OUT="results_cluster_${LSB_JOBID:-local}.csv"
PROGRESS="progress_${LSB_JOBID:-local}.txt"   # one line per started run, follow with `tail -f`
TOTAL=$(( REPS * $(echo $RANKS | wc -w) * $(echo $MODES | wc -w) ))
i=0

# Record where the job ran, needed to describe the hardware in the report
echo "Host:      $(hostname)"
echo "CPU model: $(lscpu | grep 'Model name' | sed 's/.*: *//')"
echo "LSF hosts: $LSB_HOSTS"
echo "Results:   $OUT"
echo "Progress:  $PROGRESS"

# ---- Run: rep, then ranks, then modes (modes interleaved so drift hits all equally) ----
for rep in $(seq 1 $REPS); do
  for n in $RANKS; do
    for cm in $MODES; do
      i=$((i + 1))
      echo "$(date +%H:%M:%S)  run $i/$TOTAL  rep=$rep n=$n mode=$cm" >> "$PROGRESS"
      out=$(mpirun -n $n python3 Mandelbrot.py $ARGS $cm 2>&1)
      line=$(echo "$out" | grep RESULT)
      if [ -z "$line" ]; then
        echo "FAILED: rep=$rep n=$n mode=$cm" >&2
        echo "$out" >&2
      else
        echo "$rep,$line" | tee -a "$OUT"
      fi
    done
  done
done
