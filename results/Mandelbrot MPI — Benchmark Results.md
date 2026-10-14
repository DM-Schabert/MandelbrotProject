
> [!info] Experimental record

> Data collected on DTU HPC, 14 October 2026. The figures below are **medians of three independent repetitions** unless specified otherwise. Time units are seconds; imbalance = maximum rank computation time divided by mean rank computation time (1 = perfectly balanced). All experiments use the original `Mandelbrot.py` and `benchmark.py`.

## Environment and conventions

- Python 3.13.11, Open MPI 5.0.3, Intel Xeon E5-2650 v4 @ 2.20 GHz (as reported by the LSF jobs).

- MPI ranks = processes, **not** physical nodes. The original single-node job allocated 16 slots on host `n-62-21-76`; the two-node job allocated eight slots on each of `n-62-21-75` and `n-62-21-47`.

- Full image is allocated on every process using `np.zeros(infos.size)` (float64, 8 bytes per pixel), so image-array memory alone is approximately `8 × width × height × ranks` bytes; this is an **estimate**, not measured peak RSS.

- The scripts report `t_total`, `compute_max`, `compute_mean`, `imbalance`, and `t_wait`. The `t_wait` measurement is **rank-zero time after its computation until received data are collected**, not a pure network-only timing.

- Strong scaling: $S_p = T_1 / T_p$; efficiency $E_p=100 S_p/p$%. For the later 24-rank job, relative speedup uses 8 ranks as the reference.

## A. Strong scaling — 1 to 16 ranks

**Question:** How does computation time improve as MPI ranks increase, and is blocking or non-blocking faster?

**Parameters:** 2000 × 2000 image; full Mandelbrot view (`xlim = -2.2:0.75`, `ylim = -1.3:1.3`); chunk scheduling; chunk size 10; blocking/non-blocking; ranks 1, 2, 4, 8, 16; 3 repeats; **one node**.

**Source:** `results/scaling_single_29713056.csv` (job `29713056`).

| MPI ranks | Blocking median (s) | Blocking speedup | Blocking efficiency | Blocking imbalance |
| --------- | ------------------- | ---------------- | ------------------- | ------------------ |
| 1         | 161.889             | 1.00×            | 100.0%              | 1.000              |
| 2         | 83.481              | 1.94×            | 97.0%               | 1.025              |
| 4         | 43.019              | 3.76×            | 94.1%               | 1.061              |
| 8         | 21.537              | 7.52×            | 94.0%               | 1.053              |
| 16        | 12.204              | 13.27×           | 82.9%               | 1.182              |

| **MPI ranks** | Non-blocking median (s) | Non-blocking speedup | Non-blocking efficiency | Non-blocking imbalance |
| ------------- | ----------------------- | -------------------- | ----------------------- | ---------------------- |
| 1             | 159.964                 | 1.00×                | 100.0%                  | 1.000                  |
| 2             | 83.067                  | 1.93×                | 96.3%                   | 1.010                  |
| 4             | 43.401                  | 3.69×                | 92.1%                   | 1.016                  |
| 8             | 22.052                  | 7.25×                | 90.7%                   | 1.087                  |
| 16            | 12.357                  | 12.95×               | 80.9%                   | 1.189                  |

**Interpretation:** Blocking achieves **13.27×** speedup at 16 ranks (about **82.9%** efficiency); non-blocking achieves about **12.95×** (about **81.0%**). Neither communication method dominates consistently. The slight slowdown from ideal speedup coincides with increasing load imbalance and overhead.

Blocking speedup for number of ranks $P$ is given by $T_1 / T_P$, while the load imbalance is given by $$I = \frac{\max_{r=0, \dots, P-1} T_r}{\frac{1}{P} \sum_{r=0}^{P-1} T_r},$$
which can be understood as the ratio between the slowest rank's computation time (numerator) and the mean computation time across all ranks (denominator). Say for example, you have ranks 0, 1, 2, 3 with computation times, 10, 12, 11, 25 s. The average is $T_{mean} = \frac{10+12+11+25}{4}=14.5 \text{ s}.$ Therefore $I = \frac{25}{14.5} \approx 1.72 \text{ s}.$ This means the slowest rank takes approximately 72% longer than the average rank.

Finally, the (parallel) efficiency is a measure for how effectively the MPI ranks are utilised compared to an ideal linear speedup: $$E(P) = \frac{S(P)}{P} = \frac{T_1}{P\cdot T_P},$$ where $P$ is the number of ranks used.

## B. Static scheduling — block versus cyclic chunk

**Question:** Does splitting work into smaller round-robin chunks balance irregular Mandelbrot work better than assigning contiguous row blocks?

**Parameters:** 2000 × 2000; 4, 8, 16 ranks; **blocking** communication; chunk size 10 (the block scheduler ignores this value for distribution); 3 repeats; one node.

**Source:** `results/scheduling_single_29713056.csv` (job `29713056`).


| Ranks | Block median (s) | Block imbalance | Chunk median (s) | Chunk imbalance | Block / chunk speed ratio | Runtime reduction |
| ----- | ---------------- | --------------- | ---------------- | --------------- | ------------------------- | ----------------- |
| 4     | 85.885           | 2.102           | 43.906           | 1.054           | 1.96×                     | 48.9%             |
| 8     | 50.975           | 2.518           | 21.117           | 1.051           | 2.41×                     | 58.6%             |
| 16    | 27.111           | 2.605           | 12.723           | 1.197           | 2.13×                     | 53.1%             |

**Interpretation:** Chunk scheduling is about **2×–2.4× faster**, with imbalance ratios close to 1 instead of over 2. Irregular escape-iteration costs make contiguous row blocks uneven; round-robin chunks distribute expensive regions among ranks.

## C. Chunk-size sensitivity

**Question:** How does the static round-robin chunk size affect performance and load imbalance?

**Parameters:** 2000 × 2000; **8 ranks**; blocking communication; chunk scheduling; tested sizes 1, 5, 10, 25, 50, 100; 3 repeats; one node.

**Source:** `results/chunks_single_29713056.csv` (job `29713056`).

| Chunk size | Median runtime (s) | Median imbalance | Increase over size 10 |
| ---------- | ------------------ | ---------------- | --------------------- |
| 1          | 21.956             | 1.073            | +1.9%                 |
| 5          | 21.962             | 1.072            | +1.9%                 |
| 10         | 21.543             | 1.055            | +0.0%                 |
| 25         | 22.532             | 1.087            | +4.6%                 |
| 50         | 22.661             | 1.102            | +5.2%                 |
| 100        | 26.329             | 1.266            | +22.2%                |

**Interpretation:** Chunk size 10 gives the lowest median runtime **among these tested values**. Sizes 1–50 are quite close; size 100 is about 22.2% slower and has higher imbalance. This is static chunk assignment, not on-demand dynamic scheduling. 

## D. One node versus two nodes

**Question:** Does running the same 16-rank job on two compute nodes change performance?

**Parameters:** 5000 × 5000; **16 ranks in both configurations**; chunk scheduling, size 10; blocking/non-blocking; 3 repeats. One-node placement: 16 slots on `n-62-21-76`; two-node placement: 8 slots each on `n-62-21-75` and `n-62-21-47`.

**Sources:** `results/nodes_single_5000_29713056.csv` and `results/nodes_two_5000_29713058.csv`.

| Communication | One-node median (s) | Two-node median (s) | One-node imbalance | Two-node imbalance | Two-node runtime change |
| ------------- | ------------------- | ------------------- | ------------------ | ------------------ | ----------------------- |
| Blocking      | 72.850              | 69.298              | 1.116              | 1.151              | -4.9%                   |
| Nonblocking   | 77.164              | 70.163              | 1.190              | 1.174              | -9.1%                   |

**Interpretation:** Two nodes were slightly faster in this set of measurements, despite introducing inter-node traffic. This is **not evidence that inter-node messages are faster**: the layout also changes ranks per node, contention and possibly hardware conditions. Both configurations reported the same CPU model.

## E. Follow-up scaling — 8, 16, 24 ranks

**Question:** Does scaling continue beyond 16 ranks? Were overlapping configurations comparable to the first job?

**Parameters:** 2000 × 2000; chunk scheduling, size 10; blocking/non-blocking; ranks 8, 16, 24; 3 repetitions; **separate later job** (`29714137`).

**Source:** `results_benchmark1/scaling_24ranks_29714137.csv` (path as supplied).

| Ranks | Blocking median (s) | Blocking median imbalance | Blocking relative speedup vs 8 | Blocking relative efficiency | Non-blocking median (s) | Non-blocking median imbalance | Non-blocking relative speedup vs 8 | Non-blocking relative efficiency |
| ----- | ------------------- | ------------------------- | ------------------------------ | ---------------------------- | ----------------------- | ----------------------------- | ---------------------------------- | -------------------------------- |
| 8     | 19.041              | 1.059                     | 1.00×                          | 100.0%                       | 18.940                  | 1.060                         | 1.00×                              | 100.0%                           |
| 16    | 10.663              | 1.176                     | 1.79×                          | 89.3%                        | 11.064                  | 1.197                         | 1.71×                              | 85.6%                            |
| 24    | 6.982               | 1.135                     | 2.73×                          | 90.9%                        | 6.791                   | 1.115                         | 2.79×                              | 93.0%                            |

**Comparison of repeated rank counts across the two jobs:**

| Ranks | Mode        | Original median (s) | Later median (s) | Later vs original |
| ----- | ----------- | ------------------- | ---------------- | ----------------- |
| 8     | blocking    | 21.537              | 19.041           | -11.6%            |
| 8     | nonblocking | 22.052              | 18.940           | -14.1%            |
| 16    | blocking    | 12.204              | 10.663           | -12.6%            |
| 16    | nonblocking | 12.357              | 11.064           | -10.5%            |

**Interpretation:** The new job gives **2.73× blocking** and **2.79× non-blocking** speedup from 8 to 24 ranks, corresponding to approximately **90.9%** and **92.9%** relative efficiency. Overlapping 8- and 16-rank runs were faster than in the first job, so do **not** combine the original 1-rank time with the later 24-rank time into a single strong-scaling claim without qualifying the different run conditions. The physical-core/thread allocation of the 24-rank job should be checked using its `lscpu` output.

## F. Memory context and limitations

| Image resolution | Memory of one float64 image | 16 rank copies | 24 rank copies |
| ---------------- | --------------------------- | -------------- | -------------- |
| 1000 × 1000      | 8 MB                        | 128 MB         | 192 MB         |
| 2000 × 2000      | 32 MB                       | 512 MB         | 768 MB         |
| 5000 × 5000      | 200 MB                      | 3,200 MB       | 4,800 MB       |
| 10000 × 10000    | 800 MB                      | 12,800 MB      | 19,200 MB      |

- Figures describe **main-image array allocations only**; actual resident memory also includes Python, MPI, buffers, other arrays, and process overhead.

- LSF reported job-level `Max Memory` of **2328 MB** for the single-node multi-experiment job and **3848 MB** for the two-node job, but these are resource accounting figures from entire jobs and should **not be interpreted as controlled per-rank peak-memory measurements** or compared as a clean single-vs-two-node memory experiment.

- The 5000 × 5000 jobs are the only controlled node-placement comparison; the 2000 × 2000 tests answer the scaling/scheduling/chunking questions.

## Main conclusions

- Strong scaling to 16 ranks: approximately **13× speedup** and **81–83% efficiency**.
- **Work distribution matters most:** chunk scheduling reduces load imbalance and roughly halves the runtime compared to the use of contiguous blocks.
- **Blocking vs non-blocking:** differences are small and inconsistent for the tested implementation.
- **Chunk size:** 10 is best among the tested values; very large chunks reduce load balance.
- **Node placement:** two nodes were a little faster for the tested 16-rank, 5000x5000 case, but cause cannot be isolated.
- **24-rank extension:** continued good scaling relative to 8 ranks, with a separate baseline due to different job conditions.
- **Memory:** the entire output array is duplicated per rank; no controlled per-rank RSS experiment was performed.

## Data provenance

| Experiment | Job ID | Output CSV |
|---|---|---|
| A. 1–16 scaling | `29713056` | `scaling_single_29713056.csv` |
| B. Scheduling | `29713056` | `scheduling_single_29713056.csv` |
| C. Chunk sizes | `29713056` | `chunks_single_29713056.csv` |
| D. One node | `29713056` | `nodes_single_5000_29713056.csv` |
| D. Two nodes | `29713058` | `nodes_two_5000_29713058.csv` |
| E. 24-rank extension | `29714137` | `scaling_24ranks_29714137.csv` |

> [!note] Data handling

> All timing and imbalance medians in this note are transcribed/calculated from the job logs and 24-rank CSV pasted in the conversation. For reproducible plotting and any finer analysis (variance, error bars, additional metrics), use the original CSV files.