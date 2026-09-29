# Week 05 — Project 1 kick-off: the Mandelbrot set

Condensed notes for **02616 Large-scale Modelling**, week 05.

Source material (all in `Week05/`):

- `Week05/README.md` — the week's framing
- `Week05/week05.pdf` — the lecture deck (3 pages: title + a two-slide **recap of week 04**)
- `Week05/Project.pdf` — the Project 1 assignment text (LaTeX source: `Week05/Project.tex`)
- `Week05/Mandelbrot.py` — the shipped serial reference implementation
- `Week05/Figure_1.png` — the figure reproduced in `Project.pdf`

There is no `Week05/Lectures/` and no `Week05/Labs/` folder this week, and no
`INDEX.md`. Not condensed: `Week05/mandelbrot_static.py`,
`Week05/mandelbrot_static.sh`, `Week05/test_mandelbrot.py`,
`Week05/Figure_static.png` — teacher-side worked material, deliberately excluded
so that no solution is handed to a student through these notes.

## 1. Overview

> `Week05/README.md`: *"This week contains no learning objectives. But is the
> introduction to the first project."*

**Week 05 introduces no new theory.** It is the kick-off of **Project 1 —
Mandelbrot**, the first time the student meets *a fully functional parallel
program* end-to-end rather than an isolated exercise. Its purpose is to force the
material of weeks 01–04 into a single, self-directed piece of work.

Three things happen this week:

1. **A recap of week 04** (`week05.pdf`, slides 2–3) — `mpi4py` point-to-point,
   buffers, non-blocking, `Status`, collectives. Nothing new; see
   `Week04/AI_CONDENSED.md` for the full treatment.
2. **The project is handed out** (`Project.pdf` + `Mandelbrot.py`) — a
   compute-bound problem *with load imbalance*, to be parallelised with MPI.
3. **The student writes a project plan** and **presents it in plenum for
   discussion and guidelines** (`Week05/README.md`; root `README.md`, week 05).
   The plan — deciding *what to investigate* — is itself part of the exercise.

`Week05/README.md` states the governing expectation:

> *"It is important that the student will use the prior weeks learning objectives
> to investigate parallelisation techniques/methods."*

**Target competence:** turn a serial numerical program into an MPI program,
choose and justify a work-distribution strategy, measure it honestly, and
*explain* the measurements in a written report.

## 2. Prerequisites

Everything in this week is applied prior knowledge. Nothing below is re-taught in
`Week05/`; each item is the tool the project expects to be picked up.

| Needed for the project | Where it was taught |
|---|---|
| Distributed vs. shared memory, why message passing; the six fundamental MPI calls; rank/size/communicator | `Week01/AI_CONDENSED.md` §3.3–3.6 |
| LSF batch jobs on the DTU cluster — `#BSUB -n`, `span[hosts=1]`, `span[ptile=…]`, queue `hpcintro`, walltime, `bsub`/`bjobs`/`bkill`; never run on the login node | `Week01/AI_CONDENSED.md` §3.8 |
| Compute-bound vs. memory-bound; bandwidth and latency; the cost of the network; single-node vs. multi-node | `Week02/AI_CONDENSED.md` §3.4, §3.7 |
| Rank placement and binding — `--map-by ppr:n:<resource>`, `--bind-to`, and *deliberate over-allocation for benchmarking* | `Week02/AI_CONDENSED.md` §3.8–3.9 |
| Hand-built scatter/gather patterns, process overhead, and how to time a parallel program correctly | `Week03/AI_CONDENSED.md` §3.6–3.9 |
| `mpi4py`: `Send`/`Recv` on numpy buffers vs. pickled `send`/`recv`; `Isend`/`Irecv` + `Wait`/`Test`; `MPI.Status` (`Get_count`, `.tag`, `.source`); `Bcast`/`Scatter`/`Gather`/`Reduce` | `Week04/AI_CONDENSED.md` §3.4–3.8, recapped on `week05.pdf` slides 2–3 |

Python side: `numpy` arrays and `matplotlib` (both used by `Mandelbrot.py`),
complex arithmetic, and reading `sys.argv`.

## 3. Detailed learning objectives and concepts

### 3.1 The recap slides — `week05.pdf` (slides 2–3)

The deck is titled *"recap week 4 — mpi4py — Short Introduction"*. It restates,
with no new content:

- `Recv` takes a **maximum** buffer size — a receive is an upper bound, not an
  exact amount.
- Prefer `Send` over `send`: **`Send`/`Recv` are for pre-allocated numpy
  buffers**, **`send`/`recv` are for Python objects (slow, lots of overhead)**.
- The **buffer-argument spellings**, from most to least implicit:
  ```python
  comm.Send(buf, …)                  # auto dtype + count
  comm.Send([buf, 1], …)             # auto dtype
  comm.Send([buf, MPI.INT], …)       # auto count
  comm.Send([buf, 1, MPI.INT], …)    # explicit
  ```
- **Non-blocking**: `req = comm.Isend(buf, …)`; do work that does *not* touch
  `buf`; `req.Wait()` — *"Remember to always post a Wait/Test for all requests"*.
- **`Status`**: `comm.Recv(…, status=status)` → `status.Get_count(MPI.DOUBLE)`,
  `status.tag`, `status.source`. (`Get_count` is how you learn how much of the
  buffer was actually filled — directly relevant to a dynamic scheduler.)
- **Collectives**, with the slide's own before→after sketches:
  `comm.Bcast(buf, root=0)`, `comm.Scatter(buf_send, buf_recv, root=0)`,
  `comm.Gather(buf_send, buf_recv, root=0)`,
  `comm.Reduce(buf_send, buf_recv, op=MPI.MIN, root=0)`.

For an agent: **do not treat these slides as new objectives.** They are a
checklist of the primitives the project is expected to be built from.

### 3.2 The problem — what the Mandelbrot set is (`Project.pdf`)

For each pixel, iterate a **truncation function**:

```
p_0   = x + i·y
p_n+1 = p_n² + p_0
```

where `x`, `y` are the image coordinates and `i` the imaginary unit. The
iteration is **truncated when `|p_n| > 2` or `n > n_max`**. The stored pixel
value is **`n`** — the iteration count at which it escaped (or a colour palette
derived from it).

Default (outer-most) view:

```
x ∈ [-2.2, 0.75]
y ∈ [-1.3,  1.3]
```

`Figure_1.png` in `Project.pdf` shows this default view.

**The key structural property, stated by the assignment:**

> *"The Mandelbrot example is a case of a **compute-bound** problem with
> **load-imbalance**."*

The cost of a pixel is the number of iterations before escape. Points far outside
the set escape after a couple of iterations; points **inside** the set never
escape and pay the full `n_max`. Cost therefore varies by orders of magnitude
*across the image*, and the variation is spatially clustered — which is exactly
why a naive equal-pixels-per-rank split does not give equal time-per-rank.
`Project.pdf` points at the CLI limits as the instrument for exploring this:

> *"With the current flags you can investigate several different regions of the
> image and thus scour its inherent load imbalance problems."*

### 3.3 The shipped code — `Week05/Mandelbrot.py`

A ~90-line **serial** program. The task is to *extend it*, not replace it:

> *"Please extend the shipped `Mandelbrot.py` code to use MPI and if additional
> arguments are necessary, **add them to the end of the argument list**."*

**Command-line interface** (positional, each optional, in this order):

```
Mandelbrot.py [chunk-size] [size widthXheight] [limits xmin:xmax ymin:ymax]
```

with a `help`/`-h`/`-help`/`--help` early exit. Examples from the file's own
help text:

```bash
Mandelbrot.py 10                          # chunk-size 10
Mandelbrot.py 10 100x500                  # + image 100 × 500 px
Mandelbrot.py 10 100x500 0.1:0.3 0.2:0.3  # + x ∈ [0.1,0.3], y ∈ [0.2,0.3]
```

**Defaults in the code:**

```python
chunk_size = 10
size  = 1000, 1000
xlim  = -2.2, 0.75
ylim  = -1.3, 1.3
```

**The computation:**

```python
image = np.zeros(size)
xconst = np.diff(xlim)[0] / size[0]
yconst = np.diff(ylim)[0] / size[1]

for x in range(size[0]):
    cx = complex(xlim[0] + x * xconst, 0)
    for y in range(size[1]):
        c = cx + complex(0, ylim[0] + y * yconst)
        z = 0
        for i in range(100):
            z = z*z + c
            if np.abs(z) > 2:
                image[x, y] = i
                break
```

Facts an agent should have straight about this loop:

- **`n_max = 100`** is hard-coded as `range(100)`; it is not a CLI argument.
- The **outer loop is over `x`**, i.e. over `size[0]`, i.e. over the **first index
  of `image`**. `Project.pdf` calls this the *row* direction — see §3.4.
- Pixels that **never escape keep the initial value `0`** (they are never
  assigned) — the same value as a pixel escaping at `i == 0`. This is the shipped
  behaviour, not a bug to be reported as a finding.
- The pixel work is **pure Python scalar arithmetic** (`complex`, `np.abs`), so
  the program is genuinely compute-bound in the Python interpreter.
- **`chunk_size` is parsed but never used** in the shipped code. It is a
  deliberate placeholder — see the assignment's instruction in §3.4.

**The output stage** (after the loop): imports `matplotlib.pyplot`, sets
font-size, `plt.imshow(image.T, extent=…)` — note the **transpose**, because the
first index is `x` — labels the axes `x / Re(p_0)` and `y / Im(p_0)`, saves
`Figure_1.png` and calls `plt.show()`. In an MPI version this stage concerns a
single rank only, and `plt.show()` is not meaningful inside a batch job; the
assignment does not legislate this, it is a consequence of the code as shipped.

### 3.4 What must be built — the two required implementations

`Project.pdf`, *Expected investigations*. **At least two MPI implementations:**

1. **a blocking send/receive code**
2. **a non-blocking send/receive code**

Constraint on both:

> *"In both cases they should be implemented using a **row-only distribution**,
> i.e. the image is updated along the rows (x coordinate) of the array. You may
> later investigate/describe other work distributions."*

So: the unit of work is a set of `x` values (whole rows of `image`), never a
2-D tile — at least for the required pair. Other decompositions are permitted
*in addition*, or as discussion.

> *"Use the `chunk-size` variable in the shipped code to create an initial
> load-distribution algorithm."*

`chunk_size` is the intended knob: the number of rows handed out at a time. It is
the single parameter that connects the CLI to the scheduling strategy, and it is
what makes the granularity of the distribution measurable.

### 3.5 What must be reported — the four required surveys

The report must contain **at least** these (`Project.pdf`):

1. **Load-balancing strategies — discuss, implement, analyse.**
   The assignment names two archetypes:
   - a **static scheduler** — each rank gets a pre-determined chunk-segment;
   - a **dynamic scheduler** — work is **requested** by the workers.

   > *"There is a lot of freedom in choosing how to distribute work, **remember**
   > to describe the distribution in common language."*

   And, in its own quote block:

   > *"We prefer **deep analysis of a few distributions** vs. a weak analysis on
   > many distributions! **This is not a programming competition!**"*

   Explicit escape hatch: *"If there is not enough time for implementing a
   dynamic scheduling, discuss how one could imagine it being implemented."*

2. **A performance benchmark of the different implementations** — e.g.
   scalability plots, possibly across load-balancing strategies.

   > *"**Explain why you get the plots you get!**"*

3. **A benchmark of 1 node vs. multiple nodes.** With the assignment's own hint:

   > *"the **size of the image**, i.e. the number of pixels, has an influence on
   > different performance aspects here!"*

   (This is where `Week02/AI_CONDENSED.md` §3.7–3.9 — network cost, and rank
   placement/`span[ptile=…]` — becomes load-bearing: the image is the thing that
   crosses the network at gather time, while the iteration count is not.)

4. **Code-progress and changes** — shown as **small code snippets in the
   report**, each tied to how it **influenced the performance, whether it
   worsened or improved it**. Regressions are expected content, not something to
   hide.

### 3.6 How the report is judged — the *Notes* section

Verbatim expectations from `Project.pdf` (an agent should read these as the
grading rubric in disguise):

- **"Labels on figures, and explanatory figure captions!"** — stated twice, the
  second time *"(purposefully duplicated!)"*.
- **Ensure your lines on graphs are distinguishable.**
- **Consider what you are timing, and what you compare.** (cf.
  `Week03/AI_CONDENSED.md` §3.9 on the anatomy of a timed program.)
- Write so that **fellow students not taking the course** can understand it.
- **Explain *why* you see what you see.** The assignment's own counter-example:

  > *"if a scalability plot that flattens has a reason, what's the reason?
  > Saying ``The method scales up to 4 cores.'' is not a reason! :)"*

- **Plan the experiments before running them.** The assignment does the
  arithmetic explicitly: *N* core configurations × *P* parameters × ~4 min per
  run = **4·N·P minutes**. *"You only have a limited amount of time to conduct
  experiments, AND you also have to write the report. Planning is key here!"*
  This is the direct justification for the project plan presented in plenum
  (`Week05/README.md`).
- **"Use AI responsibly! Making mistakes is a *good* thing. Let us see what
  *you* can write!"**

### 3.7 Deliverables and deadline

From the box at the top of `Project.pdf`:

> *"Your report (PDF) and the required sources (ZIP) must be handed in
> electronically!"*
> *"Deadline: latest on **Sunday, October 18, 2026 at midnight**!"*

Upload consists of:

- the **report in PDF** format;
- a **ZIP** of the Python sources needed to run the code — *"remember to tell us
  how to execute it"*.

### 3.8 Cross-references — which prior week answers which project question

| Project question | Prior material |
|---|---|
| How do I send a row/chunk of `image` without paying pickle? | `Week04/AI_CONDENSED.md` §3.5 — `Send`/`Recv` on numpy buffers |
| How does a worker say *"how much did I just receive?"* | `Week04/AI_CONDENSED.md` §3.6 — `MPI.Status`, `Get_count`, `.source`, `.tag` (`.source`/`.tag` are how a master identifies *who* asked for work in a dynamic scheduler) |
| How do I overlap the gather with the next chunk of compute? | `Week04/AI_CONDENSED.md` §3.7 — `Isend`/`Irecv`, `Wait`/`Test`, and the request-freeing pitfall |
| Should I use a collective instead of point-to-point? | `Week04/AI_CONDENSED.md` §3.8 — `Scatter`/`Gather` fit a *static* distribution; a dynamic one cannot be a collective, since the ranks do not know in advance who gets what |
| Why does the multi-node run differ from the single-node run? | `Week02/AI_CONDENSED.md` §3.7 network hierarchy, §3.4 compute- vs. memory-bound |
| How do I ask LSF for 1 node vs. several, and pin ranks? | `Week01/AI_CONDENSED.md` §3.8 (`#BSUB -n`, `span[hosts=1]`) and `Week02/AI_CONDENSED.md` §3.9 (`span[ptile=…]`, `--map-by ppr:…`, `--bind-to`) |
| What am I actually timing? | `Week03/AI_CONDENSED.md` §3.8–3.10 |
| What would this have looked like without MPI? | `Week03/AI_CONDENSED.md` — `multiprocessing`, single-node only; the contrast is legitimate report material |

### 3.9 Guidance for an AI agent assisting on this project

The course is explicit that the student must remain the author
(`Project.pdf`: *"Use AI responsibly! … Let us see what you can write!"*; root
`AGENTS.md`: *"Act as a tutor. … Ensure the human in the loop, and try not to be
the AI in the loop."*).

Consequences for an assisting agent:

- **Do not write the MPI implementation for the student.** Discuss the choice
  between static and dynamic distribution, ask what `chunk_size` should mean in
  their design, let them write it.
- **Do not produce the report text or the plots.** The stated goal is that the
  student's own explanation is visible.
- **A failing or slower implementation is a valid result** — the assignment
  *asks* for changes that worsened performance. Do not "fix" it away; help the
  student explain it.
- **Push on "why"**, since that is the stated grading criterion: a flattening
  scalability curve needs a mechanism (imbalance, gather cost, Amdahl-type serial
  fraction, network), not a description.
- **Scope discipline**: prefer helping the student analyse *two* distributions
  deeply over generating five.
