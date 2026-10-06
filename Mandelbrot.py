"""
Mandelbrot set in parallel with MPI.

Usage:
    mpirun -n <ranks> python3 Mandelbrot.py [chunk-size] [size WIDTHxHEIGHT]
        [xlim xmin:xmax] [ylim ymin:ymax] [schedule block|chunk]
        [comm blocking|nonblocking|blocking_perchunk|nonblocking_perchunk]

All settings live in one `Infos` object. Its `rows_of(r)` method answers
"which image rows belong to rank r?", so the communication functions only
need `infos` and a rank number instead of a long list of arguments.

Importing this file has no side effects: nothing runs until `main()` is called.
"""

import sys
from dataclasses import dataclass

import numpy as np
from mpi4py import MPI

COMM_MODES = ("blocking", "nonblocking", "blocking_perchunk", "nonblocking_perchunk")
SCHEDULES = ("block", "chunk")


def help_text(prog):
    return f"""\
{prog} [chunk-size] [size widthxheight] [limits xmin:xmax ymin:ymax] [schedule block|chunk] [comm {'|'.join(COMM_MODES)}]

Here are some examples:

Call it with a chunk-size of 10
$ {prog} 10

Call it with a chunk-size of 10 and image size of 100 by 500 pixels
$ {prog} 10 100x500

Call it with a chunk-size of 10 and image size of 100 by 500 pixels
spanning the coordinates x \\in 0.1-0.3 and y \\in 0.2-0.3
$ {prog} 10 100x500 0.1:0.3 0.2:0.3
"""


# ============================================================
# Work distribution
# ============================================================

def block_scheduling_static(x_dim, rank, total_rank):
    """First intuition: one contiguous block of rows per rank."""
    return np.array_split(np.arange(x_dim), total_rank)[rank]


def chunk_scheduling_static(x_dim, chunk_size, rank, total_rank):
    """Chunks of `chunk_size` rows dealt round-robin to the ranks.

    Faster than block scheduling, because the expensive rows in the middle
    of the image get spread over all ranks.
    """
    rows = np.arange(x_dim)
    chunk_id = rows // chunk_size
    return rows[chunk_id % total_rank == rank]


# ============================================================
# Settings
# ============================================================

@dataclass
class Infos:
    """All settings of one run, plus the questions that depend on them."""

    chunk_size: int = 10
    size: tuple = (1000, 1000)
    xlim: tuple = (-2.2, 0.75)
    ylim: tuple = (-1.3, 1.3)
    schedule: str = "chunk"
    comm_mode: str = "nonblocking"
    total_rank: int = 1
    max_iter: int = 100

    @property
    def x_dim(self):
        return self.size[0]

    @property
    def y_dim(self):
        return self.size[1]

    @property
    def xconst(self):
        return (self.xlim[1] - self.xlim[0]) / self.size[0]

    @property
    def yconst(self):
        return (self.ylim[1] - self.ylim[0]) / self.size[1]

    @property
    def per_chunk(self):
        return self.comm_mode.endswith("_perchunk")

    def rows_of(self, r):
        """The image rows that rank `r` computes."""
        if self.schedule == "block":
            return block_scheduling_static(self.x_dim, r, self.total_rank)
        return chunk_scheduling_static(self.x_dim, self.chunk_size, r, self.total_rank)

    def pieces_of(self, rows):
        """Split `rows` into consecutive pieces of at most `chunk_size` rows.

        The *_perchunk modes send one message per piece.
        """
        return [rows[i:i + self.chunk_size] for i in range(0, len(rows), self.chunk_size)]

    def pieces_of_rank(self, r):
        return self.pieces_of(self.rows_of(r))


# ============================================================
# Arguments
# ============================================================

def parse_arguments(argv, total_rank=1):
    """Turn the command-line arguments (without the program name) into Infos.

    Raises ValueError for an unknown schedule or comm mode.
    """
    argv = list(argv)
    infos = Infos(total_rank=total_rank)

    if argv:
        infos.chunk_size = int(argv.pop(0))
    if argv:
        infos.size = tuple(map(int, argv.pop(0).split("x")))
    if argv:
        infos.xlim = tuple(map(float, argv.pop(0).split(":")))
    if argv:
        infos.ylim = tuple(map(float, argv.pop(0).split(":")))
    if argv:
        infos.schedule = argv.pop(0)
    if argv:
        infos.comm_mode = argv.pop(0)

    if infos.comm_mode not in COMM_MODES:
        raise ValueError(f"Unknown comm mode '{infos.comm_mode}', use one of {COMM_MODES}")
    if infos.schedule not in SCHEDULES:
        raise ValueError(f"Unknown schedule '{infos.schedule}', use 'block' or 'chunk'")

    return infos


# ============================================================
# Mandelbrot calculation
# ============================================================

def compute_piece(image, piece, infos):
    """Compute the escape iteration for every pixel in the rows `piece`.

    Writes into `image` (full size, indexed by global row number).
    """
    for x in piece:
        cx = complex(infos.xlim[0] + x * infos.xconst, 0)
        for y in range(infos.y_dim):
            c = cx + complex(0, infos.ylim[0] + y * infos.yconst)
            z = 0
            for i in range(infos.max_iter):
                z = z * z + c
                if np.abs(z) > 2:
                    image[x, y] = i
                    break


# ============================================================
# Communication
# ============================================================

def post_irecvs(comm, infos):
    """Rank 0, nonblocking: post one Irecv per worker, before computing."""
    bufs, reqs = [], []
    for r in range(1, infos.total_rank):
        buf = np.empty((len(infos.rows_of(r)), infos.y_dim))
        reqs.append(comm.Irecv(buf, source=r, tag=r))
        bufs.append(buf)
    return bufs, reqs


def post_irecvs_perchunk(comm, infos):
    """Rank 0, nonblocking_perchunk: one Irecv per piece of every worker.

    The tag is the piece index within that worker.
    """
    bufs, reqs, where = [], [], []
    for r in range(1, infos.total_rank):
        for k, piece in enumerate(infos.pieces_of_rank(r)):
            buf = np.empty((len(piece), infos.y_dim))
            reqs.append(comm.Irecv(buf, source=r, tag=k))
            bufs.append(buf)
            where.append(piece)
    return bufs, reqs, where


def send_piece(comm, image, piece, k, infos):
    """Worker, per-chunk modes: ship piece number `k` to rank 0.

    Returns (request, data) for the nonblocking mode, so the caller can keep
    the data alive until the request completes. Returns (None, None) otherwise.
    """
    data = image[piece]  # fancy indexing makes a copy
    if infos.comm_mode == "blocking_perchunk":
        comm.Send(data, dest=0, tag=k)
        return None, None
    return comm.Isend(data, dest=0, tag=k), data


def collect_blocking(comm, image, rank, infos):
    if rank != 0:
        comm.Send(image[infos.rows_of(rank)], dest=0, tag=rank)
    else:
        for r in range(1, infos.total_rank):
            rows_r = infos.rows_of(r)
            buf = np.empty((len(rows_r), infos.y_dim))
            comm.Recv(buf, source=r, tag=r)
            image[rows_r] = buf


def collect_nonblocking(comm, image, rank, infos, bufs, reqs):
    if rank != 0:
        req = comm.Isend(image[infos.rows_of(rank)], dest=0, tag=rank)
        req.Wait()
    else:
        MPI.Request.Waitall(reqs)
        for r in range(1, infos.total_rank):
            image[infos.rows_of(r)] = bufs[r - 1]


def collect_blocking_perchunk(comm, image, rank, infos):
    """Workers already sent their pieces during the computation."""
    if rank == 0:
        for r in range(1, infos.total_rank):
            for k, piece in enumerate(infos.pieces_of_rank(r)):
                buf = np.empty((len(piece), infos.y_dim))
                comm.Recv(buf, source=r, tag=k)
                image[piece] = buf


def collect_nonblocking_perchunk(comm, image, rank, bufs, reqs, where, send_reqs):
    """Workers already posted their Isends during the computation."""
    if rank != 0:
        MPI.Request.Waitall(send_reqs)
    else:
        MPI.Request.Waitall(reqs)
        for piece, buf in zip(where, bufs):
            image[piece] = buf


# ============================================================
# One full run
# ============================================================

def run(comm, infos):
    """Compute and collect the image. All ranks must call this.

    Returns (image, timings). The image is only complete on rank 0.
    `timings` has t_total, t_wait (rank 0's time after computing until all
    data is in) and compute_times (one per rank, rank 0 only, else None).
    """
    rank = comm.Get_rank()
    image = np.zeros(infos.size)
    my_rows = infos.rows_of(rank)

    comm.Barrier()
    t_start = MPI.Wtime()

    bufs, reqs, where = [], [], []
    if rank == 0 and infos.comm_mode == "nonblocking":
        bufs, reqs = post_irecvs(comm, infos)
    elif rank == 0 and infos.comm_mode == "nonblocking_perchunk":
        bufs, reqs, where = post_irecvs_perchunk(comm, infos)

    send_reqs, send_bufs = [], []
    t_compute = 0.0  # pure computing time, without time spent in Send
    for k, piece in enumerate(infos.pieces_of(my_rows)):
        tp = MPI.Wtime()
        compute_piece(image, piece, infos)
        t_compute += MPI.Wtime() - tp

        # per-chunk modes: workers ship every piece as soon as it is computed
        if infos.per_chunk and rank != 0:
            req, data = send_piece(comm, image, piece, k, infos)
            if req is not None:
                send_reqs.append(req)
                send_bufs.append(data)  # keep the copy alive until Waitall

    t1 = MPI.Wtime()
    print(f"Rank {rank}: {t_compute:.2f} s for {len(my_rows)} rows", flush=True)

    if infos.comm_mode == "blocking":
        collect_blocking(comm, image, rank, infos)
    elif infos.comm_mode == "nonblocking":
        collect_nonblocking(comm, image, rank, infos, bufs, reqs)
    elif infos.comm_mode == "blocking_perchunk":
        collect_blocking_perchunk(comm, image, rank, infos)
    else:
        collect_nonblocking_perchunk(comm, image, rank, bufs, reqs, where, send_reqs)

    t_end = MPI.Wtime()
    compute_times = comm.gather(t_compute, root=0)

    timings = {
        "t_total": t_end - t_start,
        "t_wait": t_end - t1,
        "compute_times": None if compute_times is None else np.array(compute_times),
    }
    return image, timings


def result_line(infos, timings):
    """One CSV line for benchmarking (rank 0 only)."""
    ct = timings["compute_times"]
    return (f"RESULT,{infos.total_rank},{infos.schedule},{infos.comm_mode},"
            f"{infos.chunk_size},{infos.x_dim}x{infos.y_dim},"
            f"{timings['t_total']:.4f},"
            f"{ct.max():.4f},{ct.mean():.4f},{ct.max() / ct.mean():.3f},"
            f"{timings['t_wait']:.4f}")


# ============================================================
# Plotting
# ============================================================

def make_figure(image, infos):
    """Build the figure without pyplot, so no window and no backend are needed."""
    import matplotlib
    from matplotlib.figure import Figure

    with matplotlib.rc_context({"font.size": 10}):
        fig = Figure()
        ax = fig.add_subplot()
        ax.imshow(image.T, extent=np.concatenate([infos.xlim, infos.ylim]))
        ax.set_xlabel(r"x / Re(p_0)")
        ax.set_ylabel(r"y / Im(p_0)")
        ax.margins(0, 0)
    return fig


def save_image(image, infos, filename="Figure_1.png"):
    fig = make_figure(image, infos)
    fig.savefig(filename, bbox_inches="tight", pad_inches=0)


# ============================================================
# Main program
# ============================================================

def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    if any(h in argv for h in ("help", "-h", "-help", "--help")):
        if rank == 0:
            print(help_text(sys.argv[0]))
        sys.exit(0)

    try:
        infos = parse_arguments(argv, total_rank=comm.Get_size())
    except ValueError as err:
        sys.exit(str(err))

    if rank == 0:
        print(f"""\
Calculating the Mandelbrot set with these arguments:

chunk_size = {infos.chunk_size}
size = {infos.size}
xlim = {infos.xlim}
ylim = {infos.ylim}
schedule = '{infos.schedule}'
comm_mode = '{infos.comm_mode}'
""", flush=True)

    image, timings = run(comm, infos)

    if rank == 0:
        print(result_line(infos, timings), flush=True)
        save_image(image, infos, "Figure_1.png")


if __name__ == "__main__":
    main()