import sys
import numpy as np
from mpi4py import MPI
from mpi4py.util.dtlib import from_numpy_dtype


_help = f"""\
{sys.argv[0]} [chunk-size] [size widthXheight] [limits xmin:xmax ymin:ymax] [schedule block|chunk] [comm blocking|nonblocking|blocking_perchunk|nonblocking_perchunk]

Here are some examples:

Call it with a chunk-size of 10
$ {sys.argv[0]} 10

Call it with a chunk-size of 10 and image size of 100 by 500 pixels
$ {sys.argv[0]} 10 100x500

Call it with a chunk-size of 10 and image size of 100 by 500 pixels
spanning the coordinates x \\in 0.1-0.3 and y \\in 0.2-0.3
$ {sys.argv[0]} 10 100x500 0.1:0.3 0.2:0.3
"""

for h in ("help", "-h", "-help", "--help"):
    if h in sys.argv:
        print(_help)
        sys.exit(0)

# First we define all the defaults, then we let the arguments overwrite
# them.
chunk_size = 10
size = 1000, 1000
xlim = -2.2, 0.75
ylim = -1.3, 1.3
schedule = "chunk"
comm_mode = "nonblocking"


# Now grab the arguments
argv = sys.argv[1:]
if argv:
    chunk_size = int(argv.pop(0))
if argv:
    size = tuple(map(int, argv.pop(0).split("x")))
if argv:
    xlim = tuple(map(float, argv.pop(0).split(":")))
if argv:
    ylim = tuple(map(float, argv.pop(0).split(":")))
if argv:
    schedule = argv.pop(0) 
if argv:
    comm_mode = argv.pop(0) 

comm_modes = ("blocking", "nonblocking", "blocking_perchunk", "nonblocking_perchunk")
if comm_mode not in comm_modes:
    sys.exit(f"Unknown comm mode '{comm_mode}', use one of {comm_modes}")


print(f"""\
Calculating the Mandelbrot set with these arguments:

{chunk_size = }
{size = }
{xlim = }
{ylim = }
{schedule = }
{comm_mode = }
""")

# Convert to numpy arrays, not really needed...
size = np.asarray(size)
xlim = np.asarray(xlim)
ylim = np.asarray(ylim)

# Dimensions of the image
image = np.zeros(size)

xconst = np.diff(xlim)[0] / size[0]
yconst = np.diff(ylim)[0] / size[1]



'''
Task starts here now: 
'''

# MPI things:
comm = MPI.COMM_WORLD
# Get rank in the `comm` communicator.
rank = comm.Get_rank()
# Get total number of MPI ranks in the `comm` communicator
total_rank = comm.Get_size()



'''
First Intuition of dividing the work
'''
def block_scheduling_static(x_dim, rank, total_rank):
    # split the array to the ranks: [[1,2,3,4,5][6,7,8,9,10]]
    split_x = np.array_split(np.arange(x_dim), total_rank)[rank]
    return split_x

'''
How the pdf says we should do it (way faster because computation heavy rows are in the middle of image)
'''
def chunk_scheduling_static(x_dim, chunk_size, rank, total_rank):
    rows = np.arange(x_dim)                          
    chunk_id = rows // chunk_size                    
    split_x = rows[chunk_id % total_rank == rank]
    # round robin thing
    return split_x




x_dim, y_dim = np.shape(image)
if schedule not in ("block", "chunk"):
    sys.exit(f"Unknown schedule '{schedule}', use 'block' or 'chunk'")


def rows_of(r):
    if schedule == "block":
        return block_scheduling_static(x_dim, r, total_rank)
    return chunk_scheduling_static(x_dim, chunk_size, r, total_rank)


split_x = rows_of(rank)


def pieces_of(rows):
    # consecutive pieces of at most chunk_size rows; the *_perchunk modes send one message per piece
    return [rows[i:i + chunk_size] for i in range(0, len(rows), chunk_size)]

'''
First part of non Blocking: allocating buffer for each rank in rank 0
'''
def post_irecvs(total_rank):
    bufs = [np.empty((len(rows_of(r)), y_dim)) for r in range(1, total_rank)]
    reqs = []
    for r in range(1, total_rank):
        reqs.append(comm.Irecv(bufs[r - 1], source=r, tag=r))
    return bufs, reqs


def post_irecvs_perchunk(total_rank):
    # one buffer + Irecv per piece of every worker, tag = piece index of that worker
    bufs, reqs, where = [], [], []
    for r in range(1, total_rank):
        for k, piece in enumerate(pieces_of(rows_of(r))):
            buf = np.empty((len(piece), y_dim))
            reqs.append(comm.Irecv(buf, source=r, tag=k))
            bufs.append(buf)
            where.append(piece)
    return bufs, reqs, where


comm.Barrier()
t_start = MPI.Wtime()

bufs, reqs, where = [], [], []
if comm_mode == "nonblocking" and rank == 0:
    bufs, reqs = post_irecvs(total_rank)
elif comm_mode == "nonblocking_perchunk" and rank == 0:
    bufs, reqs, where = post_irecvs_perchunk(total_rank)

'''
Real Computing
'''
per_chunk = comm_mode.endswith("_perchunk")
send_reqs, send_bufs = [], []
t_compute = 0.0  # pure computing time, without time spent in Send
t0 = MPI.Wtime()

for k, piece in enumerate(pieces_of(split_x)):
    tp = MPI.Wtime()
    for x in piece:
    # now only over your part of the x and with communication
        cx = complex(xlim[0] + x * xconst, 0)
        for y in range(size[1]):
            # process (x, y)
            c = cx + complex(0, ylim[0] + y * yconst)
            z = 0
            for i in range(100):
                z = z*z + c
                if np.abs(z) > 2:
                    image[x, y] = i
                    break
    t_compute += MPI.Wtime() - tp

    # per-chunk modes: workers ship every piece as soon as it is computed
    if per_chunk and rank != 0:
        piece_rows = image[piece]
        if comm_mode == "blocking_perchunk":
            comm.Send(piece_rows, dest=0, tag=k)
        else:
            send_reqs.append(comm.Isend(piece_rows, dest=0, tag=k))
            send_bufs.append(piece_rows)  # keep the copy alive until Waitall


t1 = MPI.Wtime()
# print(f"Rank {rank}: {t1 - t0:.2f} s for Rows {split_x[0]}–{split_x[-1]}", flush=True)
print(f"Rank {rank}: {t_compute:.2f} s for {len(split_x)} rows", flush=True)



# start = split_x[0]
# end = split_x[- 1] + 1



'''
Blocking sending and recieving
'''
def collect_blocking(image, rank, total_rank):
    if rank != 0:
        comm.Send(image[split_x], dest=0, tag=rank)
    else:
        for r in range(1, total_rank):
            rows_r = rows_of(r)
            buf = np.empty((len(rows_r), y_dim))
            comm.Recv(buf, source=r, tag=r)
            image[rows_r] = buf



'''
Second part of non-blocking: Sending and Collecting buffers
'''
def collect_nonblocking(image, rank, bufs, reqs):
    if rank != 0:
        my_rows = image[split_x]
        req = comm.Isend(my_rows, dest=0, tag=rank)
        req.Wait()
    else:
        MPI.Request.Waitall(reqs)
        for r in range(1, total_rank):
            image[rows_of(r)] = bufs[r - 1]


'''
Per-chunk variants: workers already sent their pieces during the computation
'''
def collect_blocking_perchunk(image, rank, total_rank):
    if rank == 0:
        for r in range(1, total_rank):
            for k, piece in enumerate(pieces_of(rows_of(r))):
                buf = np.empty((len(piece), y_dim))
                comm.Recv(buf, source=r, tag=k)
                image[piece] = buf


def collect_nonblocking_perchunk(image, rank, bufs, reqs, where, send_reqs):
    if rank != 0:
        MPI.Request.Waitall(send_reqs)
    else:
        MPI.Request.Waitall(reqs)
        for piece, buf in zip(where, bufs):
            image[piece] = buf


if comm_mode == "blocking":
    collect_blocking(image, rank, total_rank)
elif comm_mode == "nonblocking":
    collect_nonblocking(image, rank, bufs, reqs)
elif comm_mode == "blocking_perchunk":
    collect_blocking_perchunk(image, rank, total_rank)
else:
    collect_nonblocking_perchunk(image, rank, bufs, reqs, where, send_reqs)

t_end = MPI.Wtime()


compute_times = comm.gather(t_compute, root=0)

if rank == 0:
    ct = np.array(compute_times)
    # last column: time rank 0 spends after its own computing until all data is in
    print(f"RESULT,{total_rank},{schedule},{comm_mode},{chunk_size},"
          f"{size[0]}x{size[1]},{t_end - t_start:.4f},"
          f"{ct.max():.4f},{ct.mean():.4f},{ct.max() / ct.mean():.3f},"
          f"{t_end - t1:.4f}", flush=True)
    import matplotlib.pyplot as plt
    # Increase font-size
    plt.rcParams.update({
        "font.size": 10,
    })
    plt.imshow(image.T, extent=np.concatenate([xlim, ylim]))
    plt.xlabel(r"x / Re(p_0)")
    plt.ylabel(r"y / Im(p_0)")

    # Just minimize white-space around the actual plot...
    plt.margins(0, 0)
    plt.savefig("Figure_1.png", bbox_inches="tight", pad_inches=0)
    plt.show()
