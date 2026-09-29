import sys
import numpy as np
from mpi4py import MPI
from mpi4py.util.dtlib import from_numpy_dtype


_help = f"""\
{sys.argv[0]} [chunk-size] [size widthXheight] [limits xmin:xmax ymin:ymax]

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

print(f"""\
Calculating the Mandelbrot set with these arguments:

{chunk_size = }
{size = }
{xlim = }
{ylim = }
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
# # get shape of imaage
# x_dim, y_dim = np.shape(image)
# # split the array to the ranks: [[1,2,3,4,5][6,7,8,9,10]]
# split_x = np.array_split(np.arange(x_dim), total_rank)[rank]

'''
How the pdf says we should do it (way faster because computation heavy rows are in the middle of image)
'''
x_dim, y_dim = np.shape(image)
rows = np.arange(x_dim)                          
chunk_id = rows // chunk_size                    
split_x = rows[chunk_id % total_rank == rank]
# round robin thing



'''
First part of non Blocking: allocating buffer for each rank in rank 0
'''
if rank == 0:
    bufs = [np.empty_like(image) for _ in range(1, total_rank)]  
        # n buffer that are empty for each rank
    reqs = []
    for r in range(1, total_rank):
        reqs.append(comm.Irecv(bufs[r - 1], source=r, tag=r))
        # open the connection so the process can send their results








'''
Real Computing
'''
t0 = MPI.Wtime()

for x in split_x:
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


t1 = MPI.Wtime()
# print(f"Rank {rank}: {t1 - t0:.2f} s for Rows {split_x[0]}–{split_x[-1]}", flush=True)
print(f"Rank {rank}: {t1 - t0:.2f} s for {len(split_x)} rows", flush=True)



# start = split_x[0]
# end = split_x[- 1] + 1



'''
Blocking sending and recieving
'''
# if rank != 0:
#     comm.Send(image, dest=0, tag=rank)

# else:
#     buf = np.empty_like(image)
#     for x in range(1, total_rank):
#         comm.Recv(buf, source=x, tag=x)
#         image += buf


'''
Second part of non-blocking: Sending and Collecting buffers
'''
if rank != 0:
    req = comm.Isend(image, dest=0, tag=rank)
    req.Wait()

else:
    MPI.Request.Waitall(reqs)
    for b in bufs:
            image += b











if rank == 0:
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
