"""
Mandelbrot MPI program

This program calculates the Mandelbrot set in parallel using MPI.

Usage:
    mpirun -n <number_of_ranks> python3 Mandelbrot.py [mode] [chunk_size] [size] [xlim] [ylim]

Defaults:
    mode       = blocking
    chunk_size = 10
    size       = 1000x1000
    xlim       = -2.2:0.75
    ylim       = -1.3:1.3

Examples:

    # Run with 4 ranks using all default values
    mpirun -n 4 python3 Mandelbrot.py

    # Run with 8 ranks using non-blocking communication
    mpirun -n 8 python3 Mandelbrot.py nonblocking

    # Non-blocking communication with chunk size 20
    mpirun -n 8 python3 Mandelbrot.py nonblocking 20

    # Change image size as well
    mpirun -n 8 python3 Mandelbrot.py nonblocking 20 2000x2000

The rows of the image are divided into chunks and distributed
round-robin between the MPI ranks. Each rank computes its own rows,
after which rank 0 collects the results and saves the final image.

Available communication modes:
    blocking
    nonblocking
"""

import sys
import numpy as np
from mpi4py import MPI

# ============================================================
# Arguments
# ============================================================

def read_arguments():

    # Defaults
    mode = "blocking"
    chunk_size = 10
    size = (1000, 1000)
    xlim = (-2.2, 0.75)
    ylim = (-1.3, 1.3)

    argv = sys.argv[1:]

    if argv:
        mode = argv.pop(0).lower()

    if argv:
        chunk_size = int(argv.pop(0))

    if argv:
        size = tuple(map(int, argv.pop(0).split("x")))

    if argv:
        xlim = tuple(map(float, argv.pop(0).split(":")))

    if argv:
        ylim = tuple(map(float, argv.pop(0).split(":")))

    return mode, chunk_size, size, xlim, ylim

# ============================================================
# Static row distribution
# ============================================================

def assign_rows(x_dim, chunk_size, rank, total_rank):

    rows = np.arange(x_dim)

    # Divide rows into chunks
    chunk_id = rows // chunk_size

    # Round-robin distribution of chunks
    split_x = rows[chunk_id % total_rank == rank]

    return split_x

# ============================================================
# Mandelbrot calculation
# ============================================================

def compute_rows(rows, size, xlim, ylim):

    x_dim, y_dim = size

    xconst = (xlim[1] - xlim[0]) / x_dim
    yconst = (ylim[1] - ylim[0]) / y_dim

    # Only allocate memory for THIS rank's rows
    local_image = np.zeros((len(rows), y_dim))

    for local_x, x in enumerate(rows):

        cx = complex(xlim[0] + x * xconst, 0)

        for y in range(y_dim):

            c = cx + complex(0, ylim[0] + y * yconst)

            z = 0

            for i in range(100):

                z = z * z + c

                if np.abs(z) > 2:
                    local_image[local_x, y] = i
                    break

    return local_image

# ============================================================
# Blocking communication
# ============================================================

def blocking_collect(
    comm,
    local_image,
    local_rows,
    x_dim,
    y_dim,
    chunk_size,
    rank,
    total_rank,
):

    if rank == 0:

        # Rank 0 owns the complete final image
        image = np.zeros((x_dim, y_dim))

        # Insert rank 0's own rows
        image[local_rows, :] = local_image

        # Receive results from all other ranks
        for source in range(1, total_rank):

            source_rows = assign_rows(
                x_dim,
                chunk_size,
                source,
                total_rank,
            )

            buffer = np.empty(
                (len(source_rows), y_dim)
            )

            comm.Recv(
                buffer,
                source=source,
                tag=source,
            )

            image[source_rows, :] = buffer

        return image

    else:

        comm.Send(
            local_image,
            dest=0,
            tag=rank,
        )

        return None

# ============================================================
# Non-blocking communication
# ============================================================

def nonblocking_collect(
    comm,
    local_image,
    local_rows,
    x_dim,
    y_dim,
    chunk_size,
    rank,
    total_rank,
):

    if rank == 0:

        image = np.zeros((x_dim, y_dim))

        image[local_rows, :] = local_image

        buffers = []
        requests = []
        row_lists = []

        for source in range(1, total_rank):

            source_rows = assign_rows(
                x_dim,
                chunk_size,
                source,
                total_rank,
            )

            buffer = np.empty(
                (len(source_rows), y_dim)
            )

            request = comm.Irecv(
                buffer,
                source=source,
                tag=source,
            )

            buffers.append(buffer)
            requests.append(request)
            row_lists.append(source_rows)

        MPI.Request.Waitall(requests)

        for rows, buffer in zip(row_lists, buffers):
            image[rows, :] = buffer

        return image

    else:

        request = comm.Isend(
            local_image,
            dest=0,
            tag=rank,
        )

        request.Wait()

        return None

# ============================================================
# Save figure
# ============================================================

def save_image(image, xlim, ylim, filename="Figure_1.png"):

    # Headless backend -> good for HPC
    import matplotlib
    matplotlib.use("Agg")

    import matplotlib.pyplot as plt

    plt.imshow(
        image.T,
        extent=np.concatenate([xlim, ylim]),
    )

    plt.xlabel(r"x / Re(p_0)")
    plt.ylabel(r"y / Im(p_0)")

    plt.margins(0, 0)

    plt.savefig(
        filename,
        bbox_inches="tight",
        pad_inches=0,
    )

    plt.close()

# ============================================================
# Main program
# ============================================================

def main():

    # MPI setup
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    total_rank = comm.Get_size()

    # Arguments
    mode, chunk_size, size, xlim, ylim = read_arguments()

    size = np.asarray(size)
    xlim = np.asarray(xlim)
    ylim = np.asarray(ylim)

    x_dim, y_dim = size

    # Only rank 0 prints general information
    if rank == 0:
        print(
            f"\nCalculating Mandelbrot\n"
            f"Ranks:       {total_rank}\n"
            f"Chunk size:  {chunk_size}\n"
            f"Image size:  {tuple(size)}\n"
            f"Mode:        {mode}\n"
        )

    # Decide which rows belong to this rank
    local_rows = assign_rows(
        x_dim,
        chunk_size,
        rank,
        total_rank,
    )

    # Make sure everybody is ready before timing
    comm.Barrier()

    start_time = MPI.Wtime()

    # Calculate this rank's part
    local_image = compute_rows(
        local_rows,
        size,
        xlim,
        ylim,
    )

    # Communication
    if mode == "blocking":

        image = blocking_collect(
            comm,
            local_image,
            local_rows,
            x_dim,
            y_dim,
            chunk_size,
            rank,
            total_rank,
        )

    elif mode == "nonblocking":

        image = nonblocking_collect(
            comm,
            local_image,
            local_rows,
            x_dim,
            y_dim,
            chunk_size,
            rank,
            total_rank,
        )

    else:
        raise ValueError(
            "Mode must be 'blocking' or 'nonblocking'"
        )

    # Everybody must finish before timing stops
    comm.Barrier()

    end_time = MPI.Wtime()

    if rank == 0:

        print(
            f"Runtime: {end_time - start_time:.4f} seconds"
        )

        save_image(
            image,
            xlim,
            ylim,
        )

if __name__ == "__main__":
    main()