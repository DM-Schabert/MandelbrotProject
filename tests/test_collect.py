"""
Steps 9-11: MPI tests for the four communication modes.

These only run under mpirun, for example:

    mpirun -n 3 python -m pytest --with-mpi tests/test_collect.py

Plain `pytest` skips them (that's how pytest-mpi handles @pytest.mark.mpi).
Run them with 1, 2, 3 and 4 ranks; see run_mpi_tests.sh.

Every rank calls run() in every test, and asserts only afterwards, so a
failing assert on one rank can never leave the others waiting.
"""

import numpy as np
import pytest
from mpi4py import MPI

from Mandelbrot import COMM_MODES, SCHEDULES, Infos, compute_piece, run

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


@pytest.fixture(autouse=True)
def keep_ranks_in_step():
    """All ranks start and finish each test together."""
    comm.Barrier()
    yield
    comm.Barrier()


def fill_encoded(image, piece, infos):
    """Instead of computing: write a value that says where it belongs.

    Pixel (row, col) gets row * 1000 + col, so rank 0 can check that every
    value landed in exactly the right place. Needs y_dim < 1000.
    """
    image[piece] = piece[:, None] * 1000 + np.arange(infos.y_dim)[None, :]


def expected_encoded(infos):
    rows = np.arange(infos.x_dim)[:, None]
    cols = np.arange(infos.y_dim)[None, :]
    return (rows * 1000 + cols).astype(float)


def make_infos(x_dim, y_dim, chunk_size, schedule, comm_mode):
    return Infos(size=(x_dim, y_dim), chunk_size=chunk_size,
                 schedule=schedule, comm_mode=comm_mode, total_rank=size)


ALL_COMBINATIONS = pytest.mark.parametrize("comm_mode", COMM_MODES)
ALL_SCHEDULES = pytest.mark.parametrize("schedule", SCHEDULES)


# ============================================================
# Step 10: every value lands in the right place
# ============================================================

@pytest.mark.mpi
@ALL_SCHEDULES
@ALL_COMBINATIONS
@pytest.mark.parametrize(
    "x_dim, y_dim, chunk_size",
    [
        (23, 7, 3),     # nothing divides evenly
        (40, 5, 10),    # everything divides evenly (for 2 and 4 ranks)
        (60, 4, 1),     # chunk_size 1: many pieces (= many tags) per rank
        (2, 5, 1),      # fewer rows than ranks: some ranks have no rows
        (9, 6, 100),    # one chunk bigger than the image: only rank 0 has rows
                        #   with the chunk schedule
    ],
)
def test_every_value_lands_in_the_right_place(comm_mode, schedule, x_dim, y_dim, chunk_size):
    infos = make_infos(x_dim, y_dim, chunk_size, schedule, comm_mode)
    image, _ = run(comm, infos, compute=fill_encoded)

    if rank == 0:
        np.testing.assert_array_equal(image, expected_encoded(infos))


# ============================================================
# Step 11: same image as the serial calculation
# ============================================================

@pytest.fixture(scope="module")
def serial_image():
    """The full image computed serially, once, on every rank."""
    infos = Infos(size=(16, 12))
    image = np.zeros(infos.size)
    compute_piece(image, np.arange(infos.x_dim), infos)
    return image


@pytest.mark.mpi
@ALL_SCHEDULES
@ALL_COMBINATIONS
def test_same_image_as_serial(comm_mode, schedule, serial_image):
    infos = make_infos(16, 12, 3, schedule, comm_mode)
    image, _ = run(comm, infos)

    if rank == 0:
        np.testing.assert_array_equal(image, serial_image)


# ============================================================
# Step 11: what each rank gets back
# ============================================================

@pytest.mark.mpi
@ALL_COMBINATIONS
def test_compute_times_only_on_rank_0(comm_mode):
    infos = make_infos(12, 4, 2, "chunk", comm_mode)
    _, timings = run(comm, infos, compute=fill_encoded)

    if rank == 0:
        assert len(timings["compute_times"]) == size
        assert (timings["compute_times"] >= 0).all()
    else:
        assert timings["compute_times"] is None


@pytest.mark.mpi
@ALL_COMBINATIONS
def test_workers_keep_only_their_own_rows(comm_mode):
    # A worker's image holds its own rows; it never receives anyone else's.
    infos = make_infos(20, 5, 3, "chunk", comm_mode)
    image, _ = run(comm, infos, compute=fill_encoded)

    if rank != 0:
        mine = infos.rows_of(rank)
        others = np.setdiff1d(np.arange(infos.x_dim), mine)
        np.testing.assert_array_equal(image[mine], expected_encoded(infos)[mine])
        assert (image[others] == 0).all()


@pytest.mark.mpi
@ALL_COMBINATIONS
def test_two_runs_in_a_row_dont_mix_up_messages(comm_mode):
    # The second run must not pick up anything left over from the first.
    first = make_infos(15, 4, 2, "chunk", comm_mode)
    second = make_infos(15, 4, 2, "block", comm_mode)

    def fill_second(image, piece, infos):
        fill_encoded(image, piece, infos)
        image[piece] += 0.5

    run(comm, first, compute=fill_encoded)
    image, _ = run(comm, second, compute=fill_second)

    if rank == 0:
        np.testing.assert_array_equal(image, expected_encoded(second) + 0.5)
