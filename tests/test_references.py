"""
Step 6: reference and mapping tests for compute_piece.

1. Compare against an independent, vectorised numpy implementation.
2. Check how pixel indices map to coordinates in the complex plane.
3. Check that the way rows are split up (pieces, order, ranks) never
   changes the result. This is what the MPI version relies on.
"""

import numpy as np
import pytest

from Mandelbrot import Infos, compute_piece


# ============================================================
# Helpers
# ============================================================

def reference_image(infos):
    """Independent vectorised version of the Mandelbrot calculation.

    Same maths as compute_piece, written differently: all pixels at once
    with numpy arrays instead of three Python loops.
    """
    re = infos.xlim[0] + np.arange(infos.x_dim) * infos.xconst
    im = infos.ylim[0] + np.arange(infos.y_dim) * infos.yconst

    # Build c exactly, without any rounding from multiplying by 1j
    c = np.empty((infos.x_dim, infos.y_dim), dtype=complex)
    c.real = re[:, None]
    c.imag = im[None, :]

    z = np.zeros_like(c)
    out = np.full(c.shape, float(infos.max_iter))  # never escaped -> max_iter
    active = np.ones(c.shape, dtype=bool)          # pixels that haven't escaped yet

    for i in range(infos.max_iter):
        za = z[active]
        za = za * za + c[active]
        z[active] = za

        escaped = np.zeros_like(active)
        escaped[active] = np.abs(za) > 2
        out[escaped] = i
        active &= ~escaped

        if not active.any():
            break

    return out


def full_image(infos):
    """compute_piece over every row at once."""
    image = np.zeros(infos.size)
    compute_piece(image, np.arange(infos.x_dim), infos)
    return image


# ============================================================
# 1. Comparison with the reference
# ============================================================

def test_reference_itself_on_known_points():
    # Check the reference independently first, so we're not comparing
    # against something that shares a bug. Pixels land on c = 0, 1, 2, 3.
    infos = Infos(size=(4, 1), xlim=(0.0, 4.0), ylim=(0.0, 1.0))
    assert reference_image(infos)[:, 0].tolist() == [100, 2, 1, 0]


@pytest.mark.parametrize(
    "xlim, ylim, max_iter",
    [
        ((-2.2, 0.75), (-1.3, 1.3), 100),    # the default view
        ((-0.8, -0.7), (0.05, 0.15), 100),   # zoom on the boundary, lots of detail
        ((-2.2, 0.75), (-1.3, 1.3), 30),     # a different iteration limit
    ],
)
def test_matches_reference(xlim, ylim, max_iter):
    # Non-square (50 x 37), so swapped x and y axes would be caught
    infos = Infos(size=(50, 37), xlim=xlim, ylim=ylim, max_iter=max_iter)
    image = full_image(infos)

    assert image.shape == (50, 37)
    np.testing.assert_array_equal(image, reference_image(infos))


# ============================================================
# 2. Pixel -> coordinate mapping
# ============================================================

def test_x_mapping():
    # 4 pixels across xlim = (0, 4): they sit at c = 0, 1, 2, 3.
    # Pixel 0 is at xlim[0]; xlim[1] = 4 itself is not a pixel.
    # Escape values: c=0 never, c=1 at i=2, c=2 at i=1, c=3 at i=0.
    infos = Infos(size=(4, 1), xlim=(0.0, 4.0), ylim=(0.0, 1.0))
    assert full_image(infos)[:, 0].tolist() == [100, 2, 1, 0]


def test_y_mapping():
    # Same idea on the imaginary axis: c = 0, i, 2i, 3i.
    # c=0 and c=i never escape, c=2i at i=1, c=3i at i=0.
    infos = Infos(size=(1, 4), xlim=(0.0, 1.0), ylim=(0.0, 4.0))
    assert full_image(infos)[0, :].tolist() == [100, 100, 1, 0]


def test_first_index_is_real_axis():
    # image[x, y]: the first index runs along the real axis.
    # A 2 x 2 image at c = {0, 3} x {0i, 1i}. The points with real part 3
    # escape at once, the others never do, so a transposed image would
    # put the 0s in the wrong places.
    infos = Infos(size=(2, 2), xlim=(0.0, 6.0), ylim=(0.0, 2.0))
    image = full_image(infos)
    assert image[0, 0] == 100   # c = 0
    assert image[0, 1] == 100   # c = i
    assert image[1, 0] == 0     # c = 3
    assert image[1, 1] == 0     # c = 3 + i


# ============================================================
# 3. Splitting the work never changes the result
# ============================================================

def test_only_touches_its_own_rows():
    infos = Infos(size=(10, 8))
    image = np.full(infos.size, -1.0)
    compute_piece(image, np.array([3, 7]), infos)

    untouched = [r for r in range(10) if r not in (3, 7)]
    assert (image[untouched] == -1).all()
    assert (image[[3, 7]] >= 0).all()


def test_pieces_in_any_order_and_size():
    infos = Infos(size=(40, 25))
    expected = full_image(infos)

    rng = np.random.default_rng(0)
    rows = rng.permutation(infos.x_dim)       # shuffled row order
    image = np.zeros(infos.size)
    start = 0
    while start < len(rows):
        n = int(rng.integers(1, 8))           # random piece size 1..7
        compute_piece(image, rows[start:start + n], infos)
        start += n

    np.testing.assert_array_equal(image, expected)


@pytest.mark.parametrize("schedule", ["block", "chunk"])
@pytest.mark.parametrize("total_rank", [1, 2, 3, 4, 7])
@pytest.mark.parametrize("chunk_size", [1, 3, 10, 100])
def test_split_over_ranks_gives_same_image(schedule, total_rank, chunk_size):
    # Simulate the MPI split serially: each "rank" computes its own pieces
    # into its own empty image, then we combine them. No MPI needed.
    infos = Infos(size=(30, 20), schedule=schedule,
                  total_rank=total_rank, chunk_size=chunk_size)
    expected = full_image(infos)

    combined = np.zeros(infos.size)
    for r in range(total_rank):
        part = np.zeros(infos.size)
        for piece in infos.pieces_of_rank(r):
            compute_piece(part, piece, infos)
        rows = infos.rows_of(r)
        combined[rows] = part[rows]

    np.testing.assert_array_equal(combined, expected)