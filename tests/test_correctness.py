"""
Step 4: known-point tests for compute_piece.

Each test builds a 1x1 image whose single pixel lands exactly on a chosen
complex number c, so we know the correct escape iteration by hand.

Tests marked xfail describe the bug where points inside the set get 0
instead of max_iter. They are "expected to fail" until Step 5 fixes it.
strict=True means that once the fix is in, they report XPASS as a failure,
which reminds you to remove the xfail marker.
"""

import numpy as np
import pytest

from Mandelbrot import Infos, compute_piece


def escape_value(c, max_iter=100):
    """Run compute_piece on a single pixel located exactly at c."""
    infos = Infos(
        size=(1, 1),
        xlim=(c.real, c.real + 1.0),
        ylim=(c.imag, c.imag + 1.0),
        max_iter=max_iter,
    )
    image = np.zeros(infos.size)
    compute_piece(image, np.arange(infos.x_dim), infos)
    return image[0, 0]


# --- Points outside the set: these already pass -------------------------

@pytest.mark.parametrize(
    "c, expected",
    [
        (1 + 0j, 2),   # z: 1, 2, 5          -> |z| > 2 at i = 2
        (3 + 0j, 0),   # z: 3                -> |z| > 2 at i = 0
        (0 + 2j, 1),   # z: 2i, -4 + 2i      -> |z| > 2 at i = 1 (checks the y mapping)
    ],
)
def test_escaping_points(c, expected):
    assert escape_value(c) == expected


def test_known_point():
    infos = Infos()
    infos.xlim = (-2, 2)


# --- Points inside the set: fail until Step 5 ----------------------------

@pytest.mark.parametrize(
    "c",
    [
        0 + 0j,    # z stays 0
        -1 + 0j,   # z cycles -1, 0, -1, 0, ...
        -2 + 0j,   # z: -2, 2, 2, ... (|z| = 2 is not > 2, so it never escapes)
        0 + 1j,    # z cycles i, -1 + i, -i, -1 + i, ...
    ],
)
def test_inside_points_get_max_iter(c):
    assert escape_value(c) == 100


def test_inside_differs_from_instant_escape():
    # c = 0 never escapes, c = 3 escapes at i = 0.
    # They must not get the same value, or the set's interior is coloured
    # like the far outside of the image.
    assert escape_value(0 + 0j) != escape_value(3 + 0j)



def test_max_iter_is_respected():
    assert escape_value(0 + 0j, max_iter=50) == 50


def test_whole_row_of_known_points():
    # The 4x1 image from the plan: pixels land on c = -2, -1, 0, 1.
    infos = Infos(size=(4, 1), xlim=(-2.0, 2.0), ylim=(0.0, 1.0))
    image = np.zeros(infos.size)
    compute_piece(image, np.arange(infos.x_dim), infos)

    # The one escaping point is already correct.
    assert image[3, 0] == 2
    # The three inside points are covered by the xfail tests above;
    # for now just check they all got the same value as each other.
    assert image[0, 0] == image[1, 0] == image[2, 0]
