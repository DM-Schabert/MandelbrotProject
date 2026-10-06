"""
Step 7: tests for make_figure and save_image.

Tests marked PLOT_BUG describe the two current problems:
  - the y-axis is drawn upside down relative to its labels;
  - the axis labels aren't math text, so "p_0" shows literally.
They are xfail(strict=True): once fixed, they report XPASS as a failure,
which reminds you to remove the marker.
"""

import numpy as np
import pytest
from matplotlib.backends.backend_agg import FigureCanvasAgg

from Mandelbrot import Infos, make_figure, save_image

# Asymmetric ylim, so an upside-down axis can't hide behind the symmetry
# of the Mandelbrot set.
INFOS = Infos(size=(40, 30), xlim=(-2.0, 1.0), ylim=(-0.5, 1.5))


def first_image(fig):
    """The AxesImage drawn by imshow on the main axes."""
    return fig.axes[0].images[0]


def colour_at(fig, x, y):
    """Render the figure and return the RGB colour at data point (x, y)."""
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    pixels = np.asarray(canvas.buffer_rgba())
    ax = fig.axes[0]
    px, py = ax.transData.transform((x, y))   # display coords, origin bottom-left
    row = pixels.shape[0] - 1 - int(round(py))  # image rows start at the top
    col = int(round(px))
    return pixels[row, col, :3].astype(int)


# ============================================================
# What is plotted
# ============================================================

def test_plots_the_transposed_image():
    image = np.arange(40 * 30, dtype=float).reshape(INFOS.size)
    fig = make_figure(image, INFOS)
    np.testing.assert_array_equal(np.asarray(first_image(fig).get_array()), image.T)


def test_extent_matches_limits():
    fig = make_figure(np.zeros(INFOS.size), INFOS)
    assert tuple(first_image(fig).get_extent()) == (-2.0, 1.0, -0.5, 1.5)


# ============================================================
# Orientation
# ============================================================


def test_origin_is_lower():
    fig = make_figure(np.zeros(INFOS.size), INFOS)
    assert first_image(fig).origin == "lower"



def test_low_y_is_drawn_at_the_bottom():
    # Behavioural check that doesn't care how the fix is done:
    # value 0 in the lower half of y, value 1 in the upper half.
    image = np.zeros(INFOS.size)
    image[:, INFOS.y_dim // 2:] = 1.0
    fig = make_figure(image, INFOS)

    low = colour_at(fig, -0.5, -0.4)    # near ylim[0], should show value 0
    high = colour_at(fig, -0.5, 1.4)    # near ylim[1], should show value 1

    # The default colour map (viridis) goes from dark purple (0) to yellow (1),
    # so the green channel is much higher for 1 than for 0.
    assert low[1] < 100 < high[1]


# ============================================================
# Labels
# ============================================================


def test_labels_use_math_text():
    fig = make_figure(np.zeros(INFOS.size), INFOS)
    ax = fig.axes[0]
    assert "$p_0$" in ax.get_xlabel()
    assert "$p_0$" in ax.get_ylabel()


def test_labels_name_the_axes():
    fig = make_figure(np.zeros(INFOS.size), INFOS)
    ax = fig.axes[0]
    assert "Re" in ax.get_xlabel()
    assert "Im" in ax.get_ylabel()


# ============================================================
# Saving
# ============================================================

def test_save_writes_a_png(tmp_path):
    out = tmp_path / "figure.png"
    save_image(np.random.default_rng(0).random(INFOS.size), INFOS, out)

    assert out.exists()
    data = out.read_bytes()
    assert data[:8] == b"\x89PNG\r\n\x1a\n"   # PNG file signature
    assert len(data) > 1000