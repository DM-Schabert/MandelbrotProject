"""
Step 8: tests for how the rows are divided between ranks (serial, no MPI).

Every test simulates all ranks in a loop: rank r's rows are infos.rows_of(r).
"""

import numpy as np
import pytest

from Mandelbrot import Infos, block_scheduling_static, chunk_scheduling_static

SCHEDULES = ["block", "chunk"]

# Cases chosen to include the awkward ones:
#   x_dim not divisible by chunk_size or by total_rank,
#   chunk_size = 1, chunk_size >= x_dim,
#   more ranks than rows / than chunks.
X_DIMS = [1, 7, 30, 101]
TOTAL_RANKS = [1, 2, 3, 4, 7, 150]
CHUNK_SIZES = [1, 3, 10, 1000]


def all_rows(infos):
    return [infos.rows_of(r) for r in range(infos.total_rank)]


# ============================================================
# Both schedules
# ============================================================

@pytest.mark.parametrize("schedule", SCHEDULES)
@pytest.mark.parametrize("x_dim", X_DIMS)
@pytest.mark.parametrize("total_rank", TOTAL_RANKS)
@pytest.mark.parametrize("chunk_size", CHUNK_SIZES)
def test_every_row_assigned_exactly_once(schedule, x_dim, total_rank, chunk_size):
    infos = Infos(size=(x_dim, 5), schedule=schedule,
                  total_rank=total_rank, chunk_size=chunk_size)
    combined = np.concatenate(all_rows(infos))

    assert len(combined) == x_dim                      # nothing duplicated
    assert sorted(combined.tolist()) == list(range(x_dim))  # nothing missing


@pytest.mark.parametrize("schedule", SCHEDULES)
def test_rows_of_matches_the_plain_functions(schedule):
    infos = Infos(size=(53, 5), schedule=schedule, total_rank=4, chunk_size=6)
    for r in range(4):
        if schedule == "block":
            expected = block_scheduling_static(53, r, 4)
        else:
            expected = chunk_scheduling_static(53, 6, r, 4)
        np.testing.assert_array_equal(infos.rows_of(r), expected)


@pytest.mark.parametrize("schedule", SCHEDULES)
def test_each_ranks_rows_are_in_increasing_order(schedule):
    infos = Infos(size=(101, 5), schedule=schedule, total_rank=4, chunk_size=7)
    for rows in all_rows(infos):
        assert (np.diff(rows) > 0).all()


# ============================================================
# Block schedule
# ============================================================

@pytest.mark.parametrize("x_dim", X_DIMS)
@pytest.mark.parametrize("total_rank", TOTAL_RANKS)
def test_block_gives_one_contiguous_range_per_rank(x_dim, total_rank):
    infos = Infos(size=(x_dim, 5), schedule="block", total_rank=total_rank)
    for rows in all_rows(infos):
        if len(rows) > 1:
            assert (np.diff(rows) == 1).all()


@pytest.mark.parametrize("x_dim", X_DIMS)
@pytest.mark.parametrize("total_rank", TOTAL_RANKS)
def test_block_sizes_differ_by_at_most_one(x_dim, total_rank):
    infos = Infos(size=(x_dim, 5), schedule="block", total_rank=total_rank)
    sizes = [len(rows) for rows in all_rows(infos)]
    assert max(sizes) - min(sizes) <= 1


def test_block_ranks_are_in_order():
    # Rank 0 gets the first block, rank 1 the next, and so on.
    infos = Infos(size=(10, 5), schedule="block", total_rank=3)
    assert [r.tolist() for r in all_rows(infos)] == [
        [0, 1, 2, 3], [4, 5, 6], [7, 8, 9],
    ]


def test_block_more_ranks_than_rows():
    infos = Infos(size=(3, 5), schedule="block", total_rank=5)
    assert [r.tolist() for r in all_rows(infos)] == [[0], [1], [2], [], []]


# ============================================================
# Chunk schedule
# ============================================================

@pytest.mark.parametrize("x_dim", X_DIMS)
@pytest.mark.parametrize("total_rank", TOTAL_RANKS)
@pytest.mark.parametrize("chunk_size", CHUNK_SIZES)
def test_chunks_are_dealt_round_robin(x_dim, total_rank, chunk_size):
    # Chunk k = rows [k*chunk_size, (k+1)*chunk_size) belongs entirely
    # to rank k % total_rank.
    infos = Infos(size=(x_dim, 5), schedule="chunk",
                  total_rank=total_rank, chunk_size=chunk_size)
    owner = np.empty(x_dim, dtype=int)
    for r, rows in enumerate(all_rows(infos)):
        owner[rows] = r

    chunk_id = np.arange(x_dim) // chunk_size
    np.testing.assert_array_equal(owner, chunk_id % total_rank)


def test_chunk_small_example():
    # 10 rows, chunks of 2, 3 ranks: chunks 0..4 go to ranks 0,1,2,0,1
    infos = Infos(size=(10, 5), schedule="chunk", total_rank=3, chunk_size=2)
    assert [r.tolist() for r in all_rows(infos)] == [
        [0, 1, 6, 7], [2, 3, 8, 9], [4, 5],
    ]


def test_chunk_size_larger_than_image():
    # Only one chunk exists, so rank 0 gets everything.
    infos = Infos(size=(7, 5), schedule="chunk", total_rank=3, chunk_size=1000)
    assert [r.tolist() for r in all_rows(infos)] == [list(range(7)), [], []]


# ============================================================
# pieces_of
# ============================================================

@pytest.mark.parametrize("n_rows", [0, 1, 5, 10, 11, 23])
@pytest.mark.parametrize("chunk_size", [1, 3, 10, 1000])
def test_pieces_join_back_in_order(n_rows, chunk_size):
    infos = Infos(chunk_size=chunk_size)
    rows = np.arange(100, 100 + n_rows)
    pieces = infos.pieces_of(rows)

    joined = np.concatenate(pieces) if pieces else np.array([], dtype=int)
    np.testing.assert_array_equal(joined, rows)


@pytest.mark.parametrize("n_rows", [1, 5, 10, 11, 23])
@pytest.mark.parametrize("chunk_size", [1, 3, 10, 1000])
def test_piece_sizes(n_rows, chunk_size):
    infos = Infos(chunk_size=chunk_size)
    sizes = [len(p) for p in infos.pieces_of(np.arange(n_rows))]

    # All pieces are full except possibly the last one, which isn't empty.
    assert all(s == chunk_size for s in sizes[:-1])
    assert 1 <= sizes[-1] <= chunk_size


def test_no_rows_gives_no_pieces():
    assert Infos().pieces_of(np.array([], dtype=int)) == []


@pytest.mark.parametrize("schedule", SCHEDULES)
@pytest.mark.parametrize("total_rank", [1, 3, 4])
@pytest.mark.parametrize("chunk_size", [1, 4, 10])
def test_every_piece_is_contiguous(schedule, total_rank, chunk_size):
    # With the chunk schedule each piece is exactly one chunk, and with
    # the block schedule it's a slice of one block. Either way a piece is
    # a run of consecutive rows.
    infos = Infos(size=(53, 5), schedule=schedule,
                  total_rank=total_rank, chunk_size=chunk_size)
    for r in range(total_rank):
        for piece in infos.pieces_of_rank(r):
            assert (np.diff(piece) == 1).all()


def test_chunk_pieces_are_exactly_the_chunks():
    infos = Infos(size=(25, 5), schedule="chunk", total_rank=2, chunk_size=4)
    # chunks: 0-3, 4-7, 8-11, 12-15, 16-19, 20-23, 24
    assert [p.tolist() for p in infos.pieces_of_rank(0)] == [
        [0, 1, 2, 3], [8, 9, 10, 11], [16, 17, 18, 19], [24],
    ]
    assert [p.tolist() for p in infos.pieces_of_rank(1)] == [
        [4, 5, 6, 7], [12, 13, 14, 15], [20, 21, 22, 23],
    ]