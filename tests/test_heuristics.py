# Copyright 2022 Stephen Dunn

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math

import pytest

from slidingpuzzle import *


def test_corner_tiles_distance():
    board = new_board(3, 3)
    assert corner_tiles_distance(board) == 0

    # top-left corner
    board = from_rows([6, 2, 3, 4], [7, 5, 8, 9], [10, 11, 12, 13], [14, 15, 1, 0])
    assert corner_tiles_distance(board) == 2

    board = from_rows([6, 2, 3, 4], [5, 7, 8, 9], [10, 11, 12, 13], [14, 15, 1, 0])
    assert corner_tiles_distance(board) == 4

    # top-right corner
    board = from_rows([1, 2, 3, 5], [4, 6, 7, 9], [8, 10, 11, 12], [13, 14, 15, 0])
    assert corner_tiles_distance(board) == 2

    board = from_rows([1, 2, 3, 5], [4, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 0])
    assert corner_tiles_distance(board) == 4

    # bottom-left corner
    board = from_rows([1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [15, 13, 14, 0])
    assert corner_tiles_distance(board) == 2

    board = from_rows([1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [15, 14, 13, 0])
    assert corner_tiles_distance(board) == 4


def test_euclidean_distance():
    board = new_board(3, 5)
    assert euclidean_distance(board) == 0

    swap_tiles(board, (1, 2), (0, 0))
    c = math.sqrt((2 * 1) ** 2 + (2 * 2) ** 2)
    assert euclidean_distance(board) == c


def test_hamming_distance():
    board = new_board(5, 3)
    assert hamming_distance(board) == 0

    swap_tiles(board, (0, 1), (0, 0))
    assert hamming_distance(board) == 2


def test_last_moves_distance():
    board = from_rows([1, 2, 3], [4, 5, 6], [7, 0, 8])
    assert last_moves_distance(board) == 0

    board = from_rows([8, 6, 0], [4, 5, 2], [7, 3, 1])
    assert last_moves_distance(board) == 2

    assert last_moves_distance(new_board(5, 5)) == 0

    board = from_rows(
        [1, 2, 3, 4, 5],
        [6, 7, 8, 9, 10],
        [11, 12, 13, 14, 20],
        [16, 17, 18, 19, 15],
        [21, 22, 23, 24, 0],
    )
    assert last_moves_distance(board) == 2

    board = from_rows(
        [17, 1, 20, 9, 16],
        [2, 22, 19, 14, 5],
        [15, 21, 0, 3, 24],
        [23, 18, 13, 12, 7],
        [10, 8, 6, 4, 11],
    )
    assert last_moves_distance(board) == 4


def test_linear_conflict_distance():
    board = from_rows([1, 2, 3], [4, 5, 6], [7, 0, 8])
    assert linear_conflict_distance(board, optimized=False) == 0
    assert linear_conflict_distance(board) == 1

    board = from_rows([2, 1, 3], [4, 5, 6], [7, 8, 0])
    assert linear_conflict_distance(board, optimized=False) == 2
    assert linear_conflict_distance(board) == 6

    board = from_rows([4, 2, 3], [1, 5, 6], [7, 8, 0])
    assert linear_conflict_distance(board, optimized=False) == 2
    assert linear_conflict_distance(board) == 6

    board = from_rows([2, 1, 3], [4, 5, 6], [7, 0, 8])
    assert linear_conflict_distance(board, optimized=False) == 2
    assert linear_conflict_distance(board) == 5

    board = from_rows([1, 2, 3], [6, 5, 4], [7, 8, 0])
    assert linear_conflict_distance(board, optimized=False) == 4
    assert linear_conflict_distance(board) == 8


def test_manhattan_distance():
    board = new_board(3, 5)
    assert manhattan_distance(board) == 0

    swap_tiles(board, (1, 2), (0, 0))
    assert manhattan_distance(board) == 6


def test_relaxed_adjacency_distance():
    h, w = 3, 3
    b = new_board(h, w)
    swap_tiles(b, (h - 1, w - 1), (0, 0))
    assert relaxed_adjacency_distance(b) == 1

    b = new_board(h, w)
    swap_tiles(b, (0, 0), (0, 1))
    swap_tiles(b, (0, 0), (h - 1, w - 1))
    swap_tiles(b, (0, 0), (1, 0))
    assert relaxed_adjacency_distance(b) == 3


@pytest.mark.slow
def test_heuristic_behavior():
    # we compute avg generated nodes over multiple runs to confirm that
    # heuristic behavior is in line with expectations
    generated_avg = compare(
        3,
        3,
        ha=linear_conflict_distance,
        hb=manhattan_distance,
    )
    assert generated_avg[0] < generated_avg[1]

    generated_avg = compare(
        3, 3, num_iters=4, ha=manhattan_distance, hb=hamming_distance
    )
    assert generated_avg[0] < generated_avg[1]

    # lcd/manhattan/euclidean are good contenders, so we don't compare them
    generated_avg = compare(
        3, 3, num_iters=4, ha=euclidean_distance, hb=hamming_distance
    )
    assert generated_avg[0] < generated_avg[1]


@pytest.mark.slow
@pytest.mark.parametrize("hw", [(3, 3)])
def test_heuristic_admissibility(hw):
    h, w = hw
    # validate that solutions are in line with BFS
    # this does not guarantee admissibility, it's just an empirical sanity check
    boards = [shuffle(new_board(h, w)) for _ in range(50)]
    optimal = [len(search(b, "bfs").solution) for b in boards]
    for h in (linear_conflict_distance, manhattan_distance):
        for b, o in zip(boards, optimal):
            assert len(search(b, heuristic=h).solution) == o, b


@pytest.mark.skip
@pytest.mark.slow
@pytest.mark.parametrize("hw", [(3, 3)])
def test_linear_conflict_distance_exhaustive(hw):
    h, w = hw
    start, stop = 0, None
    for i, b in enumerate(board_generator(h, w, start, stop)):
        expected = len(search(b, heuristic=manhattan_distance).solution)
        actual = linear_conflict_distance(b)
        assert expected >= actual, f"board #{start + i}: {b}"


@pytest.mark.skip
@pytest.mark.slow
@pytest.mark.parametrize("hw", [(3, 3), (4, 4)])
def test_linear_conflict_distance_exhaustive_dataset(hw):
    r"""
    This test is ths same as above, but faster if a dataset file is available
    instead of re-generating and solving all boards from scratch.
    """
    import json

    h, w = hw
    with open(f"datasets/examples_{h}x{w}.json") as fp:
        db = json.load(fp)
    for i, (board, solution) in enumerate(db):
        board = from_rows(*board)
        expected = len(solution)
        actual = linear_conflict_distance(board)
        assert expected >= actual, f"board #{i}: {board}, {solution}"
