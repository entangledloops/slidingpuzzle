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
    board = from_rows([8, 5, 3], [4, 2, 6], [7, 1, 0])
    assert corner_tiles_distance(board) == 2

    board = from_rows([8, 2, 3], [4, 5, 6], [7, 0, 1])
    assert corner_tiles_distance(board) == 4

    # top-right corner
    board = from_rows([1, 2, 8], [4, 6, 5], [7, 0, 3])
    assert corner_tiles_distance(board) == 2

    board = from_rows([1, 2, 8], [3, 5, 6], [4, 0, 7])
    assert corner_tiles_distance(board) == 4

    # bottom-left corner
    board = from_rows([1, 2, 3], [4, 7, 6], [5, 0, 8])
    assert corner_tiles_distance(board) == 2

    board = from_rows([1, 2, 3], [4, 7, 6], [5, 8, 0])
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


def test_linear_conflict_distance():
    board = from_rows([1, 2, 3], [4, 5, 6], [7, 0, 8])
    assert linear_conflict_distance(board) == 1

    board = from_rows([2, 1, 3], [4, 5, 6], [7, 8, 0])
    assert linear_conflict_distance(board) == 4

    board = from_rows([4, 2, 3], [1, 5, 6], [7, 8, 0])
    assert linear_conflict_distance(board) == 4

    board = from_rows([2, 1, 3], [4, 5, 6], [7, 0, 8])
    assert linear_conflict_distance(board) == 5

    board = from_rows([1, 2, 3], [6, 5, 4], [7, 8, 0])
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
def test_heuristic_admissibility():
    # validate that solutions are in line with BFS
    # this does not guarantee admissibility, it's just an empirical sanity check
    boards = [shuffle(new_board(3, 3)) for _ in range(50)]
    optimal = [len(search(b, "bfs").solution) for b in boards]
    for h in (linear_conflict_distance, manhattan_distance):
        for b, o in zip(boards, optimal):
            assert len(search(b, heuristic=h).solution) == o, b


@pytest.mark.skip
def test_linear_conflict_distance_exhaustive():
    start = 0
    stop = None
    for i, b in enumerate(board_generator(3, 3, start, stop)):
        expected = len(search(b, heuristic=manhattan_distance).solution)
        actual = len(search(b, heuristic=linear_conflict_distance).solution)
        assert expected == actual, f"{start + i}: {b}"
