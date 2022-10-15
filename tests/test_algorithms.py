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
import random

import pytest

from slidingpuzzle import *


def test_hamming():
    board = new_board(5, 3)
    assert hamming_distance(board) == 0

    swap_tiles(board, (0, 1), (0, 0))
    assert hamming_distance(board) == 2


def test_manhattan():
    board = new_board(5, 3)
    assert manhattan_distance(board) == 0

    swap_tiles(board, (1, 2), (0, 0))
    assert manhattan_distance(board) == 6


def test_euclidean():
    board = new_board(5, 3)
    assert euclidean_distance(board) == 0

    swap_tiles(board, (1, 2), (0, 0))
    c = math.sqrt((2 * 1) ** 2 + (2 * 2) ** 2)
    assert euclidean_distance(board) == c


def test_a_star():
    board = new_board(3, 3)
    shuffle_board_lazy(board, num_moves=10)
    assert len(search(board).solution) == len(
        search(board, A_STAR, heuristic=manhattan_distance).solution
    )


@pytest.mark.parametrize("algorithm", [IDA_STAR, IDDFS])
def test_search_slow(algorithm):
    random.seed(0)
    b = new_board(3, 3)
    shuffle_board_lazy(b, 10)
    expected_len = len(search(b).solution)
    actual = search(b, alg=algorithm, heuristic=manhattan_distance)
    # best solution should be found (using default args)
    actual_len = len(actual.solution)
    assert expected_len == actual_len


@pytest.mark.parametrize("algorithm", [A_STAR, BEAM, BFS, DFS, GREEDY])
@pytest.mark.parametrize("heuristic", [euclidean_distance, manhattan_distance])
def test_search(algorithm, heuristic):
    random.seed(0)
    board = tuple([[5, 2, 4], [3, 0, 1]])
    weight = 1
    result = search(
        board,
        alg=algorithm,
        heuristic=heuristic,
        weight=weight,
        width=4,
    )
    assert result.solution is not None
    if algorithm in (A_STAR, BEAM, BFS, IDA_STAR):
        assert len(result.solution) == 15
    print(repr(result))
    print(str(result))


def test_search_hard():
    random.seed(0)
    board = tuple([[8, 6, 7], [3, 5, 1], [2, 0, 4]])
    result = search(
        board,
        alg=A_STAR,
        heuristic=manhattan_distance,
        weight=1,
    )
    assert len(result.solution) == 27


def test_heuristic_behavior():
    random.seed(0)

    # we compute avg expanded nodes over multiple runs to confirm that
    # heuristic behavior is in line with expectations
    expanded_avg = compare(3, 3, manhattan_distance, hamming_distance)
    assert expanded_avg[0] < expanded_avg[1]

    # manhattan + euclidean are both good contenders, so we don't compare them
    expanded_avg = compare(3, 3, euclidean_distance, hamming_distance)
    assert expanded_avg[0] < expanded_avg[1]


def test_solution_as_tiles():
    h, w = 3, 3
    b = new_board(h, w)
    swap_tiles(b, (h - 1, w - 1), (h - 2, w - 1))
    swap_tiles(b, (h - 2, w - 1), (h - 2, w - 2))
    r = search(b)
    assert [5, 6] == solution_as_tiles(b, r.solution)
