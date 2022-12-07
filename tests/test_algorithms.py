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

import pytest

from slidingpuzzle import *


def test_a_star():
    board = new_board(3, 3)
    shuffle_lazy(board, num_moves=10)
    assert len(search(board).solution) == len(
        search(board, Algorithm.A_STAR, heuristic=manhattan_distance).solution
    )


@pytest.mark.parametrize("alg", [Algorithm.IDA_STAR, Algorithm.IDDFS])
def test_search_iterative_deepening(alg):
    b = new_board(3, 3)
    shuffle_lazy(b, 10)
    # best solution should be found (using default args)
    expected_len = len(search(b).solution)
    actual = search(b, alg=alg, heuristic=manhattan_distance)
    actual_len = len(actual.solution)
    assert expected_len == actual_len


@pytest.mark.parametrize(
    "algorithm",
    [Algorithm.A_STAR, Algorithm.BEAM, Algorithm.BFS, Algorithm.DFS, Algorithm.GREEDY],
)
@pytest.mark.parametrize(
    "heuristic",
    [
        euclidean_distance,
        linear_conflict_distance,
        manhattan_distance,
        relaxed_adjacency_distance,
    ],
)
def test_search(algorithm, heuristic):
    board = from_rows([5, 2, 4], [3, 0, 1])
    weight = 1
    result = search(
        board,
        alg=algorithm,
        heuristic=heuristic,
        weight=weight,
        width=128,
    )
    assert result.solution is not None
    if algorithm in (
        Algorithm.A_STAR,
        Algorithm.BEAM,
        Algorithm.BFS,
        Algorithm.IDA_STAR,
    ):
        assert len(result.solution) == 15
    print(repr(result))
    print(str(result))


def test_search_hard():
    board = from_rows([8, 6, 7], [3, 5, 1], [2, 0, 4])
    result = search(board, weight=1)
    assert len(result.solution) == 27


def test_evaluate():
    assert evaluate(3, 3, num_iters=1) > 0


@pytest.mark.slow
def test_compare():
    ha, hb = compare(3, 3, hb=manhattan_distance)
    assert ha < hb


def test_solution_as_tiles():
    h, w = 3, 3
    b = new_board(h, w)
    swap_tiles(b, (h - 1, w - 1), (h - 2, w - 1))
    swap_tiles(b, (h - 2, w - 1), (h - 2, w - 2))
    r = search(b)
    assert [5, 6] == solution_as_tiles(b, r.solution)
