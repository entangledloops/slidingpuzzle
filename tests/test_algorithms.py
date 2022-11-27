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
    shuffle_board_lazy(board, num_moves=10)
    assert len(search(board).solution) == len(
        search(board, A_STAR, heuristic=manhattan_distance).solution
    )


@pytest.mark.parametrize("alg", [IDA_STAR, IDDFS])
def test_search_iterative_deepening(alg):
    b = new_board(3, 3)
    shuffle_board_lazy(b, 10)
    # best solution should be found (using default args)
    expected_len = len(search(b).solution)
    actual = search(b, alg=alg, heuristic=manhattan_distance)
    actual_len = len(actual.solution)
    assert expected_len == actual_len


@pytest.mark.parametrize("algorithm", [A_STAR, BEAM, BFS, DFS, GREEDY])
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
    board = board_from([5, 2, 4], [3, 0, 1])
    weight = 1
    result = search(
        board,
        alg=algorithm,
        heuristic=heuristic,
        weight=weight,
        width=128,
    )
    assert result.solution is not None
    if algorithm in (A_STAR, BEAM, BFS, IDA_STAR):
        assert len(result.solution) == 15
    print(repr(result))
    print(str(result))


def test_search_hard():
    board = board_from([8, 6, 7], [3, 5, 1], [2, 0, 4])
    result = search(
        board,
        alg=A_STAR,
        heuristic=manhattan_distance,
        weight=1,
    )
    assert len(result.solution) == 27


def test_evaluate():
    assert evaluate(3, 3, num_iters=1) > 0


def test_solution_as_tiles():
    h, w = 3, 3
    b = new_board(h, w)
    swap_tiles(b, (h - 1, w - 1), (h - 2, w - 1))
    swap_tiles(b, (h - 2, w - 1), (h - 2, w - 2))
    r = search(b)
    assert [5, 6] == solution_as_tiles(b, r.solution)


@pytest.mark.slow
def test_heuristic_behavior():
    # we compute avg generated nodes over multiple runs to confirm that
    # heuristic behavior is in line with expectations
    generated_avg = compare(
        3, 3, linear_conflict_distance, manhattan_distance, num_iters=4
    )
    assert generated_avg[0] < generated_avg[1]

    generated_avg = compare(3, 3, manhattan_distance, hamming_distance, num_iters=4)
    assert generated_avg[0] < generated_avg[1]

    # lcd/manhattan/euclidean are good contenders, so we don't compare them
    generated_avg = compare(3, 3, euclidean_distance, hamming_distance, num_iters=4)
    assert generated_avg[0] < generated_avg[1]


@pytest.mark.slow
def test_heuristic_admissibility():
    # validate that solutions are in line with BFS
    # this does not guarantee admissibility, it's just an empirical sanity check
    boards = [shuffle_board(new_board(3, 3)) for _ in range(50)]
    optimal = [len(search(b, "bfs").solution) for b in boards]
    for h in (linear_conflict_distance, manhattan_distance):
        for b, o in zip(boards, optimal):
            assert len(search(b, heuristic=h).solution) == o
