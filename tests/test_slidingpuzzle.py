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


import random

import pytest

from slidingpuzzle import *


def test_newboard():
    ground_truth = ([1, 2, 3], [4, 5, 6], [7, 8, 0])
    assert new_board(3, 3) == ground_truth

    ground_truth = ([1, 2], [3, 4], [5, 0])
    assert new_board(3, 2) == ground_truth


def test_freezeboard():
    b = new_board(3, 3)
    s = set()
    frozen = freeze_board(b)
    s.add(frozen)


def test_print_board():
    print_board(new_board(5, 5))


def test_get_yx():
    board = new_board(3, 3)
    tile = 1
    for y, row in enumerate(board):
        for x in range(len(row) - 1):
            assert tile == board[y][x]
            tile += 1


def test_get_empty_yx():
    board = new_board(3, 3)
    assert get_empty_yx(board) == (2, 2)
    assert get_yx(board, EMPTY_TILE) == get_empty_yx(board)

    tmp = board[2][2]
    board[2][2] = board[0][1]
    board[0][1] = tmp
    assert get_empty_yx(board) == (0, 1)


def test_swap_tiles():
    board = new_board(3, 3)
    pos1 = (2, 2)
    pos2 = (1, 1)
    orig_pos1 = board[pos1[0]][pos1[1]]
    orig_pos2 = board[pos2[0]][pos2[1]]
    swap_tiles(board, pos1, pos2)
    assert orig_pos1 == board[pos2[0]][pos2[1]]
    assert orig_pos2 == board[pos1[0]][pos1[1]]


def test_apply_move():
    board = new_board(5, 5)
    pos1 = (2, 2)
    pos2 = get_empty_yx(board)
    orig_pos1 = board[pos1[0]][pos1[1]]
    orig_pos2 = board[pos2[0]][pos2[1]]
    apply_move(board, pos1)
    assert orig_pos1 == board[pos2[0]][pos2[1]]
    assert orig_pos2 == board[pos1[0]][pos1[1]]


@pytest.mark.parametrize("size", [(2, 2), (3, 2), (2, 3), (3, 3)])
def test_shuffle_board(size):
    random.seed(0)
    h, w = size
    board = new_board(h, w)
    shuffle_board(board)
    r = search(board, algorithm="greedy", heuristic=manhattan_distance)
    assert r.solution is not None


@pytest.mark.parametrize("size", [(2, 2), (3, 2), (2, 3), (3, 3)])
def test_shuffle_board_slow(size):
    random.seed(0)
    h, w = size
    board = new_board(h, w)
    shuffle_board_slow(board)
    r = search(board, algorithm="greedy", heuristic=manhattan_distance)
    assert r.solution is not None


def test_get_possible_moves():
    board = new_board(3, 3)
    moves = get_next_moves(board, (2, 2))
    assert moves == [(2, 1), (1, 2)]

    pos1 = (1, 1)
    pos2 = (2, 2)
    swap_tiles(board, pos1, pos2)
    moves = get_next_moves(board, pos1)
    assert len(moves) == 4


def test_get_next_states():
    board = new_board(3, 3)
    state = State(board, (2, 2), [])
    next_states = get_next_states(state)
    assert len(next_states) == 2


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
    assert euclidean_distance(board) == 10


def test_a_star():
    board = new_board(3, 3)
    shuffle_board_slow(board, num_moves=10)
    assert len(search(board).solution) == len(
        search(board, A_STAR, manhattan_distance).solution
    )


@pytest.mark.parametrize("algorithm", [IDA_STAR, IDDFS])
def test_search_slow(algorithm):
    random.seed(0)
    b = new_board(3, 3)
    shuffle_board_slow(b, 20)
    expected_len = len(search(b).solution)
    actual = search(b, algorithm=algorithm, heuristic=manhattan_distance)
    # there should be no history recorded for iterative deepening
    assert len(actual.visited) == 0
    # best solution should be found (using default args)
    actual_len = len(actual.solution)
    assert expected_len == actual_len


@pytest.mark.parametrize("algorithm", [A_STAR, BEAM, BFS, DFS, GREEDY])
@pytest.mark.parametrize(
    "heuristic", [None, euclidean_distance, manhattan_distance, random_distance]
)
def test_search(algorithm, heuristic):
    random.seed(0)
    board = tuple([[5, 2, 4], [3, 0, 1]])
    result = search(
        board,
        algorithm=algorithm,
        heuristic=heuristic,
        kwargs={"weight": 2, "width": 3},
    )
    if BFS == algorithm:
        assert len(result.solution) == 15
    else:
        assert len(result.solution) >= 15
    print(repr(result))
    print(str(result))


def test_search_hard():
    random.seed(0)
    board = tuple([[8, 6, 7], [3, 5, 1], [2, 0, 4]])
    result = search(
        board,
        algorithm=A_STAR,
        heuristic=manhattan_distance,
        kwargs={"weight": 1},
    )
    assert len(result.solution) == 27


def test_heuristic_behavior():
    random.seed(0)

    # ordered from best -> worst
    heuristics = [
        manhattan_distance,
        euclidean_distance,
        hamming_distance,
        random_distance,
    ]

    # we compute avg expanded nodes over multiple runs to confirm that
    # heuristic behavior is in line with expectations
    num_runs = 20
    expanded_avg = [0 for _ in range(len(heuristics))]
    for i, heuristic in enumerate(heuristics):
        for _ in range(num_runs):
            board = new_board(3, 2)  # largest puzzle reasonably solvable with random
            shuffle_board(board)
            result = search(board, algorithm="greedy", heuristic=heuristic)
            expanded_avg[i] += result.expanded
        expanded_avg[i] /= num_runs
    print("expanded_avg:", expanded_avg)

    # manhattan + euclidean are both good contenders, so we don't compare them
    assert expanded_avg[0] < expanded_avg[2]
    assert expanded_avg[1] < expanded_avg[2]
    assert expanded_avg[2] < expanded_avg[3]


def test_solution_as_tiles():
    h, w = 3, 3
    b = new_board(h, w)
    swap_tiles(b, (h - 1, w - 1), (h - 2, w - 1))
    swap_tiles(b, (h - 2, w - 1), (h - 2, w - 2))
    r = search(b)
    assert [5, 6] == solution_as_tiles(b, r.solution)
