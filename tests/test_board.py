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


def test_new_board():
    ground_truth = ([1, 2, 3], [4, 5, 6], [7, 8, 0])
    assert new_board(3, 3) == ground_truth

    ground_truth = ([1, 2], [3, 4], [5, 0])
    assert new_board(3, 2) == ground_truth


@pytest.mark.parametrize("h", [3, 5])
@pytest.mark.parametrize("w", [3, 5])
def test_board_from_values(h, w):
    b = board_from_values(h, w, range(1, 1 + h * w))
    b[-1][-1] = EMPTY_TILE
    assert b == new_board(h, w)


def test_freeze_board():
    b = new_board(3, 3)
    frozen = freeze_board(b)
    assert id(b) != id(frozen)
    swap_tiles(b, (0, 1), (1, 0))
    assert b[0][1] != frozen[0][1]


def test_print_board():
    print_board(new_board(5, 5))


def test_get_yx():
    board = new_board(3, 3)
    tile = 1
    for y, row in enumerate(board):
        for x in range(len(row)):
            if board[y][x] == EMPTY_TILE:
                continue
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
    r = search(board, alg="greedy", heuristic=linear_conflict_distance)
    assert r.solution is not None


@pytest.mark.parametrize("size", [(2, 2), (3, 2), (2, 3), (3, 3)])
def test_shuffle_board_lazy(size):
    random.seed(0)
    h, w = size
    board = new_board(h, w)
    shuffle_board_lazy(board)
    r = search(board, alg="greedy", heuristic=manhattan_distance)
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


@pytest.mark.parametrize("h", [3, 5])
@pytest.mark.parametrize("w", [3, 5])
def test_board_generator(h, w):
    gen = board_generator(h, w)
    assert next(gen) == board_from_values(h, w, range(h * w))
