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

from slidingpuzzle import *


def test_hamming_distance():
    board = new_board(5, 3)
    assert hamming_distance(board) == 0

    swap_tiles(board, (0, 1), (0, 0))
    assert hamming_distance(board) == 2


def test_linear_conflict_distance():
    board = board_from([2, 1, 3], [4, 5, 6], [7, 8, 0])
    assert linear_conflict_distance(board) == 4

    board = board_from([4, 2, 3], [1, 5, 6], [7, 8, 0])
    assert linear_conflict_distance(board) == 4

    board = board_from([1, 2, 3], [6, 5, 4], [7, 8, 0])
    assert linear_conflict_distance(board) == 8


def test_manhattan_distance():
    board = new_board(3, 5)
    assert manhattan_distance(board) == 0

    swap_tiles(board, (1, 2), (0, 0))
    assert manhattan_distance(board) == 6


def test_euclidean_distance():
    board = new_board(3, 5)
    assert euclidean_distance(board) == 0

    swap_tiles(board, (1, 2), (0, 0))
    c = math.sqrt((2 * 1) ** 2 + (2 * 2) ** 2)
    assert euclidean_distance(board) == c


def test_relaxed_adjacency_distance():
    h, w = 3, 3
    b = new_board(h, w)
    swap_tiles(b, (h - 1, w - 1), (0, 0))
    assert relaxed_adjacency_distance(b) == 1

    swap_tiles(b, (0, 0), (0, 1))
    swap_tiles(b, (0, 1), (h - 1, w - 1))
    assert relaxed_adjacency_distance(b) == 3
