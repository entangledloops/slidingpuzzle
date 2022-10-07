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


def euclidean_distance(board: tuple[list[int]]) -> int:
    w = len(board[0])
    dist = 0
    for y, row in enumerate(board):
        for x, tile in enumerate(row):
            a = abs(y - (tile - 1) // w)
            b = abs(x - (tile - 1) % w)
            dist += a**2 + b**2  # ignore sqrt to save time
    return dist


def hamming_distance(board: tuple[list[int]]) -> int:
    w = len(board[0])
    dist = 0
    for y, row in enumerate(board):
        for x, tile in enumerate(row):
            orig_y = (tile - 1) // w
            orig_x = (tile - 1) % w
            if y != orig_y or x != orig_x:
                dist += 1
    return dist


def manhattan_distance(board: tuple[list[int]]) -> int:
    w = len(board[0])
    dist = 0
    for y, row in enumerate(board):
        for x, tile in enumerate(row):
            orig_y = (tile - 1) // w
            orig_x = (tile - 1) % w
            dist += abs(y - orig_y) + abs(x - orig_x)
    return dist


def random_distance(board: tuple[list[int]]) -> int:
    frozen_board = tuple(tuple(row) for row in board)
    return hash(frozen_board)
