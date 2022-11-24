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

"""
This module provides heuristic functions used to evaluate board states.

Each function accepts a puzzle board as input, and returns a numeric value
that indicates the estimated distance from the goal.
"""

from typing import Callable, TypeAlias

import math

import slidingpuzzle
from slidingpuzzle.board import Board, freeze_board


Heuristic: TypeAlias = Callable[[Board], int | float]


def euclidean_distance(board: Board) -> float:
    r"""
    The distance between each tile and its destination, as measured in Euclidean space

    .. math::
        \sum_{i}^{n} \sqrt{(\text{tile}_{i, x} - \text{goal}_{i, x})^2 +
            (\text{tile}_{i, y} - \text{goal}_{i, y})^2}

    Args:
        board: The board

    Returns:
        The sum of all tile distances from their goal positions
    """
    w = len(board[0])
    dist = 0.0
    for y, row in enumerate(board):
        for x, tile in enumerate(row):
            if slidingpuzzle.EMPTY_TILE == tile:
                continue
            a = abs(y - (tile - 1) // w)
            b = abs(x - (tile - 1) % w)
            dist += math.sqrt(a**2 + b**2)
    return dist


def hamming_distance(board: Board) -> int:
    r"""
    The count of misplaced tiles.

    .. math::
        \sum_{i}^{n} \text{tile}_i \oplus \text{goal}_i

    Args:
        board: The board

    Returns:
        The number of tiles out of their goal positions
    """
    w = len(board[0])
    dist = 0
    for y, row in enumerate(board):
        for x, tile in enumerate(row):
            if slidingpuzzle.EMPTY_TILE == tile:
                continue
            orig_y = (tile - 1) // w
            orig_x = (tile - 1) % w
            if y != orig_y or x != orig_x:
                dist += 1
    return dist


def linear_conflict_distance(board: Board) -> int:
    r"""
    Starts with Manhattan distance and adds an additional 2 for every out-of-place
    pair of tiles in the same row or column that are also both in their target row
    or column. It will take at least 2 additional moves to reshuffle the tiles into
    their correct positions.

    Args:
        board: The board

    Returns:
        A slightly more accurate heuristic value than Manhattan distance.
    """
    h = len(board)
    w = len(board[0])
    dist = manhattan_distance(board)

    # check out-of-place cols in each row
    for y in range(h):
        max_conflicts = 0
        for x1 in range(w):
            if slidingpuzzle.EMPTY_TILE == board[y][x1]:
                continue
            conflicts = 0
            tile1 = board[y][x1]
            target_row1 = (tile1 - 1) // w
            if y != target_row1:
                continue
            for x2 in range(x1 + 1, w):
                if slidingpuzzle.EMPTY_TILE == board[y][x2]:
                    continue
                tile2 = board[y][x2]
                target_row2 = (tile2 - 1) // w
                if y != target_row2:
                    continue
                if target_row1 == target_row2 and tile2 < tile1:
                    conflicts += 1
            max_conflicts = max(max_conflicts, conflicts)
        dist += 2 * max_conflicts

    # check out-of-place rows in each col
    for x in range(w):
        max_conflicts = 0
        for y1 in range(h):
            if slidingpuzzle.EMPTY_TILE == board[y1][x]:
                continue
            conflicts = 0
            tile1 = board[y1][x]
            target_col1 = (tile1 - 1) % w
            if x != target_col1:
                continue
            for y2 in range(y1 + 1, h):
                if slidingpuzzle.EMPTY_TILE == board[y2][x]:
                    continue
                tile2 = board[y2][x]
                target_col2 = (tile2 - 1) % w
                if x != target_col2:
                    continue
                if target_col1 == target_col2 and tile2 < tile1:
                    conflicts += 1
            max_conflicts = max(max_conflicts, conflicts)
        dist += 2 * max_conflicts
    return dist


def manhattan_distance(board: Board) -> int:
    r"""
    The minimum number of moves needed to restore the board to the goal state, if tiles
    could be moved through each other.

    .. math::
        \sum_{i}^{n} |\text{tile}_{i, x} - \text{goal}_{i, x}| +
            |\text{tile}_{i, y} - \text{goal}_{i, y}|

    Args:
        board: The board

    Returns:
        The sum of all tile distances from their goal positions
    """
    w = len(board[0])
    dist = 0
    for y, row in enumerate(board):
        for x, tile in enumerate(row):
            if slidingpuzzle.EMPTY_TILE == tile:
                continue
            orig_y = (tile - 1) // w
            orig_x = (tile - 1) % w
            dist += abs(y - orig_y) + abs(x - orig_x)
    return dist


def random_distance(board: Board) -> int:
    r"""
    A random distance computed as a hash of the board state. Useful as a baseline.

    .. math::
        hash(\text{board})

    Args:
        board: The board

    Returns:
        A random distance that is consistent for a given board state
    """
    return hash(freeze_board(board))
