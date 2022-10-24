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

import math

import slidingpuzzle
from slidingpuzzle.board import freeze_board


def euclidean_distance(board: tuple[list[int], ...]) -> float:
    r"""
    The distance between each tile and its destination, as measured in Euclidean space

    .. math::
        \sum_{i}^{n} \sqrt{(\text{tile}_{i, x} - \text{goal}_{i, x})^2 + \
            (\text{tile}_{i, y} - \text{goal}_{i, y})^2}

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
            a = abs(y - (tile - 1) // w)
            b = abs(x - (tile - 1) % w)
            dist += math.sqrt(a**2 + b**2)
    return dist


def hamming_distance(board: tuple[list[int], ...]) -> int:
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


def manhattan_distance(board: tuple[list[int], ...]) -> int:
    r"""
    The minimum number of moves needed to restore the board to the goal state, if tiles
    could be moved through each other.

    .. math::
        \sum_{i}^{n} |\text{tile}_{i, x} - \text{goal}_{i, x}| + \
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


def random_distance(board: tuple[list[int], ...]) -> int:
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
