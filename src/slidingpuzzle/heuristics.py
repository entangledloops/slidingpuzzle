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

import numpy as np

from slidingpuzzle.board import (
    BLANK_TILE,
    Board,
    find_blank,
    get_goal_tile,
    get_goal_yx,
    freeze_board,
    is_solved,
    swap_tiles,
)


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
    h, w = len(board), len(board[0])
    dist = 0.0
    for y, row in enumerate(board):
        for x, tile in enumerate(row):
            if BLANK_TILE == tile:
                continue
            goal_y, goal_x = get_goal_yx(h, w, tile)
            a, b = abs(y - goal_y), abs(x - goal_x)
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
    h, w = len(board), len(board[0])
    dist = 0
    for y, row in enumerate(board):
        for x, tile in enumerate(row):
            if BLANK_TILE == tile:
                continue
            goal_y, goal_x = get_goal_yx(h, w, tile)
            if y != goal_y or x != goal_x:
                dist += 1
    return dist


def linear_conflict_distance(board: Board) -> int:
    r"""
    Starts with Manhattan distance and adds an additional 2 for every out-of-place
    pair of tiles in the same row or column that are also both in their target row
    or column. (It will take at least 2 additional moves to reshuffle the tiles into
    their correct positions.) Only the largest number of conflicts per row/col are
    added to the distance. This is an improvement over Manhattan distance.

    Args:
        board: The board

    Returns:
        Estimated distance to goal.
    """
    h, w = len(board), len(board[0])
    dist = manhattan_distance(board)

    # check out-of-place tiles in each row
    for y in range(h):
        max_conflicts = 0
        for x1 in range(w):
            if BLANK_TILE == board[y][x1]:
                continue
            conflicts = 0
            tile1 = board[y][x1]
            target_row1, _ = get_goal_yx(h, w, tile1)
            if y != target_row1:
                continue
            for x2 in range(x1 + 1, w):
                if BLANK_TILE == board[y][x2]:
                    continue
                tile2 = board[y][x2]
                target_row2, _ = get_goal_yx(h, w, tile2)
                if y != target_row2:
                    continue
                if target_row1 == target_row2 and tile2 < tile1:
                    conflicts += 1
            max_conflicts = max(max_conflicts, conflicts)
        dist += 2 * max_conflicts

    # check out-of-place tiles in each col
    for x in range(w):
        max_conflicts = 0
        for y1 in range(h):
            if BLANK_TILE == board[y1][x]:
                continue
            conflicts = 0
            tile1 = board[y1][x]
            _, target_col1 = get_goal_yx(h, w, tile1)
            if x != target_col1:
                continue
            for y2 in range(y1 + 1, h):
                if BLANK_TILE == board[y2][x]:
                    continue
                tile2 = board[y2][x]
                _, target_col2 = get_goal_yx(h, w, tile2)
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
    h, w = len(board), len(board[0])
    dist = 0
    for y, row in enumerate(board):
        for x, tile in enumerate(row):
            if BLANK_TILE == tile:
                continue
            goal_y, goal_x = get_goal_yx(h, w, tile)
            dist += abs(y - goal_y) + abs(x - goal_x)
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


def relaxed_adjacency_distance(board: Board) -> int:
    r"""
    Repeatedly swap misplaced tiles into their goal positions, ignoring legal board
    distance and rules. We repeat this process until the board is solved. This
    heuristic is a slight improvement to Hamming distance.

    Args:
        board: The board

    Returns:
        The estimated distance to goal.
    """
    h, w = len(board), len(board[0])
    board = np.copy(board)
    dist = 0

    def swap_first_misplaced(blank_yx: tuple[int, int]) -> None:
        for y in range(h):
            for x in range(w):
                if get_goal_yx(h, w, board[y][x]) != (y, x):
                    return swap_tiles(board, (y, x), blank_yx)

    while not is_solved(board):
        blank_yx = find_blank(board)
        if blank_yx == get_goal_yx(h, w, BLANK_TILE):
            swap_first_misplaced(blank_yx)
        else:
            swap_tiles(board, get_goal_tile(h, w, blank_yx), blank_yx)
        dist += 1

    return dist


def checkerboard_adjacency_distance(board: Board) -> int:
    r""" """
    pass
