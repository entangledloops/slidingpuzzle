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

from typing import Callable, Iterator, TypeAlias

import logging
import math

import numpy as np
import numpy.typing as npt

from slidingpuzzle.board import (
    BLANK_TILE,
    Board,
    find_blank,
    find_tile,
    get_goal_tile,
    get_goal_y,
    get_goal_x,
    get_goal_yx,
    freeze_board,
    is_solved,
    swap_tiles,
)


Heuristic: TypeAlias = Callable[[Board], int | float]

log = logging.getLogger(__name__)
manhattan_tables = {}  # (h, w): {(r, c, tile): manhattan dist}


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
    h, w = board.shape
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
    h, w = board.shape
    dist = 0
    for y, row in enumerate(board):
        for x, tile in enumerate(row):
            if BLANK_TILE == tile:
                continue
            goal_y, goal_x = get_goal_yx(h, w, tile)
            if y != goal_y or x != goal_x:
                dist += 1
    return dist


def last_moves_distance(board: Board) -> int:
    r"""
    If the tiles required for the final move are not in appropriate positions, we must
    make at least 2 more moves to shuffle them. The constraints are weakened
    intentionally so that this heuristic can be constructively combined with Manhattan
    without violating admissibility.

    Args:
        board: The board

    Returns:
        2 if the both of the last two tiles adj. to the blank corner are outside of the
        blank row/col.
    """
    h, w = board.shape
    dist = 0
    # last moves enhancement (helps with tie-breaking)
    adj1 = get_goal_tile(h, w, (h - 2, w - 1))  # target tile in row next to goal
    adj2 = get_goal_tile(h, w, (h - 1, w - 2))  # target tile in col next to goal
    adj1_y, _ = find_tile(board, adj1)  # actual locations of targets on board
    _, adj2_x = find_tile(board, adj2)
    # if the tiles that must be the final move are not located in the goal row/col,
    # there will need to be additional moves made to shuffle them before the final move
    if adj1_y != h - 1 and adj2_x != w - 1:
        dist += 2
    return dist


def corner_tiles_distance(board: Board) -> int:
    r"""
    If tiles that belong in corners are not there, but the adj. tiles are correct,
    additional reshuffling will need to occur. We must ensure that row/col conflicts
    have not already accounted for this in order to combine admissibily with the
    Linear Conflict heuristic.
    """
    h, w = board.shape
    dist = 0

    def has_row_conflict(y, x) -> bool:
        for col1 in range(w):
            if BLANK_TILE == board[y, col1]:
                continue
            for col2 in range(col1 + 1, w):
                if BLANK_TILE == board[y, col2] or (col1 != x and col2 != x):
                    continue
                if col2 < col1:
                    return True
        return False

    def has_col_conflict(y, x) -> bool:
        for row1 in range(h):
            if BLANK_TILE == board[row1, x]:
                continue
            for row2 in range(row1 + 1, h):
                if BLANK_TILE == board[row2, x] or (row1 != y and row2 != y):
                    continue
                if row2 < row1:
                    return True
        return False

    def has_conflict(y, x) -> bool:
        return has_row_conflict(y, x) or has_col_conflict(y, x)

    # top-left corner
    if get_goal_tile(h, w, (0, 0)) != board[0, 0]:
        adj1 = get_goal_tile(h, w, (0, 1))
        if board[0, 1] == adj1 and not has_conflict(0, 1):
            dist += 2
        adj2 = get_goal_tile(h, w, (1, 0))
        if board[1, 0] == adj2 and not has_conflict(1, 0):
            dist += 2

    # top-right corner
    if get_goal_tile(h, w, (0, w - 1)) != board[0, w - 1]:
        adj1 = get_goal_tile(h, w, (0, w - 2))
        if board[0, w - 2] == adj1 and not has_conflict(0, w - 2):
            dist += 2
        adj2 = get_goal_tile(h, w, (1, w - 1))
        if board[1, w - 1] == adj2 and not has_conflict(1, w - 1):
            dist += 2

    # bottom-left corner
    if get_goal_tile(h, w, (h - 1, 0)) != board[h - 1, 0]:
        adj1 = get_goal_tile(h, w, (h - 1, 1))
        if board[h - 1, 1] == adj1 and not has_conflict(h - 1, 1):
            dist += 2
        adj2 = get_goal_tile(h, w, (h - 2, 0))
        if board[h - 2, 0] == adj2 and not has_conflict(h - 2, 0):
            dist += 2

    return dist


def linear_conflict_distance(board: Board) -> int:
    r"""
    Starts with Manhattan distance, then for each row and column, the number of tiles
    "in conflict" are identified, and 2 * this number is added to the distance.
    (It will take at least 2 additional moves to reshuffle the conflicting tiles into
    their correct positions.) This is an admissible improvement over Manhattan
    distance (Hansson, Mayer, Young, 1985). Additionally, the Last Moves and Corner
    enhancements are implemented (Korf and Taylor, 1996).

    Args:
        board: The board

    Returns:
        Estimated distance to goal.
    """
    h, w = board.shape
    dist = manhattan_distance(board)

    def line_generator() -> Iterator[tuple[npt.NDArray, npt.NDArray]]:
        r"""
        Helper to generate all lines on a board (rows followed by columns) along with
        a bool specifying whether the tile is in its goal line.

        Returns:
            A tuple of line index, line values, and goal positions..
        """
        for y, row in enumerate(board):
            yield row, np.fromiter(
                (y == get_goal_y(h, w, tile) for tile in row), dtype=bool
            )
        for x, col in enumerate(board.T):
            yield col, np.fromiter(
                (x == get_goal_x(h, w, tile) for tile in col), dtype=bool
            )

    def get_line_conflicts(line, goals) -> npt.NDArray[np.integer]:
        r"""
        Helper to return a list of conflict counts for a line.

        Args:
            line: The values in the line
            goals: The goal positions for each tile in line

        Returns:
            Conflict counts for each tile
        """
        conflicts = np.zeros(len(line), dtype=int)
        for pos1, (tile1, goal1) in enumerate(zip(line, goals)):
            # check if this tile is in it's goal line
            if not goal1 or BLANK_TILE == tile1:
                continue
            # now check tiles after this
            next_pos = pos1 + 1
            for pos2, (tile2, goal2) in enumerate(
                zip(line[next_pos:], goals[next_pos:]), next_pos
            ):
                # same checks for tile2
                if not goal2 or BLANK_TILE == tile2:
                    continue
                # check if these tiles are in conflict
                if tile2 < tile1:
                    conflicts[pos1] += 1
                    conflicts[pos2] += 1
        return conflicts

    for line, goals in line_generator():
        line = np.copy(line)  # don't modify original board
        while np.any(line_conflicts := get_line_conflicts(line, goals)):
            dist += 2
            line[np.argmax(line_conflicts)] = BLANK_TILE

    dist += last_moves_distance(board)
    dist += corner_tiles_distance(board)

    return dist


def prepare_manhattan_table(h, w) -> dict[tuple[int, int, int], int]:
    log.debug(f"Preparing first use of Manhattan table {h}x{w}x{h*w}...")
    table = {}
    for y in range(h):
        for x in range(w):
            for tile in range(h * w):
                if BLANK_TILE == tile:
                    table[(y, x, tile)] = 0
                else:
                    goal_y, goal_x = get_goal_yx(h, w, tile)
                    table[(y, x, tile)] = abs(y - goal_y) + abs(x - goal_x)
    return table


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
    h, w = board.shape
    table = manhattan_tables.get((h, w), None)
    if table is None:
        table = prepare_manhattan_table(h, w)
        manhattan_tables[(h, w)] = table

    dist = 0
    for y, row in enumerate(board):
        for x, tile in enumerate(row):
            dist += table[(y, x, tile)]
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
    h, w = board.shape
    board = np.copy(board)
    dist = 0

    def swap_first_misplaced(blank_yx: tuple[int, int]) -> None:
        for y in range(h):
            for x in range(w):
                if get_goal_yx(h, w, board[y][x]) != (y, x):
                    swap_tiles(board, (y, x), blank_yx)

    while not is_solved(board):
        blank_yx = find_blank(board)
        if blank_yx == get_goal_yx(h, w, BLANK_TILE):
            swap_first_misplaced(blank_yx)
        else:
            swap_tiles(board, get_goal_tile(h, w, blank_yx), blank_yx)
        dist += 1

    return dist
