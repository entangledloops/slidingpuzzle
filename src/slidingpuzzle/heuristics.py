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

from typing import Callable, Iterator, Optional, TypeAlias

import enum
import logging
import math

import numpy as np
import numpy.typing as npt

from slidingpuzzle.board import (
    BLANK,
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
            if BLANK == tile:
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
            if BLANK == tile:
                continue
            goal_y, goal_x = get_goal_yx(h, w, tile)
            if y != goal_y or x != goal_x:
                dist += 1
    return dist


def has_row_conflict(
    board: Board, y: int, x: int, ignore_x: Optional[int] = None
) -> bool:
    r"""
    On row ``y``, we check every tile for conflict with column ``x`` (ignoring
    ``ignore_x``). The skip variable is used to ignore a specific column, if desired.

    Args:
        board: The board:
        y: The row of interest
        x: The column we will be checking for involvement in any conflicts
        ignore_x: A column to ignore during conflict checks

    Returns:
        True if column ``x`` is involved in any conflicts on row ``y``
    """
    _, w = board.shape
    for col1 in range(w):
        tile1 = board[y, col1]
        if BLANK == tile1 or col1 == ignore_x:
            continue
        for col2 in range(col1 + 1, w):
            tile2 = board[y, col2]
            # skip tiles that don't involve our x of interest
            if BLANK == tile2 or col2 == ignore_x or (col1 != x and col2 != x):
                continue
            if tile2 < tile1:
                return True
    return False


def has_col_conflict(
    board: Board, y: int, x: int, ignore_y: Optional[int] = None
) -> bool:
    r"""
    See :func:`has_row_conflict`. This is identical, but swaps rows for columns.
    """
    h, _ = board.shape
    for row1 in range(h):
        tile1 = board[row1, x]
        if BLANK == tile1 or row1 == ignore_y:
            continue
        for row2 in range(row1 + 1, h):
            tile2 = board[row2, x]
            if BLANK == tile2 or row2 == ignore_y or (row1 != y and row2 != y):
                continue
            if tile2 < tile1:
                return True
    return False


def has_conflict(
    board: Board,
    y: int,
    x: int,
    ignore_y: Optional[int] = None,
    ignore_x: Optional[int] = None,
):
    r"""
    Combines :func:`has_row_conflict` and :func:`has_col_conflict` for convenience.
    """
    return has_row_conflict(board, y, x, ignore_x) or has_col_conflict(
        board, y, x, ignore_y
    )


def last_moves_distance(board: Board) -> int:
    r"""
    If the tiles required for the final move are not in appropriate positions, we must
    make at least 2 more moves to shuffle them. This is similar to
    :func:`corner_tiles_distance`, but specifically targets the goal corner, which has
    slightly different constraints.

    Note:
        Currently this only considers the very last moves, not the moves also prior to
        the last moves (2-removed) as discussed in Korf.

    Args:
        board: The board

    Returns:
        2 if the both of the last two tiles adj. to the blank corner are outside of the
        blank row/col.
    """
    h, w = board.shape
    adj1 = get_goal_tile(h, w, (h - 2, w - 1))  # tile in row next to goal
    adj2 = get_goal_tile(h, w, (h - 1, w - 2))  # tile in col next to goal
    adj1_y, adj1_x = find_tile(board, adj1)  # coords of these tiles on cur. board
    adj2_y, adj2_x = find_tile(board, adj2)
    corner = board[-1, -1]

    # can't add anything if corner neighbors are already in their dest.
    if adj1_y == h - 1 or adj2_x == w - 1:
        return 0
    # prevent over-counting when used with linear conflict distance
    if adj1_y == h - 2 and (BLANK == corner or has_row_conflict(board, adj1_y, adj1_x)):
        return 0
    if adj2_x == w - 2 and (BLANK == corner or has_col_conflict(board, adj2_y, adj2_x)):
        return 0

    # if the tiles that must make the final move are not located next to their goal
    # row/col, they will need to shuffle before the final move
    return 2


def corner_tiles_distance(board: Board, ignore_conflicts: bool = False) -> int:
    r"""
    If tiles that belong in corners are not there, but the adjacent tiles are correct,
    additional reshuffling will need to occur. We must ensure that row/col conflicts
    have not already accounted for this in order to combine admissibily with the
    :func:`linear_conflict_distance`.

    Args:
        board: The board
        ignore_conflicts: If True, tiles adjacent to corners that have conflicts in
            their row/col will still be counted. When True, this cannot be admissibly
            combined with :func:`linear_conflict_distance`.

    Returns:
        The additional distance required to shuffle corners into position.
    """
    h, w = board.shape
    dist = 0

    def check_adjacent(y, x, corner) -> int:
        adj = get_goal_tile(h, w, (y, x))
        if adj != board[y, x] or (
            not ignore_conflicts and has_conflict(board, y, x, *corner)
        ):
            return 0
        return 2

    def check_corner(y, x, adj1, adj2) -> int:
        if board[y, x] == BLANK:
            return 0
        corner = get_goal_tile(h, w, (y, x))
        if board[y, x] != corner and board[adj1] != corner and board[adj2] != corner:
            return check_adjacent(*adj1, (y, x)) + check_adjacent(*adj2, (y, x))
        return 0

    # top-left corner
    dist += check_corner(0, 0, (0, 1), (1, 0))
    # top-right corner
    dist += check_corner(0, w - 1, (0, w - 2), (1, w - 1))
    # bottom-left corner
    dist += check_corner(h - 1, 0, (h - 1, 1), (h - 2, 0))

    return dist


def linear_conflict_distance(board: Board) -> int:
    r"""
    Starts with Manhattan distance, then for each row and column, the number of tiles
    "in conflict" are identified, and 2 * this number is added to the distance.
    (It will take at least 2 additional moves to reshuffle the conflicting tiles into
    their correct positions.) This is an admissible improvement over
    :func:`manhattan_distance` (`Hansson, Mayer, Young, 1985 <hannson_>`_).
    The :func:`last_moves_distance` and :func:`corner_tiles_distance` are included
    (`Korf and Taylor, 1996 <korf_>`_).

    Args:
        board: The board

    Returns:
        Estimated distance to goal.

    .. _hannson:
        https://academiccommons.columbia.edu/doi/10.7916/D8154QZT/download

    .. _korf:
        https://www.aaai.org/Library/AAAI/1996/aaai96-178.php
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
            if not goal1 or BLANK == tile1:
                continue
            # now check tiles after this
            next_pos = pos1 + 1
            for pos2, (tile2, goal2) in enumerate(
                zip(line[next_pos:], goals[next_pos:]), next_pos
            ):
                # same checks for tile2
                if not goal2 or BLANK == tile2:
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
            line[np.argmax(line_conflicts)] = BLANK

    dist += last_moves_distance(board)
    dist += corner_tiles_distance(board)

    return dist


def prepare_manhattan_table(h, w) -> dict[tuple[int, int, int], int]:
    log.debug(f"Preparing first use of Manhattan table {h}x{w}x{h*w}...")
    table = {}
    for y in range(h):
        for x in range(w):
            for tile in range(h * w):
                if BLANK == tile:
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
        if blank_yx == get_goal_yx(h, w, BLANK):
            swap_first_misplaced(blank_yx)
        else:
            swap_tiles(board, get_goal_tile(h, w, blank_yx), blank_yx)
        dist += 1

    return dist
