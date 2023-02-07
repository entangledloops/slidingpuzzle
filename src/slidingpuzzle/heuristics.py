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


def has_row_conflict(board: Board, y: int, x: int) -> bool:
    r"""
    On row ``y``, we check every tile for conflict with column ``x``.

    Args:
        board: The board:
        y: The row of interest
        x: The column we will be checking for involvement in any conflicts

    Returns:
        True if column ``x`` is involved in any conflicts on row ``y``
    """
    h, w = board.shape
    for col1 in range(w):
        tile1 = board[y, col1]
        # check conflict conditions
        if BLANK == tile1 or get_goal_y(h, w, tile1) != y:
            continue
        for col2 in range(col1 + 1, w):
            # skip tiles that don't involve our x of interest
            if col1 != x and col2 != x:
                continue
            tile2 = board[y, col2]
            # check conflict conditions
            if BLANK == tile2 or get_goal_y(h, w, tile2) != y:
                continue
            if get_goal_x(h, w, tile2) < get_goal_x(h, w, tile1):
                return True
    return False


def has_col_conflict(board: Board, y: int, x: int) -> bool:
    r"""
    See :func:`has_row_conflict`. This is identical, but swaps rows for columns.
    """
    h, w = board.shape
    for row1 in range(h):
        tile1 = board[row1, x]
        # check conflict conditions
        if BLANK == tile1 or get_goal_x(h, w, tile1) != x:
            continue
        for row2 in range(row1 + 1, h):
            if row1 != y and row2 != y:
                continue
            tile2 = board[row2, x]
            # check conflict conditions
            if BLANK == tile2 or get_goal_x(h, w, tile2) != x:
                continue
            if get_goal_y(h, w, tile2) < get_goal_y(h, w, tile1):
                return True
    return False


def has_conflict(board: Board, y: int, x: int):
    r"""
    Combines :func:`has_row_conflict` and :func:`has_col_conflict` for convenience.
    """
    return has_row_conflict(board, y, x) or has_col_conflict(board, y, x)


def last_moves_distance(board: Board, relaxed: bool = True) -> int:
    r"""
    Immediately before the last move, one of the tiles adjacent to the blank tile must
    be in the corner. If not, we must make at least 2 more moves to shuffle.
    This is similar to :func:`corner_tiles_distance`, but specifically targets the goal
    corner, which has slightly different constraints.This function also ensures
    admissibility with :func:`manhattan_distance` and :func:`linear_conflict_distance`.
    If you aren't using this with other heuristics, you can set ``relaxed=True``.

    Note:
        Currently this only considers the very last moves, not the moves also prior to
        the last moves (2-removed) as discussed in Korf.

    Args:
        board: The board
        relaxed: If False, can be safely combined with  :func:`manhattan_distance` and
            :func:`linear_conflict_distance`.

    Returns:
        2 if neither of the final tiles to move are in the final corner, else 0.
    """
    h, w = board.shape
    adj1 = get_goal_tile(h, w, (h - 2, w - 1))  # tile in row next to goal
    adj2 = get_goal_tile(h, w, (h - 1, w - 2))  # tile in col next to goal
    corner = board[-1, -1]
    if corner == adj1 or corner == adj2 or is_solved(board):
        return 0
    if relaxed:
        return 2

    # coords of these tiles on cur. board
    adj1_y, adj1_x = find_tile(board, adj1)
    adj2_y, adj2_x = find_tile(board, adj2)
    # prevent interaction with Manhattan
    if adj1_y == h - 1 or adj2_x == w - 1:
        return 0
    # prevent interaction with linear conflict distance
    if adj1_y == h - 2 and has_row_conflict(board, adj1_y, adj1_x):
        return 0
    if adj2_x == w - 2 and has_col_conflict(board, adj2_y, adj2_x):
        return 0

    # if the tiles that must make the final move are not located next to their goal
    # row/col, they will need to shuffle before the final move
    return 2


def corner_tiles_distance(board: Board, relaxed: bool = True) -> int:
    r"""
    If tiles that belong in corners are not there, but the adjacent tiles are correct,
    additional reshuffling will need to occur. We must ensure that row/col conflicts
    have not already accounted for this in order to combine admissibily with the
    :func:`linear_conflict_distance`.

    Note:
        This heuristic is inadmissible if used on boards of insufficient size. For
        example, on a 3x3 board, the corners share some adjacent tiles and this
        will result in overcounting.

    Args:
        board: The board
        relaxed: If False, relevant tiles will be checked for linear conflicts.

    Returns:
        The additional distance required to shuffle corners into position.
    """
    h, w = board.shape
    dist = 0

    def check_adjacent(y, x) -> int:
        r""" "
        Helper to ensure this corner-adjacent tile is in its correct position
        and is not involved in a conflict.
        """
        adj = get_goal_tile(h, w, (y, x))
        if adj != board[y, x] or (not relaxed and has_conflict(board, y, x)):
            return 0
        return 2

    def check_corner(y, x, adj1, adj2) -> int:
        r"""
        Checks if:
            - Corner tile is out of position
            - Corner adjacents are in correct position

        Returns:
            Either 0, 2, or 4 to be added to distance.
        """
        corner = get_goal_tile(h, w, (y, x))
        if BLANK != board[y, x] and corner != board[y, x]:
            return check_adjacent(*adj1) + check_adjacent(*adj2)
        return 0

    # top-left corner
    dist += check_corner(0, 0, (0, 1), (1, 0))
    # top-right corner
    dist += check_corner(0, w - 1, (0, w - 2), (1, w - 1))
    # bottom-left corner
    dist += check_corner(h - 1, 0, (h - 1, 1), (h - 2, 0))

    return dist


def linear_conflict_distance(board: Board, optimized: bool = True) -> int:
    r"""
    Starts with Manhattan distance, then for each row and column, the number of tiles
    "in conflict" are identified, and 2 * this number is added to the total distance.
    (It will take at least 2 additional moves to reshuffle the conflicting tiles into
    their correct positions.) This is an admissible improvement over
    :func:`manhattan_distance` (`Hansson, Mayer, Young, 1985 <hannson_>`_).
    The :func:`last_moves_distance` and :func:`corner_tiles_distance` are included
    (`Korf and Taylor, 1996 <korf_>`_).

    Args:
        board: The board
        optimized: If False, will not include :func:`manhattan_distance`,
            :func:`last_moves_distance` and :func:`corner_tiles_distance`. It will
            return only the base number of linear conflicts.

    Returns:
        Estimated distance to goal.

    .. _hannson:
        https://academiccommons.columbia.edu/doi/10.7916/D8154QZT/download

    .. _korf:
        https://www.aaai.org/Library/AAAI/1996/aaai96-178.php
    """
    h, w = board.shape
    dist = manhattan_distance(board) if optimized else 0

    def line_generator() -> Iterator[tuple[npt.NDArray, npt.NDArray]]:
        r"""
        Helper to generate all lines on a board (rows followed by columns) along with
        a bool specifying whether the tile is in its goal line.

        Returns:
            A tuple of line index, line values, and goal positions.
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
            line: The tile values in the line
            goals: True/False if goal position for each tile is in this line

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
        while np.any(conflicts := get_line_conflicts(line, goals)):
            dist += 2
            line[np.argmax(conflicts)] = BLANK  # remove largest conflict

    # we can't use corner distance on tiny boards, b/c it not only conflicts with
    # last moves distance, it also conflicts with itself
    if optimized:
        if h > 3 and w > 3:
            dist += corner_tiles_distance(board, relaxed=False)
        dist += last_moves_distance(board, relaxed=False)

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
