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
A collection of functions for working with sliding tile puzzle boards.
"""

from typing import Iterable, Iterator, Optional, TypeAlias

import itertools
import random
import sys


BLANK_TILE = 0
Board: TypeAlias = tuple[list[int], ...]
FrozenBoard: TypeAlias = tuple[tuple[int, ...], ...]


def is_solved(board: Board) -> bool:
    """
    Determine if a board is in the solved state.

    Args:
        board: The board

    Returns:
        True if the board is solved, otherwise False.
    """
    h, w = len(board), len(board[0])
    for y, row in enumerate(board):
        for x, tile in enumerate(row):
            if tile != get_goal_tile(h, w, (y, x)):
                return False
    return True


def new_board(h: int, w: int) -> Board:
    """
    Create a new board in the default solved state.

    Args:
        h: Height of the board.
        w: Width of the board.

    Returns:
        The new board.
    """
    board = tuple([(y * w) + x + 1 for x in range(w)] for y in range(h))
    board[-1][-1] = BLANK_TILE
    return board


def board_from(*rows: Iterable[int]) -> Board:
    """
    Constructs a new board from a provided iterable of rows.
    May throw a `ValueError` if rows are unequal in size or duplicate values
    are found.

    Args:
        rows: One or more rows that constitute a board. Each row should contain ints.

    Raises:
        TypeError: If a non-int value is found in a row.
        ValueError: If rows have unequal length or tiles are duplicated, missing, or
            have unexpected gaps.

    Returns:
        The constructed board.
    """
    board = []
    for row in rows:
        new_row = []
        for tile in row:
            # ensure all values are ints
            if not isinstance(tile, int):
                raise TypeError(
                    f"Expected tile to be an int, but received '{type(tile)}'."
                )
            new_row.append(tile)
        board.append(new_row)

    # sanity check row lengths
    for r1, r2 in zip(board[:-1], board[1:]):
        if len(r1) != len(r2):
            raise ValueError(
                f"All rows must have the same length ({len(r1)} != {len(r2)})."
            )

    # check all values are present
    tiles = list(sorted(itertools.chain(*board)))
    if tiles != list(range(len(board) * len(board[0]))):
        raise ValueError("Your board is missing tiles or contains duplicates.")

    return tuple(board)


def board_from_values(h: int, w: int, values: Iterable[int]) -> Board:
    r"""
    Given an iterable of ints, will construct a board of size ``h x w`` in
    row-major order. The number of values must exactly equal :math:`h \cdot w` or
    a :class:`ValueError` will be raised.

    Args:
        h: Height of the board to construct
        w: Width of the board to construct
        values: Values to construct board from
    """
    # construct board from iterable
    board = []
    row = []
    for value in values:
        row.append(value)
        if len(row) == w:
            board.append(row)
            row = []

    # validate board shape
    for row in board:
        if len(row) != w or len(board) != h:
            raise ValueError("Not enough values provided")
    return tuple(board)


def freeze_board(board: Board) -> FrozenBoard:
    """
    Obtain a frozen copy of the board that is hashable.

    Args:
        board: The board to freeze.

    Returns:
        A frozen copy of the board.
    """
    return tuple(tuple(row) for row in board)


def print_board(board: Board | FrozenBoard, file=sys.stdout) -> Board:
    """
    Convienance function for printing a formatted board.

    Args:
        board: The board to print.
        file: The target output file. Defaults to stdout.

    Returns:
        The input board (useful for call chaining).
    """
    board_size = len(board) * len(board[0])
    # the longest str we need to print is the largest tile number
    max_width = len(str(board_size - 1))
    for row in board:
        for tile in row:
            if tile == BLANK_TILE:
                print(" " * max_width, end=" ", file=file)
            else:
                print(str(tile).ljust(max_width), end=" ", file=file)
        print(file=file)

    return board


def get_goal_yx(h: int, w: int, tile: int) -> tuple[int, int]:
    """
    Given a board width and tile number, returns the goal (y, x) position of ``tile``.

    Args:
        h: Height of the board
        w: Width of the board
        tile: The tile number of interest

    Returns:
        A tuple (y, x) indicating the tile's goal position.
    """
    if BLANK_TILE == tile:
        return h - 1, w - 1
    return (tile - 1) // w, (tile - 1) % w


def get_goal_tile(h: int, w: int, pos: tuple[int, int]) -> int:
    """
    Given a board width and (y, x)-coord, return the tile number that belongs at (y, x).

    Args:
        h: Height of the board
        w: Width of the board
        pos: A (y, x)-position on the board.

    Returns:
        The tile number that belongs at (y, x).
    """
    if pos == (h - 1, w - 1):
        return BLANK_TILE
    return (w * pos[0]) + pos[1] + 1


def find_tile(board: Board | FrozenBoard, tile: int) -> tuple[int, int]:
    """
    Given a tile number, find the (y, x)-coord on the board.

    Args:
        board: The puzzle board.
        move: The

    Returns:
        The tile position or None if
    """
    for y, row in enumerate(board):
        for x in range(len(row)):
            if board[y][x] == tile:
                return y, x
    raise ValueError(f'There is no tile "{tile}" on the board.')


def find_blank(board: Board | FrozenBoard) -> tuple[int, int]:
    """
    Locate the blank tile's (y, x)-coord.
    Equivalent to ``find_tile(board, BLANK_TILE)``.

    Args:
        board: The puzzle board.

    Returns:
        The (y, x)-coord of the blank tile.
    """
    return find_tile(board, BLANK_TILE)


def get_next_moves(
    board: Board | FrozenBoard,
    blank_pos: Optional[tuple[int, int]] = None,
) -> list[tuple[int, int]]:
    """
    Return a list of all possible moves.

    Args:
        board: The current puzzle board.
        blank_pos: The position of the empty tile.
            If not provided, it will be located automatically.

    Returns:
        A list of (y, x)-coords that are tile positions capable of
        swapping with the blank tile.
    """
    if blank_pos is None:
        blank_pos = find_blank(board)
    y, x = blank_pos
    moves = []
    for dy, dx in ((0, -1), (0, 1), (-1, 0), (1, 0)):
        if 0 <= y + dy < len(board) and 0 <= x + dx < len(board[0]):
            moves.append((y + dy, x + dx))
    return moves


def swap_tiles(
    board: Board, tile1: tuple[int, int] | int, tile2: tuple[int, int] | int
) -> Board:
    """
    Mutates the board by swapping a pair of tiles.

    Args:
        board: The board to modify.
        pos1: The first tile position or value.
        pos2: The second tile position or value.

    Return:
        The modified board, for conveience chaining calls.
    """
    if isinstance(tile1, int):
        tile1 = find_tile(board, tile1)
    if isinstance(tile2, int):
        tile2 = find_tile(board, tile2)

    y1, x1 = tile1
    y2, x2 = tile2
    board[y1][x1], board[y2][x2] = board[y2][x2], board[y1][x1]

    return board


def count_inversions(board: Board | FrozenBoard) -> int:
    """
    From each tile, count the number of tiles that are out of place after this tile.
    Returns the sum of all counts. See :func:`is_solvable`.

    Args:
        board: The puzzle board.

    Returns:
        The count of inversions.
    """
    h, w = len(board), len(board[0])
    board_size = h * w
    inversions = 0
    for i in range(board_size):
        t1 = board[i // w][i % w]
        if t1 == BLANK_TILE:
            continue
        for j in range(i + 1, board_size):
            t2 = board[j // w][j % w]
            if t2 == BLANK_TILE:
                continue
            if t2 < t1:
                inversions += 1
    return inversions


def apply_move(
    board: Board,
    move: tuple[int, int] | int,
    blank_pos: Optional[tuple[int, int]] = None,
) -> Board:
    """
    Applies a move to the board in place.

    Args:
        board: The puzzle board.
        move: The move to apply. Can be a (y, x)-coord or a tile int.
        blank_pos: The position of the blank tile. Will be found if not provided.

    Returns:
        The modified board, for convience chaining calls.
    """
    if isinstance(move, int):
        move = find_tile(board, move)
    if blank_pos is None:
        blank_pos = find_blank(board)
    return swap_tiles(board, move, blank_pos)


def apply_random_move(
    board: Board, blank_pos: Optional[tuple[int, int]] = None
) -> Board:
    """
    Picks a random legal move and applies it to the board.

    Args:
        board: The board
        blank_pos: The position of the blank. If not provided, will be found.
    """
    moves = get_next_moves(board, blank_pos)
    return apply_move(board, random.choice(moves))


def is_solvable(board: Board | FrozenBoard) -> bool:
    """
    Determines if it is possible to solve this board.

    Note:
        The algorithm counts `inversions`_ to determine solvability.
        The "standard" algorithm has been modified here to support
        non-square board sizes.

    Args:
        board: The puzzle board.

    Returns:
        bool: True if the board is solvable, False otherwise.

    .. _inversions:
        https://www.cs.princeton.edu/courses/archive/spring21/cos226/assignments/8puzzle/specification.php
    """
    inversions = count_inversions(board)
    h, w = len(board), len(board[0])
    if w % 2 == 0:
        y, _ = find_blank(board)
        if h % 2 == 0:
            if (inversions + y) % 2 != 0:
                return True
        else:
            if (inversions + y) % 2 == 0:
                return True
    elif inversions % 2 == 0:
        return True
    return False


def shuffle_board(board: Board) -> Board:
    """
    Shuffles a board (in place) and validates that the result is solvable.

    Args:
        board: The board to shuffle.

    Returns:
        The same board for chaining convience.
    """
    h, w = len(board), len(board[0])
    while True:
        # first shuffle the board
        for y in range(h):
            for x in range(w):
                pos1 = y, x
                pos2 = random.randrange(h), random.randrange(w)
                swap_tiles(board, pos1, pos2)

        if is_solvable(board):
            break
    return board


def shuffle_board_lazy(
    board: Board, num_moves: Optional[int] = None, moves: Optional[list] = None
) -> Board:
    """
    Shuffles a board in place by making random legal moves.
    Each move is first checked to avoid repeated states, although
    this does not guarantee the

    Args:
        board: The board to shuffle.
        num_moves (int): Number of random moves to make.
            If ``None``, ``(h + w) * 2`` will be used.
        moves: If a list is provided, the moves made will be appended.

    Returns:
        The same board for chaining convience.
    """
    h, w = len(board), len(board[0])
    if num_moves is None:
        num_moves = (h + w) * 2
    blank_pos = find_blank(board)
    visited: set[FrozenBoard] = set()
    for _ in range(num_moves):
        next_moves = get_next_moves(board, blank_pos)
        random.shuffle(next_moves)
        new_move = False  # tracks if this is a new move
        # try to find a new move
        while next_moves:
            next_move = next_moves.pop()
            swap_tiles(board, blank_pos, next_move)
            # have we been here before?
            if visit(visited, board):
                swap_tiles(board, blank_pos, next_move)  # undo move
            else:
                new_move = True
                break

        # if we couldn't find a new move, we're done
        if not new_move:
            return board

        # if we're tracking which moves we made, record this move
        if moves is not None:
            moves.append(next_move)

        # update blank_pos
        blank_pos = next_move
    return board


def solution_as_tiles(board: Board, solution: list[tuple[int, int]]) -> list[int]:
    """
    Converts a list of (y, x)-coords indicating moves into tile numbers,
    given a starting board configuration.

    Args:
        board: The initial board we will apply moves to (does not alter board).
        solution: A list of move coordinates in (y, x) form.

    Returns:
        A list of ints, which indicate a sequence of tile numbers to move.
    """
    board = tuple(row.copy() for row in board)
    tiles = []
    blank_pos = find_blank(board)
    for move in solution:
        y, x = move
        tiles.append(board[y][x])
        swap_tiles(board, blank_pos, move)
        blank_pos = move
    return tiles


def visit(visited: set[FrozenBoard], board: Board) -> bool:
    """
    Helper to check if this state already exists. Otherwise, record it.
    Returns True if we have already been here, False otherwise.

    Args:
        visited: Set of boards already seen.
        board: The current board.

    Returns:
        True if we've been here before.
    """
    frozen_board = freeze_board(board)
    if frozen_board in visited:
        return True
    visited.add(frozen_board)
    return False


def board_generator(h: int, w: int) -> Iterator[Board]:
    """
    Returns a generator that yields all possible boards, in numerical order.

    Args:
        h: Height of board
        w: Width of board

    Yields:
        A board permutation
    """
    size = h * w
    for values in itertools.permutations(range(size)):
        yield board_from_values(h, w, values)
