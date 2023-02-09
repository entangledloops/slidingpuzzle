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

import numpy as np
import numpy.typing as npt


BLANK = 0
Board: TypeAlias = npt.NDArray
FrozenBoard: TypeAlias = tuple[tuple[int, ...], ...]


def new_board(h: int, w: int, dtype: Optional[npt.DTypeLike] = None) -> Board:
    r"""
    Create a new board in the default solved state.

    Args:
        h: Height of the board.
        w: Width of the board.
        dtype: An optional numpy dtype for the board. Will be inferred if not provided.

    Returns:
        The new board.
    """
    board = np.arange(1, 1 + h * w, dtype=dtype).reshape(h, w)
    board[-1, -1] = BLANK
    return board


def from_rows(*rows: Iterable[int], dtype: Optional[npt.DTypeLike] = None) -> Board:
    r"""
    Constructs a new board from a provided iterable of rows.
    May throw a :class:`ValueError` if rows are unequal in size or duplicate values
    are found.

    Args:
        rows: One or more rows that constitute a board. Each row should contain ints.
        dtype: An optional numpy dtype for the board. Will be inferred if not provided.

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
            if not (isinstance(tile, int) or isinstance(tile, np.integer)):
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

    return np.array(board, dtype=dtype)


def from_iter(h: int, w: int, iter: Iterable[int]) -> Board:
    r"""
    Given an iterable of ints, will construct a board of size ``h x w`` in
    row-major order. The number of values should equal :math:`h \cdot w` and
    all values must be provided. (Any partial extra row will be discarded.)

    Args:
        h: Height of the board to construct
        w: Width of the board to construct
        iter: Iterable of values to construct board from

    Raises:
        TypeError: If a non-int value is found in a row.
        ValueError: If rows have unequal length or tiles are duplicated, missing, or
            have unexpected gaps.

    Returns:
        The constructed board.
    """
    # construct board from iterable
    rows = []
    row = []
    for tile in iter:
        row.append(tile)
        if len(row) == w:
            rows.append(row)
            row = []
    return from_rows(*rows)


def flatten_board(board: Board | FrozenBoard) -> list[int]:
    r"""
    Flattens a board to a list. Useful for quickly compressing the board
    state. Can be reconstructed using :func:`board_from_iter`.

    Args:
        board: The board to flatten

    Returns:
        A flat sequence of ints representing the board.
    """
    return [tile for row in board for tile in row]


def freeze_board(board: Board) -> FrozenBoard:
    r"""
    Obtain a frozen copy of the board that is hashable.

    Args:
        board: The board to freeze.

    Returns:
        A frozen copy of the board.
    """
    return tuple(tuple(int(col) for col in row) for row in board)


def print_board(board: Board | FrozenBoard, file=sys.stdout) -> None:
    r"""
    Convienance function for printing a formatted board.

    Args:
        board: The board to print.
        file: The target output file. Defaults to stdout.
    """
    board_size = len(board) * len(board[0])
    # the longest str we need to print is the largest tile number
    max_width = len(str(board_size - 1))
    for row in board:
        for tile in row:
            if tile == BLANK:
                print(" " * max_width, end=" ", file=file)
            else:
                print(str(tile).ljust(max_width), end=" ", file=file)
        print(file=file)


def get_goal_y(h: int, w: int, tile: int) -> int:
    r"""
    Given a board width and tile number, returns the goal row position of ``tile``.

    Args:
        h: Height of the board
        w: Width of the board
        tile: The tile number of interest

    Returns:
        The goal row (y-coord)
    """
    if BLANK == tile:
        return h - 1
    return (tile - 1) // w


def get_goal_x(h: int, w: int, tile: int) -> int:
    r"""
    Given a board width and tile number, returns the goal column position of ``tile``.

    Args:
        h: Height of the board
        w: Width of the board
        tile: The tile number of interest

    Returns:
        The goal column (x-coord)
    """
    if BLANK == tile:
        return w - 1
    return (tile - 1) % w


def get_goal_yx(h: int, w: int, tile: int) -> tuple[int, int]:
    r"""
    Given a board width and tile number, returns the goal (y, x) position of ``tile``.

    Args:
        h: Height of the board
        w: Width of the board
        tile: The tile number of interest

    Returns:
        A tuple (y, x) indicating the tile's goal position.
    """
    return get_goal_y(h, w, tile), get_goal_x(h, w, tile)


def get_goal_tile(h: int, w: int, pos: tuple[int, int]) -> int:
    r"""
    Given a board width and (y, x)-coord, return the tile number that belongs at (y, x).

    Args:
        h: Height of the board
        w: Width of the board
        pos: A (y, x)-position on the board.

    Returns:
        The tile number that belongs at (y, x).
    """
    if pos == (h - 1, w - 1):
        return BLANK
    return (w * pos[0]) + pos[1] + 1


def is_solved(board: Board) -> bool:
    r"""
    Determine if a board is in the solved state.

    Args:
        board: The board

    Returns:
        True if the board is solved, otherwise False.
    """
    h, w = board.shape
    for y, row in enumerate(board):
        for x, tile in enumerate(row):
            if tile != get_goal_tile(h, w, (y, x)):
                return False
    return True


def find_tile(board: Board | FrozenBoard, tile: int) -> tuple[int, int]:
    r"""
    Given a tile number, find the (y, x)-coord on the board.

    Args:
        board: The puzzle board.
        move: The

    Raises:
        ValueError: If the tile is not on the board.

    Returns:
        The tile position.
    """
    if isinstance(board, np.ndarray):
        y, x = np.where(board == tile)
        return y[0], x[0]
    for y, row in enumerate(board):
        for x in range(len(row)):
            if board[y][x] == tile:
                return y, x
    raise ValueError(f'There is no tile "{tile}" on the board.')


def find_blank(board: Board | FrozenBoard) -> tuple[int, int]:
    r"""
    Locate the blank tile's (y, x)-coord.
    Equivalent to ``find_tile(board, BLANK)``.

    Args:
        board: The puzzle board.

    Returns:
        The (y, x)-coord of the blank tile.
    """
    return find_tile(board, BLANK)


def get_next_moves(
    board: Board | FrozenBoard,
    blank_pos: Optional[tuple[int, int]] = None,
) -> list[tuple[int, int]]:
    r"""
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
    board: Board,
    tile1: tuple[int, int] | int,
    tile2: Optional[tuple[int, int] | int] = None,
) -> Board:
    r"""
    Mutates the board by swapping a pair of tiles.

    Args:
        board: The board to modify.
        tile1: The first tile position or value.
        tile2: The second tile position or value. If None, the blank will be located
            and used.

    Return:
        The modified board, for conveience chaining calls.
    """
    if isinstance(tile1, int) or isinstance(tile1, np.integer):
        tile1 = find_tile(board, tile1)
    if isinstance(tile2, int) or isinstance(tile2, np.integer):
        tile2 = find_tile(board, tile2)
    elif tile2 is None:
        tile2 = find_blank(board)

    y1, x1 = tile1
    y2, x2 = tile2
    board[y1, x1], board[y2, x2] = board[y2, x2], board[y1, x1]

    return board


def random_move(board: Board, blank_pos: Optional[tuple[int, int]] = None) -> Board:
    r"""
    Picks a random legal move and applies it to the board.

    Args:
        board: The board
        blank_pos: The position of the blank. If not provided, will be found.
    """
    moves = get_next_moves(board, blank_pos)
    return swap_tiles(board, random.choice(moves), blank_pos)


def count_inversions(board: Board) -> int:
    r"""
    From each tile, count the number of tiles that are out of place after this tile.
    Returns the sum of all counts. See :func:`is_solvable`.

    Args:
        board: The puzzle board.

    Returns:
        The count of inversions.
    """
    _, w = board.shape
    board_size = np.prod(board.shape)
    inversions = 0
    for i in range(board_size):
        t1 = board[i // w, i % w]
        if t1 == BLANK:
            continue
        for j in range(i + 1, board_size):
            t2 = board[j // w, j % w]
            if t2 == BLANK:
                continue
            if t2 < t1:
                inversions += 1
    return inversions


def is_solvable(board: Board) -> bool:
    r"""
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
    h, w = board.shape
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


def shuffle(board: Board) -> Board:
    r"""
    Shuffles a board (in place). Board is always solvable.

    Args:
        board: The board to shuffle.

    Returns:
        The shuffled board.
    """
    h, w = board.shape
    while True:
        for y in range(h):
            for x in range(w):
                pos1 = y, x
                pos2 = random.randrange(h), random.randrange(w)
                swap_tiles(board, pos1, pos2)
        if is_solvable(board):
            return board


def shuffle_lazy(
    board: Board, num_moves: Optional[int] = None, moves: Optional[list] = None
) -> Board:
    r"""
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
    h, w = board.shape
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


def solution_as_tiles(board: Board, solution: Iterable[tuple[int, int]]) -> list[int]:
    r"""
    Converts a list of (y, x)-coords indicating moves into tile numbers,
    given a starting board configuration.

    Args:
        board: The initial board we will apply moves to (does not alter board).
        solution: A list of move coordinates in (y, x) form.

    Returns:
        A list of ints, which indicate a sequence of tile numbers to move.
    """
    board = np.copy(board)
    tiles = []
    blank_pos = find_blank(board)
    for move in solution:
        y, x = move
        tiles.append(int(board[y][x]))
        swap_tiles(board, blank_pos, move)
        blank_pos = move
    return tiles


def visit(visited: set[FrozenBoard], board: Board) -> bool:
    r"""
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


def board_generator(
    h: int, w: int, start: int = 0, stop: Optional[int] = None
) -> Iterator[Board]:
    r"""
    Returns a generator that yields all solvable boards in lexicographical order.

    Args:
        h: Height of board
        w: Width of board
        start: Index of board to start with
        stop: The board index to stop at, or None if we should run until completion.
            The board at ``stop`` is not included, similar to :func:`range`.

    Yields:
        A board permutation
    """
    board_id = -1
    for values in itertools.permutations(range(h * w)):
        board = from_iter(h, w, values)
        if is_solvable(board):
            board_id += 1
            if stop is not None and board_id == stop:
                return
            if board_id < start:
                continue
            yield board
