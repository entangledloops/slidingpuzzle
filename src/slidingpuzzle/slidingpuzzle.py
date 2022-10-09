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

from typing import Callable

import collections
import copy
import heapq
import random
import sys

from slidingpuzzle.heuristics import manhattan_distance
from slidingpuzzle.state import SearchResult, State


# prevent typos
EMPTY_TILE = 0
A_STAR = "a*"
BEAM = "beam"
BFS = "bfs"
DFS = "dfs"
GREEDY = "greedy"
IDA_STAR = "ida*"
IDDFS = "iddfs"
ALGORITHMS = (
    A_STAR,
    BEAM,
    BFS,
    DFS,
    GREEDY,
    IDA_STAR,
    IDDFS,
)


def new_board(h: int, w: int) -> tuple[list[int], ...]:
    """
    Create a new board in the default solved state.

    Args:
        h: Height of the board.
        w: Width of the board.

    Returns:
        The new board.
    """
    board = tuple([(y * w) + x + 1 for x in range(w)] for y in range(h))
    board[-1][-1] = 0
    return board


def print_board(board: tuple[list[int], ...], file=sys.stdout) -> None:
    """
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
            if tile == EMPTY_TILE:
                print(" " * max_width, end=" ", file=file)
            else:
                print(str(tile).ljust(max_width), end=" ", file=file)
        print(file=file)


def get_empty_yx(board: tuple[list[int], ...]) -> tuple[int, int]:
    """
    Locate the empty tile's (y, x)-coord.

    Args:
        board: The puzzle board.

    Returns:
        The (y, x)-coord of the empty tile.
    """
    for y, row in enumerate(board):
        for x, tile in enumerate(row):
            if tile == EMPTY_TILE:
                return y, x
    raise ValueError("No empty tile found.")


def get_next_moves(
    board: tuple[list[int], ...],
    empty_pos: tuple[int, int] = None,
) -> list[tuple[int, int]]:
    """
    Return a list of all possible moves.

    Args:
        board: The current puzzle board.
        empty_pos: The position of the empty tile.
            If not provided, it will be located automatically.

    Returns:
        A list of (y, x)-coords that are tile positions capable of
        swapping with the empty tile.
    """
    if empty_pos is None:
        empty_pos = get_empty_yx(board)
    y, x = empty_pos
    moves = []
    for dy, dx in ((0, -1), (0, 1), (-1, 0), (1, 0)):
        if 0 <= y + dy < len(board) and 0 <= x + dx < len(board[0]):
            moves.append((y + dy, x + dx))
    return moves


def swap_tiles(
    board: tuple[list[int], ...], pos1: tuple[int, int], pos2: tuple[int, int]
) -> None:
    """
    Mutates the board by swapping a pair of tiles.

    Args:
        board: The board to modify.
        pos1: The first tile position.
        pos2: The second tile position.
    """
    y1, x1 = pos1
    y2, x2 = pos2
    board[y1][x1], board[y2][x2] = board[y2][x2], board[y1][x1]


def count_inversions(board: tuple[list[int], ...]) -> int:
    """
    From each tile, count the number of tiles that are out of place.
    Returns the sum of all counts. See :func:`is_solvable`.

    Args:
        board: The puzzle board.

    Returns:
        The count of inversions.
    """
    h, w = len(board), len(board[0])
    board_size = h * w
    inversions = 0
    for tile1 in range(board_size):
        for tile2 in range(tile1 + 1, board_size):
            t1 = board[tile1 // w][tile1 % w]
            t2 = board[tile2 // w][tile2 % w]
            if EMPTY_TILE in (t1, t2):
                continue
            if t2 < t1:
                inversions += 1
    return inversions


def is_solvable(board: tuple[list[int], ...]) -> bool:
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
        y, _ = get_empty_yx(board)
        if h % 2 == 0:
            if (inversions + y) % 2 != 0:
                return True
        else:
            if (inversions + y) % 2 == 0:
                return True
    elif inversions % 2 == 0:
        return True
    return False


def shuffle_board(board: tuple[list[int], ...]) -> tuple[list[int], ...]:
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


def shuffle_board_slow(
    board: tuple[list[int], ...], num_moves: int = None
) -> tuple[list[int], ...]:
    """
    Shuffles a board in place by making random legal moves.

    Args:
        num_moves (int): Number of random moves to make.
            If ``None``, ``(h * w) ** 2`` will be used.

    Returns:
        The same board for chaining convience.
    """
    h, w = len(board), len(board[0])
    if num_moves is None:
        num_moves = (h * w) ** 2
    empty_pos = get_empty_yx(board)
    for _ in range(num_moves):
        next_moves = get_next_moves(board, empty_pos)
        random.shuffle(next_moves)
        next_move = next_moves.pop()
        swap_tiles(board, empty_pos, next_move)
        empty_pos = next_move
    return board


def get_next_states(state: State) -> list[State]:
    """
    Creates a list of next viable states, given the current state.
    """
    moves = get_next_moves(state.board, state.empty_pos)
    next_states = []
    for move in moves:
        # construct new altered board
        next_board = copy.deepcopy(state.board)
        swap_tiles(next_board, state.empty_pos, move)
        # record history
        next_history = copy.deepcopy(state.history)
        next_history.append(move)
        # after moving, the move_pos is now the empty_pos
        next_state = State(next_board, move, next_history)
        next_states.append(next_state)
    return next_states


def search(
    board: tuple[list[int], ...],
    algorithm: str = BFS,
    heuristic: Callable[[tuple[list[int], ...]], int | float] = None,
    **kwargs,
) -> SearchResult:
    """
    Searches for a set of moves that take the provided board state to the
    solved state.

    Requested algorithm may be one of:
        "a*", "beam", "bfs", "dfs", "greedy", "ida*", "iddfs"

    See :mod:`heuristics` for heuristic functions or provide your own.

    If a heuristic is provided with "bfs", "dfs", or "iddfs", it is only used
    to sort the locally generated nodes on each expansion, but not the entire
    open list. You may also provide a "bound" via the kwargs described below
    to limit the search depth.

    If "greedy" or "a*" is used, the entire open list is sorted. The "greedy"
    algorithm sorts only on the heuristic, while "a*" uses heuristic + length
    of the solution.

    Of the provided algorithms, only beam search is incomplete. This means it
    may miss the goal, even thought the board is solvable.

    The algorithms support some additional kwargs that can be used to
    customize their behavior. These are:

    ..  code-block:: python

        a*: {
            "bound": default is float("inf"),
                restricts search to f-values < bound
            "weight": default is 1
        }
        beam: {
            "bound": default is float("inf"),
                restricts search to f-values < bound
            "width", default is 3
        }
        bfs: {
            "bound": default is float("inf"),
                restricts search to depth < bound
        }
        dfs: {
            "bound": default is float("inf"),
                restricts search to depth < bound
        }
        ida*: {
            "bound", default is manhattan_distance(board),
                restricts search to f-values < bound
            "weight": default is 1
        }
        iddfs: {
            "bound", default is manhattan_distance(board),
                restricts search to depth < bound
        }

    Args:
        board: The initial board state to search.
        algorithm (str): The algorithm to use for search.
        heuristic: A function to evaluate the board.
            See :mod:`slidingpuzzle.heuristics`.

    Returns:
        Returns a :class:`SearchResult` containing a list of moves to solve the
        puzzle from the initial state along with some search statistics. Solution
        may be None if no solution was found.
    """

    if not is_solvable(board):
        raise ValueError("The provided board is not solvable.")
    algorithm = algorithm.strip().lower()
    if algorithm not in ALGORITHMS:
        raise ValueError(f'Unknown algorithm: "{algorithm}"')
    if heuristic is None:

        # if no heuristic is provided, treat all nodes equally
        def heuristic(*_):
            return 0

    # only relevant for some algorithms
    a_star_weight = kwargs.get("weight", 1)
    beam_width = kwargs.get("width", 3)
    # there can only ever be 4 moves possible, so anything larger reduces to 4
    if beam_width > 4:
        beam_width = 4

    if algorithm in (IDA_STAR, IDDFS):
        # initial upper bound for ida* is the minimum number of moves possible
        bound = kwargs.get("bound", manhattan_distance(board))
    else:
        bound = kwargs.get("bound", float("inf"))

    # this is used by iterative deepening algorithms to determine when the
    # search space has been exhausted and/or to set the next iteration bound
    next_bound = float("inf")

    # prepare a func that will be used to evaluate states, depending upon alg.
    if algorithm in (A_STAR, IDA_STAR):

        def evaluate(state):
            return len(state.history) + a_star_weight * heuristic(state.board)

    else:

        def evaluate(state):
            return heuristic(state.board)

    # prepare initial state
    board = copy.deepcopy(board)  # don't modify or point to orig board
    goal = new_board(len(board), len(board[0]))
    empty_pos = get_empty_yx(board)
    expanded = 0
    generated = 0
    initial_state = State(board, empty_pos, [])
    visited = set()  # closed states
    if algorithm in (BEAM, BFS):
        unvisited = collections.deque([initial_state])
    else:
        unvisited = [initial_state]

    def get_next_state() -> State:
        """
        A helper function to properly remove the next state based upon
        the underlying data structure.
        """
        if algorithm in (BEAM, BFS):
            return unvisited.popleft()
        elif algorithm in (A_STAR, GREEDY, IDA_STAR):
            return heapq.heappop(unvisited)
        else:
            return unvisited.pop()

    def add_states(states) -> None:
        """Add provided states to unvisited."""
        if algorithm in (A_STAR, GREEDY, IDA_STAR):
            for s in states:
                heapq.heappush(unvisited, s)
        else:
            unvisited.extend(states)

    while unvisited:
        # some search algorithms need to restart search with an altered state.
        # see code after this inner loop
        while unvisited:
            # breadth-first search removes from the front, all others use the end
            state = get_next_state()
            expanded += 1

            # bounds check (if not using bound, this will always pass)
            if algorithm in (A_STAR, BEAM, IDA_STAR):
                if state.f > bound:
                    next_bound = min(next_bound, state.f)  # used by ida*
                    continue
            else:
                depth = len(state.history)
                if depth > bound:
                    next_bound = depth  # used by iddfs
                    continue

            # most algorithms check for duplicate states
            if algorithm not in (IDA_STAR, IDDFS):
                frozen_board = tuple(tuple(row) for row in state.board)
                if frozen_board in visited:
                    continue
                visited.add(frozen_board)

            # goal check
            if state.board == goal:
                return SearchResult(
                    board, generated, expanded, list(unvisited), visited, state.history
                )

            # compute value of next states
            next_states = get_next_states(state)
            for next_state in next_states:
                next_state.f = evaluate(next_state)

            # update the open list
            if algorithm in (A_STAR, GREEDY, IDA_STAR):
                add_states(next_states)
            else:
                # these algorithms sort only local nodes
                next_states.sort(reverse=True)
                # we should only store the last beam_width states during beam search
                if BEAM == algorithm:
                    next_states = next_states[-beam_width:]
                add_states(next_states)
            generated += len(next_states)

        # do we need to restart search?
        if algorithm in (IDA_STAR, IDDFS) and bound != next_bound:
            bound = next_bound
            if IDA_STAR == algorithm:
                next_bound = float("inf")
            unvisited.append(initial_state)

    # if we are here, no solution was found
    return SearchResult(board, generated, expanded, list(unvisited), visited, None)


def solution_as_tiles(
    board: tuple[list[int], ...], solution: list[tuple[int, int]]
) -> list[int]:
    """
    Converts a list of (y, x)-coords indicating moves into tile numbers,
    given a starting board configuration.

    Args:
        board: The initial board we will apply moves to (does not alter board).
        solution: A list of move coordinates in (y, x) form.

    Returns:
        A list of ints, which indicate a sequence of tile numbers to move.
    """
    board = copy.deepcopy(board)
    tiles = []
    empty_pos = get_empty_yx(board)
    for move in solution:
        y, x = move
        tiles.append(board[y][x])
        swap_tiles(board, empty_pos, move)
        empty_pos = move
    return tiles
