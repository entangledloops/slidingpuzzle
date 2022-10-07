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


import collections
import copy
import random
import sys

from slidingpuzzle.state import SearchResult, State


def new_board(h, w) -> tuple[list[int]]:
    """
    Return a new board in the default solved state.
    """
    return tuple([(y * w) + x + 1 for x in range(w)] for y in range(h))


def print_board(board: tuple[list[int]], file=sys.stdout) -> None:
    empty_tile = len(board) * len(board[0])
    # the longest str we need to print is the largest tile number
    max_width = len(str(empty_tile - 1))
    for row in board:
        for tile in row:
            if tile == empty_tile:
                print(" " * max_width, end=" ", file=file)
            else:
                print(str(tile).ljust(max_width), end=" ", file=file)
        print(file=file)


def get_empty_yx(board: tuple[list[int]]) -> tuple[int, int]:
    """
    Locate the empty tile's (y, x)-coord.
    """
    empty_tile = len(board) * len(board[0])
    for y, row in enumerate(board):
        for x, tile in enumerate(row):
            if tile == empty_tile:
                return y, x
    raise ValueError("No empty tile found.")


def get_next_moves(
    board: tuple[list[int]], empty_pos: tuple[int]
) -> list[tuple[int, int]]:
    """
    Return a list of (y, x)-coords that are tile positions capable of
    swapping with the empty tile.
    """
    y, x = empty_pos
    moves = []
    for dy, dx in ((0, -1), (0, 1), (-1, 0), (1, 0)):
        if 0 <= y + dy < len(board) and 0 <= x + dx < len(board[0]):
            moves.append((y + dy, x + dx))
    return moves


def swap_tiles(board, pos1: tuple[int, int], pos2: tuple[int, int]) -> None:
    """
    Mutates the board by swapping a pair of tiles.
    """
    y1, x1 = pos1
    y2, x2 = pos2
    board[y1][x1], board[y2][x2] = board[y2][x2], board[y1][x1]


def count_inversions(board) -> int:
    """
    From each tile, count the number of tiles that are out of place.
    Returns the sum of all counts. See `is_solvable`.
    """
    h, w = len(board), len(board[0])
    empty_tile = h * w
    inversions = 0
    for tile1 in range(empty_tile):
        for tile2 in range(tile1 + 1, empty_tile):
            t1 = board[tile1 // w][tile1 % w]
            t2 = board[tile2 // w][tile2 % w]
            if empty_tile in (t1, t2):
                continue
            if t2 < t1:
                inversions += 1
    return inversions


def is_solvable(board) -> bool:
    """
    Returns True if it is possible to solve this board.

    The original idea comes from:
    https://www.cs.princeton.edu/courses/archive/spring21/cos226/assignments/8puzzle/specification.php

    This has been augmented to support non-square boards.
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


def shuffle_board(board) -> None:
    """Shuffles a board in place quickly and then verifies it is solvable. If not, repeat."""
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


def shuffle_board_slow(board, num_moves = None) -> None:
    """
    Shuffles a board in place by making random legal moves.

    :param num_moves: Number of moves to shuffle. If None, (h*w)^2 will be used.
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


def get_next_states(state: State) -> list[State]:
    """
    Returns a list of viable next states, given the current state.
    Although the state is fully captured by only the board, it is
    convenient and more efficient to also track the empty position.
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


def print_result(result: SearchResult, file=sys.stdout):
    """
    Print an abbreviated search result.
    """
    print(
        (
            f"solution_len={len(result.solution)}, "
            f"generated={result.generated}, "
            f"expanded={result.expanded}, "
            f"unvisited={len(result.unvisited)}, "
            f"visited={len(result.visited)}"
        ),
        file=file,
    )


def search(board, algorithm="bfs", heuristic=None) -> SearchResult:
    """
    Returns a `SearchResult` containing a list of moves to solve the puzzle from
    the initial state along with some search statistics.

    Requested algorithm may be one of: ["bfs", "dfs", "greedy", "a*"]
    See `heuristics.py` for available heuristic functions or provide your own.

    If a heuristic is provided with "bfs" or "dfs", it is only used to sort
    the locally generated nodes on each expansion, but not the entire open list.

    If "greedy" or "a*" is used, the entire open list is sorted. The "greedy"
    algorithm sorts only on the heuristic, while "a*" uses heuristic + length
    of the solution.
    """
    if algorithm not in ["bfs", "dfs", "greedy", "a*"]:
        raise ValueError(f"Unknown algorithm: '{algorithm}'")
    if heuristic is None:

        def heuristic(*_):
            return 0

    # prepare initial state
    empty_pos = get_empty_yx(board)
    initial_state = State(copy.deepcopy(board), empty_pos, [])
    goal = new_board(len(board), len(board[0]))
    unvisited = (
        collections.deque([initial_state]) if algorithm == "bfs" else [initial_state]
    )  # open nodes
    visited = set()  # closed nodes
    expanded = 0
    generated = 0

    while unvisited:
        state = unvisited.popleft() if algorithm == "bfs" else unvisited.pop()
        expanded += 1

        # check for duplicate states
        frozen_board = tuple(tuple(row) for row in state.board)
        if frozen_board in visited:
            continue
        visited.add(frozen_board)

        # goal check
        if state.board == goal:
            return SearchResult(generated, expanded, unvisited, visited, state.history)

        # obtain next states
        next_states = get_next_states(state)
        if "a*" == algorithm:
            unvisited.extend(next_states)
            unvisited.sort(key=lambda s: len(s.history) + heuristic(s.board), reverse=True)
        elif "greedy" == algorithm:
            unvisited.extend(next_states)
            unvisited.sort(key=lambda s: heuristic(s.board), reverse=True)
        else:
            next_states.sort(key=lambda s: heuristic(s.board), reverse=True)
            unvisited.extend(next_states)
        generated += len(next_states)

    raise RuntimeError("All possible boards explored, but no solution could be found.")
