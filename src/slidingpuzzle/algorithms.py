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
Search algorithms
"""

import collections
import heapq

from slidingpuzzle.board import (
    get_empty_yx,
    get_next_moves,
    is_solvable,
    shuffle_board,
    new_board,
    swap_tiles,
    visit,
)
from slidingpuzzle.heuristics import manhattan_distance
from slidingpuzzle.state import State, SearchResult


# all supported algorithms

A_STAR = "a*"
BEAM = "beam"
BFS = "bfs"
DFS = "dfs"
GREEDY = "greedy"
IDA_STAR = "ida*"
IDDFS = "iddfs"
ALGORITHMS = (
    A_STAR,  # f(n) = g(n) + W * h(n)
    BEAM,
    BFS,
    DFS,
    GREEDY,  # f(n) = h(n)
    IDA_STAR,
    IDDFS,
)


def get_next_states(state: State) -> list[State]:
    """
    Creates a list of next viable states, given the current state.
    """
    moves = get_next_moves(state.board, state.empty_pos)
    next_states = []
    for move in moves:
        # construct new altered board
        next_board = tuple(row.copy() for row in state.board)
        swap_tiles(next_board, state.empty_pos, move)
        # record history
        next_history = state.history.copy()
        next_history.append(move)
        # after moving, the move_pos is now the empty_pos
        next_state = State(next_board, move, next_history)
        next_states.append(next_state)
    return next_states


def a_star(board: tuple[list[int], ...], **kwargs) -> SearchResult:
    # args
    depth_bound = kwargs.get("depth_bound", float("inf"))
    f_bound = kwargs.get("f_bound", float("inf"))
    detect_dupes = kwargs.get("detect_dupes", True)
    heuristic = kwargs.get("heuristic", manhattan_distance)
    weight = kwargs.get("weight", 1)

    # initial state
    goal = new_board(len(board), len(board[0]))
    empty_pos = get_empty_yx(board)
    initial_state = State(board, empty_pos)
    unvisited = [initial_state]
    visited = set()

    # stats
    generated, expanded = 0, 0

    while unvisited:
        state = heapq.heappop(unvisited)
        expanded += 1

        # goal check
        if goal == state.board:
            return SearchResult(
                board, generated, expanded, unvisited, visited, state.history
            )

        # bound
        if len(state.history) > depth_bound or state.f > f_bound:
            continue

        # duplicate detection
        if detect_dupes and visit(visited, state.board):
            continue

        # children
        next_states = get_next_states(state)
        for state in next_states:
            state.f = len(state.history) + weight * heuristic(state.board)
            heapq.heappush(unvisited, state)
        generated += len(next_states)

    # if we are here, no solution was found
    return SearchResult(board, generated, expanded, list(unvisited), visited, None)


def beam(board: tuple[list[int], ...], **kwargs) -> SearchResult:
    # args
    depth_bound = kwargs.get("depth_bound", float("inf"))
    f_bound = kwargs.get("f_bound", float("inf"))
    detect_dupes = kwargs.get("detect_dupes", True)
    heuristic = kwargs.get("heuristic", manhattan_distance)
    width = kwargs.get("width", 3)

    # initial state
    goal = new_board(len(board), len(board[0]))
    empty_pos = get_empty_yx(board)
    initial_state = State(board, empty_pos)
    unvisited = collections.deque([initial_state])
    visited = set()

    # stats
    generated, expanded = 0, 0

    while unvisited:
        state = unvisited.popleft()
        expanded += 1

        # goal check
        if goal == state.board:
            return SearchResult(
                board, generated, expanded, unvisited, visited, state.history
            )

        # bound
        if len(state.history) > depth_bound or state.f > f_bound:
            continue

        # duplicate detection
        if detect_dupes and visit(visited, state.board):
            continue

        # children
        next_states = get_next_states(state)
        for state in next_states:
            state.f = heuristic(state.board)
        next_states.sort()
        next_states = next_states[-width:]
        unvisited.extend(next_states)
        generated += len(next_states)

    # if we are here, no solution was found
    return SearchResult(board, generated, expanded, list(unvisited), visited, None)


def bfs(board: tuple[list[int], ...], **kwargs) -> SearchResult:
    # args
    depth_bound = kwargs.get("depth_bound", float("inf"))
    detect_dupes = kwargs.get("detect_dupes", True)

    # initial state
    goal = new_board(len(board), len(board[0]))
    empty_pos = get_empty_yx(board)
    initial_state = State(board, empty_pos)
    unvisited = collections.deque([initial_state])
    visited = set()

    # stats
    generated, expanded = 0, 0

    while unvisited:
        state = unvisited.popleft()
        expanded += 1

        # goal check
        if goal == state.board:
            return SearchResult(
                board, generated, expanded, unvisited, visited, state.history
            )

        # bound
        if len(state.history) > depth_bound:
            continue

        # duplicate detection
        if detect_dupes and visit(visited, state.board):
            continue

        # children
        next_states = get_next_states(state)
        unvisited.extend(next_states)
        generated += len(next_states)

    # if we are here, no solution was found
    return SearchResult(board, generated, expanded, list(unvisited), visited, None)


def dfs(board: tuple[list[int], ...], **kwargs) -> SearchResult:
    # args
    depth_bound = kwargs.get("depth_bound", float("inf"))
    detect_dupes = kwargs.get("detect_dupes", True)

    # initial state
    goal = new_board(len(board), len(board[0]))
    empty_pos = get_empty_yx(board)
    initial_state = State(board, empty_pos)
    unvisited = [initial_state]
    visited = set()

    # stats
    generated, expanded = 0, 0

    while unvisited:
        state = unvisited.pop()
        expanded += 1

        # goal check
        if goal == state.board:
            return SearchResult(
                board, generated, expanded, unvisited, visited, state.history
            )

        # bound
        if len(state.history) > depth_bound:
            continue

        # duplicate detection
        if detect_dupes and visit(visited, state.board):
            continue

        # children
        next_states = get_next_states(state)
        unvisited.extend(next_states)
        generated += len(next_states)

    # if we are here, no solution was found
    return SearchResult(board, generated, expanded, list(unvisited), visited, None)


def greedy(board: tuple[list[int], ...], **kwargs) -> SearchResult:
    # args
    depth_bound = kwargs.get("depth_bound", float("inf"))
    f_bound = kwargs.get("f_bound", float("inf"))
    detect_dupes = kwargs.get("detect_dupes", True)
    heuristic = kwargs.get("heuristic", manhattan_distance)

    # initial state
    goal = new_board(len(board), len(board[0]))
    empty_pos = get_empty_yx(board)
    initial_state = State(board, empty_pos)
    unvisited = [initial_state]
    visited = set()

    # stats
    generated, expanded = 0, 0

    while unvisited:
        state = heapq.heappop(unvisited)
        expanded += 1

        # goal check
        if goal == state.board:
            return SearchResult(
                board, generated, expanded, unvisited, visited, state.history
            )

        # bound
        if len(state.history) > depth_bound or state.f > f_bound:
            continue

        # duplicate detection
        if detect_dupes and visit(visited, state.board):
            continue

        # children
        next_states = get_next_states(state)
        for state in next_states:
            state.f = heuristic(state.board)
            heapq.heappush(unvisited, state)
        generated += len(next_states)

    # if we are here, no solution was found
    return SearchResult(board, generated, expanded, list(unvisited), visited, None)


def ida_star(board: tuple[list[int], ...], **kwargs) -> SearchResult:
    # args
    depth_bound = kwargs.get("depth_bound", float("inf"))
    detect_dupes = kwargs.get("detect_dupes", True)
    heuristic = kwargs.get("heuristic", manhattan_distance)
    weight = kwargs.get("weight", 1)

    # initial state
    goal = new_board(len(board), len(board[0]))
    empty_pos = get_empty_yx(board)
    bound = manhattan_distance(board)
    initial_state = State(board, empty_pos)
    next_bound = float("inf")
    unvisited = [initial_state]
    visited = set()

    # stats
    generated, expanded = 0, 0

    while unvisited:
        while unvisited:
            state = unvisited.pop()
            expanded += 1

            # goal check
            if goal == state.board:
                return SearchResult(
                    board, generated, expanded, unvisited, visited, state.history
                )

            # bound
            if len(state.history) > depth_bound:
                continue
            if state.f > bound:
                next_bound = min(next_bound, state.f)
                continue

            # duplicate detection
            if detect_dupes and visit(visited, state.board):
                continue

            # children
            next_states = get_next_states(state)
            for state in next_states:
                state.f = len(state.history) + weight * heuristic(state.board)
            next_states.sort(reverse=True)
            unvisited.extend(next_states)
            generated += len(next_states)

        # do we need to restart?
        if bound != next_bound and next_bound != float("inf"):
            bound = next_bound
            next_bound = float("inf")
            unvisited.append(initial_state)
            visited.clear()

    # if we are here, no solution was found
    return SearchResult(board, generated, expanded, list(unvisited), visited, None)


def iddfs(board: tuple[list[int], ...], **kwargs) -> SearchResult:
    # args
    detect_dupes = kwargs.get("detect_dupes", True)

    # initial state
    bound = manhattan_distance(board)
    goal = new_board(len(board), len(board[0]))
    empty_pos = get_empty_yx(board)
    initial_state = State(board, empty_pos)
    next_bound = bound
    unvisited = [initial_state]
    visited = set()

    # stats
    generated, expanded = 0, 0

    while unvisited:
        while unvisited:
            state = unvisited.pop()
            expanded += 1

            # goal check
            if goal == state.board:
                return SearchResult(
                    board, generated, expanded, unvisited, visited, state.history
                )

            # bound
            if len(state.history) > bound:
                next_bound = len(state.history)
                continue

            # duplicate detection
            if detect_dupes and visit(visited, state.board):
                continue

            # children
            next_states = get_next_states(state)
            unvisited.extend(next_states)
            generated += len(next_states)

        # do we need to restart?
        if bound != next_bound:
            bound = next_bound
            unvisited.append(initial_state)
            visited.clear()

    # if we are here, no solution was found
    return SearchResult(board, generated, expanded, list(unvisited), visited, None)


ALGORITHMS_MAP = {
    A_STAR: a_star,
    BEAM: beam,
    BFS: bfs,
    DFS: dfs,
    GREEDY: greedy,
    IDA_STAR: ida_star,
    IDDFS: iddfs,
}


def search(
    board: tuple[list[int], ...],
    alg: str = A_STAR,
    **kwargs,
) -> SearchResult:
    """
    Searches for a set of moves that take the provided board state to the
    solved state.

    Requested ``alg`` may be one of:
        "a*", "beam", "bfs", "dfs", "greedy", "ida*", "iddfs"

    See :mod:`heuristics` for heuristic functions or provide your own.

    If a heuristic is provided with "bfs", "dfs", or "iddfs", it is only used
    to sort the locally generated nodes on each expansion, but not the entire
    open list. You may also provide a "bound" via the kwargs described below
    to limit the search depth.

    If "greedy" or "a*" is used, the entire open list is sorted. The "greedy"
    algorithm sorts only on the heuristic, while "a*" uses heuristic + length
    of the solution. Default heuristics are listed in the table below.

    Of the provided algorithms, only beam search is incomplete. This means it
    may miss the goal, even thought the board is solvable.

    The algorithms support some additional kwargs that can be used to
    customize their behavior. These are:

    ..  code-block:: python

        a*: {
            "depth_bound": default is float("inf"),
                restrict search to depth < depth_bound
            "detect_dupes": default is True,
                if False, will trade time for memory savings
            "f_bound": default is float("inf"),
                restricts search to f-values < bound
            "heuristic": default is manhattan_distance
            "weight": default is 1
        }
        beam: {
            "depth_bound": default is float("inf"),
                restrict search to depth < depth_bound
            "detect_dupes": default is True,
                if False, will trade time for memory savings
            "f_bound": default is float("inf"),
                restricts search to f-values < bound
            "heuristic": default is manhattan_distance
            "width", default is 3
        }
        bfs: {
            "depth_bound": default is float("inf"),
                restrict search to depth < depth_bound
            "detect_dupes": default is True,
                if False, will trade time for memory savings
        }
        dfs: {
            "depth_bound": default is float("inf")
                restrict search to depth < depth_bound
            "detect_dupes": default is True,
                if False, will trade time for memory savings
        }
        greedy: {
            "depth_bound": default is float("inf"),
                restrict search to depth < depth_bound
            "detect_dupes": default is True,
                if False, will trade time for memory savings
            "f_bound": default is float("inf"),
                restricts search to f-values < bound
        }
        ida*: {
            "depth_bound": default is float("inf")
                restrict search to depth < depth_bound
            "detect_dupes": default is True,
                if False, will trade time for memory savings
            "heuristic": default is manhattan_distance
            "weight": default is 1
        }
        iddfs: {
            "detect_dupes": default is True,
                if False, will trade time for memory savings
        }

    Args:
        board: The initial board state to search.
        algorithm (str): The algorithm to use for search.
            Use print(ALGORITHMS) to see available.
        kwargs: Algorithm arguments as described above.

    Returns:
        Returns a :class:`SearchResult` containing a list of moves to solve the
        puzzle from the initial state along with some search statistics.
    """

    if not is_solvable(board):
        raise ValueError("The provided board is not solvable.")
    alg = alg.strip().lower()
    if alg not in ALGORITHMS:
        raise ValueError(f'Unknown algorithm: "{alg}"')
    return ALGORITHMS_MAP[alg](board, **kwargs)


def eval_heuristic(
    h: int,
    w: int,
    heuristic,
    algorithm=A_STAR,
    num_iters: int = 64,
    **kwargs,
) -> float:
    """
    Runs search on ``num_iters`` random boards. Returns the average number of nodes
    expanded.

    Returns:
        The average number of states expanded.
    """
    total = 0
    for _ in range(num_iters):
        board = new_board(h, w)
        shuffle_board(board)
        result = search(board, alg=algorithm, heuristic=heuristic, **kwargs)
        total += result.expanded
    return total / num_iters


def compare(
    h: int,
    w: int,
    ha=manhattan_distance,
    hb=manhattan_distance,
    alga=A_STAR,
    algb=A_STAR,
    num_iters: int = 8,
    **kwargs,
) -> tuple[float, float]:
    """
    Runs search on ``num_iters`` random boards, trying both alga(board, ha) and
    algb(board, hb) on each board. Returns the average number of states expanded
    for both.

    Returns:
        A tuple containing (avg. A, avg. B).
    """
    total_a, total_b = 0, 0
    for _ in range(num_iters):
        board = new_board(h, w)
        shuffle_board(board)
        result = search(board, alg=alga, heuristic=ha, **kwargs)
        total_a += result.expanded
        result = search(board, alg=algb, heuristic=hb, **kwargs)
        total_b += result.expanded
    return total_a / num_iters, total_b / num_iters
