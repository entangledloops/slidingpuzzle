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

from typing import Optional

import collections
import heapq

from slidingpuzzle.board import (
    Board,
    FrozenBoard,
    get_empty_yx,
    get_next_moves,
    is_solvable,
    shuffle_board,
    new_board,
    swap_tiles,
    visit,
)
from slidingpuzzle.heuristics import linear_conflict_distance
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
    A_STAR,
    BEAM,
    BFS,
    DFS,
    GREEDY,
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


def a_star(board: Board, **kwargs) -> SearchResult:
    r"""
    A* heuristic search algorithm.
    Supports weights, depth bounds, and f-bounds.
    The distance of a search state from the goal is computed as:

    .. math::
        f(n) = g(n) + w \cdot h(n)

    or equivalently ``f(n) = len(state.history) + weight * heuristic(state.board)``
    Here, ``g(n)`` is the cost of the solution so far, ``w`` is the weight, and
    ``h(n)`` is the heuristic evaluation. When no heuristic is used, A* becomes
    breadth-first search. When ``weight != 1`` or an inadmissable heuristic is used,
    A* may return a suboptimal solution.

    Args:
        board: The board
        depth_bound (int): A limit to search depth. Default is :math:`\infty`.
        f_bound (float): A limit on state cost. Default is :math:`\infty`.
        detect_dupes (bool): Duplicate detection (i.e. track visited states).
            Default is ``True``.
        heuristic: A function that maps boards to an estimated cost-to-go.
            Default is :func:`slidingpuzzle.heuristics.linear_conflict_distance`.
        weight (float): A constant multiplier on heuristic evaluation

    Returns:
        A :class:`slidingpuzzle.state.SearchResult` with a solution and statistics
    """
    # args
    depth_bound = kwargs.get("depth_bound", float("inf"))
    f_bound = kwargs.get("f_bound", float("inf"))
    detect_dupes = kwargs.get("detect_dupes", True)
    heuristic = kwargs.get("heuristic", linear_conflict_distance)
    weight = kwargs.get("weight", 1)

    # initial state
    goal = new_board(len(board), len(board[0]))
    empty_pos = get_empty_yx(board)
    initial_state = State(board, empty_pos)
    unvisited = [initial_state]
    visited: set[FrozenBoard] = set()

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


def beam(board: Board, **kwargs) -> SearchResult:
    r"""
    Beam search is a variant of breadth-first search that sorts its children using a
    heuristic function and then drops child states to match the beam width. This search
    is incomplete, meaning it may miss a solution although it exists. It is useful for
    limiting memory usage in large search spaces.

    Args:
        board: The board
        depth_bound (int): A limit to search depth. Default is :math:`\infty`.
        f_bound (float): A limit on state cost. Default is :math:`\infty`.
        detect_dupes (bool): Duplicate detection (i.e. track visited states).
            Default is ``True``.
        heuristic: A function that maps boards to an estimated cost-to-go.
            Default is :func:`slidingpuzzle.heuristics.linear_conflict_distance`.
        width (int): The beam width. Default is ``3``.

    Returns:
        A :class:`slidingpuzzle.state.SearchResult` with a solution and statistics
    """
    # args
    depth_bound = kwargs.get("depth_bound", float("inf"))
    f_bound = kwargs.get("f_bound", float("inf"))
    detect_dupes = kwargs.get("detect_dupes", True)
    heuristic = kwargs.get("heuristic", linear_conflict_distance)
    width = kwargs.get("width", 3)

    # initial state
    goal = new_board(len(board), len(board[0]))
    empty_pos = get_empty_yx(board)
    initial_state = State(board, empty_pos)
    unvisited = collections.deque([initial_state])
    visited: set[FrozenBoard] = set()

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
        next_states = next_states[:width]
        unvisited.extend(next_states)
        generated += len(next_states)

    # if we are here, no solution was found
    return SearchResult(board, generated, expanded, list(unvisited), visited, None)


def bfs(board: Board, **kwargs) -> SearchResult:
    r"""
    Breadth-first search

    Args:
        board: The board
        depth_bound (int): A limit to search depth. Default is :math:`\infty`.
        detect_dupes (bool): Duplicate detection (i.e. track visited states).
            Default is ``True``.

    Returns:
        A :class:`slidingpuzzle.state.SearchResult` with a solution and statistics
    """
    # args
    depth_bound = kwargs.get("depth_bound", float("inf"))
    detect_dupes = kwargs.get("detect_dupes", True)

    # initial state
    goal = new_board(len(board), len(board[0]))
    empty_pos = get_empty_yx(board)
    initial_state = State(board, empty_pos)
    unvisited = collections.deque([initial_state])
    visited: set[FrozenBoard] = set()

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


def dfs(board: Board, **kwargs) -> SearchResult:
    r"""
    Depth-first search

    Args:
        board: The board
        depth_bound (int): A limit to search depth. Default is :math:`\infty`.
        detect_dupes (bool): Duplicate detection (i.e. track visited states).
            Default is ``True``.

    Returns:
        A :class:`slidingpuzzle.state.SearchResult` with a solution and statistics
    """
    # args
    depth_bound = kwargs.get("depth_bound", float("inf"))
    detect_dupes = kwargs.get("detect_dupes", True)

    # initial state
    goal = new_board(len(board), len(board[0]))
    empty_pos = get_empty_yx(board)
    initial_state = State(board, empty_pos)
    unvisited = [initial_state]
    visited: set[FrozenBoard] = set()

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


def greedy(board: Board, **kwargs) -> SearchResult:
    r"""
    Greedy best-first search. This search orders all known states using the provided
    heuristic and greedily chooses the state closest to the goal.

    Args:
        board: The board
        depth_bound (int): A limit to search depth. Default is :math:`\infty`.
        f_bound (float): A limit on state cost. Default is :math:`\infty`.
        detect_dupes (bool): Duplicate detection (i.e. track visited states).
            Default is ``True``.
        heuristic: A function that maps boards to an estimated cost-to-go.
            Default is :func:`slidingpuzzle.heuristics.linear_conflict_distance`.

    Returns:
        A :class:`slidingpuzzle.state.SearchResult` with a solution and statistics
    """
    # args
    depth_bound = kwargs.get("depth_bound", float("inf"))
    f_bound = kwargs.get("f_bound", float("inf"))
    detect_dupes = kwargs.get("detect_dupes", True)
    heuristic = kwargs.get("heuristic", linear_conflict_distance)

    # initial state
    goal = new_board(len(board), len(board[0]))
    empty_pos = get_empty_yx(board)
    initial_state = State(board, empty_pos)
    unvisited = [initial_state]
    visited: set[FrozenBoard] = set()

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


def ida_star(board: Board, **kwargs) -> SearchResult:
    r"""
    Iterative deepening A*. A depth-first search that uses an f-bound instead of depth
    to limit search. The next bound is set to the minimum increase in f-bound observed
    during the current iteration. See :func:`a_star`.

    Args:
        board: The board
        depth_bound (int): A limit to search depth. Default is :math:`\infty`.
        detect_dupes (bool): Duplicate detection (i.e. track visited states).
            Default is ``True``.
        heuristic: A function that maps boards to an estimated cost-to-go.
            Default is :func:`slidingpuzzle.heuristics.linear_conflict_distance`.
        weight (float): A constant multiplier on heuristic evaluation.
            Default is ``1``.

    Returns:
        A :class:`slidingpuzzle.state.SearchResult` with a solution and statistics
    """
    # args
    depth_bound = kwargs.get("depth_bound", float("inf"))
    detect_dupes = kwargs.get("detect_dupes", True)
    heuristic = kwargs.get("heuristic", linear_conflict_distance)
    weight = kwargs.get("weight", 1)

    # initial state
    goal = new_board(len(board), len(board[0]))
    empty_pos = get_empty_yx(board)
    bound = float(heuristic(board))
    initial_state = State(board, empty_pos)
    next_bound = float("inf")
    unvisited = [initial_state]
    visited: set[FrozenBoard] = set()

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


def iddfs(board: Board, **kwargs) -> SearchResult:
    r"""
    Iterative deepening depth first search. Same as :func:`dfs`, except that the depth
    bound is incrementally increased until a solution is found.

    Args:
        board: The board
        detect_dupes (bool): Duplicate detection (i.e. track visited states).
            Default is ``True``.

    Returns:
        A :class:`slidingpuzzle.state.SearchResult` with a solution and statistics
    """
    # args
    detect_dupes = kwargs.get("detect_dupes", True)

    # initial state
    bound = linear_conflict_distance(board)
    goal = new_board(len(board), len(board[0]))
    empty_pos = get_empty_yx(board)
    initial_state = State(board, empty_pos)
    next_bound = bound
    unvisited = [initial_state]
    visited: set[FrozenBoard] = set()

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


def search(board: Board, alg: str = A_STAR, **kwargs) -> SearchResult:
    r"""
    Searches for a set of moves that take the provided board state to the
    solved state.

    Requested ``alg`` may be one of:
        "a*", "beam", "bfs", "dfs", "greedy", "ida*", "iddfs"

    See :mod:`slidingpuzzle.heuristics` for heuristic functions or provide
    your own.

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
    customize their behavior. See the docs for individual algorithms.

    Args:
        board: The initial board state to search.
        alg: The algorithm to use for search.
            Use ``print(ALGORITHMS)`` to see available.
        kwargs: Algorithm arguments as described above.

    Returns:
        Returns a :class:`slidingpuzzle.state.SearchResult` containing a list of moves
        to solve the puzzle from the initial state along with some search statistics.
    """

    if not is_solvable(board):
        raise ValueError("The provided board is not solvable.")
    alg_ = alg.strip().lower()
    if alg_ not in ALGORITHMS:
        raise ValueError(f'Unknown algorithm: "{alg}"')
    return ALGORITHMS_MAP[alg_](board, **kwargs)


def evaluate(
    h: int,
    w: int,
    heuristic=linear_conflict_distance,
    alg=A_STAR,
    num_iters: int = 64,
    **kwargs,
) -> float:
    """
    Runs search on ``num_iters`` random boards. Returns the average number of nodes
    generated.

    Args:
        h: Height of the board
        w: Width of the board
        heuristic: Heuristic function to evaluate
        alg: Search algorithm to evaluate
        num_iters: Number of iterations to average
        kwargs: Additional args for ``algorithm``

    Returns:
        The average number of states generated.
    """
    total = 0
    for _ in range(num_iters):
        board = new_board(h, w)
        shuffle_board(board)
        result = search(board, alg=alg, heuristic=heuristic, **kwargs)
        total += result.generated
    return round(total / num_iters, 2)


def compare(
    h: int,
    w: int,
    ha=linear_conflict_distance,
    hb=linear_conflict_distance,
    alga=A_STAR,
    algb=A_STAR,
    num_iters: int = 32,
    kwargsa: Optional[dict] = None,
    kwargsb: Optional[dict] = None,
    **kwargs,
) -> tuple[float, float]:
    """
    Runs search on ``num_iters`` random boards, trying both alga(board, ha) and
    algb(board, hb) on each board. Returns the average number of states generated
    for both.

    Args:
        h: Height of the board
        w: Width of the board
        ha: First heuristic function to evaluate
        hb: Second heuristic function to evaluate
        alga: Search algorithm paired with ``ha``
        algb: Search algorithm paired with ``hb``
        num_iters: Number of iterations to average
        kwargsa: Keyword args for only ``alga``
        kwargsb: Keyword args for only ``algb``
        kwargs: Keyword args for both algorithms

    Returns:
        A tuple containing (avg. generated A, avg. generated B).
    """
    if not kwargsa:
        kwargsa = {}
    if not kwargsb:
        kwargsb = {}
    total_a, total_b = 0, 0
    for _ in range(num_iters):
        board = new_board(h, w)
        shuffle_board(board)
        result = search(board, alg=alga, heuristic=ha, **kwargsa, **kwargs)
        total_a += result.generated
        result = search(board, alg=algb, heuristic=hb, **kwargsb, **kwargs)
        total_b += result.generated
    return round(total_a / num_iters, 2), round(total_b / num_iters, 2)
