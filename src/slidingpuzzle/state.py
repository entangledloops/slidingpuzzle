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
Provides some convience classes to track search state and results.
"""

from typing import Collection, Optional

import dataclasses

import slidingpuzzle
from slidingpuzzle.board import Board, FrozenBoard


@dataclasses.dataclass(order=True)
class State:
    """
    A dataclass for representing the state during search.
    Although the ``board`` is enough to fully capture the game state,
    it is convenient to track some additional information.

    Args:
        board: The board state.
        blank_pos: The (y, x)-coord of the blank tile.
        history: A list of (y, x)-coords representing moves from the initial
            state to the current board state.
        f: The stored value of this node. Used by some search algorithms to order
            states. The value stored here has no meaning if not using a relevant
            search algorithm.
        g: For some search algorithms, this will hold the number of moves made to reach
            this state. Used to tie-break when `f` values are identical.
    """

    board: Board = dataclasses.field(compare=False)
    blank_pos: tuple[int, int] = dataclasses.field(compare=False)
    history: list[tuple[int, int]] = dataclasses.field(
        compare=False, default_factory=list
    )
    f: int | float = 0
    g: int = 0  # stored separately for tie-breaking


@dataclasses.dataclass
class SearchResult:
    """
    A dataclass for returning a puzzle solution along with some information
    concerning how the search progressed. Useful for evaluating different
    heuristics.

    Args:
        board: The original input board.
        generated: The number of states generated during search.
        expanded: The number of states evaluated during search.
        unvisited: The list of states that were never reached.
        visited: The set of boards evaluated.
        solution: The list of moves from initial position to solution.

    Note:
        The ``unvisited`` and ``visited`` attributes are traditionally known
        as "open" and "closed" in search parlance. However, Python's built-in
        function ``open`` would collide, so they have been renamed.
    """

    board: Board
    generated: int
    expanded: int
    unvisited: Collection[State]
    visited: set[FrozenBoard]
    solution: Optional[list[tuple[int, int]]]

    def __repr__(self) -> str:
        solution = (
            slidingpuzzle.solution_as_tiles(self.board, self.solution)
            if self.solution
            else "N/A"
        )
        return (
            f"solution={solution}\n"
            f"solution_len={len(self.solution) if self.solution else 'N/A'}, "
            f"generated={self.generated}, "
            f"expanded={self.expanded}, "
            f"unvisited={len(self.unvisited)}, "
            f"visited={len(self.visited)}"
        )

    def __str__(self) -> str:
        return repr(self)
