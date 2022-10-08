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

import dataclasses


@dataclasses.dataclass(order=True)
class State:
    """
    A dataclass for representing the state during search.
    Although the ``board`` is enough to fully capture the game state,
    it is convenient to track some additional information.

    Args:
        board: The board state.
        empty_pos: The (y, x)-coord of the empty tile.
        history: A list of (y, x)-coords representing moves from the initial
            state to the current board state.
        f: The stored value of this node. Used by some search algorithms.
            The value stored here has no meaning if not using a relevant
            search algorithm.
    """

    board: tuple[list[int]] = dataclasses.field(compare=False)
    empty_pos: tuple[int, int] = dataclasses.field(compare=False)
    history: list[tuple[int, int]] = dataclasses.field(compare=False)
    f: int = 0


@dataclasses.dataclass
class SearchResult:
    """
    A dataclass for returning a puzzle solution along with some information
    concerning how the search progressed. Useful for evaluating different
    heuristics.

    Args:
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

    generated: int
    expanded: int
    unvisited: list[State]
    visited: set[tuple[tuple[int, ...], ...]]
    solution: list[tuple[int, int]]
