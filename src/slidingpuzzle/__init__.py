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
A package for solving and working with sliding tile puzzles.

All submodule code is imported into the parent namespace, so there is no need
to explicitly refer to submodules.

Examples:
    >>> from slidingpuzzle import *
    >>> b = new_board(3, 3)
    >>> print_board(b)
    1 2 3
    4 5 6
    7 8
    >>> shuffle_board(b)
    >>> print_board(b)
    4 5
    6 7 2
    1 8 3
    >>> r = search(b)
    >>> print_result(r)
    solution_len=20, generated=144704, expanded=103615, unvisited=41090, visited=54466
    >>> r = search(b, algorithm="greedy", heuristic=manhattan_distance)
    >>> print_result(r)
    solution_len=36, generated=409, expanded=299, unvisited=111, visited=153
"""

from slidingpuzzle.heuristics import (
    euclidean_distance,
    hamming_distance,
    manhattan_distance,
    random_distance,
)

from slidingpuzzle.slidingpuzzle import (
    EMPTY_TILE,
    A_STAR,
    BEAM,
    BFS,
    DFS,
    GREEDY,
    IDA_STAR,
    IDDFS,
    ALGORITHMS,
    count_inversions,
    get_empty_yx,
    get_next_moves,
    get_next_states,
    is_solvable,
    new_board,
    print_board,
    search,
    shuffle_board,
    shuffle_board_slow,
    solution_as_tiles,
    swap_tiles,
)

from slidingpuzzle.state import SearchResult, State
