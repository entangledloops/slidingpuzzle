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
A package for solving sliding tile puzzles.

Examples:
    >>> from slidingpuzzle import *
    >>> b = new_board(3, 3)
    >>> print_board(b)
    1 2 3
    4 5 6
    7 8
    >>> print_board(shuffle(b))
    1 6 7
    4   8
    5 3 2
    >>> search(b, "greedy", heuristic=manhattan_distance)
    solution=[3, 2, 8, 3, 6, 7, 3, 6, 7, 1, 4, 7, 2, 5, 7, 4, 1, 2, 5, 8]
    solution_len=20, generated=829, expanded=518, unvisited=312, visited=313
"""

from .__about__ import __version__

from slidingpuzzle.heuristics import (
    corner_tiles_distance,
    euclidean_distance,
    hamming_distance,
    last_moves_distance,
    linear_conflict_distance,
    manhattan_distance,
    random_distance,
    relaxed_adjacency_distance,
)

from slidingpuzzle.board import (
    BLANK,
    board_generator,
    count_inversions,
    find_blank,
    find_tile,
    flatten_board,
    freeze_board,
    from_rows,
    from_iter,
    get_goal_tile,
    get_goal_x,
    get_goal_y,
    get_goal_yx,
    get_next_moves,
    is_solvable,
    is_solved,
    new_board,
    print_board,
    random_move,
    shuffle,
    shuffle_lazy,
    solution_as_tiles,
    visit,
)

from slidingpuzzle.algorithms import (
    Algorithm,
    compare,
    evaluate,
    get_next_states,
    search,
    swap_tiles,
)

from slidingpuzzle.state import SearchResult, State

import logging


logging.basicConfig(level=logging.INFO)
