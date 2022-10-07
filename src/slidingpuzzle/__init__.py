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


from slidingpuzzle.heuristics import (
    euclidean_distance,
    hamming_distance,
    manhattan_distance,
    random_distance,
)

from slidingpuzzle.slidingpuzzle import (
    count_inversions,
    get_empty_yx,
    get_next_moves,
    get_next_states,
    is_solvable,
    new_board,
    print_board,
    print_result,
    search,
    shuffle_board,
    shuffle_board_slow,
    swap_tiles,
)

from slidingpuzzle.state import SearchResult, State
