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


import dataclasses


@dataclasses.dataclass
class State:
    board: tuple[list[int]]
    empty_pos: tuple[int, int]
    history: list[tuple[int, int]]


@dataclasses.dataclass
class SearchResult:
    generated: int
    expanded: int
    unvisited: list[State]
    visited: set[tuple[tuple[int]]]
    solution: list[tuple[int]]
