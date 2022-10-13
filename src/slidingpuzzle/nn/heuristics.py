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
Defines neural network-guided heuristics.
"""

import slidingpuzzle.nn.model as model
import slidingpuzzle.nn.paths as paths


MODEL_HEURISTICS = {}


def get_heuristic_key(h: int, w: int, version: str):
    board_size_str = paths.get_board_size_str(h, w)
    return f"{board_size_str}_{version}"


def set_heuristic(h, w, version, heuristic):
    key = get_heuristic_key(h, w, version)
    MODEL_HEURISTICS[key] = heuristic


def get_heuristic(h, w, version):
    key = get_heuristic_key(h, w, version)
    heuristic = MODEL_HEURISTICS.get(key, None)
    if heuristic is None:
        model.load_model(h, w, version)
        heuristic = MODEL_HEURISTICS[key]
    return heuristic


def v1_distance(board: tuple[list[int], ...]) -> float:
    h, w = len(board), len(board[0])
    heuristic = get_heuristic(h, w, model.VERSION_1)
    return heuristic(board)
