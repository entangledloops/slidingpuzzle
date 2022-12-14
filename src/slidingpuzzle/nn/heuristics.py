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
import torch

import slidingpuzzle.nn.models as models
import slidingpuzzle.nn.paths as paths
from slidingpuzzle.board import Board
from slidingpuzzle.heuristics import Heuristic


model_heuristics = {}


def get_heuristic_key(h: int, w: int, version: str):
    board_size_str = paths.get_board_size_str(h, w)
    return f"{board_size_str}_{version}"


def make_heuristic(model: torch.nn.Module | torch.ScriptModule):
    device = models.DEVICE
    dtype = models.DTYPE

    def heuristic(board: tuple[list[int], ...]) -> float:
        tensor = torch.tensor(board, dtype=dtype).unsqueeze(0).to(device)
        return model(tensor).item()

    return heuristic


def set_heuristic(model: torch.nn.Module | torch.ScriptModule):
    key = get_heuristic_key(model.w, model.h, model.version)
    heuristic = make_heuristic(model)
    model_heuristics[key] = heuristic
    return heuristic


def get_heuristic(h: int, w: int, version: str) -> Heuristic:
    key = get_heuristic_key(h, w, version)
    heuristic = model_heuristics.get(key, None)
    if heuristic is None:
        model = models.load_model(h, w, version)
        key = get_heuristic_key(h, w, version)
        heuristic = make_heuristic(model)
        model_heuristics[key] = heuristic
    return heuristic


##################################################################
# methods beyond here correspond to predefined model classes
# you can add your own model heuristics below
##################################################################


def v1_distance(board: Board) -> float:
    """
    A neural network that estimates distance to goal

    Args:
        board: The board

    Returns:
        An estimated number of moves to reach the goal
    """
    h, w = len(board), len(board[0])
    heuristic = get_heuristic(h, w, models.VERSION_1)
    return heuristic(board)


def v2_distance(board: Board) -> float:
    """
    A neural network that estimates distance to goal

    Args:
        board: The board

    Returns:
        An estimated number of moves to reach the goal
    """
    h, w = len(board), len(board[0])
    heuristic = get_heuristic(h, w, models.VERSION_2)
    return heuristic(board)
