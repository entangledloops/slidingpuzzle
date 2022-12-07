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
Defines a PyTorch model to evaluate sliding puzzle boards.
"""

from typing import Optional

import torch
import torch.nn as nn

import slidingpuzzle.nn.paths as paths
import slidingpuzzle.board as board_


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float32
VERSION_1 = "v1"


class Model_v1(nn.Module):
    """
    A stack of linear layers that accepts a board as input and outputs the
    estimated distance to the goal.
    """

    def __init__(self, h: int, w: int) -> None:
        super().__init__()
        self.version = VERSION_1  # required
        self.h = h  # required
        self.w = w  # required
        size = h * w
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(size, size * 8, dtype=DTYPE)
        self.linear2 = nn.Linear(size * 8, size * 8, dtype=DTYPE)
        self.linear3 = nn.Linear(size * 8, size * 4, dtype=DTYPE)
        self.linear4 = nn.Linear(size * 4, size, dtype=DTYPE)
        self.linear5 = nn.Linear(size, 1, dtype=DTYPE)

    def forward(self, x):
        x = self.flatten(x)
        x = torch.relu(self.linear1(x))
        x = torch.relu(self.linear2(x))
        x = torch.relu(self.linear3(x))
        x = torch.relu(self.linear4(x))
        x = self.linear5(x)
        return x


def save_model(model: nn.Module, device: Optional[str] = None) -> None:
    """
    Save a frozen version of the model into the "models" dir.

    Args:
        model: The trained model.
        device: The device the model is currently loaded on. If not provied, it will
            be guessed.
    """
    if device is None:
        device = DEVICE
    model.eval()
    model.to(device)
    path = paths.get_model_path(model.h, model.w, model.version)
    board = board_.new_board(model.h, model.w)
    example_inputs = torch.tensor(board, dtype=torch.float32).unsqueeze(0).to(device)
    traced_model = torch.jit.trace(model, example_inputs)
    frozen_model = torch.jit.freeze(traced_model)
    frozen_model.save(str(path))


def load_model(
    h: int, w: int, version: str, device: Optional[str] = None
) -> torch.ScriptModule:
    """
    Reload a pre-trained frozen model.
    """
    if device is None:
        device = DEVICE
    model_path = paths.get_model_path(h, w, version)
    model = torch.jit.load(str(model_path), map_location=device)
    model.eval()
    return model
