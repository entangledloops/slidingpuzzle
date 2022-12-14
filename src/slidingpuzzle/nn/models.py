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
VERSION_2 = "v2"


class Model_v1(nn.Module):
    """
    A stack of linear layers that accepts a board as input and outputs the
    estimated distance to the goal.

    Trained with::

        SGD(lr = 0.001, momentum = 0.9)
        MSELoss

    Total parameters: 1322496
    """

    def __init__(self, h: int, w: int) -> None:
        super().__init__()
        self.version = VERSION_1  # required
        self.h = h  # required
        self.w = w  # required
        hidden_size = 512
        self.input = nn.Sequential(
            nn.Flatten(),
            nn.Linear(h * w, hidden_size, dtype=DTYPE),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size, dtype=DTYPE),
        )
        n_layers = 5
        self.linears = nn.Sequential(
            *[
                nn.Sequential(
                    nn.Linear(hidden_size, hidden_size, bias=False, dtype=DTYPE),
                    nn.ReLU(),
                    nn.BatchNorm1d(hidden_size, dtype=DTYPE),
                )
                for _ in range(n_layers)
            ]
        )
        self.output = nn.Linear(hidden_size, 1, bias=False, dtype=DTYPE)

    def forward(self, x):
        x = self.input(x)
        x = self.linears(x)
        x = self.output(x)
        return x


class Model_v2(nn.Module):
    """
    A multi-headed attention-based network with about the same number of parameters
    as v1. (WIP)

    Trained with::

        SGD(lr = 0.0001, momentum = 0.9)
        MSELoss

    Total parameters: 1170767
    """

    def __init__(self, h: int, w: int) -> None:
        super().__init__()
        self.version = VERSION_2  # required
        self.h = h  # required
        self.w = w  # required
        size = h * w
        hidden_size = size * 32  # should be chosen so that is divisible by h*w
        self.flatten = nn.Flatten()
        self.embed = nn.ModuleList(
            nn.Linear(1, hidden_size, dtype=DTYPE) for _ in range(size)
        )
        self.first_attention = nn.MultiheadAttention(
            hidden_size, size, batch_first=True, dtype=DTYPE
        )
        self.first_attention_linear = nn.Linear(hidden_size, hidden_size, dtype=DTYPE)
        self.first_attention_norm = nn.BatchNorm1d(size)

        n_layers = 1
        self.attentions = nn.ModuleList(
            nn.MultiheadAttention(hidden_size, size, batch_first=True, dtype=DTYPE)
            for _ in range(n_layers)
        )
        self.linears = nn.ModuleList(
            nn.Linear(hidden_size, hidden_size, dtype=DTYPE) for _ in range(n_layers)
        )
        self.norms = nn.ModuleList(
            nn.BatchNorm1d(size, dtype=DTYPE) for _ in range(n_layers)
        )

        self.last_attention = nn.MultiheadAttention(
            hidden_size, size, batch_first=True, dtype=DTYPE
        )
        self.last_attention_linear = nn.Linear(hidden_size, 1, dtype=DTYPE)
        self.output = nn.Linear(size, 1, dtype=DTYPE)

    def forward(self, x):
        # first embed and create initial q, k, v
        x = self.flatten(x)
        x = torch.split(x, 1, dim=-1)  # break up board into individual tile tensors
        x = torch.stack([torch.relu(e(t)) for e, t in zip(self.embed, x)], dim=1)

        # attention layers
        x, _ = self.first_attention(x, x, x)
        x = torch.relu(self.first_attention_linear(x))
        x = self.first_attention_norm(x)

        for atn, lin, nrm in zip(self.attentions, self.linears, self.norms):
            x, _ = atn(x, x, x)
            x = torch.relu(lin(x))
            x = nrm(x)

        x, _ = self.last_attention(x, x, x)
        x = torch.relu(self.last_attention_linear(x))

        # output
        x = self.flatten(x)
        x = self.output(x)
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


def get_num_params(model: nn.Module) -> int:
    """
    Compute the total number of parameters in a model.

    Args:
        model: A model

    Returns:
        The total number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
