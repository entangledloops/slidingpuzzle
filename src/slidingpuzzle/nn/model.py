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

import torch
import torch.nn as nn


class Model_v1(nn.Module):
    def __init__(self, h, w) -> None:
        super().__init__()
        self.h = h
        self.w = w
        size = h * w
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(size, size * 4, dtype=torch.float32)
        self.linear2 = nn.Linear(size * 4, size * 4, dtype=torch.float32)
        self.linear3 = nn.Linear(size * 4, size, dtype=torch.float32)
        self.linear4 = nn.Linear(size, 1, dtype=torch.float32)

    def forward(self, x):
        x = self.flatten(x)
        x = torch.relu(self.linear1(x))
        x = torch.relu(self.linear2(x))
        x = torch.relu(self.linear3(x))
        x = self.linear4(x)
        return x
