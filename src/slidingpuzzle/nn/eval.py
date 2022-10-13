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
Module used to evaluate model performance vs. other heuristics.
"""

import torch
import torch.utils


def evaluate(
    model: torch.nn.Module, criterion, dataset: torch.utils.data.Dataset, device
) -> float:
    loss = 0.0
    dataloader = torch.utils.data.DataLoader(dataset)
    model.eval()
    with torch.no_grad():
        for batch, expected in iter(dataloader):
            predicted = model(batch.to(device))
            expected = expected.to(device)
            loss += criterion(predicted, expected).item()
    return loss / len(dataset)
