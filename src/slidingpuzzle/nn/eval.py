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


def accuracy(expected, predicted) -> float:
    diff = torch.abs(expected - torch.abs(predicted))
    diff /= 1 + diff
    diff = torch.nan_to_num(diff, nan=0, posinf=0, neginf=0)
    diff = 1 - diff
    return torch.sum(diff).item()


def evaluate(
    model: torch.nn.Module, criterion, dataset: torch.utils.data.Dataset
) -> tuple[float, float]:
    """
    Runs the model on provided datatset computing average loss and accuracy.
    """
    model.eval()
    device = next(model.parameters()).device
    running_loss = 0.0
    running_accuracy = 0.0
    dataloader = torch.utils.data.DataLoader(dataset)
    with torch.no_grad():
        for batch, expected in iter(dataloader):
            predicted = model(batch.to(device))
            expected = expected.to(device)
            running_loss += criterion(predicted, expected).item()
            running_accuracy += accuracy(expected, predicted)
    return running_loss / len(dataset), running_accuracy / len(dataset)


def evaluate_checkpoint(
    model: torch.nn.Module,
    epoch: int = None,
    num_iters: int = None,
    device: str = None,
    **kwargs,
) -> float:
    """
    Loads the provided model from the checkpoint at ``epoch`` and runs
    ``evaluate_heuristic``, returning the result. If ``epoch`` is not provided, the
    latest checkpoint is used.
    """
    from slidingpuzzle.algorithms import evaluate_heuristic
    from slidingpuzzle.nn.train import load_checkpoint
    from slidingpuzzle.nn.heuristics import set_heuristic

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    load_checkpoint(model, epoch=epoch)
    model.to(device)
    heuristic = set_heuristic(model)
    if num_iters is None:
        return evaluate_heuristic(model.h, model.w, heuristic=heuristic, **kwargs)
    else:
        return evaluate_heuristic(
            model.h, model.w, heuristic=heuristic, num_iters=num_iters, **kwargs
        )
