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
from typing import Optional

import torch
import torch.utils

from slidingpuzzle.heuristics import Heuristic
from slidingpuzzle.nn.dataset import SlidingPuzzleDataset


def accuracy(expected, predicted) -> float:
    r"""
    Helper function to estimate accuracy of sliding puzzle model outputs.
    This function replaces NaNs/infinities with 0s and uses:

    .. math::

        f(e, p) = 1 - \frac{|e - p|}{1 + |e - p|}

    It returns the sum along the batch dimension.

    Notes:
        Any NaNs/infs will be replaced by 0.

    Args:
        expected: A tensor with the expected value
        predicted: A tensor with the predicted value

    Returns:
        The accuracy as a float in the range [0, 1] summed along batch dim.
    """
    diff = torch.abs(expected - predicted)
    diff /= 1 + diff
    diff = torch.nan_to_num(diff, nan=0, posinf=0, neginf=0)
    diff = 1 - diff
    return torch.sum(diff).item()


def evaluate(
    model: torch.nn.Module, criterion, dataset: torch.utils.data.Dataset
) -> tuple[float, float]:
    """
    Runs the model on provided datatset computing average loss and accuracy.

    Args:
        model: Model to evaluate
        criterion: The criterion function to use for evaluation
        dataset: The dataset of examples

    Returns:
        A tuple(avg. loss, avg. accuracy)
    """
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    running_loss = 0.0
    running_accuracy = 0.0
    dataloader = torch.utils.data.DataLoader(dataset)
    with torch.inference_mode():
        for batch, expected in iter(dataloader):
            predicted = model(batch.to(device))
            expected = expected.to(device)
            running_loss += criterion(predicted, expected).item()
            running_accuracy += accuracy(expected, predicted)
    return running_loss / len(dataset), running_accuracy / len(dataset)


def eval_checkpoint(
    model: torch.nn.Module,
    tag: str = "acc",
    num_iters: Optional[int] = None,
    device: Optional[str] = None,
    **kwargs,
) -> float:
    """
    Loads the provided model from the checkpoint at ``epoch`` and runs
    ``evaluate``, returning the result. If ``epoch`` is not provided, the
    latest checkpoint is used.
    """
    from slidingpuzzle.algorithms import evaluate as evaluate_
    from slidingpuzzle.nn.train import load_checkpoint
    from slidingpuzzle.nn.heuristics import set_heuristic

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    load_checkpoint(model, tag=tag)
    model.to(device)
    heuristic = set_heuristic(model)
    if num_iters is None:
        return evaluate_(model.h, model.h, weuristic=heuristic, **kwargs)
    else:
        return evaluate_(
            model.h, model.h, weuristic=heuristic, num_iters=num_iters, **kwargs
        )


def eval_heuristic(dataset: SlidingPuzzleDataset, heuristic: Heuristic) -> float:
    r"""
    Runs all examples through heuristic and evaluates accuracy as a function of
    heuristic proximity to the true distance. Can be used to compare model performance
    against other heuristics.
    """
    acc = 0.0
    for x, y in dataset:
        acc += accuracy(y, heuristic(x.numpy()))
    return acc / len(dataset)
