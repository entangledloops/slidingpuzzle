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
Utilities for training a new sliding tile puzzle model for computing heuristic value.
"""

from typing import Collection, Optional

import collections
import itertools
import logging
import math
import random

import numpy as np
import tensorboard
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import tqdm
from tqdm.contrib.logging import tqdm_logging_redirect

import slidingpuzzle.nn.paths as paths
from slidingpuzzle.nn.dataset import load_or_build_dataset
from slidingpuzzle.nn.eval import accuracy, evaluate


log = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    """
    Set all standard random seeds.

    Args:
        seed (int): The seed to use
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_state_dict(model: nn.Module, optimizer: optim.Optimizer, **kwargs) -> dict:
    return {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "board_height": model.h,
        "board_width": model.w,
        **kwargs,
    }


def load_checkpoint(
    model: nn.Module,
    tag: str = "acc",
    optimizer: Optional[optim.Optimizer] = None,
) -> dict:
    """
    Loads a model and optimizer state from the specified epoch.
    If no epoch provided, latest trained model is used.
    In addition to the model and optimizer state, a checkpoint contains:

    * ``epoch``: The epoch this checkpoint is from
    * ``test_accuracy``: The test accuracy at that time
    * ``board_width/height``: The board size

    Note:
        The board size is extracted from ``model.h`` and ``model.w``, which are
        expected to be present.

    Args:
        model: The module to load weights into
        optimizer: The optimizer state to restore. If ``None``, ignored.
        tag: The checkpoint tag to load. If ``None``, latest checkpo

    Returns:
        The checkpoint ``dict``. If not found, returns ``dict`` with default values
        that can be used at the start of search.
    """
    checkpoint_path = paths.get_checkpoint_path(model.h, model.w, tag)
    try:
        checkpoint = torch.load(str(checkpoint_path))
        model.load_state_dict(checkpoint["model_state_dict"])
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        return checkpoint
    except FileNotFoundError:
        log.info("No model checkpoint found.")
        return {"epoch": 0, "test_accuracy": float("-inf")}


def save_checkpoint(state: dict, tag: Optional[str] = None) -> None:
    h = state["board_height"]
    w = state["board_width"]
    checkpoint_path = paths.get_checkpoint_path(h, w, tag)
    torch.save(state, str(checkpoint_path))


def launch_tensorboard(dirname: str = paths.TENSORBOARD_DIR) -> str:
    """
    Launches tensorboard in the specified directory.

    Returns:
        The launch URL.
    """
    tensorboard.program.logger.setLevel(logging.ERROR)
    tb = tensorboard.program.TensorBoard()
    tb.configure(argv=[None, "--logdir", dirname])
    url = tb.launch()
    werkzeug = logging.getLogger("werkzeug")
    werkzeug.setLevel(logging.ERROR)
    return url


def linear_regression_beta(data):
    r"""
    Calculates the :math:`\beta` (slope) of the linear regresion of ``data``.
    Used to determine which direction a trend appears to be moving. The formula
    derived below comes from linear regression. We wish to solve for :math:`\beta`
    in the equation

    .. math::
        y = \alpha + \beta x

    where the formula for :math:`\beta` is

    .. math::
        \beta = \frac{\sum^{n}_{i=1}{(x_i - \overline{x})(y_i - \overline{y})}}
            {\sum^{n}_{i=1}{(x_i - \overline{x})^2}}
            \text{.}

    In our case, the values for :math:`x` are the range :math:`[1, n]`, where
    :math:`n` is ``len(data)``. We can compute the sum of this range using the
    standard formula,

    .. math::
        n(n+1)/2

    and to compute the average :math:`\overline{x}`,

    .. math::
        \overline{x} = n(n+1)/2n = (n+1)/2 \text{.}
    """
    x_mean = (len(data) + 1) / 2
    y_mean = sum(data) / len(data)
    numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(range(len(data)), data))
    denominator = sum((x - x_mean) ** 2 for x in data)
    return numerator / denominator


def train(
    model: nn.Module,
    num_examples: int = 2**14,
    dataset: Optional[torch.utils.data.Dataset] = None,
    train_fraction: float = 0.95,
    num_epochs: int = 0,
    batch_size: int = 256,
    early_quit_epochs: int = 256,
    early_quit_threshold: float = 1e-5,
    device: Optional[str] = None,
    checkpoint_freq: int = 50,
    tensorboard_dir: str = paths.TENSORBOARD_DIR,
    seed: int = 0,
    **kwargs,
) -> None:
    """
    Trains a model for ``num_epochs``. If no prior model is found, a new one is
    created. If a prior model is found, training will be resumed. If no dataset is
    found, one will be built. Will automatically launch tensorboard and checkpoint
    periodically. The train/test split is done randomly, but in a reproducible way
    so that training may be paused and resumed without leaking test data.

    Note:
        If you change your dataset between runs, the train/test split will no longer
        be consistent with original training, so training will need to be restarted
        to prevent leaking test data.

    Args:
        model: The model instance to train
        num_examples: Total number of examples to use for training / test. Ignored
            if ``dataset`` is provided
        dataset: A custom dataset to use. If not provided, it will be generated.
        train_fraction: The train/test split
        num_epochs: Total number of epochs model should be trained for. Use 0 to run
            until the ``early_quit_epochs`` logic is hit.
        batch_size: Batch size to use for training
        early_quit_epochs: We will hold an ``early_quit_epochs``-sized window of test
            loss values. If the linear regression slope of these data points is
            > ``early_quit_threshold``, we will terminate training early. Use 0 to
            disable this feature and always run until ``num_epochs`` of training have
            completed. If both this and ``num_epochs`` are 0, training will run until
            manually interrupted.
        early_quit_beta: If the slope of the test loss rises above this value, training
            is terminated.
        device: Device to train on. Default will autodetect CUDA or use CPU
        checkpoint_freq: Model will be checkpointed each time this many epochs
            elapse. If 0, no epoch checkpointint will be used. (Highest test acc. will
            still be checkpointed and also final checkpoint on termination.)
        tensorboard_dir: The root tensorboard dir for logs. Default is "tensorboard".
        seed: Seed used for torch random utilities for reproducibility
        kwargs: Additional args that may be passed to
            :func:`slidingpuzzle.nn.dataset.make_examples` when generating a new
            dataset. Can be used to customize the search algorithm used to find
            training examples if, for example, it is taking too long.
    """
    set_seed(seed)
    h, w = model.w, model.h

    # prepare tensorboard to record training
    url = launch_tensorboard(tensorboard_dir)
    log.info(f"Tensorboard launched: {url}")
    log_dir = paths.get_log_dir(tensorboard_dir, h, w)
    comment = f"EXAMPLES_{num_examples}_BS_{batch_size}"
    writer = SummaryWriter(log_dir, comment=comment)
    layout = {
        model.version: {
            "loss": ["Multiline", ["loss/train", "loss/test"]],
            "accuracy": ["Multiline", ["accuracy/train", "accuracy/test"]],
        }
    }
    writer.add_custom_scalars(layout)

    # prepare dataset
    train_size = math.floor(num_examples * train_fraction)
    test_size = num_examples - train_size
    if dataset is None:
        dataset = load_or_build_dataset(h, w, num_examples, **kwargs)
    train_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, test_size], generator=torch.Generator().manual_seed(seed)
    )

    # prepare model
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)
    highest_acc = load_checkpoint(model, tag="acc", optimizer=optimizer)[
        "test_accuracy"
    ]
    epoch = load_checkpoint(model, tag="latest", optimizer=optimizer)["epoch"]
    criterion = nn.MSELoss(reduction="sum")
    test_loss_window: Collection[float] = collections.deque(maxlen=early_quit_epochs)

    # prepare training loop
    if 0 == num_epochs:
        log.info(f"initial epoch={epoch}, batch_size={batch_size}")
        epoch_iter = itertools.count(start=epoch)
    else:
        log.info(
            f"initial epoch={epoch}/{num_epochs}, "
            f"batch_size={batch_size}, "
            f"highest_acc={highest_acc}"
        )
        epoch_iter = range(epoch, num_epochs)

    total_epochs = num_epochs if num_epochs else None
    pbar = tqdm.tqdm(total=total_epochs, initial=epoch)
    try:
        for epoch in epoch_iter:
            model.train()  # every epoch b/c eval happens every epoch
            running_loss = 0.0
            running_accuracy = 0.0
            num_train_examples = 0

            for batch, expected in iter(
                DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            ):
                num_train_examples += batch.shape[0]
                batch = batch.to(device)
                expected = expected.to(device)
                optimizer.zero_grad()
                predicted = model(batch)
                loss = criterion(predicted, expected)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                running_accuracy += accuracy(expected, predicted)

            # need to normalize loss and acc for the examples we actually saw
            running_loss /= num_train_examples
            running_accuracy /= num_train_examples
            test_loss, test_accuracy = evaluate(model, criterion, test_dataset)
            test_loss_window.append(test_loss)
            writer.add_scalar("loss/training", running_loss, epoch)
            writer.add_scalar("accuracy/training", running_accuracy, epoch)
            writer.add_scalar("loss/test", test_loss, epoch)
            writer.add_scalar("accuracy/test", test_accuracy, epoch)
            pbar.set_description(f"test/acc: {test_accuracy:0.5f}")
            pbar.update(1)

            if test_accuracy > highest_acc:
                # save a tagged checkpoint for highest acc
                highest_acc = test_accuracy
                state = get_state_dict(
                    model, optimizer, epoch=epoch, test_accuracy=test_accuracy
                )
                save_checkpoint(state, "acc")
            if checkpoint_freq > 0 and epoch % checkpoint_freq == checkpoint_freq - 1:
                # save latest model state + epoch labeled checkpoint
                state = get_state_dict(
                    model, optimizer, epoch=epoch, test_accuracy=test_accuracy
                )
                save_checkpoint(state, f"epoch_{epoch}")
            if early_quit_epochs > 0 and len(test_loss_window) == early_quit_epochs:
                # if we are using early quitting, check the overall trendline is down
                if linear_regression_beta(test_loss_window) > early_quit_threshold:
                    with tqdm_logging_redirect():
                        log.info(
                            f"Early quit threshold reached at epoch {epoch}, "
                            f"highest_acc={highest_acc}"
                        )
                    break
            if num_epochs and epoch == num_epochs - 1:
                with tqdm_logging_redirect():
                    log.info("Max epochs reached.")
                break

    except KeyboardInterrupt:
        with tqdm_logging_redirect():
            log.info("Training interrupted.")
    finally:
        pbar.close()

    # save final state in case we were interrupted
    state = get_state_dict(model, optimizer, epoch=epoch, test_accuracy=test_accuracy)
    save_checkpoint(state, "latest")
