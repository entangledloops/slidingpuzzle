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

The models compute h(board) -> [0, 1], where
"""

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

import slidingpuzzle.nn.paths as paths
from slidingpuzzle.nn.dataset import (
    SlidingPuzzleDataset,
    load_dataset,
    load_examples,
    make_examples,
)
from slidingpuzzle.nn.eval import accuracy, evaluate


def load_checkpoint(
    model: nn.Module, optimizer: optim.Optimizer = None, epoch: int = None
) -> int:
    """
    Loads a model and optimizer state from the specified epoch.
    If no epoch provided, latest trained model is used.

    Note:
        The board size is extracted from ``model.h`` and ``model.w``,
        which are expected to be present.

    Returns:
        The epoch number found in the checkpoint. Default is 1.
    """
    checkpoint_path = paths.get_checkpoint_path(model.h, model.w, epoch)
    try:
        checkpoint = torch.load(str(checkpoint_path))
        model.load_state_dict(checkpoint["model_state_dict"])
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        return checkpoint["epoch"]
    except FileNotFoundError:
        print("No model checkpoint found.")
        return 1


def save_checkpoint(model: nn.Module, optimizer: optim.Optimizer, epoch: int) -> None:
    state = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "board_height": model.h,
        "board_width": model.w,
    }
    # save "latest" model along with epoch labeled checkpoint
    torch.save(state, str(paths.get_checkpoint_path(model.h, model.w)))
    torch.save(state, str(paths.get_checkpoint_path(model.h, model.w, epoch)))


def launch_tensorboard(dirname: str) -> str:
    """
    Launches tensorboard in the specified directory.

    Returns:
        The launch URL.
    """
    tb = tensorboard.program.TensorBoard()
    tb.configure(argv=[None, "--logdir", dirname])
    return tb.launch()


def train(
    model: nn.Module,
    num_examples: int = 2**14,
    train_fraction: float = 0.95,
    num_epochs: int = 100_000,
    batch_size: int = 128,
    device: str = None,
    dataset: torch.utils.data.Dataset = None,
    tensorboard_dir: str = paths.TENSORBOARD_DIR,
    seed: int = 0,
    checkpoint_freq: int = 50,
) -> nn.Module:
    """
    Trains a model for ``num_epochs``. If no prior model is found, a new one is
    created. If a prior model is found, training will be resumed. If no dataset is
    found, one will be built. Will automatically launch tensorboard and checkpoint
    periodically. The train/test split is done randomly, but in a reproducible way
    so that training may be paused and resumed without leaking test data.

    Note:
        If you change your dataaset between runs, the train/test split will no longer
        be consistent with original training, so training will need to be restarted
        to prevent leaking test data.

    Args:
        h: Height of board to train the model for
        w: Width of the board to train the model for
        num_examples: Total number of examples to use for training / test. Ignored
            if ``dataset`` is provided
        num_epochs: Total number of epochs model should be trained for
        device: Device to train on. Default will autodetect CUDA or use CPU
        batch_size: Batch size to use for training
        dataset: A custom dataset to use
        tensorboard_dir: The root tensorboard dir for logs. Default is "tensorboard".
        seed: Seed used for torch random utilities for reproducibility
        checkpoint_freq: Model will be checkpointed every time this many epochs elapses

    Returns:
        The trained model. May be loaded on GPU if GPU is available. The model is also
        saved to disk before returning in the checkpoints directory.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    h, w = model.h, model.w

    # prepare tensorboard to record training
    url = launch_tensorboard(tensorboard_dir)
    print(f"tensorboard launched: {url}")
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
    dataset = load_dataset(h, w, num_examples) if dataset is None else dataset
    train_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, test_size], generator=torch.Generator().manual_seed(seed)
    )
    # test_examples = make_examples(h, w, test_size)
    # test_dataset = SlidingPuzzleDataset(test_examples)
    # train_examples = make_examples(h, w, train_size, test_examples)
    # train_dataset = SlidingPuzzleDataset(train_examples)

    # prepare model
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)
    epoch = load_checkpoint(model, optimizer)
    criterion = nn.MSELoss()

    print(f"initial epoch={epoch}/{num_epochs}, batch_size={batch_size}")
    try:
        for epoch in tqdm.tqdm(range(epoch, num_epochs + 1)):
            model.train()  # every epoch b/c eval happens every epoch
            running_loss = 0.0
            running_accuracy = 0.0
            num_train_examples = 0

            for batch, expected in iter(
                DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            ):
                num_train_examples += batch_size
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
            writer.add_scalar("loss/training", running_loss, epoch)
            writer.add_scalar("accuracy/training", running_accuracy, epoch)
            writer.add_scalar("loss/test", test_loss, epoch)
            writer.add_scalar("accuracy/test", test_accuracy, epoch)

            if epoch % checkpoint_freq == 0:
                save_checkpoint(model, optimizer, epoch)
            if epoch % 100 == 0:
                print(
                    f"training/loss: {running_loss:0.5f}, "
                    f"test/loss: {test_loss:0.5f}, "
                    f"training/acc: {running_accuracy:0.5f}, "
                    f"test/acc: {test_accuracy:0.5f}"
                )
    except KeyboardInterrupt:
        print("training interrupted")

    # save final state
    save_checkpoint(model, optimizer, epoch)

    return model
