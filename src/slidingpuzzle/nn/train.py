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

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from slidingpuzzle.nn.dataset import load_dataset
from slidingpuzzle.nn.model import Model_v1


def get_board_size_str(h: int, w: int) -> str:
    return f"{h}x{w}"


def get_checkpoint_filename(h: int, w: int) -> str:
    board_size_str = get_board_size_str(h, w)
    return f"checkpoints/checkpoint_{board_size_str}"


def get_tensorboard_filename(h: int, w: int) -> str:
    board_size_str = get_board_size_str(h, w)
    return f"tensorboard/slidingpuzzle_{board_size_str}"


def load_checkpoint(model, optimizer) -> int:
    print("Loading model...")
    checkpoint_filename = get_checkpoint_filename(model.h, model.w)
    epoch = 0
    try:
        checkpoint = torch.load(checkpoint_filename)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        epoch = checkpoint["epoch"]
    except FileNotFoundError:
        print("No model checkpoint found.")
    return epoch


def save_checkpoint(model: nn.Module, optimizer: optim.Optimizer, epoch: int) -> None:
    print("Saving model...")
    checkpoint_filename = get_checkpoint_filename(model.h, model.w)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "board_height": model.h,
            "board_width": model.w,
        },
        checkpoint_filename,
    )


def train(h, w, num_epochs=3, batch_size=1):
    tensorboard_filename = get_tensorboard_filename(h, w)
    writer = SummaryWriter(tensorboard_filename)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    ds = load_dataset(h, w)
    model = Model_v1(h, w)
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    epoch = load_checkpoint(model, optimizer)
    criterion = nn.MSELoss()
    model.to(device)

    print(f"Training started, epoch: {epoch}")
    iters = 0
    for epoch in range(epoch, num_epochs):
        running_loss = 0.0

        for batch in DataLoader(ds, batch_size=batch_size, shuffle=True):
            optimizer.zero_grad()
            outputs = model(batch)
            expected = 0
            loss = criterion(outputs, expected)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        if epoch % 100 == 99:
            save_checkpoint(model, optimizer, epoch)
            running_loss /= 100
            writer.add_scalar("training_loss", running_loss, epoch)
            print(f"[{epoch + 1}, {iters + 1:5d}] loss: {running_loss:.3f}")
            running_loss = 0.0

    save_checkpoint(model, optimizer, epoch)
    print(f"finished after {iters} iterations")
