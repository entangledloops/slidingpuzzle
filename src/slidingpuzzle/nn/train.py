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

import tensorboard
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import tqdm

from slidingpuzzle.nn.dataset import load_dataset
from slidingpuzzle.nn.paths import get_checkpoint_path, get_tensorboard_path
from slidingpuzzle.nn.model import Model_v1
from slidingpuzzle.nn.paths import TENSORBOARD_DIR


def load_checkpoint(model, optimizer) -> int:
    checkpoint_path = get_checkpoint_path(model.h, model.w)
    epoch = 0
    try:
        checkpoint = torch.load(str(checkpoint_path))
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        epoch = checkpoint["epoch"]
    except FileNotFoundError:
        print("No model checkpoint found.")
    return epoch


def save_checkpoint(model: nn.Module, optimizer: optim.Optimizer, epoch: int) -> None:
    checkpoint_path = get_checkpoint_path(model.h, model.w)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "board_height": model.h,
            "board_width": model.w,
        },
        str(checkpoint_path),
    )


def launch_tensorboard(dirname=TENSORBOARD_DIR):
    tb = tensorboard.program.TensorBoard()
    tb.configure(argv=[None, "--logdir", dirname])
    url = tb.launch()
    print(f"Tensorboard launched: {url}")


def train(h, w, num_epochs=30, batch_size=1):
    # prepare tensorboard to record training
    tensorboard_path = get_tensorboard_path(h, w)
    launch_tensorboard()
    comment = f"EPOCHS_{num_epochs}_BS_{batch_size}"
    writer = SummaryWriter(tensorboard_path, comment=comment)

    # prepare dataset and model
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    ds = load_dataset(h, w)
    model = Model_v1(h, w)
    model.to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    epoch = load_checkpoint(model, optimizer)
    criterion = nn.MSELoss()

    print(f"Training started at epoch: {epoch}")
    for epoch in tqdm.tqdm(range(epoch, num_epochs)):
        running_loss = 0.0

        for batch, expected in iter(
            DataLoader(ds, batch_size=batch_size, shuffle=True)
        ):
            batch = batch.to(device)
            expected = expected.to(device)
            optimizer.zero_grad()
            outputs = model(batch)
            loss = criterion(outputs, expected)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        save_checkpoint(model, optimizer, epoch)
        running_loss /= len(ds)
        writer.add_scalar("training_loss", running_loss, epoch)
        running_loss = 0.0

    save_checkpoint(model, optimizer, epoch)
