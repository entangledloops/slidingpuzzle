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
Utilities for dealing with paths required during model training and inference.
"""

from typing import Optional

import pathlib


CHECKPOINT_DIR = "checkpoints"
DATASET_DIR = "datasets"
MODELS_DIR = "models"
TENSORBOARD_DIR = "tensorboard"


def get_board_size_str(h: int, w: int) -> str:
    return f"{h}x{w}"


def get_path(dirname: str, filename: str) -> pathlib.Path:
    """
    Creates intermediate directorys for dirname and returns the full path from dirname
    to filename.

    Args:
        dirname: A directory or path to a directory.
        filename: The filename the path will point to.

    Returns:
        The path object.
    """
    dirpath = pathlib.Path(dirname)
    dirpath.mkdir(exist_ok=True, parents=True)
    return dirpath / filename


def get_checkpoint_path(h: int, w: int, tag: str) -> pathlib.Path:
    """
    Get the path to a checkpoint, given a board size and optional tag.

    Args:
        h: Board height
        w: Board width
        tag: Checkpoint tag to load
    """
    checkpoint_name = f"{get_board_size_str(h, w)}_{tag}"
    return get_path(CHECKPOINT_DIR, f"checkpoint_{checkpoint_name}")


def get_examples_path(h: int, w: int) -> pathlib.Path:
    board_size_str = get_board_size_str(h, w)
    return get_path(DATASET_DIR, f"examples_{board_size_str}.json")


def get_model_path(h: int, w: int, version: str) -> pathlib.Path:
    board_size_str = get_board_size_str(h, w)
    return get_path(MODELS_DIR, f"{version}_{board_size_str}.pt")


def get_log_dir(tensorboard_dir: str, h: int, w: int) -> pathlib.Path:
    board_size_str = get_board_size_str(h, w)
    return get_path(tensorboard_dir, f"slidingpuzzle_{board_size_str}")
