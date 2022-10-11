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

import pathlib


CHECKPOINT_DIR = "checkpoints"
DATASET_DIR = "datasets"
TENSORBOARD_DIR = "tensorboard"


def get_board_size_str(h: int, w: int) -> str:
    return f"{h}x{w}"


def get_path(dirname, filename) -> pathlib.Path:
    dirpath = pathlib.Path(dirname)
    dirpath.mkdir(exist_ok=True, parents=True)
    return dirpath / filename


def get_checkpoint_path(h: int, w: int) -> pathlib.Path:
    board_size_str = get_board_size_str(h, w)
    return get_path(CHECKPOINT_DIR, f"checkpoint_{board_size_str}")


def get_examples_path(h: int, w: int) -> pathlib.Path:
    board_size_str = get_board_size_str(h, w)
    return get_path(DATASET_DIR, f"examples_{board_size_str}.json")


def get_tensorboard_path(h: int, w: int) -> pathlib.Path:
    board_size_str = get_board_size_str(h, w)
    return get_path(TENSORBOARD_DIR, f"slidingpuzzle_{board_size_str}")
