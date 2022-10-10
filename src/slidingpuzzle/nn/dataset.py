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
Utilities for creating, saving, and loading board datasets.
"""

import pickle

import torch
import torch.utils


def make_dataset(h, w) -> torch.utils.data.Dataset:
    return None


def load_dataset(h, w) -> torch.utils.data.Dataset:
    print("Loading dataset...")
    board_size_str = f"{h}x{w}"
    board_db_file = f"board_{board_size_str}.db"

    try:
        with open(board_db_file, "rb") as fp:
            db = pickle.load(fp)
    except FileNotFoundError:
        db = make_dataset(h, w)
        with open(board_db_file, "wb") as fp:
            pickle.dump(db, fp)

    return db
