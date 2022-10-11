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

import json
import math

import torch
import torch.utils
import tqdm

from slidingpuzzle.slidingpuzzle import (
    apply_move,
    freeze_board,
    new_board,
    search,
    shuffle_board,
)
from slidingpuzzle.heuristics import euclidean_distance, manhattan_distance
from slidingpuzzle.nn.paths import get_examples_path


class SlidingPuzzleDataset(torch.utils.data.Dataset):
    def __init__(self, examples) -> None:
        super().__init__()
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        x, y = self.examples[idx]
        return (
            torch.tensor(x, dtype=torch.float32),
            torch.tensor([y], dtype=torch.float32),
        )


def make_examples(h, w, num_examples) -> list[tuple]:
    """
    Constructs a list of training examples, which are tuples of:
        (board, num_moves_to_goal)

    Args:
        h: Height of board.
        w: Width of board.
        num_examples: Number of examples to produce.

    Returns:
        The list of training examples.
    """
    examples = []
    visited = set()

    def visit(board) -> bool:
        """
        Helper to check if this state already exists. Otherwise, record it.

        Returns:
            True if we've been here before.
        """
        frozen_board = freeze_board(board)
        if frozen_board in visited:
            return True
        visited.add(frozen_board)
        return False

    # if the board is large, we need weighted a* to obtain solutions this century
    weight = max(1, math.floor(math.sqrt(h * w) - 2))

    pbar = tqdm.tqdm(total=num_examples)
    while len(examples) < num_examples:
        board = shuffle_board(new_board(h, w))
        if visit(board):
            continue

        # find a path to use as an accurate training reference
        result = search(board, "a*", manhattan_distance, weight=weight)

        # we can use all intermediate boards as examples
        while len(examples) < num_examples:
            distance = len(result.solution)
            examples.append((board, distance))
            pbar.update(1)
            if not len(result.solution):
                break
            move = result.solution.pop(0)
            apply_move(board, move)
    pbar.close()

    return examples


def load_examples(h: int, w: int, examples_file: str = None) -> list:
    """
    Loads examples from a JSON file.
    """
    if examples_file is None:
        examples_file = get_examples_path(h, w)
    with open(examples_file, "rt") as fp:
        return json.load(fp)


def save_examples(h: int, w: int, examples, examples_file: str = None) -> None:
    """
    Save a list of examples to disk as JSON.
    """
    if examples_file is None:
        examples_file = get_examples_path(h, w)
    with open(examples_file, "wt") as fp:
        json.dump(examples, fp)


def load_dataset(
    h: int, w: int, examples_file=None, num_examples=1000
) -> torch.utils.data.Dataset:
    """
    Loads examples, constructs a SlidingPuzzleDataset from them, and returns it.
    If no examples are found, it will first build an examples database.

    Args:
        h: The height of the board to locate a dataset for
        w: The width of the board to locate a dataset for
        examples_file: The name of the examples file to save or load.
        num_examples: If no examples are found, the number to construct.

    Returns:
        A dataset for the requested puzzle size.
    """
    print("Loading dataset...")
    try:
        examples = load_examples(h, w, examples_file)
    except FileNotFoundError:
        print("Failed. Building new dataset...")
        examples = make_examples(h, w, num_examples)
        save_examples(h, w, examples, examples_file)

    return SlidingPuzzleDataset(examples)
