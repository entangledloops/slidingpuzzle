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
    shuffle_board_lazy,
    visit,
)
import slidingpuzzle.nn.paths as paths


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


def make_examples(h, w, num_examples, prior_examples=None) -> list[tuple]:
    """
    Constructs a list of training examples, which are tuples of:
        (board, num_moves_to_goal)

    Args:
        h: Height of board.
        w: Width of board.
        num_examples: Number of examples to produce.
        prior_examples: Previous examples that we should avoid reproducing.

    Returns:
        The list of training examples.
    """
    examples = [] if prior_examples is None else list(prior_examples)

    if prior_examples is None:
        visited = set()
    else:
        visited = set(freeze_board(example[0]) for example in prior_examples)

    pbar = tqdm.tqdm(total=num_examples)
    while len(examples) < num_examples:
        board = shuffle_board_lazy(new_board(h, w), h * w * 2)
        if visit(visited, board):
            continue

        # find a path to use as an accurate training reference
        result = search(board)

        # we can use all intermediate boards as examples
        while len(examples) < num_examples:
            distance = len(result.solution)
            examples.append((freeze_board(board), distance))
            pbar.update(1)
            if not len(result.solution):
                break
            move = result.solution.pop(0)
            apply_move(board, move)
            if visit(visited, board):
                break
    pbar.close()

    return examples


def load_examples(h: int, w: int, examples_file: str = None) -> list:
    """
    Loads examples from a JSON file.
    """
    if examples_file is None:
        examples_file = paths.get_examples_path(h, w)
    with open(examples_file, "rt") as fp:
        return json.load(fp)


def save_examples(h: int, w: int, examples, examples_file: str = None) -> None:
    """
    Save a list of examples to disk as JSON.
    """
    if examples_file is None:
        examples_file = paths.get_examples_path(h, w)
    with open(examples_file, "wt") as fp:
        json.dump(examples, fp)


def load_dataset(
    h: int,
    w: int,
    num_examples: int,
    examples_file=None,
) -> torch.utils.data.Dataset:
    """
    Loads examples, constructs a SlidingPuzzleDataset from them, and returns it.
    If there is a mismatch between the requested num_examples and the provided
    dataset, examples may be truncated or new examples may be constructed and
    saved to the dataset.

    Args:
        h: The height of the board to locate a dataset for
        w: The width of the board to locate a dataset for
        num_examples: The total number of examples desired. If there are too many,
            examples will be truncated. If there are too few, new examples will be
            constructed.
        examples_file: The name of the examples file to save or load.

    Returns:
        A dataset for the requested puzzle size.
    """
    print("Loading dataset...")
    try:
        examples = load_examples(h, w, examples_file)
        print(f"Dataset loaded with {len(examples)} examples.")
    except FileNotFoundError:
        print("No dataset found.")
        examples = []

    if len(examples) > num_examples:
        examples = examples[:num_examples]
    if len(examples) < num_examples:
        num_new_examples = num_examples - len(examples)
        print(f"Constructing {num_new_examples} new examples...")
        examples = make_examples(h, w, num_new_examples)
        save_examples(h, w, examples, examples_file)

    return SlidingPuzzleDataset(examples)
