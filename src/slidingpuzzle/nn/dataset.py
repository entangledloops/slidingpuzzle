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

from typing import Iterable, Optional, TypeAlias

import collections
import gc
import json
import logging

import torch
import torch.utils
import tqdm

import slidingpuzzle.algorithms as algorithms
import slidingpuzzle.nn.paths as paths
from slidingpuzzle.board import (
    Board,
    FrozenBoard,
    find_blank,
    freeze_board,
    new_board,
    shuffle,
    solution_as_tiles,
    swap_tiles,
    visit,
)
from slidingpuzzle.state import State


Example: TypeAlias = tuple[FrozenBoard, list[int]]
log = logging.getLogger(__name__)


class SlidingPuzzleDataset(torch.utils.data.Dataset):
    def __init__(self, examples: Iterable[Example]) -> None:
        super().__init__()
        self.examples = [
            (
                torch.tensor(x, dtype=torch.float32),
                torch.tensor([len(y)], dtype=torch.float32),
            )
            for x, y in examples
        ]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


def all_examples(
    h: int, w: int, start: int = 0, stop: Optional[int] = None
) -> list[Example]:
    """
    Helper to generate all training examples for a board size by performing reverse BFS.

    .. note::
        It is safe to interrupt this function any time with a keyboard interrupt and
        receive back the examples generated so far.

    Args:
        h: Height of the board
        w: Width of the board
        start: Start index (0 starts at the initial solved board)
        stop: Index to stop at (``None`` will product all boards)

    Returns:
         A list of examples.
    """
    visited: set[FrozenBoard] = set()
    examples = []
    board = new_board(h, w)
    initial_state = State(board, find_blank(board))
    unvisited = collections.deque([initial_state])
    count = 0
    pbar = tqdm.tqdm()
    try:
        while unvisited and (stop is None or count < stop):
            count += 1
            state = unvisited.popleft()
            if count < start or visit(visited, state.board):
                continue
            board = freeze_board(state.board)
            solution = tuple(solution_as_tiles(state.board, reversed(state.history)))
            examples.append((board, solution))
            pbar.update(1)
            next_states = algorithms.get_next_states(state)
            # patch the history to reflect that this search is backwards
            for ns in next_states:
                ns.history[-1] = state.blank_pos
            unvisited.extend(next_states)
    except KeyboardInterrupt:
        log.info(f"Example generation interrupted, returning {len(examples)} examples.")
    finally:
        pbar.close()
    return examples


def random_examples(
    h: int,
    w: int,
    num_examples: int,
    ignore_examples: Optional[list[Example]] = None,
    **kwargs,
) -> list[Example]:
    """
    Constructs a list of training examples, which are tuples of:
        (board, solution)

    Args:
        h: Height of board.
        w: Width of board.
        num_examples: Number of examples to produce.
        ignore_examples: If any example produced matches one in ``ignore_examples``,
            it will be discarded.
        kwargs: Args to pass to the search algorithm used to find board solutions.

    Returns:
        The list of training examples.
    """
    visited: set[FrozenBoard] = set()
    examples = []
    if ignore_examples is not None:
        dupe_found = False
        for board, _ in ignore_examples:
            if visit(visited, board):
                dupe_found = True
        if dupe_found:
            log.warning("Duplicate found in prior examples.")

    def add_examples(board: Board) -> None:
        # free memory from prior searches
        gc.collect()

        # find a path to use as an accurate training reference
        result = algorithms.search(board, **kwargs)

        # we can use all intermediate boards as examples
        while len(examples) < num_examples:
            solution = tuple(solution_as_tiles(result.board, result.solution))
            examples.append((freeze_board(board), solution))
            pbar.update(1)
            if not len(result.solution):
                break
            swap_tiles(board, result.solution.pop(0))
            if visit(visited, board):
                break

    try:
        with tqdm.tqdm(total=num_examples) as pbar:
            while len(examples) < num_examples:
                board = shuffle(new_board(h, w))
                if not visit(visited, board):
                    add_examples(board)
    except KeyboardInterrupt:
        log.info(f"Example generation interrupted, returning {len(examples)} examples.")

    return examples


def load_examples(h: int, w: int, examples_file: Optional[str] = None) -> list[Example]:
    """
    Loads examples from a JSON file.
    """
    if examples_file is None:
        examples_file = paths.get_examples_path(h, w)
    with open(examples_file, "rt") as fp:
        return json.load(fp)


def save_examples(examples: list[Example], examples_file: Optional[str] = None) -> None:
    """
    Save a list of examples to disk as JSON.
    """
    if 0 == len(examples):
        return
    if examples_file is None:
        board, _ = examples[0]
        h, w = len(board), len(board[0])
        examples_file = paths.get_examples_path(h, w)
    with open(examples_file, "wt") as fp:
        json.dump(examples, fp)


def get_examples(
    h: int, w: int, num_examples: int, prior_examples: list[Example], **kwargs
) -> list[Example]:
    """
    Returns ``num_examples`` total unique examples, starting with ``prior_examples`` if
    they are provided. May truncate or produce new examples as necessary.
    """
    examples = list(prior_examples)
    if len(examples) > num_examples:
        examples = examples[:num_examples]
    if len(examples) < num_examples:
        num_new_examples = num_examples - len(prior_examples)
        log.info(f"Constructing {num_new_examples} new examples...")
        examples.extend(
            random_examples(h, w, num_new_examples, prior_examples, **kwargs)
        )
    return examples


def build_or_load_dataset(
    h: int,
    w: int,
    num_examples: Optional[int] = None,
    examples_file: Optional[str] = None,
    **kwargs,
) -> SlidingPuzzleDataset:
    """
    Loads examples, constructs a SlidingPuzzleDataset from them, and returns it.
    If there is a mismatch between the requested num_examples and the loaded
    examples file, examples may be truncated or new examples may be constructed and
    saved to disk. No duplicate examples are produced.

    Args:
        h: The height of the board to locate a dataset for
        w: The width of the board to locate a dataset for
        num_examples: The total number of examples desired. If there are too many,
            examples will be truncated. If there are too few, new examples will be
            constructed. If ``None``, the entire dataset loaded will be used.
        examples_file: The name of the examples file to save or load.
        kwargs: Will be forwaded to :func:`make_examples` if it is called.

    Returns:
        A dataset for the requested puzzle size.
    """
    log.info("Loading dataset...")
    try:
        examples = load_examples(h, w, examples_file)
        log.info(f"Dataset loaded with {len(examples)} examples.")
    except FileNotFoundError:
        log.info("No dataset found.")
        examples = []

    # if a specific number was not requested, use all loaded examples
    if num_examples is None:
        num_examples = len(examples)

    # if we were asked for a different number of examples than we have on hand
    if len(examples) != num_examples:
        new_examples = get_examples(h, w, num_examples, examples, **kwargs)
        # if we made new examples, save them for next time
        if len(new_examples) > len(examples):
            print(new_examples)
            save_examples(new_examples, examples_file)
            log.info("Dataset saved.")
        examples = new_examples

    return SlidingPuzzleDataset(examples)


def load_dataset(
    h: int, w: int, examples_file: Optional[str] = None
) -> SlidingPuzzleDataset:
    """
    Loads examples, constructs a SlidingPuzzleDataset from them, and returns it.

    Args:
        h: The height of the board to locate a dataset for
        w: The width of the board to locate a dataset for
        examples_file: Path to file. Default will check local ``datasets`` directory.

    Returns:
        A dataset for the requested puzzle size.
    """
    log.info("Loading dataset...")
    try:
        examples = load_examples(h, w, examples_file)
        log.info(f"Dataset loaded with {len(examples)} examples.")
    except FileNotFoundError:
        log.info("No dataset found.")
        examples = []
    return SlidingPuzzleDataset(examples)
