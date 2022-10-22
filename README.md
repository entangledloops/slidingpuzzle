# Sliding Puzzle

[![docs](https://readthedocs.org/projects/slidingtilepuzzle/badge/?version=latest)](https://slidingtilepuzzle.readthedocs.io/en/latest/?badge=latest)
![tests](https://github.com/entangledloops/slidingpuzzle/actions/workflows/tests.yaml/badge.svg)
<a href="https://www.buymeacoffee.com/entangledloops" target="_blank"><img src="https://www.buymeacoffee.com/assets/img/custom_images/orange_img.png" alt="Buy Me A Coffee" ></a>

- [Sliding Puzzle](#sliding-puzzle)
  - [Installation](#installation)
  - [Documentation](#documentation)
  - [Examples](#examples)
  - [Algorithms](#algorithms)
  - [Heuristics](#heuristics)
    - [Neural Nets](#neural-nets)
  - [Custom Models](#custom-models)
  - [Creating a Pull Request](#creating-a-pull-request)

A package for solving and working with sliding tile puzzles.

## Installation

```bash
pip install slidingpuzzle
```

## Documentation

https://slidingtilepuzzle.readthedocs.io/en/latest/slidingpuzzle.html

## Examples

```python
>>> from slidingpuzzle import *
>>> b = new_board(3, 3)
>>> print_board(b)
1 2 3
4 5 6
7 8
>>> shuffle_board(b)
>>> print_board(b)
1 6 7
4   8
5 3 2
```

The boards are just a `tuple` of `list[int]`. The number `0` is reserved for the blank. You can easily build your own board:

```python
>>> b = ([4, 5, 6], [7, 8, 0], [1, 2, 3])
>>> print_board(b)
4 5 6
7 8
1 2 3
>>> manhattan_distance(b)
11
>>> is_solvable(b)
False
```

Not all board configurations are solvable. The [`search()`](https://slidingtilepuzzle.readthedocs.io/en/latest/slidingpuzzle.html#slidingpuzzle.slidingpuzzle.search) routine will validate the board before beginning, and may throw a `ValueError` if the board is illegal.

The default search is `A*` with `manhattan_distance` as the heuristic:

```python
>>> search(b)
solution=[3, 2, 8, 3, 6, 7, 3, 6, 7, 1, 4, 7, 2, 5, 7, 4, 1, 2, 5, 8]
solution_len=20, generated=829, expanded=518, unvisited=312, visited=313
>>> search(b, "bfs")
solution=[3, 2, 8, 3, 6, 7, 3, 6, 7, 1, 4, 7, 2, 5, 7, 4, 1, 2, 5, 8]
solution_len=20, generated=165616, expanded=120653, unvisited=44964, visited=62277
>>> search(b, "greedy")
solution=[8, 2, 3, 8, 2, 7, 6, 2, 7, 3, 8, 5, 4, 7, 5, 4, 7, 5, 3, 6, 2, 3, 4, 8, 6, 2, 3, 1, 5, 4, 2, 6, 8, 7, 4, 5, 1, 2, 5, 4, 7, 8]
solution_len=42, generated=711, expanded=490, unvisited=222, visited=258
```

As expected, greedy search finds a solution must faster than BFS, but the solution is of lower quality.

We can get a rough comparison of two heuristics like this:
```python
>>> compare(3, 3, ha=manhattan_distance, hb=euclidean_distance)
(1594.8666666666666, 3377.5)
```

The numbers are the average number of states expanded over `num_iters` runs for each heuristic.

Or we can compare two algorithms:

```python
>>> compare(3, 3, alga="a*", algb="greedy")
(2907.5, 618.0)
```

The solutions are actually stored as a list of (y, x)-coords of moves, indicating which tile is to be moved next:

```python
>>> result = search(b)
>>> result.solution
[(2, 1), (2, 2), (1, 2), (1, 1), (0, 1), (0, 2), (1, 2), (1, 1), (0, 1), (0, 0), (1, 0), (1, 1), (2, 1), (2, 0), (1, 0), (0, 0), (0, 1), (1, 1), (2, 1), (2, 2)]
```

If you are working with a physical puzzle and actual tile numbers would be easier to read, you can obtain them the same way `str(SearchResult)` does internally:

```python
>>> solution_as_tiles(result.board, result.solution)
[8, 2, 3, 8, 2, 7, 6, 2, 7, 3, 8, 5, 4, 7, 5, 4, 7, 5, 3, 6, 2, 3, 4, 8, 6, 2, 3, 1, 5, 4, 2, 6, 8, 7, 4, 5, 1, 2, 5, 4, 7, 8]
```

## Algorithms

```python
>>> print(ALGORITHMS)
('a*', 'beam', 'bfs', 'dfs', 'greedy', 'ida*', 'iddfs')
```

The available algorithms for [`search()`](https://slidingtilepuzzle.readthedocs.io/en/latest/slidingpuzzle.html#slidingpuzzle.slidingpuzzle.search) are:
- `"a*"` - [A* search](https://en.wikipedia.org/wiki/A*_search_algorithm)
- `"beam"` - [Beam search](https://en.wikipedia.org/wiki/Beam_search)
- `"bfs"` (*default*) - [Breadth-first search](https://en.wikipedia.org/wiki/Breadth-first_search)
- `"dfs"` - [Depth-first search](https://en.wikipedia.org/wiki/Depth-first_search)
- `"greedy"` - [Greedy best-first search](https://en.wikipedia.org/wiki/Best-first_search#Greedy_BFS)
- `"ida*"` - [Iterative deepening A*](https://en.wikipedia.org/wiki/Iterative_deepening_A*)
- `"iddfs"` - [Iterative deepening depth-first search](https://en.wikipedia.org/wiki/Iterative_deepening_depth-first_search)

Some algorithms support additional customization via `kwargs`. These are:

    a*: {
        "depth_bound": default is float("inf"),
            restrict search to depth < depth_bound
        "detect_dupes": default is True,
            if False, will trade time for memory savings
        "f_bound": default is float("inf"),
            restricts search to f-values < bound
        "heuristic": default is manhattan_distance
        "weight": default is 1
    }
    beam: {
        "depth_bound": default is float("inf"),
            restrict search to depth < depth_bound
        "detect_dupes": default is True,
            if False, will trade time for memory savings
        "f_bound": default is float("inf"),
            restricts search to f-values < bound
        "heuristic": default is manhattan_distance
        "width", default is 3
    }
    bfs: {
        "depth_bound": default is float("inf"),
            restrict search to depth < depth_bound
        "detect_dupes": default is True,
            if False, will trade time for memory savings
    }
    dfs: {
        "depth_bound": default is float("inf")
            restrict search to depth < depth_bound
        "detect_dupes": default is True,
            if False, will trade time for memory savings
    }
    greedy: {
        "depth_bound": default is float("inf"),
            restrict search to depth < depth_bound
        "f_bound": default is float("inf"),
            restricts search to f-values < bound
    }
    ida*: {
        "depth_bound": default is float("inf")
            restrict search to depth < depth_bound
        "detect_dupes": default is True,
            if False, will trade time for memory savings
        "heuristic": default is manhattan_distance
        "weight": default is 1
    }
    iddfs: {
        "detect_dupes": default is True,
            if False, will trade time for memory savings
    }

Example:

```python
>>> for weight in range(1, 16):
...     r = search(b, weight=weight)
...     print(f"weight: {weight}, solution_len: {len(r.solution)}, expanded: {r.expanded}")
...
weight: 1, solution_len: 27, expanded: 1086
weight: 2, solution_len: 35, expanded: 1026
weight: 3, solution_len: 35, expanded: 814
weight: 4, solution_len: 43, expanded: 640
weight: 5, solution_len: 45, expanded: 531
weight: 6, solution_len: 45, expanded: 563
weight: 7, solution_len: 43, expanded: 407
weight: 8, solution_len: 43, expanded: 578
weight: 9, solution_len: 43, expanded: 648
weight: 10, solution_len: 43, expanded: 689
weight: 11, solution_len: 43, expanded: 764
weight: 12, solution_len: 43, expanded: 601
weight: 13, solution_len: 43, expanded: 786
weight: 14, solution_len: 43, expanded: 733
weight: 15, solution_len: 43, expanded: 782
```

With the default parameters, we see that A* finds the optimal solution after expanding `1086` states. If we are willing to sacrifice optimality of the solution, we can try different weights to tradeoff solution quality for search time. Note that for a given problem and quality criteria, there will be an optimal weight. Beyond a certain point, the weight will have no positive effect. In the example above, using weights beyond `~38` produces further change in neither solution quality nor nodes expanded.

Of the provided algorithms, only beam search is incomplete. This means it
may miss the goal, even thought the board is solvable.

## Heuristics

The available heuristics are:
- `euclidean_distance` - The straight line distance in Euclidean space between two tiles. This is essentially the hypotenuse of a right triangle. (The square root is not used as it does not affect the sorting order.)
- `hamming_distance` - Count of how many tiles are in the correct position
- `manhattan_distance` - Count of how many moves it would take each tile to arrive in the correct position, if other tiles could be ignored
- `random_distance` - This is a random number (but a *consistent* random number for a given board state). It is useful as a baseline.
- Neural net heuristics from `slidingpuzzle.nn` submodule (see section below)
- Any heuristic you want! Just pass any function that accepts a board and returns a number. The lower the number, the closer the board is to the goal (lower = better).

### Neural Nets

Well-trained neural networks are generally superior to the other heuristics. Pre-trained nets will be available for download soon. For now, you can follow the steps below to train and use your own net from scratch using the models defined in `slidingpuzzle/nn/models.py`.

```console
pip install -r requirements-nn.txt
```

You can then train a new network easily:

```python
>>> import slidingpuzzle.nn as nn
>>> model = nn.Model_v1(3, 3)
>>> nn.train(model)
```

**Note**: Unless you are providing your own dataset, for model sizes larger than `3 x 3` you probably need to pass `kwargs` to `train()` so that the search algorithm used for generating training example can find solutions in a reasonable timeframe. For example:

```python
>>> import slidingpuzzle.nn as nn
>>> model = nn.Model_v1(4, 4, weight=2)  # use Weighted A* with weight of 2; all kwargs forwarded to search()
>>> nn.train(model)
```

The default behavior of `train()` runs until it appears test accuracy has been declining for "a while". See the docs for `train()` for details.

You will now have various model checkpoints available from training.

The model with highest accuracy on the test data is tagged `"acc"`.

```python
>>> checkpoint = nn.load_checkpoint(model, tag="acc")
>>> checkpoint["epoch"]
3540
```

Or to load a specific epoch:

```python
>>> checkpoint = nn.load_checkpoint(model, tag="epoch_1499")
```

(See the `checkpoints` directory for all trained models available to load by `tag`.)

You can then register the model:

```python
>>> nn.set_heuristic(model)
```

Your model is now available as `nn.v1_distance`. (These are associated behind the scenes via the `model.version` string.)

You can save your model to disk to be used automatically anytime you use this package as the default for `nn.v1_distance`:

```python
>>> nn.save_model(model)
```

Your model will now be available whenever you import `slidingpuzzle.nn`.

```python
>>> compare(3, 3, ha=nn.v1_distance, hb=manhattan_distance, num_iters=128, weight=7)
(73.375, 514.1328125)
```

Training uses GPU if available and falls back to CPU otherwise.

## Custom Models

First define your `torch.nn.Module` somewhere.
Your model class must:
- have a unique `self.version` string that is safe to use in filenames (e.g. `"my_model_v1"`)
- have `self.h` and `self.w` indicating the input board dimensions it expects,
- have a `forward()` that accepts board as a tensor constructed by:
  - `torch.tensor(board, dtype=torch.float32)`
  - (The tensor above does not include the batch dimension.)
  - For example, expect: `model(board)`

Train and save your model as above.

You can now copy-paste the model-based heuristic function below:

```python
def my_model_distance(board) -> float:
    h, w = len(board), len(board[0])
    heuristic = nn.get_heuristic(h, w, "my_model_version")
    return heuristic(board)
```

Just change `"my_model_version` to the string you used in your model class.

And you use it as expected:

```python
>>> search(board, "a*", heuristic=my_model_distance)
```

You can add your `my_model_distance()` function to the bottom of `nn/heuristics.py` to make it permanently available.

During training, tensorboard will show your training/test loss and accuracy.
After training is complete, you can also evaluate each checkpoint for comparison.

For example, to evaluate a specific epoch:

```python
>>> nn.eval_checkpoint(model, tag="epoch_649")
```

Or the highest accuracy observed:

```python
>>> nn.eval_checkpoint(model, tag="acc", num_iters=128)
```

Or the latest model epoch:

```python
>>> nn.eval_checkpoint(model, tag="latest", num_iters=128)
```

The call to `eval_checkpoint()` will automatically load the model weights from the checkpoint file and run `eval_heuristic()`.


## Creating a Pull Request

First of all, thanks for contributing!
Setup your dev environment:

```console
pip install -r requirements-dev.txt
```

First and **most importantly** verify you haven't broken anything by running [`pytest`](https://pypi.org/project/pytest/):
```console
pytest
```

Don't forget to add new tests for anything you've added.

You can also run `mypy` and look for any new violations:
```console
mypy src
```

Finally, check that the docs look correct:
```console
cd docs
./make html
```

[`Black`](https://pypi.org/project/black/) and [`flake8`](https://pypi.org/project/flake8/) are used for formatting and linting, but they are automatically run by the pre-commit hooks installed in the Git repo.

If you made changes in the `nn` submodule, you also need to install `requirements-nn.txt` and run `pytest tests/test_nn.py` to validate.
