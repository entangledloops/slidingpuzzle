# Sliding Puzzle

![docs](https://readthedocs.org/projects/slidingtilepuzzle/badge/?version=latest)
![tests](https://github.com/entangledloops/slidingpuzzle/actions/workflows/tests.yaml/badge.svg)
![PyPI - Version](https://img.shields.io/pypi/v/slidingpuzzle.svg)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/slidingpuzzle)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
<a href="https://www.buymeacoffee.com/entangledloops" target="_blank"><img src="https://www.buymeacoffee.com/assets/img/custom_images/orange_img.png" alt="Buy Me A Coffee" height=32></a>

[Installation](https://slidingtilepuzzle.readthedocs.io/en/latest/install.html) | Documentation ([Latest](https://slidingtilepuzzle.readthedocs.io/en/latest/slidingpuzzle.html) | [Stable](https://slidingtilepuzzle.readthedocs.io/en/stable/slidingpuzzle.html))

A package for solving sliding tile puzzles.

- [Sliding Puzzle](#sliding-puzzle)
  - [Installation](#installation)
  - [Examples](#examples)
  - [Algorithms](#algorithms)
  - [Heuristics](#heuristics)
    - [Neural Nets](#neural-nets)
  - [Custom Models](#custom-models)
  - [Contributing](#contributing)


## Installation

```console
pip install slidingpuzzle
```

## Examples

```python
>>> from slidingpuzzle import *
>>> board = new_board(3, 3)
>>> print_board(board)
1 2 3
4 5 6
7 8
>>> print_board(shuffle_board(board))
8 3 1 
4   2 
5 6 7 
```

Regular boards are stored as [numpy arrays](https://numpy.org/doc/stable/reference/generated/numpy.array.html). The number `0` is reserved for the blank. 

```python
>>> board
array([[8, 3, 1],
       [4, 5, 6],
       [5, 6, 7]])
```

You can easily build your own boards using numpy or any of the provided convenience methods:

```python
>>> board = board_from([4, 5, 6], [7, 8, 0], [1, 2, 3])
>>> print_board(board)
4 5 6
7 8
1 2 3
>>> board = board_from_iter(3, 3, [4, 5, 6, 7, 8, 0, 1, 2, 3])
>>> print_board(board)
4 5 6
7 8
1 2 3
>>> manhattan_distance(board)
11
>>> is_solvable(board)
False
```

Not all board configurations are solvable. The [`search()`](https://slidingtilepuzzle.readthedocs.io/en/latest/slidingpuzzle.html#slidingpuzzle.algorithms.search) routine will validate the board before beginning, and may throw a `ValueError` if the board is illegal.

The default search is [`A*`](https://slidingtilepuzzle.readthedocs.io/en/latest/slidingpuzzle.html#slidingpuzzle.algorithms.a_star) with [`linear_conflict_distance()`](https://slidingtilepuzzle.readthedocs.io/en/latest/slidingpuzzle.html#slidingpuzzle.heuristics.linear_conflict_distance) as the heuristic:

```python
>>> board = shuffle_board(new_board(3, 3))
>>> search(board)
solution=[5, 4, 1, 2, 6, 5, 3, 7, 4, 1, 2, 3, 5, 8, 7, 4, 1, 2, 3, 6]
solution_len=20, generated=360, expanded=164, unvisited=197, visited=136
```

- `generated` is the total number of nodes generated during the search
- `expanded` is the total number of nodes that were evaluated (removed from search frontier)
- `unvisited` is the number of nodes that we never reached because search terminated early (search frontier, "open")
- `visited` is the number of unique states visited (`expanded` minus duplicate state expansions, "closed")


```python
>>> search(b, "bfs")
solution=[3, 7, 5, 4, 1, 2, 6, 8, 7, 5, 4, 1, 2, 3, 5, 4, 1, 2, 3, 6]
solution_len=20, generated=125450, expanded=88173, unvisited=37278, visited=45763
>>> search(b, "greedy")
solution=[3, 7, 5, 4, 1, 3, 4, 1, 3, 2, 6, 8, 7, 4, 2, 6, 8, 7, 4, 5, 1, 2, 5, 4, 7, 8]
solution_len=26, generated=556, expanded=374, unvisited=183, visited=202
```

Notice how many states are generated for BFS to find a solution. Greedy search finds a solution quickly, but the solution is of lower quality.

There are numerous convenience functions available for working with boards. Below are a few examples.

```python
>>> find_blank(board)
(1, 1)
>>> find_tile(board, 3)
(1, 0)
>>> moves = get_next_moves(board)
>>> moves
[(1, 0), (1, 2), (0, 1), (2, 1)]
>>> swap_tiles(board, moves[0])
array([[7, 5, 4],
       [0, 3, 1],
       [8, 6, 2]])
```

```python
>>> result = search(board)
>>> result.solution
[(0, 1), (0, 2), (1, 2), (2, 2), (2, 1), (1, 1), (1, 0), (0, 0), (0, 1), (0, 2), (1, 2), (1, 1), (2, 1), (2, 0), (1, 0), (0, 0), (0, 1), (0, 2), (1, 2), (2, 2)]
```

If you are working with a physical puzzle and actual tile numbers would be easier to read, you can obtain them using the convenience function [`solution_as_tiles()`](https://slidingtilepuzzle.readthedocs.io/en/latest/slidingpuzzle.html#slidingpuzzle.board.solution_as_tiles):

```python
>>> solution_as_tiles(result.board, result.solution)
[5, 4, 1, 2, 6, 5, 3, 7, 4, 1, 2, 3, 5, 8, 7, 4, 1, 2, 3, 6]
```

We can [`compare()`](https://slidingtilepuzzle.readthedocs.io/en/latest/slidingpuzzle.html#slidingpuzzle.algorithms.compare) two heuristics like this:
```python
>>> compare(3, 3, ha=manhattan_distance, hb=euclidean_distance)
(1594.87, 3377.5)
```

The numbers are the average number of states generated over `num_iters` runs for each heuristic.

Or we can compare two algorithms:

```python
>>> compare(3, 3, alga="a*", algb="greedy")
(2907.5, 618.0)
```

The solutions are actually stored as a list of (y, x)-coords of moves, indicating which tile is to be moved next:

## Algorithms

```python
>>> list(Algorithm)
[<Algorithm.A_STAR: 'a*'>, <Algorithm.BEAM: 'beam'>, <Algorithm.BFS: 'bfs'>, <Algorithm.DFS: 'dfs'>, <Algorithm.GREEDY: 'greedy'>, <Algorithm.IDA_STAR: 'ida*'>, <Algorithm.IDDFS: 'iddfs'>]
```

The `search()` method accepts the enum values or the `str` name.

The available algorithms are:
- `"a*"` (*default*) - [Docs](https://slidingtilepuzzle.readthedocs.io/en/latest/slidingpuzzle.html#slidingpuzzle.algorithms.a_star), [Wiki](https://en.wikipedia.org/wiki/A*_search_algorithm)
- `"beam"` - [Docs](https://slidingtilepuzzle.readthedocs.io/en/latest/slidingpuzzle.html#slidingpuzzle.algorithms.beam), [Wiki](https://en.wikipedia.org/wiki/Beam_search)
- `"bfs"` - [Docs](https://slidingtilepuzzle.readthedocs.io/en/latest/slidingpuzzle.html#slidingpuzzle.algorithms.bfs), [Wiki](https://en.wikipedia.org/wiki/Breadth-first_search)
- `"dfs"` - [Docs](https://slidingtilepuzzle.readthedocs.io/en/latest/slidingpuzzle.html#slidingpuzzle.algorithms.dfs), [Wiki](https://en.wikipedia.org/wiki/Depth-first_search)
- `"greedy"` - [Docs](https://slidingtilepuzzle.readthedocs.io/en/latest/slidingpuzzle.html#slidingpuzzle.algorithms.greedy), [Wiki](https://en.wikipedia.org/wiki/Best-first_search#Greedy_BFS)
- `"ida*"` - [Docs](https://slidingtilepuzzle.readthedocs.io/en/latest/slidingpuzzle.html#slidingpuzzle.algorithms.ida_star), [Wiki](https://en.wikipedia.org/wiki/Iterative_deepening_A*)
- `"iddfs"` - [Docs](https://slidingtilepuzzle.readthedocs.io/en/latest/slidingpuzzle.html#slidingpuzzle.algorithms.iddfs), [Wiki](https://en.wikipedia.org/wiki/Iterative_deepening_depth-first_search)

All algorithms support behavior customization via `kwargs`. See the docs for individual algorithms linked above.

Of the provided algorithms, only beam search is incomplete by default. This means it
may miss the goal, even thought the board is solvable.

## Heuristics

The available heuristics are:
- [`corner_tiles_heuristic`](https://slidingtilepuzzle.readthedocs.io/en/latest/slidingpuzzle.html#slidingpuzzle.heuristics.corner_tiles_distance) - Adds additional distance penalty when corners are incorrect, but the neighbors are in goal position. Additional moves are needed to shuffle the corners out and the neighbors back.
- [`euclidean_distance`](https://slidingtilepuzzle.readthedocs.io/en/latest/slidingpuzzle.html#slidingpuzzle.heuristics.euclidean_distance) - The straight line distance in Euclidean space between two tiles. This is essentially the hypotenuse of a right triangle. (The square root is not used as it does not affect the sorting order.)
- [`hamming_distance`](https://slidingtilepuzzle.readthedocs.io/en/latest/slidingpuzzle.html#slidingpuzzle.heuristics.hamming_distance) - Count of how many tiles are in the correct position
- [`last_moves_distance`](https://slidingtilepuzzle.readthedocs.io/en/latest/slidingpuzzle.html#slidingpuzzle.heuristics.last_moves_distance) - Adds an additional 2 penalty if the neighbors of the blank goal corner are not in the blank's goal row/col, as they will need to be moved their just before the final move.
- [`linear_conflict_distance`](https://slidingtilepuzzle.readthedocs.io/en/latest/slidingpuzzle.html#slidingpuzzle.heuristics.linear_conflict_distance) - This is an augmented Manhattan distance. Roughly speaking, adds an additional 2 to each pair of inverted tiles that are already on their goal row/column. (This includes the Last Moves heuristic and the Corner Tiles heuristic.)
- [`manhattan_distance`](https://slidingtilepuzzle.readthedocs.io/en/latest/slidingpuzzle.html#slidingpuzzle.heuristics.manhattan_distance) - Count of how many moves it would take each tile to arrive in the correct position, if other tiles could be ignored
- [`random_distance`](https://slidingtilepuzzle.readthedocs.io/en/latest/slidingpuzzle.html#slidingpuzzle.heuristics.random_distance) - This is a random number (but a *consistent* random number for a given board state). It is useful as a baseline.
- [`relaxed_adjacency_distance`](https://slidingtilepuzzle.readthedocs.io/en/latest/slidingpuzzle.html#slidingpuzzle.heuristics.relaxed_adjacency_distance) - This is a slight improvement over Hamming distance that includes a penalty for swapping out-of-place tiles.
- Neural net heuristics from [`slidingpuzzle.nn`](https://slidingtilepuzzle.readthedocs.io/en/latest/slidingpuzzle.nn.html) submodule (see section below)
- Any heuristic you want. Just pass any function that accepts a board and returns a number. The lower the number, the closer the board is to the goal (lower = better).

There are two simple provided utility functions for evaluating algorithm/heuristic performance: [`evaluate()`](https://slidingtilepuzzle.readthedocs.io/en/latest/slidingpuzzle.html#slidingpuzzle.algorithms.evaluate) and [`compare()`](https://slidingtilepuzzle.readthedocs.io/en/latest/slidingpuzzle.html#slidingpuzzle.algorithms.compare).

For example, here we use `compare()` with a trivial custom heuristic to see how it fares against Manhattan Distance:

```python
>>> def max_distance(board):
...     return max(manhattan_distance(board), relaxed_adjacency_distance(board))
... 
>>> compare(3, 3, ha=manhattan_distance, hb=max_distance)
(3020.5, 2857.53)
```

We can reasonably conclude that on average the max of these two heuristics is slightly better than either one alone ([Hansson et al., 1985](https://www.sciencedirect.com/science/article/abs/pii/002002559290070O)).

We can use `evaluate()` to study an algorithm's behavior when tweaking a parameter.

```python
import matplotlib.pyplot as plt
import numpy as np
x = np.linspace(1, 32, num=256)
y = [evaluate(3, 3, weight=w) for w in x]
plt.plot(x, y)
plt.title("Average Nodes Generated vs. A* Weight")
plt.xlabel("Weight")
plt.ylabel("Generated")
plt.show()
```

![Generated vs. Weight](https://raw.githubusercontent.com/entangledloops/slidingpuzzle/master/media/generated_vs_weight.png))

### Neural Nets

Well-trained neural networks are generally superior to the other heuristics. Pre-trained nets will be available for download soon. For now, you can follow the steps below to train and use your own net from scratch using the models defined in [`slidingpuzzle.nn.models`](https://slidingtilepuzzle.readthedocs.io/en/latest/slidingpuzzle.nn.html#module-slidingpuzzle.nn.models).

```console
pip install -r requirements-nn.txt
```
> **_Note:_**  This will install the CUDA 11.6 version of PyTorch. If you want another version, you will need to follow the [official instructions](https://pytorch.org/).

You can then train a new network easily:

```python
>>> import slidingpuzzle.nn as nn
>>> nn.set_seed(0)  # if you want reproducible weights
>>> h, w = 3, 3
>>> dataset = nn.build_or_load_dataset(h, w, 2**14)
>>> model = nn.Model_v1(h, w)
>>> nn.train(model, dataset)
```
> **_Note:_**  Unless you are providing your own dataset, for model sizes larger than `3 x 3` you probably need to pass `kwargs` to [`train()`](https://slidingtilepuzzle.readthedocs.io/en/latest/slidingpuzzle.nn.html#slidingpuzzle.nn.train.train) so that the search algorithm used for generating training example can find solutions in a reasonable timeframe. For example:

```python
>>> import slidingpuzzle.nn as nn
>>> h, w = 4, 4
>>> dataset = nn.build_or_load_dataset(h, w, 2**14, weight=2)  # use Weighted A* with weight of 2; all kwargs forwarded to search()
>>> model = nn.Model_v1(h, w)
>>> nn.train(model, dataset)
```

The default behavior runs until it appears test accuracy has been declining for "a while". See the docs for for details.

If you left the default settings for ``checkpoint_freq``, you will now have various model checkpoints available from training.

The model with estimated highest accuracy on the test data is tagged `"acc"` in the checkpoints directory.

You can evaluate a checkpoint similar to `evaluate`:
```python
>>> nn.eval_checkpoint(model, tag="acc")
417.71875
```

Or the latest model epoch:
```python
>>> nn.eval_checkpoint(model, tag="latest", num_iters=128)
```

The call to `eval_checkpoint()` will load the model weights from the appropriate checkpoint file and run `evaluate()`.

You can also manually load checkpoints:
```python
>>> checkpoint = nn.load_checkpoint(model, tag="epoch_1499")
>>> checkpoint["epoch"]
1499
```

(See the `checkpoints` directory for all trained models available to load by `tag`.)

You can then register the model:
```python
>>> nn.set_heuristic(model)
```

Your model is now available as [`nn.v1_distance`](https://slidingtilepuzzle.readthedocs.io/en/latest/slidingpuzzle.nn.html#slidingpuzzle.nn.heuristics.v1_distance) if you are using the default provided model [`Model_v1`](https://slidingtilepuzzle.readthedocs.io/en/latest/slidingpuzzle.nn.html#slidingpuzzle.nn.models.Model_v1). (These are associated behind the scenes via the `model.version` property.)

You can freeze your preferred model to disk to be used as the default for `nn.v1_distance`:
```python
>>> nn.save_model(model)
```
> **_Note:_** This may overwrite a previously saved model.

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

Just change `"my_model_version"` to the string you used in your model class.

And you use it as expected:

```python
>>> search(board, "a*", heuristic=my_model_distance)
```

You can add your `my_model_distance()` function to the bottom of [`nn/heuristics.py`](https://slidingtilepuzzle.readthedocs.io/en/latest/_modules/slidingpuzzle/nn/heuristics.html#v1_distance) to make it permanently available.

During training, tensorboard will show your training/test loss and accuracy.
After training is complete, you can also evaluate each checkpoint for comparison as shown above.

## Contributing

First of all, thanks for contributing!
Setup your dev environment:

```console
pip install -r requirements-dev.txt
pre-commit install
```

After you've added your new code, verify you haven't broken anything by running [`pytest`](https://pypi.org/project/pytest/):
```console
pytest
```

If you changed anything in the `slidingpuzzle.nn` package, also run:
```console
pip install -r requirements-nn.txt
pytest tests/test_nn.py
```

Don't forget to add new tests for anything you've added.

Finally, check that the docs look correct:
```console
cd docs
make html
```
> **_Note:_**  Use `./make html` on Windows.

You can also run `mypy` and look for any new violations:
```console
mypy src
```

[`Black`](https://pypi.org/project/black/) and [`flake8`](https://pypi.org/project/flake8/) are used for formatting and linting, but they are automatically run by the pre-commit hooks you installed earlier in the Git repo.
