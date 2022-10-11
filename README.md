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
([1, 6, 7], [4, 0, 8], [5, 3, 2])
>>> print_board(b)
1 6 7
4   8
5 3 2
>>> result = search(b)
>>> result
solution=[3, 2, 8, 3, 6, 7, 3, 6, 7, 1, 4, 7, 2, 5, 7, 4, 1, 2, 5, 8]
solution_len=20, generated=165616, expanded=120653, unvisited=44964, visited=62277
>>> result = search(b, "greedy", heuristic=manhattan_distance)
>>> result
solution=[8, 2, 3, 8, 2, 7, 6, 2, 7, 3, 8, 5, 4, 7, 5, 4, 7, 5, 3, 6, 2, 3, 4, 8, 6, 2, 3, 1, 5, 4, 2, 6, 8, 7, 4, 5, 1, 2, 5, 4, 7, 8]
solution_len=42, generated=711, expanded=490, unvisited=222, visited=258
```

As expected, greedy search finds a solution must faster but the solution is of lower quality than the optimal solution found by breadth-first search (the default).

Solutions are stored as a list of (y, x)-coords of moves, indicating which tile is to be moved next.

```python
>>> result.solution
[(1, 2), (2, 2), (2, 1), (1, 1), (1, 2), (0, 2), (0, 1), (1, 1), (1, 2), (2, 2), (2, 1), (2, 0), (1, 0), (1, 1), (2, 1), (2, 0), (1, 0), (1, 1), (1, 2), (0, 2), (0, 1), (1, 1), (2, 1), (2, 2), (1, 2), (0, 2), (0, 1), (0, 0), (1, 0), (1, 1), (1, 2), (2, 2), (2, 1), (2, 0), (1, 0), (0, 0), (0, 1), (1, 1), (1, 0), (2, 0), (2, 1), (2, 2)]
```

If you are working with a physical puzzle and actual tile numbers would be easier to read, you can obtain them the same way as `repr(SearchResult)` / `str(SearchResult)`:

```python
>>> solution_as_tiles(result.board, result.solution)
[8, 2, 3, 8, 2, 7, 6, 2, 7, 3, 8, 5, 4, 7, 5, 4, 7, 5, 3, 6, 2, 3, 4, 8, 6, 2, 3, 1, 5, 4, 2, 6, 8, 7, 4, 5, 1, 2, 5, 4, 7, 8]
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
        "bound": default is float("inf"),
            restricts search to f-values < bound
        "weight": default is 1
    }
    beam: {
        "bound": default is float("inf"),
            restricts search to f-values < bound
        "width", default is 3
    }
    bfs: {
        "bound": default is float("inf"),
            restricts search to depth < bound
    }
    dfs: {
        "bound": default is float("inf"),
            restricts search to depth < bound
    }
    ida*: {
        "bound", default is manhattan_distance(board),
            restricts search to f-values < bound
        "weight": default is 1
    }
    iddfs: {
        "bound", default is manhattan_distance(board),
            restricts search to depth < bound
    }

Example:

```python
result = search(board, "beam", manhattan_distance, width=3)
```

Of the provided algorithms, only beam search is incomplete. This means it
may miss the goal, even thought the board is solvable.

## Heuristics

Heuristics are most relevant for `"greedy"`, `"a*"`, and `"ida*"` as they are used to sort all known moves before selecting the next action.

For other algorithms, the heuristic is only used to sort the local nodes. For example, when the `2 <= N <= 4` nearby available moves are generated they will be immediately sorted using the heuristic.

The available heuristics are:
- `euclidean_distance` - The straight line distance in Euclidean space between two tiles. This is essentially the hypotenuse of a right triangle. (The square root is not used as it does not affect the sorting order.)
- `hamming_distance` - Count of how many tiles are in the correct position
- `manhattan_distance` - Count of how many moves it would take each tile to arrive in the correct position, if other tiles could be ignored
- `random_distance` - This is a random number (but a *consistent* random number for a given board state). It is useful as a baseline.
- Any heuristic you want! Just pass any function that accepts a board and returns a number. The lower the number, the closer the board is to the goal (lower = better).


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
