# Sliding Puzzle

[![Documentation Status](https://readthedocs.org/projects/slidingtilepuzzle/badge/?version=latest)](https://slidingtilepuzzle.readthedocs.io/en/latest/?badge=latest)

A package for solving and working with sliding tile puzzles.

## Installation

```bash
pip install slidingpuzzle
```

## Documentation

https://slidingtilepuzzle.readthedocs.io/en/latest/slidingpuzzle.html

## Examples

All submodule code is imported into the parent namespace, so there is no need
to explicitly refer to submodules.

```python
>>> from slidingpuzzle import *
>>> b = new_board(3, 3)
>>> print_board(b)
1 2 3
4 5 6
7 8
>>> shuffle_board(b)
>>> print_board(b)
4 5
6 7 2
1 8 3
>>> r = search(b)
>>> print_result(r)
solution_len=20, generated=144704, expanded=103615, unvisited=41090, visited=54466
>>> r = search(b, algorithm="greedy", heuristic=manhattan_distance)
>>> print_result(r)
solution_len=36, generated=409, expanded=299, unvisited=111, visited=153
```

Solutions are stored as a list of (y, x)-coords of moves, indicating which tile is to be moved next.

```python
>>> r.solution
[(1, 2), (2, 2), (2, 1), (1, 1), (0, 1), (0, 0), (1, 0), (2, 0), (2, 1), (1, 1), (0, 1), (0, 0), (1, 0), (1, 1), (0, 1), (0, 2), (1, 2), (1, 1), (2, 1), (2, 2)]
```

If you are working with a physical puzzle and actual tile numbers would be easier to read, you can obtain them:

```python
>>> solution_as_tiles(b, r.solution)
[2, 3, 8, 7, 5, 4, 6, 1, 7, 5, 4, 6, 1, 4, 6, 2, 3, 6, 5, 8]
```

The boards are just a `tuple` of `list[int]`. The largest number is reserved for the blank. You can therefore easily build your own board, e.g.:

```python
>>> b = ([4, 5, 6], [7, 8, 9], [1, 2, 3])
>>> print_board(b)
4 5 6
7 8
1 2 3
>>> manhattan_distance(b)
12
>>> is_solvable(b)
False
```

Not all board configurations are solvable. The `search()` routine will validate the board before beginning, and may throw a `ValueError` if the board is illegal.

## Algorithms

The available algorithms for `search(... algorithm="choose from below")` are:
- `"bfs"` (default) - [Breadth-first search](https://en.wikipedia.org/wiki/Breadth-first_search)
- `"dfs"` - [Depth-first search](https://en.wikipedia.org/wiki/Depth-first_search)
- `"greedy"` - [Greedy best-first search](https://en.wikipedia.org/wiki/Best-first_search#Greedy_BFS)
- `"a"*` - [A* search](https://en.wikipedia.org/wiki/A*_search_algorithm)

## Heuristics

Heuristics are most relevant for `"greedy"` and `"a*"`, as they are used to sort all known moves before selecting the next action.

When used with either `"bfs"` or `"dfs"`, the heuristic is only used to sort the local nodes. For example, when the `N` nearby available moves are examined, the algorithm will first sort those `N` next board states using the heuristic.

The available heuristics are:
- `euclidean_distance` - The straight line distance in Euclidean space between two tiles. This is essentially the hypotenuse of a right triangle. (The square root is not used as it does not affect the sorting order.)
- `hamming_distance` - Count of how many tiles are in the correct position
- `manhattan_distance` - Count of how many moves it would take each tile to arrive in the correct position, if other tiles could be ignored
- `random_distance` - This is a random number (but a *consistent* random number for a given board state). It is useful as a baseline.
- Any heuristic you want! Just pass any function that accepts a board and returns a number. The lower the number, the closer the board is to the goal (lower = better).