# Sliding Tile Puzzle

Usage:

```python
>>> import stp
>>> import heuristics
>>> board = stp.new_board(3, 3)
>>> stp.shuffle_board(board)
>>> result = stp.search(board)
>>> stp.print_result(result)
solution_len=20, generated=111664, expanded=77083, unvisited=34582, visited=40844
>>> result = stp.search(board, algorithm="greedy", heuristic=heuristics.manhattan_distance) 
>>> print_result(result)
solution_len=80, generated=461, expanded=303, unvisited=159, visited=173
```