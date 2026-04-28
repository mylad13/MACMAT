from hetmarl.envs.gridworld.frontier.utils import generate_square
import numpy as np
from .utils import *
import random


def utility_goal(map, unexplored, loc, steps, edge_len=7):
    _, vis = bfs(map, loc, steps)
    H, W = map.shape
    utility = np.zeros((H, W), dtype=np.int32)
    frontiers = []
    for x in range(H):
        for y in range(W):
            if map[x, y] == 2 and vis[x, y]:
                mat = generate_square(H, W, (x, y), edge_len)
                utility[x, y] = unexplored[mat == 1].sum()
                frontiers.append((x, y))
                
    if len(frontiers) > 0:
        mx = utility.max()
        value = [utility[x, y] for x, y in frontiers]
        candidates = [(x, y) for i, (x, y) in enumerate(frontiers) if value[i] == mx]
        goal = random.choice(candidates)
    else:
        # Fallback 1: Are there any unexplored cells left at all?
        unexplored_cells = np.argwhere(unexplored == 1)
        if len(unexplored_cells) > 0:
            # Go to the closest unexplored cell (Manhattan distance)
            distances = np.abs(unexplored_cells[:, 0] - loc[0]) + np.abs(unexplored_cells[:, 1] - loc[1])
            goal = tuple(unexplored_cells[np.argmin(distances)])
        else:
            # Fallback 2: Map is 100% explored. Stay in place to avoid blocking corridors.
            goal = loc
            
    return goal