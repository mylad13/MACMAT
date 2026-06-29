"""Semantic codes for grid cells in the SearchAndRescue environment.

The occupancy/observation maps store an integer per cell. Most codes are the
MiniGrid object index scaled by 20 (``OBJECT_TO_IDX[type] * 20`` from
``gym_minigrid/minigrid.py``); agent footprints are written with the small raw
codes 10/11/12. ``CellCode`` is an ``IntEnum`` so every member compares and
behaves exactly like its underlying ``int`` (including inside NumPy array
comparisons), making the replacement of the scattered magic numbers a pure,
value-preserving rename.

If you change ``OBJECT_TO_IDX`` or the ``* 20`` scaling in ``minigrid.py``, keep
the values here in sync (``tests/test_semantics.py`` guards them).
"""
from enum import IntEnum


class CellCode(IntEnum):
    # OBJECT_TO_IDX[type] * 20  (see WorldObj.encode in minigrid.py)
    UNSEEN = 0       # OBJECT_TO_IDX['unseen']  (0) * 20
    EMPTY = 20       # OBJECT_TO_IDX['empty']   (1) * 20
    WALL = 40        # OBJECT_TO_IDX['wall']    (2) * 20
    BASE = 60        # OBJECT_TO_IDX['base']    (3) * 20
    DOOR = 80        # OBJECT_TO_IDX['door']    (4) * 20
    OBSTACLE = 160   # OBJECT_TO_IDX['obstacle'](8) * 20
    TARGET = 180     # OBJECT_TO_IDX['target']  (9) * 20
    HAZARD = 200     # OBJECT_TO_IDX['lava']   (10) * 20

    # Agent footprints written directly onto the local map (not * 20 scaled).
    AGENT_RESCUER = 10
    AGENT_SCOUT = 11
    AGENT_CLEANER = 12
