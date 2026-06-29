"""Guard the CellCode values against accidental drift.

These are a pure rename of the magic numbers previously scattered through the
env and runner, so the values must stay exactly equal to the original literals
(and to OBJECT_TO_IDX * 20 for the scaled object codes).
"""
from hetmarl.envs.gridworld.semantics import CellCode
from hetmarl.envs.gridworld.gym_minigrid.minigrid import OBJECT_TO_IDX


def test_literal_values():
    assert int(CellCode.UNSEEN) == 0
    assert int(CellCode.EMPTY) == 20
    assert int(CellCode.WALL) == 40
    assert int(CellCode.BASE) == 60
    assert int(CellCode.DOOR) == 80
    assert int(CellCode.OBSTACLE) == 160
    assert int(CellCode.TARGET) == 180
    assert int(CellCode.HAZARD) == 200
    assert int(CellCode.AGENT_RESCUER) == 10
    assert int(CellCode.AGENT_SCOUT) == 11
    assert int(CellCode.AGENT_CLEANER) == 12


def test_scaled_codes_match_object_index_times_20():
    assert int(CellCode.UNSEEN) == OBJECT_TO_IDX["unseen"] * 20
    assert int(CellCode.EMPTY) == OBJECT_TO_IDX["empty"] * 20
    assert int(CellCode.WALL) == OBJECT_TO_IDX["wall"] * 20
    assert int(CellCode.BASE) == OBJECT_TO_IDX["base"] * 20
    assert int(CellCode.DOOR) == OBJECT_TO_IDX["door"] * 20
    assert int(CellCode.OBSTACLE) == OBJECT_TO_IDX["obstacle"] * 20
    assert int(CellCode.TARGET) == OBJECT_TO_IDX["target"] * 20
    assert int(CellCode.HAZARD) == OBJECT_TO_IDX["lava"] * 20
