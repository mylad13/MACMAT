#!/usr/bin/env python
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import time
from hetmarl.envs.gridworld.gym_minigrid.minigrid import *
import cv2
import random
import copy
import numpy as np

from hetmarl.envs.gridworld.frontier.apf import APF
from hetmarl.envs.gridworld.frontier.utility import utility_goal
from hetmarl.envs.gridworld.frontier.rrt import rrt_goal
from hetmarl.envs.gridworld.frontier.nearest import nearest_goal
from hetmarl.envs.gridworld.frontier.voronoi import voronoi_goal

from hetmarl.envs.gridworld.gym_minigrid.register import register
from hetmarl.envs.gridworld.semantics import CellCode

from hetmarl.utils import astar
from hetmarl.utils.util import get_connected_agents
from hetmarl.utils import dstarlite, plotting, env
import pyastar2d


def l1distance(x, y):
    return abs(x[0] - y[0]) + abs(x[1] - y[1])

def euclideandistance(x, y):
    return math.sqrt((x[0] - y[0])**2 + (x[1] - y[1])**2)

# Map of direction indices to vectors
DIR_TO_VEC = [
    # Pointing right (positive X)
    np.array((1, 0)),
    # Down (positive Y)
    np.array((0, 1)),
    # Pointing left (negative X)
    np.array((-1, 0)),
    # Up (negative Y)
    np.array((0, -1)),
]
VEC_TO_DIR = {tuple(v): k for k, v in enumerate(DIR_TO_VEC)}

class SearchAndRescueEnv(MiniGridEnv):
    """
    Classic 4 rooms gridworld environment.
    Can specify agent and goal position, if not it set at random.
    """

    def __init__(
        self,
        grid_size,
        max_steps,
        local_step_num,
        agent_view_size,
        num_obstacles,
        max_num_targets,
        agent_classes_list,
        num_agents=2,
        n_agent_types=2,
        agent_pos=None,
        goal_pos=None,
        use_full_comm=True,
        use_partial_comm=False,
        use_orientation=False,
        use_same_location=True,
        use_time_penalty=False,
        use_energy_penalty=False,
        use_auxiliary_rewards=False,
        use_auxiliary_rewards_decay=True,
        max_auxiliary_reward_episode = 2000,
        use_agent_obstacle = False,
        algorithm_name = 'amat',
        comm_sigma = 5,
        comm_radius = 10,
        block_doors = False,
        block_chance = 0.2,
        use_slam_noise = False,
        slam_noise_prob = 0.05,
        use_perception_noise = False,
        perception_noise_base_value = 0.1,
        perception_noise_distance_factor = 0.05,
        action_size = 5,
        dynamic_targets = False,
        target_behavior = 'default',
        dynamic_obstacles = False,
        obstacle_behavior = 'oscillate',
        spawn_hazards = False,
        max_num_hazards = 5,
        exploration_only=False,
    ):
        self.initial_grid_size = grid_size
        self._agent_default_pos = agent_pos
        self._goal_default_pos = goal_pos
        self.door_size = 1
        self.max_steps = max_steps
        self.use_same_location = use_same_location
        self.use_time_penalty = use_time_penalty
        self.use_energy_penalty = use_energy_penalty
        self.use_auxiliary_rewards = use_auxiliary_rewards
        self.use_auxiliary_rewards_decay = use_auxiliary_rewards_decay
        self.max_auxiliary_reward_episode = max_auxiliary_reward_episode

        self.use_full_comm = use_full_comm
        self.use_partial_comm = use_partial_comm
        self.use_agent_obstacle = use_agent_obstacle
        
        self.n_agent_types = n_agent_types
        self.original_agent_classes_list = agent_classes_list
        self.comm_sigma = comm_sigma
        self.comm_radius = comm_radius
        self.block_doors = block_doors
        self.block_chance = block_chance
        self.use_slam_noise = use_slam_noise
        self.slam_noise_prob = slam_noise_prob
        self.use_perception_noise = use_perception_noise
        self.perception_noise_base_value = perception_noise_base_value
        self.perception_noise_distance_factor = perception_noise_distance_factor
        self.action_size = action_size

        self.max_num_targets = max_num_targets
        self.dynamic_targets = dynamic_targets
        self.target_behavior = target_behavior
        self.dynamic_obstacles = dynamic_obstacles
        self.obstacle_behavior = obstacle_behavior
        self.spawn_hazards = spawn_hazards
        self.max_num_hazards = max_num_hazards
        self.exploration_only = exploration_only
        self.algorithm_name = algorithm_name

        if num_obstacles <= grid_size/2 + 1:
            self.num_obstacles = int(num_obstacles)
        else:
            self.num_obstacles = int(grid_size**2/30)

        super().__init__(grid_size=grid_size,
                         max_steps=max_steps,
                         num_agents=num_agents,
                         agent_view_size=agent_view_size,
                         use_full_comm=use_full_comm,
                         use_partial_comm=use_partial_comm,
                         )
        self.dynamics_rng = np.random.RandomState(seed=self.random_seed) # RNG used for dynamic targets and obstacles to ensure consistent behavior across runs
        self.comm_rng = np.random.RandomState(seed=self.random_seed + 1) # RNG used for communication noise
        self.step_rng = np.random.RandomState(seed=self.random_seed + 2) # RNG used for events at every step to ensure consistency across evaluation runs

        self.explored_ratio = 0
        # temporary fix to resuming runs is to set the episode number manually
        self.episode_n = 0
        self.mission_completed = 0
        # self.episode_n = 8767

        self.agent_exploration_reward = np.zeros((num_agents))
        self.max_steps = max_steps

        self.training_stage = 20 # 1: 1 target, 2: 1 target and 1 hazard, 3: 2 targets 1 hazard, 4: 2 targets 2 hazards, 10: set for testing hazardous SAR, 20: for non-hazardous SAR 
        self.mission_history = [] # Track last 100 mission completions
        
    def overall_gen_grid(self, width, height):
        # Create the grid
        self.grid = Grid(width, height)
        grid_size = min(width, height)
        # self.maxNum = grid_size // 5 + 1
        # self.minNum = grid_size // 8
        if self.num_obstacles <= grid_size/2 + 1:
            self.num_obstacles = int(self.num_obstacles) // 3
        else:
            self.num_obstacles = int(grid_size**2/30)
        self.num_dynamic_obstacles = self.num_obstacles // 3 if self.dynamic_obstacles else 0

        # Set agent classes list for rendering purposes
        self.grid.agent_classes_list = self.agent_classes_list


        # Generate the surrounding walls
        self.grid.horz_wall(0, 0)
        self.grid.horz_wall(0, height - 1)
        self.grid.vert_wall(0, 0)
        self.grid.vert_wall(width - 1, 0)

        # Initialize our sets
        self.doorways = set()
        self.doorways_adjacent_cells = set()
        self.gt_target_pos_set = set()
        self.gt_target_obj_set = set()
        self.gt_hazard_pos_set = set()
        self.gt_hazard_obj_set = set()
        self.initial_agent_pos = set()
        
        # Generate rooms with random sizes using recursive subdivision
        min_room_size = 3
        rooms, self.room_doorways = self.generate_rooms(1, 1, width-2, height-2, min_room_size)

        # Select a starting room for agents (preferably a corner room)
        starting_room = self.select_starting_room(rooms)
        
        # Place base of operations on an edge wall of the starting room
        base_pos = self.place_base_of_operations(starting_room)
        
        # Place agents in the starting room near edges
        if self._agent_default_pos is not None and len(self._agent_default_pos) > 0:
            self.agent_pos = self._agent_default_pos
            for i in range(self.num_agents):
                self.grid.set(*self._agent_default_pos[i], None)
            self.agent_dir = [self._rand_int(0, 4) for i in range(self.num_agents)]  # assuming random start direction
        else:
            self.place_agents_in_room(starting_room)
        
                
        # Define target rejection function to ensure targets are not too close to base
        def target_reject_function(self, pos):
            x, y = pos
            # Reject if too close to base
            if l1distance(pos, base_pos) < 10:
                return True
            # Reject if in doorway area
            if tuple(pos) in self.doorways_adjacent_cells:
                return True
            # Reject if surrounded by walls/obstacles
            for adjacent_cell in self.adjacent_cells(x, y):
                if self.grid.get(*adjacent_cell) is None:
                    return False
                # If none of the adjacent cells are empty, reject the position
                return True
            return False
        def obstacle_reject_function(self, pos):
            """
            Function to filter out bad obstacle positions
            """
            if tuple(pos) in self.doorways_adjacent_cells: # reject near doorways
                return True
            elif tuple(pos) in self.adjacent_cells(*self.base_position): # reject near base
                return True
            # reject near initial agent positions
            for agent_pos in self.initial_agent_pos:
                if l1distance(agent_pos, pos) < 2:
                    return True
            # reject near target positions
            for target_pos in self.gt_target_pos_set:
                if l1distance(target_pos, pos) < 2:
                    return True

            # reject if placing this obstacle would create 3 obstacles in any 3x3 grid
            x, y = pos
            for dx in range(-2, 1):  # Check 3x3 grids where this position could be part of
                for dy in range(-2, 1):
                    # Define the 3x3 grid with top-left corner at (x+dx, y+dy)
                    grid_x, grid_y = x + dx, y + dy
                    obstacle_count = 0
                    
                    # Count existing obstacles in this 3x3 grid
                    for gx in range(grid_x, grid_x + 3):
                        for gy in range(grid_y, grid_y + 3):
                            for obstacle in self.static_obstacles_list + self.dynamic_obstacles_list:
                                if obstacle.cur_pos is not None and tuple(obstacle.cur_pos) == (gx, gy):
                                    obstacle_count += 1
                    
                    # Count the new obstacle we're trying to place
                    if grid_x <= x < grid_x + 3 and grid_y <= y < grid_y + 3:
                        obstacle_count += 1
                    
                    # If this would create 3 or more obstacles in a 3x3 grid, reject
                    if obstacle_count >= 3:
                        return True
            
            return False
        
        def lava_reject_function(self, pos):
            """
            Function to put lava near doorways, but not too close to base
            """
            if l1distance(base_pos, pos) < 10:
                return True
            if tuple(pos) in self.doorways_adjacent_cells:
                return False
            return True

        if self.training_stage == 1:
            self.num_targets = 1
            self.num_hazards = 1
        elif self.training_stage == 2:
            self.num_targets = 2
            self.num_hazards = 1
        elif self.training_stage == 3:
            self.num_targets = 2
            self.num_hazards = 2
        elif self.training_stage == 4: # randomize targets and hazards between 2 and 3
            self.num_targets = self._rand_int(2, 4)
            if grid_size == 32:
                self.num_targets = 3
            if self.spawn_hazards:
                self.num_hazards = self._rand_int(2, 4)
        elif self.training_stage == 10: # for eval of hazardous SAR
            if grid_size == 15:
                self.num_targets = 2
                self.num_hazards = 1
            elif grid_size == 20:
                self.num_targets = 3
                self.num_hazards = 2
            elif grid_size == 32:
                self.num_targets = 3
                self.num_hazards = 2
        elif self.training_stage == 20: # for non-hazardous SAR
            if grid_size == 15:
                self.num_targets = 2
                self.num_hazards = 0
            elif grid_size == 20:
                self.num_targets = 3
                self.num_hazards = 0
            elif grid_size == 25:
                self.num_targets = 4
                self.num_hazards = 0
            elif grid_size == 28:
                self.num_targets = 4
                self.num_hazards = 0
            elif grid_size == 32:
                self.num_targets = 4
                self.num_hazards = 0
        # Place random obstacles, but do not block doorways or spawn obstacles too close to initial agent and target positions
        self.static_obstacles_list = []
        self.dynamic_obstacles_list = []
        if self.dynamic_obstacles:
                for i in range(self.num_dynamic_obstacles):
                    dynamic_obstacle = Obstacle(dynamic=True, dir=self._rand_int(0,4))
                    init_pos = self.place_obj(dynamic_obstacle, max_tries=100, reject_fn=obstacle_reject_function)
                    dynamic_obstacle.init_pos = init_pos
                    self.dynamic_obstacles_list.append(dynamic_obstacle)
        for i in range(self.num_obstacles):
                static_obstacle = Obstacle()
                init_pos = self.place_obj(static_obstacle, max_tries=100, reject_fn=obstacle_reject_function)
                static_obstacle.init_pos = init_pos
                self.static_obstacles_list.append(static_obstacle)
        
        # Place Lava randomly
        if self.spawn_hazards:
            single_door_rooms = [room for room in rooms if len(self.room_doorways[id(room)]) == 1]
            double_door_rooms = [room for room in rooms if len(self.room_doorways[id(room)]) == 2]
            # print(f"Total rooms: {len(rooms)}, Single door rooms: {len(single_door_rooms)}, Double door rooms: {len(double_door_rooms)}")
            num_hazards_placed = 0
            # if there are no single door rooms, try to block double door rooms
            if len(single_door_rooms) == 0:
                if len(double_door_rooms) > 0:
                    random.shuffle(double_door_rooms)
                    for room in double_door_rooms:
                        # print(f"Room at x={room['x']}, y={room['y']}, w={room['width']}, h={room['height']} has two doorways.")
                        if num_hazards_placed >= self.num_hazards:
                            break
                        # Place two hazards, each adjacent to a doorway of this room
                        doorways = list(self.room_doorways[id(room)])
                        for doorway in doorways:
                            adjacent_positions = self.adjacent_cells(*doorway)
                            for pos in adjacent_positions:
                                if lava_reject_function(self, pos):
                                    continue
                                lava = Lava()
                                lava_pos = self.place_obj(lava, top=pos, size=(1, 1))
                                if lava_pos[0] != -1:
                                    self.gt_hazard_obj_set.add(lava)
                                    self.gt_hazard_pos_set.add(tuple(pos))
                                    num_hazards_placed += 1
                                    break
                            if num_hazards_placed >= self.num_hazards:
                                break
            else:
                for room in single_door_rooms:
                    # print(f"Room at x={room['x']}, y={room['y']}, w={room['width']}, h={room['height']} has a single doorway.")
                    if num_hazards_placed >= self.num_hazards:
                        break
                    # Place a hazard adjacent to the only doorway of this room
                    doorway = list(self.room_doorways[id(room)])[0]
                    adjacent_positions = self.adjacent_cells(*doorway)
                    for pos in adjacent_positions:
                        if lava_reject_function(self, pos):
                            continue
                        lava = Lava()
                        lava_pos = self.place_obj(lava, top=pos, size=(1, 1))
                        if lava_pos[0] != -1:
                            self.gt_hazard_obj_set.add(lava)
                            self.gt_hazard_pos_set.add(tuple(pos))
                            num_hazards_placed += 1
                            break

            for i in range(self.num_hazards-num_hazards_placed): # Place n-1 hazards normally
                lava = Lava()
                lava_pos = self.place_obj(lava, max_tries=100, reject_fn=lava_reject_function)
                if lava_pos[0] != -1:
                    self.gt_hazard_obj_set.add(lava)
                    self.gt_hazard_pos_set.add(tuple(lava_pos))
            # After placing hazards
            blocked_rooms = []
            hazard_positions = set(self.gt_hazard_pos_set)
            for room in rooms:
                if self.is_room_blocked_by_hazards(room, hazard_positions):
                    # print(f"Room at x={room['x']}, y={room['y']}, w={room['width']}, h={room['height']} is blocked by hazards.")
                    blocked_rooms.append(room)
            # print(f"number of targets are {self.num_targets}, number of hazards are {self.num_hazards}, number of blocked rooms are {len(blocked_rooms)}")
            if blocked_rooms:
                # Pick one blocked room
                target_room = random.choice(blocked_rooms)
                # Place a target in this room
                target = Goal()
                target_pos = self.place_obj(target, top=(target_room['x'], target_room['y']),
                                        size=(target_room['width'], target_room['height']))
                target.init_pos = target_pos
                if self.dynamic_targets:
                        target.dynamic = True
                        target.dir = self._rand_int(0, 4)
                self.gt_target_obj_set.add(target)
                if target_pos[0] != -1:
                    self.gt_target_pos_set.add(tuple(target_pos))
                for i in range(self.num_targets-1): # Place remaining targets normally
                    target = Goal()
                    target_pos = self.place_obj(target, reject_fn=target_reject_function)
                    target.init_pos = target_pos
                    if self.dynamic_targets:
                        target.dynamic = True
                        target.dir = self._rand_int(0, 4)
                    self.gt_target_obj_set.add(target)
                    if target_pos[0] != -1:
                        self.gt_target_pos_set.add(tuple(target_pos))
            else:
                # If no blocked rooms, place targets normally
                for i in range(self.num_targets):
                    target = Goal()
                    target_pos = self.place_obj(target, reject_fn=target_reject_function)
                    target.init_pos = target_pos
                    if self.dynamic_targets:
                        target.dynamic = True
                        target.dir = self._rand_int(0, 4)
                    self.gt_target_obj_set.add(target)
                    if target_pos[0] != -1:
                        self.gt_target_pos_set.add(tuple(target_pos))
        else:
            # Place targets
            if self._goal_default_pos is not None:
                target = Goal()
                self.put_obj(target, *self._goal_default_pos)
                target.init_pos, target.cur_pos = self._goal_default_pos
            else:
                for i in range(self.num_targets):
                    target = Goal()
                    target_pos = self.place_obj(target, reject_fn=target_reject_function)
                    if self.dynamic_targets:
                        target.dynamic = True
                        target.dir = self._rand_int(0, 4)
                    self.gt_target_obj_set.add(target)
                    if target_pos[0] != -1:
                        self.gt_target_pos_set.add(tuple(target_pos))
                    target.init_pos = target_pos

        if self.exploration_only:
            self.mission = 'Scouts should explore the environment asap.'
        else:
            self.mission = 'rescuer robots should reach the targets asap.'


    def generate_rooms(self, x, y, width, height, min_size):
        """Generate rooms using recursive subdivision with proper dimension checks.
        Returns (rooms, room_doorways) where room_doorways is a dict: room_id -> set of doorway positions.
        """
        rooms = []
        room_doorways = dict()
        # Base case: If the space is too small to divide further, return it as a room
        if width < min_size*2 or height < min_size*2:
            room = {"x": x, "y": y, "width": width, "height": height}
            rooms.append(room)
            room_doorways[id(room)] = set()
            return rooms, room_doorways

        split_vertical = None
        if width < min_size*2:
            split_vertical = False  # Force horizontal split
        elif height < min_size*2:
            split_vertical = True   # Force vertical split
        else:
            if width > height + 2:
                split_vertical = True
            elif height > width + 2:
                split_vertical = False
            else:
                split_vertical = self._rand_float(0, 1) < 0.5

        if (split_vertical and width < min_size*2) or (not split_vertical and height < min_size*2):
            room = {"x": x, "y": y, "width": width, "height": height}
            rooms.append(room)
            room_doorways[id(room)] = set()
            return rooms, room_doorways

        try:
            if split_vertical:
                split_pos = self._rand_int(x + min_size, x + width - min_size)
                for i in range(y, y + height):
                    self.grid.set(split_pos, i, Wall())
                left_rooms, left_doorways = self.generate_rooms(x, y, split_pos - x, height, min_size)
                right_rooms, right_doorways = self.generate_rooms(split_pos + 1, y, width - (split_pos - x + 1), height, min_size)
                valid_door_positions = []
                for i in range(y + 1, y + height - 1):
                    left_cell = self.grid.get(split_pos - 1, i)
                    right_cell = self.grid.get(split_pos + 1, i)
                    if left_cell is None and right_cell is None:
                        valid_door_positions.append(i)
                new_doors = set()
                if valid_door_positions:
                    if height > 10 and len(valid_door_positions) >= 5:
                        number_of_segments = height // 5
                        segment_length = len(valid_door_positions) // number_of_segments
                        for seg in range(number_of_segments):
                            start_idx = seg * segment_length
                            end_idx = (seg + 1) * segment_length if seg < number_of_segments - 1 else len(valid_door_positions)
                            segment = valid_door_positions[start_idx:end_idx]
                            door_y = segment[self._rand_int(0, len(segment) - 1)]
                            door_pos = (split_pos, door_y)
                            self.grid.set(*door_pos, Door(color='grey', is_open=True))
                            self.doorways.add(door_pos)
                            self.doorways_adjacent_cells = self.doorways_adjacent_cells.union(self.adjacent_cells(*door_pos))
                            new_doors.add(door_pos)
                    else:
                        door_y = valid_door_positions[self._rand_int(0, len(valid_door_positions) - 1)]
                        door_pos = (split_pos, door_y)
                        self.grid.set(*door_pos, Door(color='grey', is_open=True))
                        self.doorways.add(door_pos)
                        self.doorways_adjacent_cells = self.doorways_adjacent_cells.union(self.adjacent_cells(*door_pos))
                        new_doors.add(door_pos)
                else:
                    for i in range(y + 1, y + height - 1):
                        left_cell = self.grid.get(split_pos - 1, i)
                        right_cell = self.grid.get(split_pos + 1, i)
                        if (left_cell is None or left_cell.type == 'door') and (right_cell is None or right_cell.type == 'door'):
                            door_pos = (split_pos, i)
                            self.grid.set(*door_pos, Door(color='grey', is_open=True))
                            self.doorways.add(door_pos)
                            self.doorways_adjacent_cells = self.doorways_adjacent_cells.union(self.adjacent_cells(*door_pos))
                            new_doors.add(door_pos)
                            break
                # Assign doors to rooms
                for room in left_rooms:
                    room_doorways[id(room)] = left_doorways.get(id(room), set())
                    # Add doors on left side
                    for door in new_doors:
                        if door[0] == split_pos and room["x"] <= door[0]-1 < room["x"]+room["width"] and room["y"] <= door[1] < room["y"]+room["height"]:
                            room_doorways[id(room)].add(door)
                for room in right_rooms:
                    room_doorways[id(room)] = right_doorways.get(id(room), set())
                    # Add doors on right side
                    for door in new_doors:
                        if door[0] == split_pos and room["x"] <= door[0]+1 < room["x"]+room["width"] and room["y"] <= door[1] < room["y"]+room["height"]:
                            room_doorways[id(room)].add(door)
                rooms.extend(left_rooms)
                rooms.extend(right_rooms)
            else:
                split_pos = self._rand_int(y + min_size, y + height - min_size)
                for i in range(x, x + width):
                    self.grid.set(i, split_pos, Wall())
                top_rooms, top_doorways = self.generate_rooms(x, y, width, split_pos - y, min_size)
                bottom_rooms, bottom_doorways = self.generate_rooms(x, split_pos + 1, width, height - (split_pos - y + 1), min_size)
                valid_door_positions = []
                for i in range(x + 1, x + width - 1):
                    top_cell = self.grid.get(i, split_pos - 1)
                    bottom_cell = self.grid.get(i, split_pos + 1)
                    if top_cell is None and bottom_cell is None:
                        valid_door_positions.append(i)
                new_doors = set()
                if valid_door_positions:
                    if width > 10 and len(valid_door_positions) >= 5:
                        number_of_segments = width // 5
                        segment_length = len(valid_door_positions) // number_of_segments
                        for seg in range(number_of_segments):
                            start_idx = seg * segment_length
                            end_idx = (seg + 1) * segment_length if seg < number_of_segments - 1 else len(valid_door_positions)
                            segment = valid_door_positions[start_idx:end_idx]
                            door_x = segment[self._rand_int(0, len(segment) - 1)]
                            door_pos = (door_x, split_pos)
                            self.grid.set(*door_pos, Door(color='grey', is_open=True))
                            self.doorways.add(door_pos)
                            self.doorways_adjacent_cells = self.doorways_adjacent_cells.union(self.adjacent_cells(*door_pos))
                            new_doors.add(door_pos)
                    else:
                        door_x = valid_door_positions[self._rand_int(0, len(valid_door_positions) - 1)]
                        door_pos = (door_x, split_pos)
                        self.grid.set(*door_pos, Door(color='grey', is_open=True))
                        self.doorways.add(door_pos)
                        self.doorways_adjacent_cells = self.doorways_adjacent_cells.union(self.adjacent_cells(*door_pos))
                        new_doors.add(door_pos)
                else:
                    for i in range(x + 1, x + width - 1):
                        top_cell = self.grid.get(i, split_pos - 1)
                        bottom_cell = self.grid.get(i, split_pos + 1)
                        if (top_cell is None or top_cell.type == 'door') and (bottom_cell is None or bottom_cell.type == 'door'):
                            door_pos = (i, split_pos)
                            self.grid.set(*door_pos, Door(color='grey', is_open=True))
                            self.doorways.add(door_pos)
                            self.doorways_adjacent_cells = self.doorways_adjacent_cells.union(self.adjacent_cells(*door_pos))
                            new_doors.add(door_pos)
                            break
                # Assign doors to rooms
                for room in top_rooms:
                    room_doorways[id(room)] = top_doorways.get(id(room), set())
                    for door in new_doors:
                        if door[1] == split_pos and room["y"] <= door[1]-1 < room["y"]+room["height"] and room["x"] <= door[0] < room["x"]+room["width"]:
                            room_doorways[id(room)].add(door)
                for room in bottom_rooms:
                    room_doorways[id(room)] = bottom_doorways.get(id(room), set())
                    for door in new_doors:
                        if door[1] == split_pos and room["y"] <= door[1]+1 < room["y"]+room["height"] and room["x"] <= door[0] < room["x"]+room["width"]:
                            room_doorways[id(room)].add(door)
                rooms.extend(top_rooms)
                rooms.extend(bottom_rooms)
        except ValueError as e:
            new_room = {"x": x, "y": y, "width": width, "height": height}
            rooms.append(new_room)
            room_doorways[id(new_room)] = set()
            if len(self.doorways) > 0:
                wall_options = []
                if x > 1:
                    for j in range(y + 1, y + height - 1):
                        if self.grid.get(x-1, j) is None:
                            wall_options.append((x, j))
                if x + width < self.width - 1:
                    for j in range(y + 1, y + height - 1):
                        if self.grid.get(x+width, j) is None:
                            wall_options.append((x + width - 1, j))
                if y > 1:
                    for i in range(x + 1, x + width - 1):
                        if self.grid.get(i, y-1) is None:
                            wall_options.append((i, y))
                if y + height < self.height - 1:
                    for i in range(x + 1, x + width - 1):
                        if self.grid.get(i, y+height) is None:
                            wall_options.append((i, y + height - 1))
                if wall_options:
                    door_wall = wall_options[self._rand_int(0, len(wall_options) - 1)]
                    self.grid.set(*door_wall, Door(color='grey', is_open=True))
                    self.doorways.add(door_wall)
                    self.doorways_adjacent_cells = self.doorways_adjacent_cells.union(self.adjacent_cells(*door_wall))
                    room_doorways[id(new_room)].add(door_wall)
        return rooms, room_doorways
    
    def select_starting_room(self, rooms):
        """Select a room adjacent to outer boundaries of the grid for agents to start in"""
        # Find rooms that are adjacent to the outer walls of the grid
        starting_rooms = []
        for room in rooms:
            x, y = room["x"], room["y"]
            w, h = room["width"], room["height"]
            if (x == 1 or x + w == self.width - 1 or y == 1 or y + h == self.height - 1):
                starting_rooms.append(room)
        
        # If no suitable starting rooms found, return a random room
        if not starting_rooms:
            return rooms[self._rand_int(0, len(rooms) - 1)]
        
        # Randomly select one of the starting rooms
        return starting_rooms[self._rand_int(0, len(starting_rooms) - 1)]


    def place_base_of_operations(self, room):
        """Place a base of operations on an edge wall of the given room"""
        x, y = room["x"], room["y"]
        w, h = room["width"], room["height"]
        if x == 1:
            # Place on left wall
            base_x = x - 1
            base_y = self._rand_int(y + 1, y + h - 1)
        elif x+w == self.width-1:
            # Place on right wall
            base_x = x + w
            base_y = self._rand_int(y + 1, y + h - 1)
        elif y == 1:
            # Place on top wall
            base_x = self._rand_int(x + 1, x + w - 1)
            base_y = y - 1
        elif y+h == self.height-1:
            # Place on bottom wall
            base_x = self._rand_int(x + 1, x + w - 1)
            base_y = y + h
        # Create a base object
        self.grid.set(base_x, base_y, Base())
        self.base_position = (base_x, base_y)
        return (base_x, base_y)

    def place_agents_in_room(self, room):
        """Place all agents in the given room"""
        x, y = room["x"], room["y"]
        w, h = room["width"], room["height"]
        top_x = max(x, max(1, self.base_position[0] - 2))
        top_y = max(y, max(1, self.base_position[1] - 2))
        size_x = min(w, 3)
        size_y = min(h, 3)

        agent_positions = self.place_agent(use_same_location=self.use_same_location,
                         top=(top_x,top_y), size=(size_x, size_y))
        for agent_pos in agent_positions:
            self.initial_agent_pos.add(tuple(agent_pos))
            
        
    def move_targets(self):
        """Move all targets like agents but with brownian motion"""
        sorted_targets = sorted(list(self.gt_target_obj_set), key=lambda t: (t.init_pos[0], t.init_pos[1]))

        for target_obj in sorted_targets:

            if self.target_behavior == 'default':
                # Normal target movement chance (training value)
                chance_to_act = 0.50
                stop_distance = 2
            elif self.target_behavior == 'hyperT':
                # # Eval Mode A: Hyper-speed (always moves)
                chance_to_act = 1.0 
                stop_distance = 1
                # # Eval Mode B: Variable Speed (Simulating fatigue or bursts of energy)
                # # Use a global step counter 'self.step_count'
                # import math
                # chance_to_act = 0.5 + 0.4 * math.sin(self.step_count * 0.1)
            else:
                chance_to_act = 0.50  # default fallback
                
            rand_number = self.dynamics_rng.uniform(0, 1)
            if rand_number < chance_to_act:
                # ALWAYS burn the action RNG, even if the target decides not to use it
                if self.target_behavior == 'default':
                    action = self.dynamics_rng.choice([0, 1, 2], p=[0.2, 0.2, 0.6])
                    # Staying near Rescuers/Cleaners Mechanism   
                    is_too_close = False
                    # Check if the target is too close to any rescuer or cleaner agents
                    for agent_id in range(self.num_agents):
                        if self.agent_classes_list[agent_id] == 0 or self.agent_classes_list[agent_id] == 2:  # rescuer or Cleaner
                            if abs(target_obj.cur_pos[0] - self.agent_pos[agent_id][0]) <= stop_distance and \
                            abs(target_obj.cur_pos[1] - self.agent_pos[agent_id][1]) <= stop_distance:
                                is_too_close = True
                                break
                    if is_too_close:
                        continue

                elif self.target_behavior == 'hyperT':
                    action = self.dynamics_rng.choice([0, 1, 2], p=[0.1, 0.1, 0.8])
                    
                if action == 0: # left
                    target_obj.dir = (target_obj.dir + 3) % 4
                elif action == 1: # right
                    target_obj.dir = (target_obj.dir + 1) % 4
                elif action == 2: # forward
                    old_pos = target_obj.cur_pos
                    dir_vec = DIR_TO_VEC[target_obj.dir]
                    new_pos = (old_pos[0] + dir_vec[0], old_pos[1] + dir_vec[1])
                    # Combined check for valid position: empty cell,  within bounds, not a hazard
                    if ((self.grid.get(*new_pos) is None or self.grid.get(*new_pos).can_overlap()) and \
                        (0 <= new_pos[0] < self.width) and (0 <= new_pos[1] < self.height)) and \
                            not (new_pos in self.gt_hazard_pos_set):  # Ensure not moving into lava
                        # Position is valid - move the target
                        self.grid.set(*old_pos, None)
                        self.grid.set(*new_pos, target_obj)
                        target_obj.cur_pos = new_pos

                    elif self.target_behavior == 'hyperT':
                        # Can't move forward - randomly turn left or right
                        turn_direction = self.dynamics_rng.randint(0, 2)  # 0 for left, 1 for right
                        if turn_direction == 0:
                            target_obj.dir = (target_obj.dir + 3) % 4  # turn left
                        else:
                            target_obj.dir = (target_obj.dir + 1) % 4  # turn right
            

    def move_obstacles(self, dynamic_obstacles_list):
        """Move selected obstacles with brownian motion"""

        for obstacle in dynamic_obstacles_list:
            old_pos = obstacle.cur_pos
            new_pos = old_pos # Default to staying put
        
            # 1. OSCILLATORY BEHAVIOR (Patrolling)
            if self.obstacle_behavior == 'oscillate':
                # Store direction in obstacle object (initialized earlier)
                # 0: Up, 1: Down, 2: Left, 3: Right
                dir_vec = DIR_TO_VEC[obstacle.dir] 
                
                # Propose move
                proposed_pos = (old_pos[0] + dir_vec[0], old_pos[1] + dir_vec[1])
                
                # Check collision with walls or static obstacles (not agents yet)
                if ((self.grid.get(*proposed_pos) is None or self.grid.get(*proposed_pos).can_overlap()) and
                        (0 <= proposed_pos[0] < self.width) and (0 <= proposed_pos[1] < self.height)):
                    new_pos = proposed_pos
                else: # Collision detected, flip direction
                    obstacle.dir = (obstacle.dir + 2) % 4 # Flip direction
                    # Try moving in new direction immediately? Or wait 1 step?
                    # Let's wait 1 step to simulate a "stop and turn"
                    new_pos = old_pos

            elif self.obstacle_behavior == 'random':
                # 2. RANDOM WALK BEHAVIOR
                # moves = [(0, 1), (0, -1), (1, 0), (-1, 0), (0,0)]  # up, down, right, left, stay
                # # chance_to_move = 0.25
                # # if self._rand_float(0, 1) < chance_to_move:
                # for _ in range(10): # try 10 times to move
                #     move = moves[self._rand_int(0, len(moves))]
                # # Check collision with walls or static obstacles (not agents yet)
                #     if ((self.grid.get(*proposed_pos) is None or self.grid.get(*proposed_pos).can_overlap()) and
                #         (0 <= proposed_pos[0] < self.width) and (0 <= proposed_pos[1] < self.height)):
                #         new_pos = (old_pos[0] + move[0], old_pos[1] + move[1])
                action = self.dynamics_rng.choice([0, 1, 2], p=[0.2, 0.2, 0.6])
                if action == 0: # left
                    obstacle.dir = (obstacle.dir + 3) % 4
                elif action == 1: # right
                    obstacle.dir = (obstacle.dir + 1) % 4
                elif action == 2: # forward
                    dir_vec = DIR_TO_VEC[obstacle.dir]
                    proposed_pos = (old_pos[0] + dir_vec[0], old_pos[1] + dir_vec[1])
                    # Check collision with walls or static obstacles (not agents yet)
                    if ((self.grid.get(*proposed_pos) is None or self.grid.get(*proposed_pos).can_overlap()) and
                        (0 <= proposed_pos[0] < self.width) and (0 <= proposed_pos[1] < self.height)):
                        new_pos = proposed_pos
                        
                        
            if new_pos[0] != old_pos[0] or new_pos[1] != old_pos[1]:
                agent_collision = False
                for agent_id in range(self.num_agents):
                    if new_pos[0] == self.agent_pos[agent_id][0] and new_pos[1] == self.agent_pos[agent_id][1]:
                        agent_collision = True
                        break
                
                if not agent_collision:
                    self.grid.set(*old_pos, None)
                    self.grid.set(*new_pos, obstacle)
                    obstacle.cur_pos = new_pos


    def is_room_blocked_by_hazards(self, room, hazard_positions):
        # Check if all doorways of the room are blocked by hazards
        blocked = True
        for doorway in self.room_doorways[id(room)]:
            # Check if doorway is not adjacent to any hazard
            is_adjacent_to_hazard = any(
                l1distance(doorway, hazard_pos) == 1 for hazard_pos in hazard_positions
            )
            if not is_adjacent_to_hazard:
                blocked = False
        return blocked

    def _apply_slam_noise(self, occupied_map, explored_map):
            """
            Flip obstacle/free in explored cells with probability self.slam_noise_prob.
            occupied_map: np.array, 0=free, 1=obstacle
            explored_map: np.array, 0=unexplored, 1=explored
            """
            noisy_map = occupied_map.copy()
            mask = (explored_map > 0)
            flip = self.step_rng.rand(*occupied_map.shape) < self.slam_noise_prob 
            flip_mask = mask & flip
            noisy_map[flip_mask] = 1 - noisy_map[flip_mask]
            return noisy_map
    

    def _init_observation_state(self):
        """Reset the per-episode observation, target/hazard tracking and pose
        bookkeeping (the self.* containers consumed while building agent
        observations). Called once at the start of reset()."""
        # init local map
        self.explored_each_map = []
        self.obstacle_each_map = []
        self.previous_explored_each_map = []
        self.agent_local_views = []
        self.agent_local_obstacles = []

        self.target_found_step = np.nan
        self.found_switch = 0
        self.mission_completed = 0
        self.mission_completed_step = np.nan
        self.fully_explored = 0
        self.fully_explored_step = np.nan
        self.mission_failed = 0

        self.agent_target_dicts = []
        self.all_target_dict = {}
        self.agent_targets_reached = []
        self.all_targets_reached = {}
        self.n_targets_found = []
        self.target_rescue_times = []

        self.agent_hazard_dicts = []
        self.all_hazard_dict = {}
        self.agent_hazards_reached = []
        self.all_hazards_reached = {}
        self.n_hazards_found = []
        self.agent_cleaning_hazard = {}  # Maps agent_id to hazard object being cleaned
        self.agent_cleaning_hazard_pos = {}  # Maps agent_id to target position for cleaning
        self.agent_cleaning_progress = {}  # Maps agent_id to remaining time steps

        # APF repeat penalty.
        self.ft_goals = [None for _ in range(self.num_agents)]
        self.apf_penalty = np.zeros((
            self.num_agents,
            self.width + 2*self.agent_view_size,
            self.height + 2*self.agent_view_size
        ))

        self.agent_dir_cos_sin = np.zeros((self.num_agents, 2))
        self.agent_theta = np.zeros((self.num_agents))
        self.distance_traversed = np.zeros((self.num_agents))

    def reset(self, seed=None):
        if seed is not None:
            self.random_seed = seed
        episode_seed = self.random_seed + (self.episode_n * 1000)
        self.dynamics_rng = np.random.RandomState(seed=episode_seed + 1)
        self.comm_rng = np.random.RandomState(seed=episode_seed + 2)
        self.step_rng = np.random.RandomState(seed=episode_seed + 3)
        self.explorable_size = 0
        # Agents
        self.agent_groups = np.eye(self.num_agents)
        self.agent_classes_list = np.array(self.original_agent_classes_list)
        for index in range(self.num_agents):
            # if self.training_stage == 1: # 1 target, no hazards
            #     if self.original_agent_classes_list[index] == 2 or self.original_agent_classes_list[index] == 10 or self.original_agent_classes_list[index] == 210:
            #         new_type = self._rand_int(0, 2) # 0: rescuer, 1: scout
            #         self.agent_classes_list[index] = new_type
            # el
            if self.original_agent_classes_list[index] == 10: # replace the code 10 agent randomly with 0 or 1
                new_type = self._rand_int(0, 2)# 0: rescuer, 1: scout
                self.agent_classes_list[index] = new_type
            elif self.original_agent_classes_list[index] == 210:
                new_type = self._rand_int(0, 3) # 0: rescuer, 1: scout, 2: cleaner
                self.agent_classes_list[index] = new_type
        self.agent_count = np.zeros((self.n_agent_types)) # count of each agent type
        self.agent_alive = np.ones((self.num_agents))
        
        if self.episode_n > 1:
            self.mission_history.append(int(self.mission_completed))
            if len(self.mission_history) > 100:
                self.mission_history.pop(0)
            if len(self.mission_history) == 100:
                # print("Mission history: ", self.mission_history)
                # print("Average mission success rate: ", np.mean(self.mission_history))
                completion_rate = np.mean(self.mission_history)
                if completion_rate > 0.95 and self.training_stage < 4:
                    # self.training_stage += 1
                    # restart the mission history
                    self.mission_history = []
                    # print("Training stage increased to: ", self.training_stage)


        obs = MiniGridEnv.reset(self, seed=seed, choose=True)

        # if the target set is empty, it means that not target is not placed successfully and map should be ignored
        if self.gt_target_obj_set == set():
            self.agent_alive = np.zeros((self.num_agents))
            print("No target could be placed, ignoring this map.")
        
        self.previous_number_of_targets = len(self.gt_target_obj_set)
        # for obj in self.gt_target_obj_set:
        #     print("target object is: ", obj)
        #     print("position of target: ", obj.cur_pos)


        self.num_step = 0
        self.episode_n += 1

        self._init_observation_state()
        current_agent_pos = []
        temp_agent_target_dicts = []  # temporary dict
        temp_agent_hazard_dicts = []  # temporary dict

        for i in range(self.num_agents):
            # TODO: Make sure this is consistent with the shared odometry / coordinate frame of the robots
            self.agent_count[self.agent_classes_list[i]] += 1
            direction = self.agent_dir[i]
            if direction == 0: # Facing right
                self.agent_dir_cos_sin[i] = [1, 0]
                self.agent_theta[i] = 0
            elif direction == 1: # Facing down
                self.agent_dir_cos_sin[i] = [0, -1]
                self.agent_theta[i] = -np.pi/2
            elif direction == 2: # Facing left
                self.agent_dir_cos_sin[i] = [-1, 0]
                self.agent_theta[i] = np.pi
            elif direction == 3: # Facing up
                self.agent_dir_cos_sin[i] = [0, 1]
                self.agent_theta[i] = np.pi/2


            temp_agent_target_dicts.append({})
            self.agent_target_dicts.append({})
            self.agent_targets_reached.append({})
            self.n_targets_found.append(0)
            temp_agent_hazard_dicts.append({})
            self.agent_hazard_dicts.append({})
            self.agent_hazards_reached.append({})
            self.n_hazards_found.append(0)
            self.agent_cleaning_hazard[i] = None
            self.agent_cleaning_hazard_pos[i] = None
            self.agent_cleaning_progress[i] = 0

            self.explored_each_map.append(
                np.zeros((self.width + 2*self.agent_view_size, self.height + 2*self.agent_view_size)))
            self.obstacle_each_map.append(
                np.zeros((self.width + 2*self.agent_view_size, self.height + 2*self.agent_view_size)))
            self.previous_explored_each_map.append(
                np.zeros((self.width + 2*self.agent_view_size, self.height + 2*self.agent_view_size)))
            
            # if self.agent_classes_list[i] == 0: # rescuer agents have a smaller view size
            #     agent_view_size = self.agent_view_size
            # else:
            agent_view_size = self.agent_view_size  
            

            pos = [self.agent_pos[i][1] + self.agent_view_size,
                   self.agent_pos[i][0] + self.agent_view_size]
            current_agent_pos.append(pos)

            ### Front camera view
            local_map = np.rot90(obs[i]['image'][:, :, 0].T, 3)
            local_map = np.rot90(local_map, 4-direction) # adjusts local map's angle with the global map
            if self.agent_classes_list[i] == 0 or self.agent_classes_list[i] == 2: # rescuer and cleaner agents
                local_obstacles = (local_map == CellCode.UNSEEN) | (local_map == CellCode.WALL) | (local_map == CellCode.OBSTACLE) | (local_map == CellCode.BASE)| (local_map == CellCode.TARGET) 
            elif self.agent_classes_list[i] == 1: # scout agents
                local_obstacles = (local_map == CellCode.UNSEEN) | (local_map == CellCode.WALL) | (local_map == CellCode.OBSTACLE) | (local_map == CellCode.BASE)
            for j in range(self.num_agents): #adding seen agents to the local view
                if j == i: # skip self
                    continue
                # Calculate global relative position
                dx = self.agent_pos[j][0] - self.agent_pos[i][0]
                dy = self.agent_pos[j][1] - self.agent_pos[i][1]
                V = agent_view_size
                agent_cell = None
                
                if direction == 0: # Facing Right (+x)
                    if dx >=0 and dx < V and abs(dy) <= V//2:
                        agent_cell = [dy + V//2, dx]
                elif direction == 1: # Facing Down (+y)
                    if abs(dx) <= V//2 and dy >= 0 and abs(dy) < V:
                        agent_cell = [dy, dx + V//2]
                elif direction == 2: # Facing Left (-x)
                    if dx <= 0 and abs(dx) < V and abs(dy) <= V//2:
                        agent_cell = [dy + V//2, dx + V - 1]
                elif direction == 3: # Facing Up (-y)
                    if abs(dx) <= V//2 and dy <= 0 and abs(dy) < V:
                        agent_cell = [dy + V - 1, dx + V//2]

                if agent_cell is not None and local_map[agent_cell[0], agent_cell[1]] != 0 and \
                      local_map[agent_cell[0], agent_cell[1]] != CellCode.TARGET and \
                        local_map[agent_cell[0], agent_cell[1]] != CellCode.HAZARD: # if seen, and not on a target nor a hazard
                    if self.agent_classes_list[j] == 0: # rescuer agents
                        local_map[agent_cell[0], agent_cell[1]] = CellCode.AGENT_RESCUER
                    elif self.agent_classes_list[j] == 1: # scout agents
                        local_map[agent_cell[0], agent_cell[1]] = CellCode.AGENT_SCOUT
                    elif self.agent_classes_list[j] == 2: # cleaner agents
                        local_map[agent_cell[0], agent_cell[1]] = CellCode.AGENT_CLEANER
                    if self.use_agent_obstacle and self.agent_classes_list[i] != 1: # if not a scout agent
                        if self.agent_classes_list[j] == 0 or self.agent_classes_list[j] == 2: # rescuer or cleaner agents
                            local_obstacles[agent_cell[0], agent_cell[1]] = True
                    
            realligned_local_map = np.rot90(local_map, direction+1)
            realligned_local_obstacles = np.rot90(local_obstacles, direction+1)
            self.agent_local_views.append(realligned_local_map)
            self.agent_local_obstacles.append(realligned_local_obstacles)

            # ### 360 degrees view
            # local_map = obs[i]['image'][:, :, 0].T
            # local_obstacles = (local_map == 0) | (local_map == 40) | (local_map == 160) | (local_map == 180)
            # for j in range(self.num_agents): #adding seen agents to the local view
            #     if j != i:
            #         relative_pos = [self.agent_pos[j][0] - self.agent_pos[i][0], self.agent_pos[j][1] - self.agent_pos[i][1]]
            #         if abs(relative_pos[0]) <= self.agent_view_size//2 and abs(relative_pos[1]) <= self.agent_view_size//2:
            #             agent_cell = [relative_pos[1] + self.agent_view_size//2, relative_pos[0] + self.agent_view_size//2]
            #             if local_map[agent_cell[0], agent_cell[1]] != 0: # if not unseen
            #                 local_map[agent_cell[0], agent_cell[1]] = 220
            #                 if self.use_agent_obstacle:
            #                     local_obstacles[agent_cell[0], agent_cell[1]] = True
            # self.agent_local_views.append(local_map)
            # self.agent_local_obstacles.append(local_obstacles)
            
            


            ## Front camera view
            for x in range(agent_view_size):
                    for y in range(agent_view_size):
                        if local_map[x][y] == 0:
                            continue
                        elif direction == 0: # Facing right
                            self.explored_each_map[i][x+pos[0] -
                                                    agent_view_size//2][y+pos[1]] = 1
                            if local_map[x][y] != CellCode.EMPTY:
                                if local_map[x][y] == CellCode.TARGET: # target
                                    target_pos = [y+pos[1]-agent_view_size, x+pos[0] - 3*agent_view_size//2]
                                    target_obj = self.grid.get(*target_pos)
                                    temp_agent_target_dicts[i][target_obj] = target_pos
                                elif local_map[x][y] == CellCode.HAZARD: # hazard
                                    hazard_pos = [y+pos[1]-agent_view_size, x+pos[0] - 3*agent_view_size//2]
                                    hazard_obj = self.grid.get(*hazard_pos)
                                    temp_agent_hazard_dicts[i][hazard_obj] = hazard_pos
                                elif local_map[x][y] == CellCode.OBSTACLE or local_map[x][y] == CellCode.WALL:
                                    self.obstacle_each_map[i][x+pos[0] - agent_view_size//2][y+pos[1]] = 1
                        elif direction == 1: # Facing down
                            self.explored_each_map[i][x+pos[0]][y+pos[1] -
                                                                agent_view_size//2] = 1
                            if local_map[x][y] != CellCode.EMPTY:
                                if local_map[x][y] == CellCode.TARGET:
                                    target_pos = [y+pos[1]-3*agent_view_size//2, x+pos[0] - agent_view_size]
                                    target_obj = self.grid.get(*target_pos)
                                    temp_agent_target_dicts[i][target_obj] = target_pos
                                elif local_map[x][y] == CellCode.HAZARD:
                                    hazard_pos = [y+pos[1]-3*agent_view_size//2, x+pos[0] - agent_view_size]
                                    hazard_obj = self.grid.get(*hazard_pos)
                                    temp_agent_hazard_dicts[i][hazard_obj] = hazard_pos
                                elif local_map[x][y] == CellCode.OBSTACLE or local_map[x][y] == CellCode.WALL:
                                    self.obstacle_each_map[i][x+pos[0]][y+pos[1] - agent_view_size//2] = 1
                        elif direction == 2: # Facing left
                            self.explored_each_map[i][x+pos[0]-agent_view_size //
                                                    2][y+pos[1]-agent_view_size+1] = 1
                            if local_map[x][y] != CellCode.EMPTY:
                                if local_map[x][y] == CellCode.TARGET:
                                    target_pos = [y+pos[1]-2*agent_view_size+1, x+pos[0] - 3*agent_view_size//2]
                                    target_obj = self.grid.get(*target_pos)
                                    temp_agent_target_dicts[i][target_obj] = target_pos
                                elif local_map[x][y] == CellCode.HAZARD:
                                    hazard_pos = [y+pos[1]-2*agent_view_size+1, x+pos[0] - 3*agent_view_size//2]
                                    hazard_obj = self.grid.get(*hazard_pos)
                                    temp_agent_hazard_dicts[i][hazard_obj] = hazard_pos
                                elif local_map[x][y] == CellCode.OBSTACLE or local_map[x][y] == CellCode.WALL:
                                    self.obstacle_each_map[i][x+pos[0]-agent_view_size // 2][y+pos[1]-agent_view_size+1] = 1
                        elif direction == 3: # Facing up 
                            self.explored_each_map[i][x+pos[0]-agent_view_size +
                                                    1][y+pos[1]-agent_view_size//2] = 1
                            if local_map[x][y] != CellCode.EMPTY:
                                if local_map[x][y] == CellCode.TARGET:
                                    target_pos = [y+pos[1]-3*agent_view_size//2, x+pos[0] - 2*agent_view_size+1]
                                    target_obj = self.grid.get(*target_pos)
                                    temp_agent_target_dicts[i][target_obj] = target_pos
                                elif local_map[x][y] == CellCode.HAZARD:
                                    hazard_pos = [y+pos[1]-3*agent_view_size//2, x+pos[0] - 2*agent_view_size+1]
                                    hazard_obj = self.grid.get(*hazard_pos)
                                    temp_agent_hazard_dicts[i][hazard_obj] = hazard_pos
                                elif local_map[x][y] == CellCode.OBSTACLE or local_map[x][y] == CellCode.WALL:
                                    self.obstacle_each_map[i][x+pos[0]-agent_view_size + 1][y+pos[1]-agent_view_size//2] = 1

            # ### 360 degrees view
            # for x in range(agent_view_size):
            #     for y in range(agent_view_size):
            #         if local_map[x][y] == 0:
            #             continue
            #         else:
            #             self.explored_each_map[i][x+pos[0] - agent_view_size//2][y+pos[1] - agent_view_size//2] = 1
            #             if local_map[x][y] != 20:
            #                 if local_map[x][y] == 180:
            #                     target_pos = [y+pos[1]-3*agent_view_size//2, x+pos[0] - 3*agent_view_size//2]
            #                     temp_agent_target_sets[i].add(tuple(target_pos))
            #                 elif local_map[x][y] == 280:
            #                     pass
            #                     # self.agent_trace_sets[i].add(tuple([y+pos[1]-3*agent_view_size//2, x+pos[0] - 3*agent_view_size//2]))
            #                 elif local_map[x][y] == 300:
            #                     pass
            #                 elif local_map[x][y] == 80: #door
            #                     pass
            #                 elif local_map[x][y] == 220: #another agent
            #                     pass
            #                 else:
            #                     self.obstacle_each_map[i][x+pos[0] - agent_view_size//2][y+pos[1] - agent_view_size//2] = 1
            
            # if temp_agent_target_sets[i] != set():
            #     print("Agent ", i, " found ", len(temp_agent_target_sets[i]), " targets.")


            for j in range(i+1, self.num_agents):
                if self.agent_alive[j]:
                    d_ij = euclideandistance(self.agent_pos[i], self.agent_pos[j])
                    if d_ij < self.comm_radius:
                        comm_prob = np.exp(-d_ij**2/(self.comm_sigma**2))
                    else:
                        comm_prob = 0
                    
                    # Unconditionally sample the RNG so it advances exactly the same way in full_comm and no_comm
                    comm_success = self.comm_rng.uniform(0, 1) < comm_prob
                    
                    # Only build the adjacency matrix if we are actively using partial communication
                    if self.use_partial_comm and comm_success:
                        self.agent_groups[i,j] = 1
                        self.agent_groups[j,i] = 1
        
        explored_all_map = np.zeros((self.width + 2*self.agent_view_size,
                                    self.height + 2*self.agent_view_size))
        obstacle_all_map = np.zeros((self.width + 2*self.agent_view_size,
                                    self.height + 2*self.agent_view_size))

        # self.previous_all_map = np.zeros(
        #     (self.width + 2*self.agent_view_size, self.width + 2*self.agent_view_size))


        for i in range(self.num_agents):
            # self.each_agent_trajectory_map[i][current_agent_pos[i][0]-1:current_agent_pos[i][0]+2, 
            #                             current_agent_pos[i][1]-1:current_agent_pos[i][1]+2] = 1
            # self.each_agent_trajectory_map[i][current_agent_pos[i][0], current_agent_pos[i][1]] = 1
            explored_all_map = np.maximum( explored_all_map, self.explored_each_map[i])
            obstacle_all_map = np.maximum( obstacle_all_map, self.obstacle_each_map[i])
        
        
     
        if self.use_partial_comm:
            connected_agent_groups = get_connected_agents(self.agent_groups) # connected agents encompass agent groups
            self.obstacle_shared_map = []
            self.explored_shared_map = []
            shared_target_dicts = []
            shared_hazard_dicts = []
            counter = 0
            for group in connected_agent_groups:
                obstacle_shared_map = np.zeros((self.width + 2*self.agent_view_size, self.height + 2*self.agent_view_size))
                explored_shared_map = np.zeros((self.width + 2*self.agent_view_size, self.height + 2*self.agent_view_size))

                shared_target_dicts.append({})
                shared_hazard_dicts.append({})
                for i in group:
                    obstacle_shared_map = np.maximum(obstacle_shared_map, self.obstacle_each_map[i])
                    explored_shared_map = np.maximum(explored_shared_map, self.explored_each_map[i])
                    shared_target_dicts[counter].update(temp_agent_target_dicts[i])
                    shared_hazard_dicts[counter].update(temp_agent_hazard_dicts[i])

                self.obstacle_shared_map.append(obstacle_shared_map)
                self.explored_shared_map.append(explored_shared_map)
                counter += 1
        elif self.use_full_comm:
            connected_agent_groups = [[agent_id for agent_id in range(self.num_agents)]]
        
    
        # based on agent communications, each_maps and other info are shared and target_each_maps are formed.
        if self.use_full_comm:
            for i in range(self.num_agents):
                self.all_target_dict.update(temp_agent_target_dicts[i])
                self.all_hazard_dict.update(temp_agent_hazard_dicts[i])
                self.explored_each_map[i] = explored_all_map.copy()
                self.obstacle_each_map[i] = obstacle_all_map.copy()
                self.previous_explored_each_map[i] = self.explored_each_map[i].copy()
            
            for i in range(self.num_agents):
                self.agent_target_dicts[i] = self.all_target_dict.copy()
                self.n_targets_found[i] = len(self.all_target_dict)
                self.agent_hazard_dicts[i] = self.all_hazard_dict.copy()
                self.n_hazards_found[i] = len(self.all_hazard_dict)
                


        elif self.use_partial_comm:
            counter = 0
            for group in connected_agent_groups:
                for i in group:
                    self.agent_target_dicts[i].update(shared_target_dicts[counter])
                    self.n_targets_found[i] = len(self.agent_target_dicts[i])
                    self.agent_hazard_dicts[i].update(shared_hazard_dicts[counter])
                    self.n_hazards_found[i] = len(self.agent_hazard_dicts[i])
                
                    self.obstacle_each_map[i] = self.obstacle_shared_map[counter]
                    self.explored_each_map[i] = self.explored_shared_map[counter]
                    self.previous_explored_each_map[i] = self.explored_each_map[i] 
                counter += 1
        
        reward_explored_all_map = explored_all_map.copy()
        reward_explored_all_map[reward_explored_all_map != 0] = 1

        reward_obstacle_all_map = obstacle_all_map.copy()
        reward_obstacle_all_map[reward_obstacle_all_map != 0] = 1

        delta_reward_all_map = reward_explored_all_map - reward_obstacle_all_map

        
        self.reward_previous_all_map = delta_reward_all_map.copy()

        self.previous_explored_all_map = explored_all_map.copy()

        self.explored_map = np.array(explored_all_map).astype(int)[
            self.agent_view_size: self.width+self.agent_view_size, self.agent_view_size: self.width+self.agent_view_size]
        
        
        occupied_all_map = np.copy(obstacle_all_map)
        # for target in self.all_target_dict:
        #     target_pos = self.all_target_dict[target]
        #     occupied_all_map[target_pos[1]+self.agent_view_size, target_pos[0]+self.agent_view_size] = 1
        occupied_all_map[self.base_position[1]+self.agent_view_size, self.base_position[0]+self.agent_view_size] = 1

        occupied_each_map = np.copy(self.obstacle_each_map)
        for i in range(self.num_agents):
            if self.algorithm_name[:2] == "ft" or self.algorithm_name == "amat":
                for hazard in self.agent_hazard_dicts[i]:
                    hazard_pos = self.agent_hazard_dicts[i][hazard]
                    occupied_each_map[i][hazard_pos[1]+self.agent_view_size, hazard_pos[0]+self.agent_view_size] = 1
                for target in self.agent_target_dicts[i]:
                    target_pos = self.agent_target_dicts[i][target]
                    occupied_each_map[i][target_pos[1]+self.agent_view_size, target_pos[0]+self.agent_view_size] = 1
            elif self.agent_classes_list[i] == 0 or self.agent_classes_list[i] == 2: # rescuer and cleaner agents
                for target in self.agent_target_dicts[i]:
                    target_pos = self.agent_target_dicts[i][target]
                    occupied_each_map[i][target_pos[1]+self.agent_view_size, target_pos[0]+self.agent_view_size] = 1
            occupied_each_map[i][self.base_position[1]+self.agent_view_size, self.base_position[0]+self.agent_view_size] = 1
        
        # APF penalty
        for i in range(self.num_agents):
            x, y = current_agent_pos[i]
            self.apf_penalty[i, x, y] = 5.0  # constant
        


        self.info = {}
        if self.use_slam_noise:
            noisy_occupied_all_map = self._apply_slam_noise(
                np.array(occupied_all_map), np.array(explored_all_map)
            )
            noisy_occupied_each_map = np.array([
                self._apply_slam_noise(occupied_each_map[i], self.explored_each_map[i])
                for i in range(self.num_agents)
            ])
            # Ensure agent positions are not marked as obstacles for now.
            for i in range(self.num_agents):
                pos = self.agent_pos[i]
                # Adjust for padding if necessary
                x = pos[1] + self.agent_view_size
                y = pos[0] + self.agent_view_size
                noisy_occupied_all_map[x, y] = 0
                noisy_occupied_each_map[i][x, y] = 0
            self.info['noisy_occupied_all_map'] = noisy_occupied_all_map
            self.info['noisy_occupied_each_map'] = noisy_occupied_each_map
        self.info['agent_pos'] = np.array(self.agent_pos)
        self.info['current_agent_pos'] = np.array(current_agent_pos)
        self.info['agent_direction'] = np.array(self.agent_dir)
        self.info['agent_dir_cos_sin'] = np.array(self.agent_dir_cos_sin)
        self.info['agent_theta'] = np.array(self.agent_theta)
        self.info['distance_traversed'] = np.array(self.distance_traversed)
        self.info['explored_each_map'] = np.array(self.explored_each_map)
        self.info['explored_all_map'] = np.array(explored_all_map)
        self.info['occupied_each_map'] = np.array(occupied_each_map)
        self.info['occupied_all_map'] = np.array(occupied_all_map)
        self.info['obstacle_each_map'] = np.array(self.obstacle_each_map)
        self.info['obstacle_all_map'] = np.array(obstacle_all_map)

        self.info['agent_explored_reward'] = self.agent_exploration_reward
        self.info['auxiliary_reward'] = np.zeros((self.num_agents))
        self.info['n_targets_found'] = np.array(self.n_targets_found)
        self.info['n_hazards_found'] = np.array(self.n_hazards_found)
        self.info['agent_cleaning_hazard'] = dict(self.agent_cleaning_hazard)

        self.info['agent_groups'] = self.agent_groups
        self.info['connected_agent_groups'] = connected_agent_groups
        self.info['agent_alive'] = self.agent_alive
        self.info['agent_classes_list'] = self.agent_classes_list
        self.info['agent_count'] = self.agent_count


        self.info['agent_local_views'] = self.agent_local_views
        self.info['agent_local_obstacles'] = self.agent_local_obstacles
        self.info['agent_target_dicts'] = self.agent_target_dicts
        self.info['agent_targets_reached'] = self.agent_targets_reached
        self.info['all_target_dict'] = self.all_target_dict
        self.info['gt_target_obj_set'] = self.gt_target_obj_set
        self.info['target_rescue_times'] = self.target_rescue_times
        self.info['num_targets'] = self.num_targets

        self.info['agent_hazard_dicts'] = self.agent_hazard_dicts
        self.info['agent_hazards_reached'] = self.agent_hazards_reached
        self.info['all_hazard_dict'] = self.all_hazard_dict
        self.info['gt_hazard_obj_set'] = self.gt_hazard_obj_set

        self.info['base_position'] = self.base_position

        self.info['fully_explored'] = self.fully_explored
        self.info['fully_explored_step'] = self.fully_explored_step
        self.info['mission_completed'] = self.mission_completed
        self.info['mission_completed_step'] = self.mission_completed_step
        self.info['merge_explored_ratio'] = self.explored_ratio

        self.info['training_stage'] = self.training_stage
        # self.explored_ratio = 0
        # self.merge_reward = 0
        self.agent_exploration_reward = np.zeros((self.num_agents))
        return obs, self.info

    def step(self, action):
        self.auxiliary_reward = np.zeros((self.num_agents))
        self.milestone_reward = np.zeros((self.num_agents))
        obs, reward, done = MiniGridEnv.step(self, action)
        
        self.explored_each_map_t = []
        self.obstacle_each_map_t = []
        current_agent_pos = []
        self.num_step += 1

        reward_explored_each_map = np.zeros(
            (self.num_agents, self.width + 2*self.agent_view_size, self.height + 2*self.agent_view_size))
        step_reward_each_map = np.zeros(
            (self.num_agents, self.width + 2*self.agent_view_size, self.height + 2*self.agent_view_size))
        explored_all_map = np.zeros((self.width + 2*self.agent_view_size,
                                    self.height + 2*self.agent_view_size))
        obstacle_all_map = np.zeros((self.width + 2*self.agent_view_size,
                                    self.height + 2*self.agent_view_size))

        self.agent_groups = np.eye(self.num_agents)
        temp_agent_target_dicts = [] # temporary dict
        temp_agent_target_pos_sets = []
        temp_agent_hazard_dicts = [] # temporary dict
        temp_agent_hazard_pos_sets = []
        self.agent_local_views = []
        self.agent_local_obstacles = []

        current_number_of_targets = len(self.gt_target_obj_set)
        if current_number_of_targets < self.previous_number_of_targets:
            for _ in range(self.previous_number_of_targets - current_number_of_targets):
                self.target_rescue_times.append(self.step_count)
            self.previous_number_of_targets = current_number_of_targets
        
        
        for i in range(self.num_agents):
            direction = self.agent_dir[i]
            if direction == 0: # Facing right
                self.agent_dir_cos_sin[i] = [1, 0]
                self.agent_theta[i] = 0
            elif direction == 1: # Facing down
                self.agent_dir_cos_sin[i] = [0, -1]
                self.agent_theta[i] = -np.pi/2
            elif direction == 2: # Facing left
                self.agent_dir_cos_sin[i] = [-1, 0]
                self.agent_theta[i] = np.pi
            elif direction == 3: # Facing up
                self.agent_dir_cos_sin[i] = [0, 1]
                self.agent_theta[i] = np.pi/2

            temp_agent_target_dicts.append({})
            temp_agent_hazard_dicts.append({})
            self.explored_each_map_t.append(
                np.zeros((self.width + 2*self.agent_view_size, self.height + 2*self.agent_view_size)))
            self.obstacle_each_map_t.append(
                np.zeros((self.width + 2*self.agent_view_size, self.height + 2*self.agent_view_size)))
            
            # if self.agent_classes_list[i] == 0: # rescuer agents have a smaller view size
            #     agent_view_size = self.agent_view_size
            #     # agent_view_size = self.agent_view_size // 2
            # else:
            agent_view_size = self.agent_view_size
            
            pos = [self.agent_pos[i][1] + self.agent_view_size,
                   self.agent_pos[i][0] + self.agent_view_size]
            current_agent_pos.append(pos)

            
            ### Front camera view
            local_map = np.rot90(obs[i]['image'][:, :, 0].T, 3)
            local_map = np.rot90(local_map, 4-direction) # adjusts local map's angle with the global map

            if self.agent_classes_list[i] == 0 or self.agent_classes_list[i] == 2: # rescuer and cleaner agents
                local_obstacles = (local_map == CellCode.UNSEEN) | (local_map == CellCode.WALL) | (local_map == CellCode.OBSTACLE) | (local_map == CellCode.TARGET) | (local_map == CellCode.BASE)
            elif self.agent_classes_list[i] == 1: # scout agents
                local_obstacles = (local_map == CellCode.UNSEEN) | (local_map == CellCode.WALL) | (local_map == CellCode.OBSTACLE) | (local_map == CellCode.BASE)
            
            for j in range(self.num_agents): #adding seen agents to the local view
                if j == i: # if agent is dead, skip
                    continue

                # Calculate global relative position
                dx = self.agent_pos[j][0] - self.agent_pos[i][0]
                dy = self.agent_pos[j][1] - self.agent_pos[i][1]
                V = agent_view_size
                agent_cell = None
                
                if direction == 0: # Facing Right (+x)
                    if dx >=0 and dx < V and abs(dy) <= V//2:
                        agent_cell = [dy + V//2, dx]
                elif direction == 1: # Facing Down (+y)
                    if abs(dx) <= V//2 and dy >= 0 and abs(dy) < V:
                        agent_cell = [dy, dx + V//2]
                elif direction == 2: # Facing Left (-x)
                    if dx <= 0 and abs(dx) < V and abs(dy) <= V//2:
                        agent_cell = [dy + V//2, dx + V - 1]
                elif direction == 3: # Facing Up (-y)
                    if abs(dx) <= V//2 and dy <= 0 and abs(dy) < V:
                        agent_cell = [dy + V - 1, dx + V//2]

                if agent_cell is not None and local_map[agent_cell[0], agent_cell[1]] != 0 and \
                      local_map[agent_cell[0], agent_cell[1]] != CellCode.TARGET and \
                        local_map[agent_cell[0], agent_cell[1]] != CellCode.HAZARD: # if seen, and not on a target nor a hazard
                    if self.agent_classes_list[j] == 0: # rescuer agents
                        local_map[agent_cell[0], agent_cell[1]] = CellCode.AGENT_RESCUER
                    elif self.agent_classes_list[j] == 1: # scout agents
                        local_map[agent_cell[0], agent_cell[1]] = CellCode.AGENT_SCOUT
                    elif self.agent_classes_list[j] == 2: # cleaner agents
                        local_map[agent_cell[0], agent_cell[1]] = CellCode.AGENT_CLEANER
                    if self.use_agent_obstacle and self.agent_classes_list[i] != 1: # if not a scout agent
                        if self.agent_classes_list[j] == 0 or self.agent_classes_list[j] == 2: # rescuer or cleaner agents
                            local_obstacles[agent_cell[0], agent_cell[1]] = True

            realligned_local_map = np.rot90(local_map, direction+1)
            realligned_local_obstacles = np.rot90(local_obstacles, direction+1)

            if self.use_perception_noise:
                # There is a chance (that increases with distance to agent) for a perceived cell to be unseen
                # Idea: only make the Rescuer blind to distant cells to help specialization
                ego_agent_pos = (agent_view_size-1, agent_view_size//2)
                if self.use_perception_noise == 'rescuers_only':
                    if self.agent_classes_list[i] == 0: # rescuer agents
                        for x in range(agent_view_size):
                            for y in range(agent_view_size):
                                cell_value = realligned_local_map[x][y]
                                
                                d = euclideandistance(ego_agent_pos, (x,y))
                                prob_unseen = min(1.0, self.perception_noise_base_value + d * self.perception_noise_distance_factor)
                                
                                # ALWAYS burn the RNG to keep the sequence synchronized
                                is_unseen = self.step_rng.uniform(0, 1) < prob_unseen
                                
                                if cell_value != 0 and is_unseen:
                                    realligned_local_map[x][y] = 0
                                    realligned_local_obstacles[x][y] = True
                else:
                    for x in range(agent_view_size):
                        for y in range(agent_view_size):
                            cell_value = realligned_local_map[x][y]
                            
                            d = euclideandistance(ego_agent_pos, (x,y))
                            prob_unseen = min(1.0, self.perception_noise_base_value + d * self.perception_noise_distance_factor)
                            
                            # ALWAYS burn the RNG to keep the sequence synchronized
                            is_unseen = self.step_rng.uniform(0, 1) < prob_unseen
                            
                            if cell_value != 0 and is_unseen:
                                realligned_local_map[x][y] = 0
                                realligned_local_obstacles[x][y] = True
                local_map = np.rot90(realligned_local_map, 4 - (direction + 1))
                local_obstacles = np.rot90(realligned_local_obstacles, 4 - (direction + 1))

            self.agent_local_views.append(realligned_local_map)
            self.agent_local_obstacles.append(realligned_local_obstacles)
            
            if not self.agent_alive[i]:
                continue
            # ### 360 degrees view
            # local_map = obs[i]['image'][:, :, 0].T
            # local_obstacles = (local_map == 0) | (local_map == 40) | (local_map == 160) | (local_map == 180)
            # for j in range(self.num_agents): #adding seen agents to the local view
            #     if j != i:
            #         relative_pos = [self.agent_pos[j][0] - self.agent_pos[i][0], self.agent_pos[j][1] - self.agent_pos[i][1]]
            #         if abs(relative_pos[0]) <= self.agent_view_size//2 and abs(relative_pos[1]) <= self.agent_view_size//2:
            #             agent_cell = [relative_pos[1] + self.agent_view_size//2, relative_pos[0] + self.agent_view_size//2]
            #             if local_map[agent_cell[0], agent_cell[1]] != 0:
            #                 local_map[agent_cell[0], agent_cell[1]] = 220
            #                 if self.use_agent_obstacle:
            #                     local_obstacles[agent_cell[0], agent_cell[1]] = True
            # self.agent_local_views.append(local_map)
            # self.agent_local_obstacles.append(local_obstacles)


            

            ## Front camera view
            for x in range(agent_view_size):
                    for y in range(agent_view_size):
                        if local_map[x][y] == 0:
                            continue
                        elif direction == 0: # Facing right
                            self.explored_each_map_t[i][x+pos[0] -
                                                    agent_view_size//2][y+pos[1]] = 1
                            cell_pos = [y+pos[1]-agent_view_size, x+pos[0] - 3*agent_view_size//2]
                            if local_map[x][y] != CellCode.EMPTY:
                                if local_map[x][y] == CellCode.TARGET:
                                    target_pos = cell_pos
                                    target_obj = self.grid.get(*target_pos)
                                    temp_agent_target_dicts[i][target_obj] = target_pos
                                elif local_map[x][y] == CellCode.HAZARD:
                                    hazard_pos = cell_pos
                                    hazard_obj = self.grid.get(*hazard_pos)
                                    temp_agent_hazard_dicts[i][hazard_obj] = hazard_pos
                                elif local_map[x][y] == CellCode.OBSTACLE or local_map[x][y] == CellCode.WALL:
                                    self.obstacle_each_map_t[i][x+pos[0] - agent_view_size//2][y+pos[1]] = 1
                            else:
                                for known_target in list(self.agent_target_dicts[i].keys()):
                                    # print("temp_agent_target_dicts[i]: ", temp_agent_target_dicts[i])
                                    # print("self.agent_target_dicts[i][known_target]: ", self.agent_target_dicts[i][known_target])
                                    # print("cell_pos: ", cell_pos)
                                    if self.agent_target_dicts[i][known_target][0] == tuple(cell_pos)[0] and self.agent_target_dicts[i][known_target][1] == tuple(cell_pos)[1]:
                                        # print("Popping target: ", known_target)
                                        # print("self.agent_target_dicts[i] before popping: ", self.agent_target_dicts[i])
                                        # print("self.all_target_dict before popping: ", self.all_target_dict)
                                        self.agent_target_dicts[i].pop(known_target, None)
                                        # print("self.agent_target_dicts[i] after popping: ", self.agent_target_dicts[i])
                                        # print("self.all_target_dict after popping: ", self.all_target_dict)
                                        self.all_target_dict.pop(known_target, None)
                                    
                                
                        elif direction == 1: # Facing down
                            self.explored_each_map_t[i][x+pos[0]][y+pos[1] -
                                                                agent_view_size//2] = 1
                            cell_pos = [y+pos[1]-3*agent_view_size//2, x+pos[0] - agent_view_size]
                            if local_map[x][y] != CellCode.EMPTY:
                                if local_map[x][y] == CellCode.TARGET:
                                    target_pos = cell_pos
                                    target_obj = self.grid.get(*target_pos)
                                    temp_agent_target_dicts[i][target_obj] = target_pos
                                elif local_map[x][y] == CellCode.HAZARD:
                                    hazard_pos = cell_pos
                                    hazard_obj = self.grid.get(*hazard_pos)
                                    temp_agent_hazard_dicts[i][hazard_obj] = hazard_pos
                                elif local_map[x][y] == CellCode.OBSTACLE or local_map[x][y] == CellCode.WALL:
                                    self.obstacle_each_map_t[i][x+pos[0]][y+pos[1] - agent_view_size//2] = 1
                            else:
                                for known_target in list(self.agent_target_dicts[i].keys()):
                                    if self.agent_target_dicts[i][known_target][0] == tuple(cell_pos)[0] and self.agent_target_dicts[i][known_target][1] == tuple(cell_pos)[1]:
                                        self.agent_target_dicts[i].pop(known_target, None)
                                        self.all_target_dict.pop(known_target, None)
                        
                        elif direction == 2: # Facing left
                            self.explored_each_map_t[i][x+pos[0]-agent_view_size //
                                                    2][y+pos[1]-agent_view_size+1] = 1
                            cell_pos = [y+pos[1]-2*agent_view_size+1, x+pos[0] - 3*agent_view_size//2]
                            if local_map[x][y] != CellCode.EMPTY:
                                if local_map[x][y] == CellCode.TARGET:
                                    target_pos = cell_pos
                                    target_obj = self.grid.get(*target_pos)
                                    temp_agent_target_dicts[i][target_obj] = target_pos
                                elif local_map[x][y] == CellCode.HAZARD:
                                    hazard_pos = cell_pos
                                    hazard_obj = self.grid.get(*hazard_pos)
                                    temp_agent_hazard_dicts[i][hazard_obj] = hazard_pos
                                elif local_map[x][y] == CellCode.OBSTACLE or local_map[x][y] == CellCode.WALL:
                                    self.obstacle_each_map_t[i][x+pos[0]-agent_view_size // 2][y+pos[1]-agent_view_size+1] = 1
                            else:
                                for known_target in list(self.agent_target_dicts[i].keys()):
                                    if self.agent_target_dicts[i][known_target][0] == tuple(cell_pos)[0] and self.agent_target_dicts[i][known_target][1] == tuple(cell_pos)[1]:
                                        self.agent_target_dicts[i].pop(known_target, None)
                                        self.all_target_dict.pop(known_target, None)

                        elif direction == 3: # Facing up 
                            self.explored_each_map_t[i][x+pos[0]-agent_view_size +
                                                    1][y+pos[1]-agent_view_size//2] = 1
                            cell_pos = [y+pos[1]-3*agent_view_size//2, x+pos[0] - 2*agent_view_size+1]
                            if local_map[x][y] != CellCode.EMPTY:
                                if local_map[x][y] == CellCode.TARGET:
                                    target_pos = cell_pos
                                    target_obj = self.grid.get(*target_pos)
                                    temp_agent_target_dicts[i][target_obj] = target_pos
                                elif local_map[x][y] == CellCode.HAZARD:
                                    hazard_pos = cell_pos
                                    hazard_obj = self.grid.get(*hazard_pos)
                                    temp_agent_hazard_dicts[i][hazard_obj] = hazard_pos
                                elif local_map[x][y] == CellCode.OBSTACLE or local_map[x][y] == CellCode.WALL:
                                    self.obstacle_each_map_t[i][x+pos[0]-agent_view_size + 1][y+pos[1]-agent_view_size//2] = 1
                            else:
                                for known_target in list(self.agent_target_dicts[i].keys()):
                                    if self.agent_target_dicts[i][known_target][0] == tuple(cell_pos)[0] and self.agent_target_dicts[i][known_target][1] == tuple(cell_pos)[1]:
                                        self.agent_target_dicts[i].pop(known_target, None)
                                        self.all_target_dict.pop(known_target, None)
                                
            # ### 360 degrees view
            # for x in range(agent_view_size):
            #     for y in range(agent_view_size):
            #         if local_map[x][y] == 0:
            #             continue
            #         else:
            #             self.explored_each_map[i][x+pos[0] - agent_view_size//2][y+pos[1] - agent_view_size//2] = 1
            #             if local_map[x][y] != 20:
            #                 if local_map[x][y] == 180:
            #                     target_pos = [y+pos[1]-3*agent_view_size//2, x+pos[0] - 3*agent_view_size//2]
            #                     temp_agent_target_sets[i].add(tuple(target_pos))
            #                 elif local_map[x][y] == 280:
            #                     pass
            #                     # self.agent_trace_sets[i].add(tuple([y+pos[1]-3*agent_view_size//2, x+pos[0] - 3*agent_view_size//2]))
            #                 elif local_map[x][y] == 300:
            #                     pass
            #                 elif local_map[x][y] == 80: #door
            #                     pass
            #                 elif local_map[x][y] == 220: #another agent
            #                     pass
            #                 else:
            #                     self.obstacle_each_map[i][x+pos[0] - agent_view_size//2][y+pos[1] - agent_view_size//2] = 1
            
            # print("All target dict after potential pops: ", self.all_target_dict)

            if self.aux_w > 0.0:
                if self.agent_classes_list[i] == 1 and temp_agent_target_dicts[i] != {}: # Scout agent i is seeing a target
                    for target in temp_agent_target_dicts[i]:
                        if self.dynamic_targets > 0.0: # Tracking Reward
                            is_already_tracked = any(
                                self.agent_classes_list[j] == 1 and target in temp_agent_target_dicts[j]
                                for j in range(i)
                            )
                            if not is_already_tracked:
                                # reward[i] += 0.1  # Tracking Reward
                                # self.auxiliary_reward[i] += 0.0005  # Tracking Reward (0.005 to put the max at 0.1)
                                # self.auxiliary_reward[i] += 0.0025  # Tracking Reward (0.0025 to put the max at 0.5)
                                # self.auxiliary_reward[i] += 0.005  # Tracking Reward (0.005 to put the max at 1)
                                # self.auxiliary_reward[i] += 0.01  # Tracking Reward (0.01 to put the max at 2)
                                # self.auxiliary_reward += 0.0025  # Tracking Reward (0.0025 to put the max at 0.5)
                                self.auxiliary_reward[i] += 0.2  # Tracking Reward 
                        else: # First-time Discovery reward
                            is_already_found = any(
                                self.agent_classes_list[j] == 1 and target in temp_agent_target_dicts[j]
                                for j in range(i)
                            )
                            if (not is_already_found and target not in self.agent_target_dicts[i]):
                                self.agent_target_dicts[i][target] = temp_agent_target_dicts[i][target]
                                self.auxiliary_reward[i] += 50/self.num_targets # First-time Discovery reward
                                # print(f"agent {i} discovered new target {target}")

            for j in range(i+1, self.num_agents):
                if self.agent_alive[j]:
                    d_ij = euclideandistance(self.agent_pos[i], self.agent_pos[j])
                    if d_ij < self.comm_radius:
                        comm_prob = np.exp(-d_ij**2/(self.comm_sigma**2))
                    else:
                        comm_prob = 0
                    
                    # Unconditionally sample the RNG so it advances exactly the same way in full_comm and no_comm
                    comm_success = self.comm_rng.uniform(0, 1) < comm_prob
                    
                    # Only build the adjacency matrix if we are actively using partial communication
                    if self.use_partial_comm and comm_success:
                        self.agent_groups[i,j] = 1
                        self.agent_groups[j,i] = 1
                                    
        
        for i in range(self.num_agents):
            if not self.agent_alive[i]:
                continue
            # Update the explored and obstacle each maps
            self.explored_each_map[i] = np.maximum(
                self.explored_each_map[i], self.explored_each_map_t[i])
            self.obstacle_each_map[i] = np.maximum(
                self.obstacle_each_map[i], self.obstacle_each_map_t[i])

            
            # Update the explored and obstacle all maps
            explored_all_map = np.maximum(explored_all_map, self.explored_each_map[i])
            obstacle_all_map = np.maximum(obstacle_all_map, self.obstacle_each_map[i])

            # Reward Calculation
            reward_explored_each_map[i] = self.explored_each_map[i].copy()
            reward_explored_each_map[i][reward_explored_each_map[i] != 0] = 1

            reward_previous_explored_each_map = self.previous_explored_each_map[i].copy()
            reward_previous_explored_each_map[reward_previous_explored_each_map != 0] = 1

            step_reward_each_map[i] = np.array(reward_explored_each_map[i]) - np.array(reward_previous_explored_each_map)
            step_reward_each_map[i][step_reward_each_map[i] != 0] = 1



        
        if self.use_partial_comm:
            connected_agent_groups = get_connected_agents(self.agent_groups)
            self.explored_shared_map = []
            self.obstacle_shared_map = []
            shared_target_dicts = []
            shared_targets_reached = []
            shared_hazard_dicts = []
            shared_hazards_reached = []
            counter = 0
            for group in connected_agent_groups:
                explored_shared_map = np.zeros((self.width + 2*self.agent_view_size, self.height + 2*self.agent_view_size))
                obstacle_shared_map = np.zeros((self.width + 2*self.agent_view_size, self.height + 2*self.agent_view_size))
                shared_target_dicts.append({})
                shared_targets_reached.append({})
                shared_hazard_dicts.append({})
                shared_hazards_reached.append({})
                for i in group:
                    explored_shared_map = np.maximum(explored_shared_map, self.explored_each_map[i])
                    obstacle_shared_map = np.maximum(obstacle_shared_map, self.obstacle_each_map[i])
                    shared_target_dicts[counter].update(temp_agent_target_dicts[i])
                    shared_targets_reached[counter].update(self.agent_targets_reached[i])
                    shared_hazard_dicts[counter].update(temp_agent_hazard_dicts[i])
                    shared_hazards_reached[counter].update(self.agent_hazards_reached[i])
                self.explored_shared_map.append(explored_shared_map)
                self.obstacle_shared_map.append(obstacle_shared_map)
                counter += 1
        elif self.use_full_comm:
            connected_agent_groups = [[agent_id for agent_id in range(self.num_agents)]]
        
        # based on agent communications, each_maps and other info are shared.
        if self.use_full_comm:
            for i in range(self.num_agents):
                if not self.agent_alive[i]:
                    continue
                self.all_target_dict.update(temp_agent_target_dicts[i])
                self.all_targets_reached.update(self.agent_targets_reached[i])
                self.all_hazard_dict.update(temp_agent_hazard_dicts[i])
                self.all_hazards_reached.update(self.agent_hazards_reached[i])
                # self.all_trace_set.update
                self.explored_each_map[i] = explored_all_map.copy()
                self.obstacle_each_map[i] = obstacle_all_map.copy()
                self.previous_explored_each_map[i] = self.explored_each_map[i] 
            
            # Remove reached targets and hazards from all agent_target_dicts and all_hazard_dicts
            reached_targets_set = set(self.all_targets_reached.keys())
            reached_hazards_set = set(self.all_hazards_reached.keys())
            for i in range(self.num_agents):
                if not self.agent_alive[i]:
                    continue
                self.agent_target_dicts[i] = {k: v for k, v in self.all_target_dict.items() if k not in reached_targets_set}
                self.agent_targets_reached[i] = self.all_targets_reached.copy()
                self.n_targets_found[i] = len(self.all_target_dict)
                self.agent_hazard_dicts[i] = {k: v for k, v in self.all_hazard_dict.items() if k not in reached_hazards_set}
                self.agent_hazards_reached[i] = self.all_hazards_reached.copy()
                self.n_hazards_found[i] = len(self.all_hazard_dict)
        
        elif self.use_partial_comm:
            counter = 0
            for group in connected_agent_groups:
                # Gather all reached targets and hazards in the group
                reached_targets_set = set()
                reached_hazards_set = set()
                for i in group:
                    reached_targets_set.update(shared_targets_reached[counter].keys())
                    reached_hazards_set.update(shared_hazards_reached[counter].keys())
                for i in group:
                    self.agent_target_dicts[i].update(shared_target_dicts[counter])
                    self.agent_targets_reached[i].update(shared_targets_reached[counter])
                    self.agent_hazard_dicts[i].update(shared_hazard_dicts[counter])
                    self.agent_hazards_reached[i].update(shared_hazards_reached[counter])
                    # Remove reached targets and hazards
                    self.agent_target_dicts[i] = {k: v for k, v in self.agent_target_dicts[i].items() if k not in reached_targets_set}
                    self.n_targets_found[i] = len(self.agent_target_dicts[i])
                    self.agent_hazard_dicts[i] = {k: v for k, v in self.agent_hazard_dicts[i].items() if k not in reached_hazards_set}
                    self.n_hazards_found[i] = len(self.agent_hazard_dicts[i])
                    self.explored_each_map[i] = self.explored_shared_map[counter]
                    self.obstacle_each_map[i] = self.obstacle_shared_map[counter]
                    self.previous_explored_each_map[i] = self.explored_each_map[i]
                counter += 1

        reward_explored_all_map = explored_all_map.copy()
        reward_explored_all_map[reward_explored_all_map != 0] = 1

        reward_obstacle_all_map = obstacle_all_map.copy()
        reward_obstacle_all_map[reward_obstacle_all_map != 0] = 1

        delta_reward_all_map = reward_explored_all_map - reward_obstacle_all_map


        step_reward_all_map = delta_reward_all_map - self.reward_previous_all_map

        self.reward_previous_all_map = delta_reward_all_map.copy()
        
        self.explored_map = np.array(explored_all_map).astype(int)[
            self.agent_view_size: self.width + self.agent_view_size, self.agent_view_size: self.width + self.agent_view_size]


        occupied_all_map = np.copy(obstacle_all_map)
        # for target in self.all_target_dict:
        #     target_pos = self.all_target_dict[target]
        #     occupied_all_map[target_pos[1]+self.agent_view_size, target_pos[0]+self.agent_view_size] = 1
        occupied_all_map[self.base_position[1]+self.agent_view_size, self.base_position[0]+self.agent_view_size] = 1

        occupied_each_map = np.copy(self.obstacle_each_map)
        for i in range(self.num_agents):
            if self.algorithm_name == "amat":
                for hazard in self.agent_hazard_dicts[i]:
                    hazard_pos = self.agent_hazard_dicts[i][hazard]
                    occupied_each_map[i][hazard_pos[1]+self.agent_view_size, hazard_pos[0]+self.agent_view_size] = 1
                for target in self.agent_target_dicts[i]:
                    target_pos = self.agent_target_dicts[i][target]
                    occupied_each_map[i][target_pos[1]+self.agent_view_size, target_pos[0]+self.agent_view_size] = 1
            elif self.agent_classes_list[i] == 0 or self.agent_classes_list[i] == 2: # rescuer and cleaner agents
                for target in self.agent_target_dicts[i]:
                    target_pos = self.agent_target_dicts[i][target]
                    occupied_each_map[i][target_pos[1]+self.agent_view_size, target_pos[0]+self.agent_view_size] = 1
            occupied_each_map[i][self.base_position[1]+self.agent_view_size, self.base_position[0]+self.agent_view_size] = 1


        self.info = {}
        if self.use_slam_noise:
            noisy_occupied_all_map = self._apply_slam_noise(
                np.array(occupied_all_map), np.array(explored_all_map)
            )
            noisy_occupied_each_map = np.array([
                self._apply_slam_noise(occupied_each_map[i], self.explored_each_map[i])
                for i in range(self.num_agents)
            ])
            # Ensure agent positions are not marked as obstacles for now.
            for i in range(self.num_agents):
                pos = self.agent_pos[i]
                # Adjust for padding if necessary
                x = pos[1] + self.agent_view_size
                y = pos[0] + self.agent_view_size
                noisy_occupied_all_map[x, y] = 0
                noisy_occupied_each_map[i][x, y] = 0
            self.info['noisy_occupied_all_map'] = noisy_occupied_all_map
            self.info['noisy_occupied_each_map'] = noisy_occupied_each_map
        self.info['agent_pos'] = np.array(self.agent_pos)
        self.info['current_agent_pos'] = np.array(current_agent_pos)
        self.info['agent_direction'] = np.array(self.agent_dir)
        self.info['agent_dir_cos_sin'] = np.array(self.agent_dir_cos_sin)
        self.info['agent_theta'] = np.array(self.agent_theta)
        self.info['distance_traversed'] = np.array(self.distance_traversed)
        self.info['explored_each_map'] = np.array(self.explored_each_map)
        self.info['explored_all_map'] = np.array(explored_all_map)
        self.info['occupied_each_map'] = np.array(occupied_each_map)
        self.info['occupied_all_map'] = np.array(occupied_all_map)
        self.info['obstacle_each_map'] = np.array(self.obstacle_each_map)
        self.info['obstacle_all_map'] = np.array(obstacle_all_map)
        self.info['n_targets_found'] = self.n_targets_found
        self.info['n_hazards_found'] = self.n_hazards_found
        self.info['agent_cleaning_hazard'] = dict(self.agent_cleaning_hazard)


        self.info['agent_groups'] = self.agent_groups
        self.info['connected_agent_groups'] = connected_agent_groups
        self.info['agent_alive'] = self.agent_alive
        self.info['agent_count'] = self.agent_count
        
        self.info['agent_local_views'] = self.agent_local_views
        self.info['agent_local_obstacles'] = self.agent_local_obstacles
        self.info['agent_target_dicts'] = self.agent_target_dicts
        self.info['agent_targets_reached'] = self.agent_targets_reached
        self.info['all_target_dict'] = self.all_target_dict
        self.info['gt_target_obj_set'] = self.gt_target_obj_set
        self.info['target_rescue_times'] = self.target_rescue_times
        self.info['num_targets'] = self.num_targets


        self.info['agent_hazard_dicts'] = self.agent_hazard_dicts
        self.info['agent_hazards_reached'] = self.agent_hazards_reached
        self.info['all_hazard_dict'] = self.all_hazard_dict
        self.info['gt_hazard_obj_set'] = self.gt_hazard_obj_set

        self.info['base_position'] = self.base_position
        
        self.info['training_stage'] = self.training_stage

        


        # print("All target dict: ", self.all_target_dict)
        # print("Agent target dicts from last step: ", self.agent_target_dicts)
        
        # if self.target_found.any() and self.found_switch == 0:
        #     self.target_found_step = self.num_step
        #     self.found_switch = 1
        # self.info['target_found_step'] = self.target_found_step
        

        # Overlapping explored cells are not rewarded
        each_agent_pure_exp_rewards = np.zeros(self.num_agents) # Initialize with zeros
        for i in range(self.num_agents):
            if self.agent_classes_list[i] == 1: # scout agents
                # Calculate pure reward for scout agent i
                current_agent_reward_map = step_reward_each_map[i].copy()
                for j in range(self.num_agents):
                    if j != i and self.agent_classes_list[j] == 1: # Agent j is also a scout agent
                        current_agent_reward_map -= step_reward_each_map[j]
                each_agent_pure_exp_rewards[i] = (current_agent_reward_map > 0).sum()


        # Calculate pure reward for the team.
        team_exploration_reward = (step_reward_all_map > 0).sum()

        # exploration_reward_weight = 10/self.no_wall_size # 1 is the maximum total reward for exploration
        # exploration_reward_weight = 0.5/self.no_wall_size # 0.5 is the maximum total reward for exploration
        # exploration_reward_weight = 0.1/self.no_wall_size # 0.1 is the maximum total reward for exploration
        exploration_reward_weight = 0.5
        # self.info['agent_explored_reward'] = team_exploration_reward * exploration_reward_weight
        self.info['agent_explored_reward'] = np.array(each_agent_pure_exp_rewards) * exploration_reward_weight


        if self.aux_w > 0.0: # Exploration Reward
            # # Set a maximum value for this reward and clip it
            # np.clip(self.info['agent_explored_reward'], 0, 20, out=self.info['agent_explored_reward'])
            self.auxiliary_reward += self.info['agent_explored_reward']
        self.agent_exploration_reward = self.info['agent_explored_reward']
        # print("Agent auxiliary rewards: ", self.auxiliary_reward)
        # reward += np.expand_dims(self.auxiliary_reward, axis=1)  # Add auxiliary rewards to the main reward
        self.info['auxiliary_reward'] = self.auxiliary_reward


        self.explored_ratio = delta_reward_all_map.sum() / self.no_wall_size  # (self.width * self.height)
        self.info['merge_explored_ratio'] = self.explored_ratio
        if self.explored_ratio >= 0.99 and self.fully_explored == 0:
            self.fully_explored_step = self.num_step
            self.fully_explored = 1
            if self.exploration_only:
                self.mission_completed_step = self.num_step
                self.mission_completed = 1
                reward += 500*self._reward()
        self.info['fully_explored_step'] = self.fully_explored_step
        self.info['fully_explored'] = self.fully_explored

        if self.gt_target_obj_set == set():
            if self.mission_completed == 0:
                self.mission_completed_step = self.num_step
                self.mission_completed = 1
                # print("Mission completed at step: ", self.num_step)
                reward += 500*self._reward() # Time-decayed team reward for mission success
                # done = True
        
        self.info['mission_completed'] = self.mission_completed
        self.info['mission_completed_step'] = self.mission_completed_step
        
        # if self.num_step == 20:
        #     self.agent_alive[1] = 0
        # for i in range(self.num_agents):
        #     if self.step_rng.uniform(0, 1) < 0.01:
        #         self.agent_alive[i] = 0
        # print("Current env step is : ", self.num_step)
        if self.num_step >= self.max_steps-1 or self.agent_count[0] == 0:
            if self.mission_completed == 0 and self.mission_failed == 0:
                # reward -= 2.5 #max_steps*time_penalty
                # reward -= (self.max_steps - self.num_step)*0.01 # still getting the rest of the time penalty if rescuers all died.
                self.mission_failed = 1
                # print("Mission failed at step: ", self.num_step)
                # print("Agent count at failure: ", self.agent_count)
            # done = True
        if self.use_auxiliary_rewards:
            reward += self.aux_w * np.expand_dims(self.auxiliary_reward, axis=1)  # Add auxiliary rewards to the main reward

        return obs, reward, done, self.info

    def get_short_term_action(self, inputs):
        actions = []
        tie_breakers = [self.step_rng.uniform(0, 1) for _ in range(self.num_agents)]
        for agent_id in range(self.num_agents):
            if self.agent_alive[agent_id] == 0:
                actions.append(3) # stop action
                continue
            if self.algorithm_name == "macmat":
                ### Front camera view, where inputs are either naviagation goals or primary actions. Next move is found by planning the path in agent's local view
                # If the macro-action is a primary action
                if inputs[agent_id][2] != -1: # Actions are either turn left (0), turn right (1), or toggle/interact (4)
                    # print(f"Agent {agent_id} chose to turn")
                    if inputs[agent_id][2] == 0: # Turn left
                        actions.append(0)
                        continue
                    elif inputs[agent_id][2] == 1: # Turn right
                        actions.append(1)
                        continue
                    elif inputs[agent_id][2] == 2: # Toggle (Interact with Item)
                        actions.append(4)
                        continue
                
                # Otherwise, the macro-action is coordinates on the global map, or the stop action
                
                goal = [int(inputs[agent_id][0]), int(inputs[agent_id][1])]
                agent_pos = [self.agent_pos[agent_id][0], self.agent_pos[agent_id][1]]
                if goal[0]==agent_pos[0] and goal[1]==agent_pos[1]: # Stop action
                    actions.append(3)
                    continue
                goal = [goal[0] - agent_pos[0] , goal[1] - agent_pos[1]]
                agent_dir = self.agent_dir[agent_id]
                if agent_dir == 0: # Facing right
                    goal = [goal[1], -goal[0]]
                elif agent_dir == 1: # Facing down
                    goal = [-goal[0], -goal[1]]
                elif agent_dir == 2: # Facing left
                    goal = [-goal[1], goal[0]]
                elif agent_dir == 3: # Facing up
                    goal = [goal[0], goal[1]]
                agent_fixed_pos = np.array([self.agent_view_size//2, self.agent_view_size-1])
                goal_pos = agent_fixed_pos + np.array(goal)
                # print(f"Agent {agent_id} goal_pos: ", goal_pos)
                # print(f"Agent {agent_id} agent_fixed_pos: ", agent_fixed_pos)
                # print(f"agent {agent_id} obstacle list: ", obs_list)

                local_map = self.agent_local_views[agent_id]
                # print(f"Agent {agent_id} local map: ", local_map)
                # local_map = np.rot90(local_map, self.agent_dir[agent_id] + 3) # adjusts local map's angle with the global map
                if self.agent_classes_list[agent_id] == 0 or self.agent_classes_list[agent_id] == 2: # rescuer and cleaner agents
                    obstacle = ((local_map == CellCode.WALL) | (local_map == CellCode.OBSTACLE) | (local_map == CellCode.BASE) | (local_map == CellCode.TARGET) | (local_map == CellCode.AGENT_RESCUER) | (local_map == CellCode.AGENT_CLEANER)).astype(np.int32)
                elif self.agent_classes_list[agent_id] == 1: # scout agents
                    obstacle = ((local_map == CellCode.WALL) | (local_map == CellCode.OBSTACLE) | (local_map == CellCode.BASE)).astype(np.int32)


                # print(f"Agent {agent_id} obstacle: ", obstacle)
                obs_list = []
                for x in range(-self.agent_view_size, 3*self.agent_view_size//2 + 1):
                    # enclosing the local map for the path planner
                    obs_list.append((x,-self.agent_view_size))
                    obs_list.append((x,self.agent_view_size))
                    if x>=0 and x < self.agent_view_size:
                        for y in range(0, self.agent_view_size):
                            if obstacle[x, y] == True:
                                obs_list.append((y, x))
                for y in range(-self.agent_view_size, self.agent_view_size+1):
                    obs_list.append((-self.agent_view_size,y))
                    obs_list.append((3*self.agent_view_size//2,y))
                path_planner = astar.AStar(obs_list, tuple(agent_fixed_pos), tuple(goal_pos), "manhattan")
                path, _ = path_planner.searching()
                path = path[::-1]
                # print(f"Agent {agent_id} path: ", path)
                if len(path) == 2 and path[0] == path[1]:
                    # if inputs[agent_id][2] == 1:
                    #     actions.append(4) # interact
                    # else:
                    actions.append(3) # stop, goal is reached
                    continue
                if len(path) == 1:
                    actions.append(3) # Goal is unreachable, stop
                    continue

                relative_pos = np.array(path[1]) - agent_fixed_pos

                if relative_pos[1] == 0:
                    if relative_pos[0] == 1:
                        actions.append(1) # turn right
                        continue
                    elif relative_pos[0] == -1:
                        actions.append(0) # turn left
                        continue
                elif relative_pos[0] == 0:
                    if relative_pos[1] == 1:
                        if tie_breakers[agent_id] < 0.5: # randomly turn left or right
                            actions.append(1)
                        else:
                            actions.append(0)
                        continue 
                    elif relative_pos[1] == -1:
                        actions.append(2) # forward
                        continue
            
            elif self.algorithm_name == "amat":
                # if self.use_full_comm:
                #     # explored = (self.info['explored_all_map'] > 0).astype(np.int32)[
                #     #         self.agent_view_size:self.agent_view_size+self.width, self.agent_view_size:self.agent_view_size+self.height]
                #     if self.use_slam_noise:
                #         obstacle = (self.info['noisy_occupied_all_map'] > 0).astype(np.int32)[
                #             self.agent_view_size:self.agent_view_size+self.width, self.agent_view_size:self.agent_view_size+self.height]
                #     else:
                #         obstacle = (self.info['occupied_all_map'] > 0).astype(np.int32)[
                #             self.agent_view_size:self.agent_view_size+self.width, self.agent_view_size:self.agent_view_size+self.height]
                # elif self.use_partial_comm:
                #     # explored = (self.info['explored_each_map'][agent_id] > 0).astype(np.int32)[
                #     #         self.agent_view_size:self.agent_view_size+self.width, self.agent_view_size:self.agent_view_size+self.height]
                #     if self.use_slam_noise:
                #         obstacle = (self.info['noisy_occupied_each_map'][agent_id] > 0).astype(np.int32)[
                #             self.agent_view_size:self.agent_view_size+self.width, self.agent_view_size:self.agent_view_size+self.height]
                #     else:
                #         obstacle = (self.info['occupied_each_map'][agent_id] > 0).astype(np.int32)[
                #             self.agent_view_size:self.agent_view_size+self.width, self.agent_view_size:self.agent_view_size+self.height]
                # else:
                #     raise NotImplementedError
                if self.use_slam_noise:
                    obstacle = (self.info['noisy_occupied_each_map'][agent_id] > 0).astype(np.int32)[
                        self.agent_view_size:self.agent_view_size+self.width, self.agent_view_size:self.agent_view_size+self.height]
                else:
                    obstacle = (self.info['occupied_each_map'][agent_id] > 0).astype(np.int32)[
                        self.agent_view_size:self.agent_view_size+self.width, self.agent_view_size:self.agent_view_size+self.height]
                
                if self.use_agent_obstacle:
                    if self.agent_classes_list[agent_id] != 1: # Scouts do not see other agents as obstacles
                        for a in range(self.num_agents):
                            if (a != agent_id and self.agent_classes_list[a] != 1): # Scouts are not seen as obstacles by other agents
                                obstacle[self.agent_pos[a][1], self.agent_pos[a][0]] = 1
                goal = [int(inputs[agent_id][1]), int(inputs[agent_id][0])]

                agent_pos = [self.agent_pos[agent_id][1], self.agent_pos[agent_id][0]]
                agent_dir = self.agent_dir[agent_id]
                
                obs_list = []
                for x in range(self.width):
                    #enclosing the map for the path planner
                    obs_list.append((x,-1))
                    obs_list.append((x,self.height))
                    for y in range(self.height):
                        if obstacle[x, y] == 1:
                            obs_list.append((x, y))
                for y in range(self.height):
                    obs_list.append((-1,y))
                    obs_list.append((self.width,y))

                
                # path_planner = dstarlite.DStar(obs_list, tuple(agent_pos), tuple(goal), "manhattan")
                # path_planner.ComputePath()
                # path = path_planner.extract_path()
                path_planner = astar.AStar(obs_list, tuple(agent_pos), tuple(goal), "manhattan")
                path, _ = path_planner.searching()
                
                path = path[::-1]
                
                # path_cost = path_planner.cost(tuple(agent_pos), tuple(goal))
                
                if len(path) == 2 and path[0] == path[1]:
                    # if inputs[agent_id][2] == 1:
                    #     actions.append(4) # interact
                    # else:
                    actions.append(3) # stop, goal is reached
                    continue
                if len(path) == 1:
                    
                    actions.append(4) #goal is unreachable
                    continue
                relative_pos = np.array(path[1]) - np.array(agent_pos)
                
                
                # first quadrant
                if relative_pos[0] < 0 and relative_pos[1] > 0:
                    if agent_dir == 0 or agent_dir == 3:
                        actions.append(2)  # forward
                        continue
                    if agent_dir == 1:
                        actions.append(0)  # turn left
                        continue
                    if agent_dir == 2:
                        actions.append(1)  # turn right
                        continue
                # second quadrant
                if relative_pos[0] > 0 and relative_pos[1] > 0:
                    if agent_dir == 0 or agent_dir == 1:
                        actions.append(2)  # forward
                        continue
                    if agent_dir == 2:
                        actions.append(0)  # turn left
                        continue
                    if agent_dir == 3:
                        actions.append(1)  # turn right
                        continue
                # third quadrant
                if relative_pos[0] > 0 and relative_pos[1] < 0:
                    if agent_dir == 1 or agent_dir == 2:
                        actions.append(2)  # forward
                        continue
                    if agent_dir == 3:
                        actions.append(0)  # turn left
                        continue
                    if agent_dir == 0:
                        actions.append(1)  # turn right
                        continue
                # fourth quadrant
                if relative_pos[0] < 0 and relative_pos[1] < 0:
                    if agent_dir == 2 or agent_dir == 3:
                        actions.append(2)  # forward
                        continue
                    if agent_dir == 0:
                        actions.append(0)  # turn left
                        continue
                    if agent_dir == 1:
                        actions.append(1)  # turn right
                        continue
                if relative_pos[0] == 0 and relative_pos[1] == 0:
                    # turn around
                    actions.append(1)
                    continue
                if relative_pos[0] == 0 and relative_pos[1] > 0:
                    if agent_dir == 0:
                        actions.append(2)
                        continue
                    if agent_dir == 1:
                        actions.append(0)
                        continue
                    if agent_dir == 2:
                        if self.step_rng.uniform(0, 1) < 0.5:
                            actions.append(1)
                        else:
                            actions.append(0)
                    else:
                        actions.append(1)
                        continue
                if relative_pos[0] == 0 and relative_pos[1] < 0:
                    if agent_dir == 2:
                        actions.append(2)
                        continue
                    if agent_dir == 1:
                        actions.append(1)
                        continue
                    if agent_dir == 0:
                        if self.step_rng.uniform(0, 1) < 0.5:
                            actions.append(1)
                        else:
                            actions.append(0)
                    else:
                        actions.append(0)
                        continue
                if relative_pos[0] > 0 and relative_pos[1] == 0:
                    if agent_dir == 1:
                        actions.append(2)
                        continue
                    if agent_dir == 0:
                        actions.append(1)
                        continue
                    if agent_dir == 3:
                        if self.step_rng.uniform(0, 1) < 0.5:
                            actions.append(1)
                        else:
                            actions.append(0)
                    else:
                        actions.append(0)
                        continue
                if relative_pos[0] < 0 and relative_pos[1] == 0:
                    if agent_dir == 3:
                        actions.append(2)
                        continue
                    if agent_dir == 0:
                        actions.append(0)
                        continue
                    if agent_dir == 1:
                        if self.step_rng.uniform(0, 1) < 0.5:
                            actions.append(1)
                        else:
                            actions.append(0)
                    else:
                        actions.append(1)
                        continue

            
        return actions
    
    def ft_get_short_term_goals(self, args, mode=""):
        '''
        frontier-based methods compute actions
        '''
        # self.info = self.ft_info
        replan = [False for _ in range(self.num_agents)]
        if self.use_full_comm:
            current_agent_pos = self.info["current_agent_pos"]
            location_lists = []
            for agent_id in range(self.num_agents):
                location_lists.append(current_agent_pos)
        elif self.use_partial_comm: #only the position of connected agents are known to each agent
            current_agent_pos = self.info["current_agent_pos"]
            connected_agent_groups = get_connected_agents(self.agent_groups) # connected agents encompass agent groups
            location_lists = []
            for agent_id in range(self.num_agents):
                location_list = []
                for group in connected_agent_groups:
                    if agent_id in group:
                        for agent in group:
                            location_list.append(current_agent_pos[agent])
                location_lists.append(location_list)

        goals = [None for _ in range(self.num_agents)]
        for agent_id in range(self.num_agents):
            explored = (self.info['explored_each_map'][agent_id] > 0).astype(np.int32)
            if self.use_slam_noise:
                obstacle = (self.info['noisy_occupied_each_map'][agent_id] > 0).astype(np.int32)
                obstacle_reduced = obstacle[self.agent_view_size:self.agent_view_size+self.width, self.agent_view_size:self.agent_view_size+self.height]
            else:
                obstacle = (self.info['occupied_each_map'][agent_id] > 0).astype(np.int32)
                obstacle_reduced = obstacle[self.agent_view_size:self.agent_view_size+self.width, self.agent_view_size:self.agent_view_size+self.height]
            if self.use_full_comm:
                # explored = (self.info['explored_all_map'] > 0).astype(np.int32)
                # if self.use_slam_noise:
                #     obstacle = (self.info['noisy_occupied_all_map'] > 0).astype(np.int32)
                #     obstacle_reduced = obstacle[self.agent_view_size:self.agent_view_size+self.width, self.agent_view_size:self.agent_view_size+self.height]
                # else:
                #     obstacle = (self.info['occupied_all_map'] > 0).astype(np.int32)
                #     obstacle_reduced = obstacle[self.agent_view_size:self.agent_view_size+self.width, self.agent_view_size:self.agent_view_size+self.height]
                if self.use_agent_obstacle and self.agent_classes_list[agent_id] != 1: # scouts don't see others as obstacles
                    if mode != 'voronoi': # make an exception for voronoi goal
                        for a in range(self.num_agents):
                            if a != agent_id and self.agent_classes_list[a] != 1: # scout agents are not obstacles
                                obstacle[current_agent_pos[a][0], current_agent_pos[a][1]] = 1
                                obstacle_reduced[current_agent_pos[a][0]-self.agent_view_size, current_agent_pos[a][1]-self.agent_view_size] = 1
            elif self.use_partial_comm:
                # explored = (self.info['explored_each_map'][agent_id] > 0).astype(np.int32)
                # if self.use_slam_noise:
                #     obstacle = (self.info['noisy_occupied_each_map'][agent_id] > 0).astype(np.int32)
                #     obstacle_reduced = obstacle[self.agent_view_size:self.agent_view_size+self.width, self.agent_view_size:self.agent_view_size+self.height]
                # else:
                #     obstacle = (self.info['occupied_each_map'][agent_id] > 0).astype(np.int32)
                #     obstacle_reduced = obstacle[self.agent_view_size:self.agent_view_size+self.width, self.agent_view_size:self.agent_view_size+self.height]
                if self.use_agent_obstacle and self.agent_classes_list[agent_id] != 1: # scouts don't see others as obstacles
                    if mode != 'voronoi': # make an exception for voronoi goal
                        for group in connected_agent_groups:
                            if agent_id in group:
                                for a in group:
                                    if a != agent_id and self.agent_classes_list[a] != 1: # scout agents are not obstacles
                                        obstacle[current_agent_pos[a][0], current_agent_pos[a][1]] = 1
                                        obstacle_reduced[current_agent_pos[a][0]-self.agent_view_size, current_agent_pos[a][1]-self.agent_view_size] = 1
            else:
                raise NotImplementedError
            H, W = explored.shape
            steps = [(-1, 0), (1, 0), (0, -1), (0, 1)]
            map = np.ones((H, W)).astype(np.int32) * 3  # 3 for unknown area
            map[explored == 1] = 0  # 0 for explored area
            map[obstacle == 1] = 1  # 1 for obstacles
            # print(f"Agent {agent_id} map is: \n{map}")
            # Set frontiers.
            for x in range(H):
                for y in range(W):
                    if map[x, y] == 0:
                        neighbors = [(x+dx, y+dy) for dx, dy in steps]
                        if sum([(map[u, v] == 3) for u, v in neighbors]) > 0:
                            map[x, y] = 2  # 2 for targets (frontiers)
            map[:self.agent_view_size, :] = 1
            map[H-self.agent_view_size:, :] = 1
            map[:, :self.agent_view_size] = 1
            map[:, W-self.agent_view_size:] = 1
            unexplored = (map == 3).astype(np.int32)
            map[map == 3] = 0  # set unknown area to explorable
            # print(f"map is {map}")
            if self.num_step >= 1:
                # print("ft goals for agent ", agent_id, " is: ", self.ft_goals[agent_id])
                # print("map value for agent ", agent_id, " is: ", map[self.ft_goals[agent_id][0], self.ft_goals[agent_id][1]])
                # print("current agent pos: ", current_agent_pos[agent_id])
                # print("unexplored value for agent ", agent_id, " is: ", unexplored[self.ft_goals[agent_id][0], self.ft_goals[agent_id][1]])
                if self.ft_goals is not None and self.ft_goals[agent_id] is not None and (map[self.ft_goals[agent_id][0], self.ft_goals[agent_id][1]] != 2) and\
                (unexplored[self.ft_goals[agent_id][0], self.ft_goals[agent_id][1]] == 0):
                    replan[agent_id] = True
                if self.ft_goals is not None and self.ft_goals[agent_id] is not None and current_agent_pos[agent_id][0] == self.ft_goals[agent_id][0] and current_agent_pos[agent_id][1] == self.ft_goals[agent_id][1]:
                    replan[agent_id] = True
                    map[self.ft_goals[agent_id][0], self.ft_goals[agent_id][1]] = 0
                    # print(f"replanning is triggered for agent {agent_id} at step {self.num_step}, ft_goal: {self.ft_goals[agent_id]}, current_agent_pos: {current_agent_pos[agent_id]}")
                    
            

            if self.agent_classes_list[agent_id] == 0 and len(self.agent_target_dicts[agent_id]) > 0:
                
                # 1. Sort all known targets by distance to the agent
                targets_sorted = []
                for target_id, pos in self.agent_target_dicts[agent_id].items():
                    dist = l1distance(self.agent_pos[agent_id], pos)
                    targets_sorted.append((dist, pos))
                targets_sorted.sort(key=lambda x: x[0]) # Sort by distance ascending
                
                agent_pos = self.agent_pos[agent_id]
                
                # Setup obstacles for A*
                obs_list = []
                for x in range(self.width):
                    obs_list.append((x,-1))
                    obs_list.append((x,self.height))
                    for y in range(self.height):
                        if obstacle_reduced[x, y] == 1:
                            obs_list.append((y, x))
                for y in range(self.height):
                    obs_list.append((-1,y))
                    obs_list.append((self.width,y))

                goal_found = False
                
                # 2. Iterate through targets until we find one that is reachable
                for _, target_pos in targets_sorted:
                    target_neighbors = [(target_pos[0]+dx, target_pos[1]+dy) for dx, dy in steps]
                    shortest_path = float('inf')
                    
                    for goal in target_neighbors:
                        if obstacle_reduced[goal[1], goal[0]] == 1:
                            continue
                        path_planner = astar.AStar(obs_list, tuple(agent_pos), tuple(goal), "manhattan")
                        path, _ = path_planner.searching()
                        
                        if path is None:
                            continue
                            
                        if len(path) != 1 and len(path) < shortest_path:
                            shortest_path = len(path)
                            goals[agent_id] = [goal[1]+self.agent_view_size, goal[0]+self.agent_view_size]
                    
                    if shortest_path != float('inf'):
                        # We successfully found a path to this target! Stop searching.
                        goal_found = True
                        break 
                
                # 3. Fallback: If no targets are reachable, explore frontiers instead of freezing
                if not goal_found:
                    goals[agent_id] = utility_goal(map, unexplored, current_agent_pos[agent_id], steps)
            elif self.agent_classes_list[agent_id] == 2 and len(self.agent_hazard_dicts[agent_id]) > 0:
                # Implement the same logic for cleaner agents to clean hazards
                pass
            
            elif replan[agent_id] or self.ft_goals[agent_id] is None:
                # print(f"Agent {agent_id} replanning given current pos {current_agent_pos[agent_id]} and map as \n {map}")
                if mode == 'apf':
                    apf = APF(args)
                    path = apf.schedule(map, location_lists[agent_id], steps,
                                        current_agent_pos[agent_id], self.apf_penalty[agent_id])
                    goal = path[-1]
                if mode == 'utility':
                    goal = utility_goal(map, unexplored, current_agent_pos[agent_id], steps)
                    
                    is_exploring = True
                    
                    # 1. Completion Threshold: If < 2% of the free space is unexplored, stop chasing wall glitches
                    total_free_space = (map != 1).sum()
                    if unexplored.sum() < (total_free_space * 0.02):
                        is_exploring = False
                        
                    # 2. Reached but stuck: If the goal is exactly where we are, we can't clear the shadow
                    elif goal[0] == current_agent_pos[agent_id][0] and goal[1] == current_agent_pos[agent_id][1]:
                        is_exploring = False
                        
                    # 3. Reachability Check: Can A* actually path to this frontier?
                    else:
                        agent_pos_unpadded = self.agent_pos[agent_id]
                        # Convert padded map coordinates back to unpadded A* coordinates
                        goal_unpadded = (goal[1] - self.agent_view_size, goal[0] - self.agent_view_size)
                        
                        obs_list = []
                        for x in range(self.width):
                            obs_list.append((x,-1))
                            obs_list.append((x,self.height))
                            for y in range(self.height):
                                if obstacle_reduced[x, y] == 1:
                                    obs_list.append((y, x))
                        for y in range(self.height):
                            obs_list.append((-1,y))
                            obs_list.append((self.width,y))
                            
                        path_planner = astar.AStar(obs_list, tuple(agent_pos_unpadded), tuple(goal_unpadded), "manhattan")
                        path, _ = path_planner.searching()
                        path = path[::-1]
                        if len(path) == 1:
                            is_exploring = False

                    # If exploration is done or impossible, cascade to Relay or Random Search
                    if not is_exploring:
                        if len(self.agent_target_dicts[agent_id]) > 0:
                            # Escort the nearest known target to provide vision/comm relay
                            nearest_dist = float('inf')
                            best_target = None
                            for tgt in self.agent_target_dicts[agent_id].values():
                                dist = l1distance(self.agent_pos[agent_id], tgt)
                                if dist < nearest_dist:
                                    nearest_dist = dist
                                    best_target = tgt
                            
                            if best_target is not None:
                                goal = (best_target[1] + self.agent_view_size, best_target[0] + self.agent_view_size)
                        else:
                            # FINAL FALLBACK: Randomly sample an explored, obstacle-free cell
                            free_cells = np.argwhere(map == 0) 
                            if len(free_cells) > 0:
                                # Pick a random index from the free cells
                                random_idx = np.random.choice(len(free_cells))
                                goal = (free_cells[random_idx][0], free_cells[random_idx][1])
                            else:
                                # Absolute last resort (should virtually never happen)
                                goal = current_agent_pos[agent_id]
                elif mode == 'nearest':
                    goal = nearest_goal(map, current_agent_pos[agent_id], steps)
                elif mode == 'rrt':
                    goal = rrt_goal(map, unexplored, current_agent_pos[agent_id])
                elif mode == 'voronoi':
                    goal = voronoi_goal(map, unexplored, current_agent_pos, agent_id, steps)
                goals[agent_id] = goal
                # print(f"Agent {agent_id} at step {self.num_step} replanned goal: {goal}, ft_goal: {self.ft_goals[agent_id]}")
            else:
                goals[agent_id] = self.ft_goals[agent_id]
        
        self.ft_goals = goals.copy()
        # print("goals are: ", goals)
        return goals

    def ft_get_short_term_actions(self, inputs):
        actions = []
        for agent_id in range(self.num_agents):
            if self.use_slam_noise:
                obstacle = (self.info['noisy_occupied_each_map'][agent_id] > 0).astype(np.int32)[
                    self.agent_view_size:self.agent_view_size+self.width, self.agent_view_size:self.agent_view_size+self.height]
            else:
                obstacle = (self.info['occupied_each_map'][agent_id] > 0).astype(np.int32)[
                    self.agent_view_size:self.agent_view_size+self.width, self.agent_view_size:self.agent_view_size+self.height]
            
            if self.use_agent_obstacle and agent_id != 1: # if we consider agents as obstacles, scouts don't see others as obstacles
                for a in range(self.num_agents):
                    if a != agent_id and self.agent_classes_list[a] != 1: # scout agents are not obstacles
                        obstacle[self.agent_pos[a][1], self.agent_pos[a][0]] = 1
            goal = [int(inputs[agent_id][1]), int(inputs[agent_id][0])]

            agent_pos = [self.agent_pos[agent_id][1], self.agent_pos[agent_id][0]]
            agent_dir = self.agent_dir[agent_id]
            
            obs_list = []
            for x in range(self.width):
                #enclosing the map for the path planner
                obs_list.append((x,-1))
                obs_list.append((x,self.height))
                for y in range(self.height):
                    if obstacle[x, y] == 1:
                        obs_list.append((x, y))
            for y in range(self.height):
                obs_list.append((-1,y))
                obs_list.append((self.width,y))

            path_planner = astar.AStar(obs_list, tuple(agent_pos), tuple(goal), "manhattan")
            path, _ = path_planner.searching()
            
            path = path[::-1]
            # print(f"Agent {agent_id} obstacle map: ", obstacle)
            # print(f"Agent {agent_id} pos: ", agent_pos)
            # print(f"Agent {agent_id} goal: ", goal)
            # print(f"Agent {agent_id} path: ", path)


            
            if len(path) == 2 and path[0] == path[1]:
                actions.append(3) # stop, goal is reached
                continue
            if len(path) == 1:
                actions.append(3) #goal is unreachable
                continue
            relative_pos = np.array(path[1]) - np.array(agent_pos)
            
            
            # first quadrant
            if relative_pos[0] < 0 and relative_pos[1] > 0:
                if agent_dir == 0 or agent_dir == 3:
                    actions.append(2)  # forward
                    continue
                if agent_dir == 1:
                    actions.append(0)  # turn left
                    continue
                if agent_dir == 2:
                    actions.append(1)  # turn right
                    continue
            # second quadrant
            if relative_pos[0] > 0 and relative_pos[1] > 0:
                if agent_dir == 0 or agent_dir == 1:
                    actions.append(2)  # forward
                    continue
                if agent_dir == 2:
                    actions.append(0)  # turn left
                    continue
                if agent_dir == 3:
                    actions.append(1)  # turn right
                    continue
            # third quadrant
            if relative_pos[0] > 0 and relative_pos[1] < 0:
                if agent_dir == 1 or agent_dir == 2:
                    actions.append(2)  # forward
                    continue
                if agent_dir == 3:
                    actions.append(0)  # turn left
                    continue
                if agent_dir == 0:
                    actions.append(1)  # turn right
                    continue
            # fourth quadrant
            if relative_pos[0] < 0 and relative_pos[1] < 0:
                if agent_dir == 2 or agent_dir == 3:
                    actions.append(2)  # forward
                    continue
                if agent_dir == 0:
                    actions.append(0)  # turn left
                    continue
                if agent_dir == 1:
                    actions.append(1)  # turn right
                    continue
            if relative_pos[0] == 0 and relative_pos[1] == 0:
                # turn around
                actions.append(1)
                continue
            if relative_pos[0] == 0 and relative_pos[1] > 0:
                if agent_dir == 0:
                    actions.append(2)
                    continue
                if agent_dir == 1:
                    actions.append(0)
                    continue
                if agent_dir == 2:
                    if self.step_rng.uniform(0, 1) < 0.5:
                        actions.append(1)
                    else:
                        actions.append(0)
                else:
                    actions.append(1)
                    continue
            if relative_pos[0] == 0 and relative_pos[1] < 0:
                if agent_dir == 2:
                    actions.append(2)
                    continue
                if agent_dir == 1:
                    actions.append(1)
                    continue
                if agent_dir == 0:
                    if self.step_rng.uniform(0, 1) < 0.5:
                        actions.append(1)
                    else:
                        actions.append(0)
                else:
                    actions.append(0)
                    continue
            if relative_pos[0] > 0 and relative_pos[1] == 0:
                if agent_dir == 1:
                    actions.append(2)
                    continue
                if agent_dir == 0:
                    actions.append(1)
                    continue
                if agent_dir == 3:
                    if self.step_rng.uniform(0, 1) < 0.5:
                        actions.append(1)
                    else:
                        actions.append(0)
                else:
                    actions.append(0)
                    continue
            if relative_pos[0] < 0 and relative_pos[1] == 0:
                if agent_dir == 3:
                    actions.append(2)
                    continue
                if agent_dir == 0:
                    actions.append(0)
                    continue
                if agent_dir == 1:
                    if self.step_rng.uniform(0, 1) < 0.5:
                        actions.append(1)
                    else:
                        actions.append(0)
                else:
                    actions.append(1)
                    continue
    
        # self.paths = paths
        # print("agent 1s paths is ", paths[0])
        return actions
    
    def _get_auxiliary_weight(self):
        """Get the current auxiliary reward weight based on decay settings"""
        if self.use_auxiliary_rewards_decay:
            if self.episode_n < 2000:
                return 1 - (self.episode_n - 1) / self.max_auxiliary_reward_episode
            else:
                return 0.0
        else:
            return 1.0