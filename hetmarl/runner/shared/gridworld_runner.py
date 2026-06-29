import json
import time, datetime
import os
import copy
import numpy as np
from itertools import chain
import torch
import imageio
import matplotlib.pyplot as plt
import cv2
from collections import defaultdict, deque
from hetmarl.utils.util import update_linear_schedule, get_shape_from_act_space, get_shape_from_obs_space, AsynchControl
from hetmarl.runner.shared.base_runner import Runner
from hetmarl.envs.gridworld.semantics import CellCode
import torch.nn as nn

import hydra

def _t2n(x): #tensor to numpy array

    return x.detach().cpu().numpy()


class GridWorldRunner(Runner):
    def __init__(self, config):
        super(GridWorldRunner, self).__init__(config)
        self.init_hyperparameters()
        self.init_map_variables()
        self.init_keys()
    
    def get_available_actions(self):
        available_actions = np.ones((self.n_rollout_threads, self.num_agents, self.act_dim), dtype=np.int32)
        for e in range(self.n_rollout_threads):
            for a in range(self.num_agents):
                                #TODO: If action_size is not agent_view_size//2, then the difference has to be taken into account, which isn't done now.
                if self.algorithm_name == "macmat":
                    ### Front camera view
                    occupied = self.local_obstacles[e][a]
                    # print(f"agent {a} in env {e} has local obstacles: ", occupied)
                    for x in range(0, self.agent_view_size):
                        for y in range(0, self.agent_view_size):
                            if occupied[x, y] == 1:
                                available_actions[e, a, (x)*(self.agent_view_size) + (y)] = 0
                    # ### 360 degree view
                    # occupied = self.local_obstacles[e][a].T
                    # for x in range(-self.agent_view_size//2+1, self.agent_view_size//2+1):
                    #     for y in range(-self.agent_view_size//2+1, self.agent_view_size//2+1):
                            
                    #         if occupied[x + self.agent_view_size//2, y + self.agent_view_size//2] == True:
                    #             available_actions[e, a, (x + self.agent_view_size//2)*(self.agent_view_size) + (y + self.agent_view_size//2)] = 0
                elif self.algorithm_name == "amat":
                    def adjacent_cells(x, y, height, width, surround=False):
                        """
                        Get the list of cells adjacent to a given cell
                        """
                        adj = []
                        if surround:
                            steps = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
                        else:
                            steps = [(-1, 0), (1, 0), (0, -1), (0, 1)]
                        for dx, dy in steps:
                            if x+dx >= 0 and x+dx < width and y+dy >= 0 and y+dy < height:
                                adj.append((x+dx, y+dy))
                        return adj
                    
                    occupied = self.occupied_each_map[e][a][self.agent_view_size:self.full_w - self.agent_view_size,
                                                            self.agent_view_size:self.full_h - self.agent_view_size]
                    occupied = occupied.T
                    for x in range(-self.action_size, self.action_size+1):
                        for y in range(-self.action_size, self.action_size+1):
                            coord = np.array([x, y]) + self.agent_pos[e,a]
                            if coord[0] < 1 or coord[0] >= self.map_size - 1 or coord[1] < 1 or coord[1] >= self.map_size - 1: #outside map boundaries and edges of the map
                                available_actions[e, a, (x + self.action_size)*(2*self.action_size+1) + (y + self.action_size)] = 0
                            elif occupied[coord[0], coord[1]] == 1: #Moving to occupied space is not available
                                available_actions[e, a, (x + self.action_size)*(2*self.action_size+1) + (y + self.action_size)] = 0
                            
                            else: 
                                if self.use_agent_obstacle:
                                    for j in range(self.num_agents):
                                        if j != a:
                                            if coord[0] == self.agent_pos[e, j][0] and coord[1] == self.agent_pos[e, j][1]:
                                                available_actions[e, a, (x + self.action_size)*(2*self.action_size+1) + (y + self.action_size)] = 0
                                                break
                                for adj_cells in adjacent_cells(coord[0], coord[1], self.map_size, self.map_size):
                                    if occupied[adj_cells[0], adj_cells[1]] == 0:
                                        break
                                else:
                                    available_actions[e, a, (x + self.action_size)*(2*self.action_size+1) + (y + self.action_size)] = 0

        # print("available action are chosen")
        # available_actions = np.ones((self.num_agents, *act_dim))
        # print(f"Available actions are: {available_actions} with shape {available_actions.shape}")
        return available_actions

    def correct_ma_bounds(self, macro_action):
        return np.clip(macro_action, 0, self.map_size - 1)
        
    def run(self):
        
        if self.asynch:
            def generate_random_period(min_t,max_t):
                return np.random.randint(min_t, max_t)
            self.asynch_control = AsynchControl(num_envs=self.n_rollout_threads, num_agents=self.num_agents,
                                                limit=self.episode_length, random_fn=generate_random_period,
                                                min_wait=self.all_args.async_min_wait, max_wait=self.all_args.async_max_wait,
                                                rest_time=self.all_args.max_ma_duration)

        start = time.time()
        episodes = int(self.num_env_steps) // self.max_steps // self.n_rollout_threads
        
        for episode in range(self.starting_episode,episodes):
            episode_start_time = time.time()
            ep_data_collection_time = 0
            ep_learning_time = 0
            ep_total_time = 0

            self.init_env_info()
            self.init_map_variables()
            period_rewards = np.zeros((self.n_rollout_threads, self.num_agents, 1))
            self.done_envs = np.zeros((self.n_rollout_threads,)).astype(bool)
            is_last_step = np.full((self.n_rollout_threads, self.num_agents), False)
            if self.use_linear_lr_decay:
                self.trainer.policy.lr_decay(episode, episodes, warmup = self.lr_warmup)
            if self.use_linear_entropy_decay:
                self.trainer.entropy_decay(episode, episodes)

            if self.asynch:
                self.asynch_control.reset()

            values, actions, action_log_probs, rnn_states, rnn_states_critic = self.warmup()

            async_global_macro_step = 0 # This is tau in the paper

            self.current_episode = episode 
            for step in range(self.max_steps - 1):  # -1 because the last step seems to be a reset step in the vectorized environments.
                local_step = step % self.local_step_num
                global_step = (step // self.local_step_num) % self.episode_length
                
                actions_env = self.envs.get_short_term_action(self.macro_action)
                
            
                # Obser reward and next obs
                dict_obs, rewards, dones, infos = self.envs.step(actions_env) #dones seems to be broken when env is done early, so we manually set dones.
            
                
                self.connected_agent_groups = []
                for e in range(self.n_rollout_threads):
                    env_done, rewards[e] = self.update_env_info(e, infos[e], rewards[e], step)
                    # print(f"At step {step}, env {e} has done status after update: {env_done}")
                    if env_done:
                        dones[e] = True
                    # Update state tracking 
                    self.agent_groups[e] = infos[e]['agent_groups']                    
                    self.connected_agent_groups.append(infos[e]['connected_agent_groups'])
                    self.agent_pos[e] = infos[e]['agent_pos']
                    self.agent_dir[e] = infos[e]['agent_direction']
                    self.agent_alive[e] = infos[e]['agent_alive']
                    self.agent_local_views[e] = infos[e]['agent_local_views']
                    self.local_obstacles[e] = infos[e]['agent_local_obstacles']
                    if self.algorithm_name == "amat":
                        self.occupied_each_map[e] = infos[e]['occupied_each_map']

                # Handles Agent Activation
                if self.asynch:
                    self.asynch_control.step()
                    # print(f"Active agents, standby agents and the rest times at step {step} are: {self.asynch_control.active}, {self.asynch_control.standby} and {self.asynch_control.rest}")
                    for e in range(self.n_rollout_threads):
                        for a in range(self.num_agents):
                            if (self.done_envs[e] == True) and is_last_step[e, a] == False:
                                is_last_step[e,a] = True
                                # self.asynch_control.standby[e, a] = 1
                                self.asynch_control.activate(e, a)
                                # print("All agent rewards at this last step are:", rewards[e])
                                # print(f"agent {e,a} is activated because last step {step} is reached.")
                            elif self.agent_alive[e][a] == 0:
                                self.asynch_control.active[e, a] = 0
                                self.asynch_control.standby[e, a] = 0
                                continue

                            if self.agent_classes_list[e, a] == 2 and infos[e]['agent_cleaning_hazard'].get(a) is not None:
                                # If the agent is a cleaner and cleaning a hazard, don't end macro-action and stay there
                                # print(f"agent {a} is a cleaner and cleaning a hazard, so it continues its macro-action.")
                                self.asynch_control.active[e, a] = 0
                                self.asynch_control.standby[e, a] = 0
                                continue
                            if not self.asynch_control.active[e, a] and not self.asynch_control.standby[e, a]:
                                # Checks if agent has reached its short term goal / stopped and puts it on standby if so
                                #TODO: If the agent has reached its ultimate objective, instead of putting it on standby, it should be completely deactivated
                                # Under the assumption of full communication in training, this is added to ensure the final MA of all agents finish at the same time
                  
                                if actions_env[e, a] == 3: # If agent stops
                                    self.asynch_control.standby[e, a] = 1
                                    if self.asynch_control.wait[e, a] <= 0:
                                        self.asynch_control.activate(e, a)

                                elif self.algorithm_name == "macmat" and self.macro_action[e, a][2] in {0, 1, 2}:  # If the macro-action is to turn left or right or toggles
                                    self.asynch_control.standby[e, a] = 1
                                    self.macro_action[e,a] = [self.agent_pos[e,a][0], self.agent_pos[e,a][1], -1] # Rewriting the macro action to take the stop action
                                    if self.asynch_control.wait[e, a] <= 0:
                                        self.asynch_control.activate(e, a)

                                    
                                if infos[e]['n_targets_found'][a] > self.n_targets_found[e,a]:
                                    # print(f"agent {a} knew {self.n_targets_found[e,a]} targets and now knows {infos[e]['n_targets_found'][a]} targets.")
                                    self.n_targets_found[e,a] = infos[e]['n_targets_found'][a]
                                    self.asynch_control.standby[e, a] = 1
                                    if self.asynch_control.wait[e, a] <= 0:
                                        self.asynch_control.activate(e, a)
                                    # self.asynch_control.activate(e, a)
                                    # print(f"agent {a} is standby because a new target is found.")
                                

                    if np.any(self.asynch_control.standby) and np.any(self.asynch_control.active): #Activates on-standby agents based on communication model
                        for thread in self.asynch_control.active_agents_threads():
                            if len(thread) > 1:
                                if self.use_partial_comm:
                                    connected_agents = infos[thread[0]]['connected_agent_groups']
                                    # print("connected agent groups are" , connected_agents)
                                    for agent_id in thread[1:]:
                                        for group in connected_agents:
                                            if agent_id in group:
                                                for i in group:
                                                    # if self.asynch_control.standby[thread[0], i]:
                                                    if self.asynch_control.standby[thread[0], i] and self.asynch_control.wait[thread[0], i] <= generate_random_period(2, 4):
                                                        self.asynch_control.activate(thread[0], i)
                                elif self.use_full_comm:
                                    for i in range(self.num_agents):
                                        # if self.asynch_control.standby[thread[0], i]:
                                        if self.asynch_control.standby[thread[0], i] and self.asynch_control.wait[thread[0], i] <= generate_random_period(2, 4):
                                            self.asynch_control.activate(thread[0], i)
                    # print(f"Active agents, standby agents and the rest times at step {step} after manipulation are: {self.asynch_control.active}, {self.asynch_control.standby} and {self.asynch_control.rest}")
                period_rewards += rewards
                
                if (not self.asynch and local_step == self.local_step_num - 1) or (self.asynch and np.any(self.asynch_control.active)):
                
                    if self.use_action_masking:
                        available_actions = self.get_available_actions()
                        self.available_actions = available_actions
                    else:
                        available_actions = None
                        self.available_actions = None
                    

                    data = dict_obs, period_rewards, dones, infos, values, actions, action_log_probs, rnn_states, rnn_states_critic, self.agent_groups, available_actions
                    
                    # insert data into buffer
                    if not self.asynch:
                        self.insert(data, step)
                        period_rewards = np.zeros((self.n_rollout_threads, self.num_agents, 1))
                    else:
                        self.insert(data, step, active_agents=self.asynch_control.active_agents())
                        for e, a, s in self.asynch_control.active_agents():
                            period_rewards[e, a, 0] = 0.
                    # print("active agents are: ", self.asynch_control.active_agents())
                    if not self.asynch:
                        values, actions, action_log_probs, rnn_states, rnn_states_critic = self.compute_global_goal(step=global_step + 1)
                    else:
                        async_global_macro_step += 1
                        async_values, async_actions, async_action_log_probs, async_rnn_states, async_rnn_states_critic = self.compute_global_goal(step=async_global_macro_step)
                        active_mask = (self.asynch_control.active == 1)                      
                        values[active_mask] = async_values[active_mask]
                        actions[active_mask] = async_actions[active_mask]
                        action_log_probs[active_mask] = async_action_log_probs[active_mask]

                        if self.use_rnn:
                            if self._rnn_uses_tuple_state:
                                # rnn_states[0][active_mask] = async_rnn_states[0][active_mask]
                                # rnn_states[1][active_mask] = async_rnn_states[1][active_mask]
                                rnn_states_critic[0][active_mask] = async_rnn_states_critic[0][active_mask]
                                rnn_states_critic[1][active_mask] = async_rnn_states_critic[1][active_mask]
                            else:
                                # rnn_states[active_mask] = async_rnn_states[active_mask]
                                rnn_states_critic[active_mask] = async_rnn_states_critic[active_mask]
                        
                self.prev_agent_alive = np.copy(self.agent_alive) # save previous alive agents
                if np.all(dones): # If all envs are done, finish the episode
                    break
            ep_data_collection_time = time.time() - episode_start_time
            learning_start_time = time.time()
            # compute returns and update the network
            if self.asynch:
                self.buffer.update_mask(self.asynch_control.cnt)
            else:
                self.buffer.active_masks = self.buffer.masks # to ensure correct computation of returns
            # print("values which were updated the last time a decision was made are: ", values)
            self.compute(values)
            train_infos = self.train()
            
            ep_learning_time = time.time() - learning_start_time

            # post process
            total_num_steps = (episode + 1) * (self.max_steps-1) * self.n_rollout_threads

            # save model
            if (episode % self.save_interval == 0 or episode == episodes - 1):
                if episode % 1000*self.save_interval == 0:
                    self.save(episode, save_separately=True)
                else:
                    self.save(episode, save_separately=False)

            # log information
            self.convert_info()
            
            ep_total_time = time.time() - episode_start_time
            if episode % self.log_interval == 0:
                end = time.time()
                current_total_fps = int(total_num_steps / (end - start))

                total_time_elapsed = end - start
                avg_time_per_ep = total_time_elapsed / (episode - self.starting_episode + 1)
                remaining_episodes = episodes - episode - 1
                expected_remaining_time = avg_time_per_ep * remaining_episodes
                
                # Format times as strings
                str_total_time = str(datetime.timedelta(seconds=int(total_time_elapsed)))
                str_remaining_time = str(datetime.timedelta(seconds=int(expected_remaining_time)))
                # ----------------------------------------

                print("\n Scenario {} Algo {} Exp {} updates {}/{} episodes, total num timesteps {}/{}, FPS {}.\n"
                      .format(self.all_args.scenario_name,
                              self.algorithm_name,
                              self.experiment_name,
                              episode,
                              episodes,
                              total_num_steps,
                              self.num_env_steps,
                              current_total_fps))
                print(f"Total Training Time: {str_total_time} | Expected Remaining: {str_remaining_time}")
                if self.use_wandb:
                    import wandb
                    wandb.log({
                        "time/data_collection_s": ep_data_collection_time,
                        "time/learning_s": ep_learning_time,
                        "time/total_s": ep_total_time,
                        "performance/avg_time_per_ep_s": avg_time_per_ep,
                        }, step=total_num_steps)
                self.log_env(self.env_infos, total_num_steps)
                self.log_train(train_infos, total_num_steps)
                
         

    def _convert(self, dict_obs, infos, step, active_agents=None):
        obs = {}
        
        # Calculate timespans universally for ALL models, log them for eval, 
        # and update the step trackers.
        computed_timespans = np.zeros((len(dict_obs), self.num_agents), dtype=np.float32)
        if active_agents is not None:
            for e in range(len(dict_obs)):
                for a in range(self.num_agents):
                    timespan = step - self.last_active_step[e, a]
                    computed_timespans[e, a] = timespan
                    
                    if self.asynch_control.active[e, a]:
                        self.last_active_step[e, a] = step
                        
                    if self.use_eval:
                        self.timespan_list.append(timespan)

        if self.algorithm_name == "amat":
            obs['agent_class_identifier'] = np.zeros((len(dict_obs), self.num_agents, self.n_agent_types), dtype=int)
            if self.spawn_hazards:
                obs['global_agent_map'] = np.zeros((len(dict_obs), self.num_agents, 9, self.map_size, self.map_size), dtype=np.float32)
                obs['local_agent_map'] = np.zeros((len(dict_obs), self.num_agents, 8, 7, 7), dtype=np.float32)
            else:
                obs['global_agent_map'] = np.zeros((len(dict_obs), self.num_agents, 7, self.map_size, self.map_size), dtype=np.float32)
                obs['local_agent_map'] = np.zeros((len(dict_obs), self.num_agents, 6, 7, 7), dtype=np.float32)
        
            current_agent_pos = np.zeros((len(dict_obs), self.num_agents, 2), dtype=np.int32)
            agent_pos_map = np.zeros((len(dict_obs), self.num_agents, self.full_w, self.full_h), dtype=np.float32)
            global_agent_pos_map = np.zeros((len(dict_obs), self.num_agents, self.full_w, self.full_h), dtype=np.float32)
            global_type0_agent_pos_map = np.zeros((len(dict_obs), self.num_agents, self.full_w, self.full_h), dtype=np.float32) #other agents of type 0
            global_type1_agent_pos_map = np.zeros((len(dict_obs), self.num_agents, self.full_w, self.full_h), dtype=np.float32) #other agents of type 1
            type0_agent_pos_map = np.zeros((len(dict_obs), self.num_agents, self.full_w, self.full_h), dtype=np.float32) #other agents of type 0
            type1_agent_pos_map = np.zeros((len(dict_obs), self.num_agents, self.full_w, self.full_h), dtype=np.float32) #other agents of type 1
            self.agent_goal_history = np.zeros((self.n_rollout_threads, self.num_agents, self.full_w, self.full_h), dtype=np.float32)
            global_target_map = np.zeros((len(dict_obs), self.num_agents, self.full_w, self.full_h), dtype=np.float32)
            target_map = np.zeros((len(dict_obs), self.num_agents, self.full_w, self.full_h), dtype=np.float32)
            if self.spawn_hazards:
                global_type2_agent_pos_map = np.zeros((len(dict_obs), self.num_agents, self.full_w, self.full_h), dtype=np.float32) #other agents of type 2
                type2_agent_pos_map = np.zeros((len(dict_obs), self.num_agents, self.full_w, self.full_h), dtype=np.float32) #other agents of type 2
                global_hazard_map = np.zeros((len(dict_obs), self.num_agents, self.full_w, self.full_h), dtype=np.float32)
                hazard_map = np.zeros((len(dict_obs), self.num_agents, self.full_w, self.full_h), dtype=np.float32)

            
            if self.use_full_comm:
                for e in range(len(dict_obs)):
                    for agent_id in range(self.num_agents):
                        current_agent_pos[e, agent_id] = infos[e]['current_agent_pos'][agent_id]
                        agent_pos_map[e, agent_id, current_agent_pos[e,agent_id][0],
                                        current_agent_pos[e,agent_id][1]] = 1
                        global_agent_pos_map[e, agent_id, current_agent_pos[e,agent_id][0]-1:current_agent_pos[e,agent_id][0]+2,
                                        current_agent_pos[e,agent_id][1]-1:current_agent_pos[e,agent_id][1]+2] = 1
                        self.agent_goal_history[e,agent_id, self.macro_action[e, agent_id][1]+self.agent_view_size,
                                                        self.macro_action[e, agent_id][0]+self.agent_view_size] = 1
                        self.agent_goal_history[e,agent_id,0,0] = 0
                    for agent_id in range(self.num_agents):
                        for j in range(self.num_agents):
                            if j != agent_id:
                                if self.agent_classes_list[e,j] == 0:
                                    type0_agent_pos_map[e,agent_id] = np.maximum(type0_agent_pos_map[e,agent_id], agent_pos_map[e,j])
                                    global_type0_agent_pos_map[e,agent_id] = np.maximum(global_type0_agent_pos_map[e,agent_id], global_agent_pos_map[e,j])
                                elif self.agent_classes_list[e,j] == 1:
                                    type1_agent_pos_map[e,agent_id] = np.maximum(type1_agent_pos_map[e,agent_id], agent_pos_map[e,j])
                                    global_type1_agent_pos_map[e,agent_id] = np.maximum(global_type1_agent_pos_map[e,agent_id], global_agent_pos_map[e,j])
                                elif self.spawn_hazards and self.agent_classes_list[e,j] == 2:
                                    type2_agent_pos_map[e,agent_id] = np.maximum(type2_agent_pos_map[e,agent_id],agent_pos_map[e,j])
                                    global_type2_agent_pos_map[e,agent_id] = np.maximum(global_type2_agent_pos_map[e,agent_id],global_agent_pos_map[e,j])
                for e in range(len(dict_obs)):
                    obs['agent_class_identifier'][e] = self.agent_class_identifier[e]
                    for agent_id in range(self.num_agents):
                        if self.agent_alive[e][agent_id] == 0:
                            continue
                        # Forming target and hazard maps
                        found_targets = list(infos[e]['agent_target_dicts'][agent_id].items())
                        for idx, (target, target_pos) in enumerate(found_targets):
                            global_target_map[e, agent_id, self.agent_view_size+target_pos[1]-1:self.agent_view_size+target_pos[1]+2,
                                              self.agent_view_size+target_pos[0]-1:self.agent_view_size+target_pos[0]+2] = 1
                            target_map[e, agent_id, self.agent_view_size+target_pos[1], self.agent_view_size+target_pos[0]] = 1
                        if self.spawn_hazards:
                            found_hazards = list(infos[e]['agent_hazard_dicts'][agent_id].items())
                            for idx, (hazard, hazard_pos) in enumerate(found_hazards):
                                global_hazard_map[e, agent_id, self.agent_view_size+hazard_pos[1]-1:self.agent_view_size+hazard_pos[1]+2,
                                                  self.agent_view_size+hazard_pos[0]-1:self.agent_view_size+hazard_pos[0]+2] = 1
                                hazard_map[e, agent_id, self.agent_view_size+hazard_pos[1], self.agent_view_size+hazard_pos[0]] = 1

                        # New AMAT
                        obs['global_agent_map'][e, agent_id, 0] = infos[e]['explored_each_map'][agent_id][self.agent_view_size:self.full_w -
                                                                            self.agent_view_size, self.agent_view_size:self.full_w-self.agent_view_size] 
                        if self.use_slam_noise:
                            obs['global_agent_map'][e, agent_id, 1] = infos[e]['noisy_occupied_each_map'][agent_id][self.agent_view_size:self.full_w -
                                                                            self.agent_view_size, self.agent_view_size:self.full_w-self.agent_view_size]
                            obs['local_agent_map'][e, agent_id, 1] = infos[e]['noisy_occupied_each_map'][agent_id][current_agent_pos[e, agent_id][0]-self.agent_view_size//2:current_agent_pos[e, agent_id][0]
                                                                    +self.agent_view_size//2+1, current_agent_pos[e, agent_id][1]-self.agent_view_size//2:current_agent_pos[e, agent_id][1]+self.agent_view_size//2+1]
                        else:
                            
                            obs['global_agent_map'][e, agent_id, 1] = infos[e]['occupied_each_map'][agent_id][self.agent_view_size:self.full_w -
                                                                            self.agent_view_size, self.agent_view_size:self.full_w-self.agent_view_size]
                            obs['local_agent_map'][e, agent_id, 1] = infos[e]['occupied_each_map'][agent_id][current_agent_pos[e, agent_id][0]-self.agent_view_size//2:current_agent_pos[e, agent_id][0]
                                                                    +self.agent_view_size//2+1, current_agent_pos[e, agent_id][1]-self.agent_view_size//2:current_agent_pos[e, agent_id][1]+self.agent_view_size//2+1]
                        obs['global_agent_map'][e, agent_id, 2] = global_agent_pos_map[e, agent_id][self.agent_view_size:self.full_w -
                                                                        self.agent_view_size, self.agent_view_size:self.full_w-self.agent_view_size]
                        obs['global_agent_map'][e, agent_id, 3] = self.agent_goal_history[e,agent_id][self.agent_view_size:self.full_w -
                                                self.agent_view_size, self.agent_view_size:self.full_w-self.agent_view_size]
                        obs['global_agent_map'][e, agent_id, 4] = global_target_map[e, agent_id][self.agent_view_size:self.full_w -
                                                                        self.agent_view_size, self.agent_view_size:self.full_w-self.agent_view_size]
                        obs['global_agent_map'][e, agent_id, 5] = global_type0_agent_pos_map[e,agent_id][self.agent_view_size:self.full_w -
                                                                        self.agent_view_size, self.agent_view_size:self.full_w-self.agent_view_size]
                        obs['global_agent_map'][e, agent_id, 6] = global_type1_agent_pos_map[e,agent_id][self.agent_view_size:self.full_w -
                                                                        self.agent_view_size, self.agent_view_size:self.full_w-self.agent_view_size]
                        if self.spawn_hazards:
                            obs['global_agent_map'][e, agent_id, 7] = global_type2_agent_pos_map[e,agent_id][self.agent_view_size:self.full_w -
                                                                                           self.agent_view_size, self.agent_view_size:self.full_w-self.agent_view_size]
                            obs['global_agent_map'][e, agent_id, 8] = global_hazard_map[e, agent_id][self.agent_view_size:self.full_w -
                                                            self.agent_view_size, self.agent_view_size:self.full_w-self.agent_view_size]

                        obs['local_agent_map'][e, agent_id, 0] = infos[e]['explored_each_map'][agent_id][current_agent_pos[e, agent_id][0]-self.agent_view_size//2:current_agent_pos[e, agent_id][0]
                                                                    +self.agent_view_size//2+1, current_agent_pos[e, agent_id][1]-self.agent_view_size//2:current_agent_pos[e, agent_id][1]+self.agent_view_size//2+1]
                        obs['local_agent_map'][e, agent_id, 2] = self.agent_goal_history[e,agent_id][current_agent_pos[e, agent_id][0]-self.agent_view_size//2:current_agent_pos[e, agent_id][0]
                                                                    +self.agent_view_size//2+1, current_agent_pos[e, agent_id][1]-self.agent_view_size//2:current_agent_pos[e, agent_id][1]+self.agent_view_size//2+1]
                        obs['local_agent_map'][e, agent_id, 3] = target_map[e, agent_id][current_agent_pos[e, agent_id][0]-self.agent_view_size//2:current_agent_pos[e, agent_id][0]
                                                                    +self.agent_view_size//2+1, current_agent_pos[e, agent_id][1]-self.agent_view_size//2:current_agent_pos[e, agent_id][1]+self.agent_view_size//2+1]
                        obs['local_agent_map'][e, agent_id, 4] = type0_agent_pos_map[e,agent_id][current_agent_pos[e, agent_id][0]-self.agent_view_size//2:current_agent_pos[e, agent_id][0]
                                                                    +self.agent_view_size//2+1, current_agent_pos[e, agent_id][1]-self.agent_view_size//2:current_agent_pos[e, agent_id][1]+self.agent_view_size//2+1]
                        obs['local_agent_map'][e, agent_id, 5] = type1_agent_pos_map[e,agent_id][current_agent_pos[e, agent_id][0]-self.agent_view_size//2:current_agent_pos[e, agent_id][0]
                                                                    +self.agent_view_size//2+1, current_agent_pos[e, agent_id][1]-self.agent_view_size//2:current_agent_pos[e, agent_id][1]+self.agent_view_size//2+1]
                        if self.spawn_hazards:
                            obs['local_agent_map'][e,agent_id,6] = type2_agent_pos_map[e,agent_id][current_agent_pos[e, agent_id][0]-self.action_size:current_agent_pos[e, agent_id][0]
                                                                        +self.action_size+1, current_agent_pos[e, agent_id][1]-self.action_size:current_agent_pos[e, agent_id][1]+self.action_size+1]
                            obs['local_agent_map'][e,agent_id,7] = hazard_map[e, agent_id][current_agent_pos[e, agent_id][0]-self.action_size:current_agent_pos[e, agent_id][0]
                                                                       +self.action_size+1, current_agent_pos[e, agent_id][1]-self.action_size:current_agent_pos[e, agent_id][1]+self.action_size+1]
            else:
                raise ValueError("No communication model is chosen.")

        elif self.algorithm_name == "macmat":
            obs['agent_class_identifier'] = np.zeros((len(dict_obs), self.num_agents, self.n_agent_types), dtype=int)
        
            if step == 0:
                for e in range(len(dict_obs)):
                    self.base_position[e][0],self.base_position[e][1] = infos[e]['base_position'][1], infos[e]['base_position'][0]
                    for a in range(self.num_agents):
                        relative_agent_position = infos[e]['current_agent_pos'][a] - self.agent_view_size - self.base_position[e]


            obs['agent_pose'] = np.zeros((len(dict_obs), self.num_agents, 4), dtype=np.float32)
            obs['target_positions'] = np.empty((len(dict_obs), self.num_agents), dtype=object)
            if self.spawn_hazards:
                obs['hazard_positions'] = np.empty((len(dict_obs), self.num_agents), dtype=object)
                obs['local_agent_view'] = np.zeros((len(dict_obs), self.num_agents, 9, self.agent_view_size, self.agent_view_size), dtype=np.float32) # We have hazards and cleaners
            else:
                obs['local_agent_view'] = np.zeros((len(dict_obs), self.num_agents, 7, self.agent_view_size, self.agent_view_size), dtype=np.float32)

            # Insert pre-calculated timespans into the observation dictionary for continuous-time RNNs
            if self.use_rnn and ('CfC' in self.rnn_type or 'NCP' in self.rnn_type):
                obs['timespan'] = np.zeros((len(dict_obs), self.num_agents, 1), dtype=np.float32)
                if active_agents is not None:
                    for e in range(len(dict_obs)):
                        for a in range(self.num_agents):
                            obs['timespan'][e, a] = computed_timespans[e, a]
                
            for e in range(len(dict_obs)):
                obs['agent_class_identifier'][e] = self.agent_class_identifier[e]
                for agent_id in range(self.num_agents):
                    if self.agent_alive[e][agent_id] == 0:
                        continue
                    
                    # Putting the last macro-action on the agent's local view grid
                    if self.macro_action[e, agent_id, 2] != -1: # the macro-action was to turn left or right
                        last_macro_action_i_j = [self.agent_view_size-1, self.agent_view_size//2] # the center-bottom cell
                    else:
                        ax, ay = self.agent_pos[e, agent_id]
                        mx, my = self.macro_action[e,agent_id,:2]
                        direction = infos[e]['agent_direction'][agent_id]
                        dx = mx - ax
                        dy = my - ay
                        # Rotate according to agent's direction
                        if direction == 0:   # right
                            local_x, local_y = dy, -dx
                        elif direction == 1: # down
                            local_x, local_y = -dx, -dy
                        elif direction == 2: # left
                            local_x, local_y = -dy, dx
                        elif direction == 3: # up
                            local_x, local_y = dx, dy
                        else:
                            raise ValueError("Invalid direction")
                        # Check bounds
                        if -self.agent_view_size//2 <= local_x <= self.agent_view_size//2 and -self.agent_view_size +1 <= local_y <=0:
                            last_macro_action_i_j = (local_y + self.agent_view_size-1, local_x + self.agent_view_size//2)
                            if self.spawn_hazards:
                                obs['local_agent_view'][e, agent_id, 8][last_macro_action_i_j] = 1
                            else:
                                obs['local_agent_view'][e, agent_id, 6][last_macro_action_i_j] = 1
                        else:
                            pass

                    agent_local_view = infos[e]['agent_local_views'][agent_id]
                    for i in range(self.agent_view_size): #rows
                        for j in range(self.agent_view_size): #cols
                            # Channel 0: visible/not_visible, Channel 1: walls and obstacles, Channel 2: doors, Channel 3: targets, Channel 4-6: other agents
                            if agent_local_view[i,j] != 0: # visible
                                obs['local_agent_view'][e, agent_id, 0][i,j] = 1
                                if agent_local_view[i,j] == CellCode.WALL or agent_local_view[i,j] == CellCode.OBSTACLE: # Walls and obstacles
                                    obs['local_agent_view'][e, agent_id, 1][i,j] = 1
                                elif agent_local_view[i,j] == CellCode.DOOR: # Doors
                                    obs['local_agent_view'][e, agent_id, 2][i,j] = 1
                                elif agent_local_view[i,j] == CellCode.TARGET: # Targets
                                    obs['local_agent_view'][e, agent_id, 3][i,j] = 1
                                elif agent_local_view[i,j] == CellCode.AGENT_RESCUER: # rescuer agents
                                    obs['local_agent_view'][e, agent_id, 4][i,j] = 1
                                elif agent_local_view[i,j] == CellCode.AGENT_SCOUT: # scout agents
                                    obs['local_agent_view'][e, agent_id, 5][i,j] = 1
                                elif self.spawn_hazards and agent_local_view[i,j] == CellCode.AGENT_CLEANER: # cleaner agents
                                    obs['local_agent_view'][e, agent_id, 6][i,j] = 1
                                elif self.spawn_hazards and agent_local_view[i,j] == CellCode.HAZARD: # Hazards
                                    obs['local_agent_view'][e, agent_id, 7][i,j] = 1
                    if self.agent_classes_list[e, agent_id] == 0:
                        obs['local_agent_view'][e, agent_id, 4][self.agent_view_size-1, self.agent_view_size//2] = 1
                    elif self.agent_classes_list[e, agent_id] == 1:
                        obs['local_agent_view'][e, agent_id, 5][self.agent_view_size-1, self.agent_view_size//2] = 1
                    elif self.spawn_hazards and self.agent_classes_list[e, agent_id] == 2:
                        obs['local_agent_view'][e, agent_id, 6][self.agent_view_size-1, self.agent_view_size//2] = 1

                    if self.use_localization == True:
                        ### Forming the agent_pose observation
                        # "relative" means relative to the shared coordinate frame. Assuming x-axis of the shared coordinate frame is +x direction of our maps.
                        relative_agent_position = infos[e]['current_agent_pos'][agent_id] - self.agent_view_size - self.base_position[e]
                        agent_theta = infos[e]['agent_theta'][agent_id]
                        if self.use_localization_noise:
                            # Add first-order Gaussian random-walk to the agent position
                            d_t = infos[e]['distance_traversed'][agent_id]
                            noise = np.random.normal(
                                loc=0.0,
                                scale=[np.sqrt(self.position_drift_rate**2 * d_t), np.sqrt(self.position_drift_rate**2 * d_t), np.sqrt(self.orientation_drift_rate**2 * d_t)]
                            )
                            sensed_position = relative_agent_position + noise[:2]
                            sensed_theta = agent_theta + noise[2]
                        else:
                            sensed_position = relative_agent_position
                            sensed_theta = agent_theta
                        obs['agent_pose'][e, agent_id][:2] = np.round((sensed_position) / self.map_size, 4)
                        obs['agent_pose'][e, agent_id][2:4] = np.round([np.cos(sensed_theta), np.sin(sensed_theta)], 4) # orientation in cos and sin
                        
                        ### Forming the target_positions observation
                        current_found_targets = infos[e]['agent_target_dicts'][agent_id]
                        current_reached_targets = infos[e]['agent_targets_reached'][agent_id]

                        # Mark currently found targets as 'known'
                        for target_id, pos in current_found_targets.items():
                            self.all_known_targets[e][agent_id][target_id] = {'pos': pos, 'status': 'known'}

                        # Mark reached targets as 'rescued'
                        for target_id, pos in current_reached_targets.items():
                            self.all_known_targets[e][agent_id][target_id] = {'pos': pos, 'status': 'rescued'}

                        # Identify and mark lost targets
                        known_ids = set(self.all_known_targets[e][agent_id].keys())
                        current_ids = set(current_found_targets.keys()) | set(current_reached_targets.keys())
                        lost_target_ids = known_ids - current_ids
                        
                        for target_id in lost_target_ids:
                            # Only mark as 'lost' if it was previously 'known'
                            if self.all_known_targets[e][agent_id][target_id]['status'] == 'known':
                                self.all_known_targets[e][agent_id][target_id]['status'] = 'lost'

                        # Build observation from the complete set of known targets
                        if not self.all_known_targets[e][agent_id]:
                            # If there are no known targets, add a single zero vector
                            obs['target_positions'][e][agent_id] = [np.zeros(5, dtype=np.float32)]
                        else:
                            for target_id, data in self.all_known_targets[e][agent_id].items():
                                status = data['status']
                                pos = data['pos']
                                abs_pos = (pos[1] - self.base_position[e][0], pos[0] - self.base_position[e][1])
                                relative_pos = (abs_pos[0] - relative_agent_position[0], abs_pos[1] - relative_agent_position[1])

                                if self.use_localization_noise:
                                    sensed_pos = relative_pos + noise[:2]
                                else:
                                    sensed_pos = relative_pos
                                
                                scaled_pos = np.round(np.array(sensed_pos, dtype=float) / self.map_size, 4)
                                
                                if status == 'known':
                                    target_status_vec = np.array([1, 0, 0], dtype=np.float32)
                                elif status == 'rescued':
                                    target_status_vec = np.array([0, 1, 0], dtype=np.float32)
                                elif status == 'lost':
                                    target_status_vec = np.array([0, 0, 1], dtype=np.float32)
                                    scaled_pos = (0, 0) # Position is unknown for lost targets
                                
                                
                                # Add the target information to the observation
                                if obs['target_positions'][e][agent_id] is None:
                                    obs['target_positions'][e][agent_id] = [np.concatenate([target_status_vec, scaled_pos]).astype(np.float32)]
                                else:
                                    obs['target_positions'][e][agent_id].append(np.concatenate([target_status_vec, scaled_pos]).astype(np.float32))


                        ### Forming the hazard_positions observation
                        if self.spawn_hazards:
                            current_found_hazards = infos[e]['agent_hazard_dicts'][agent_id]
                            current_reached_hazards = infos[e]['agent_hazards_reached'][agent_id]

                            for hazard_id, pos in current_found_hazards.items():
                                self.all_known_hazards[e][agent_id][hazard_id] = {'pos': pos, 'status': 'known'}
                            for hazard_id, pos in current_reached_hazards.items():
                                self.all_known_hazards[e][agent_id][hazard_id] = {'pos': pos, 'status': 'rescued'}
                            
                            known_ids = set(self.all_known_hazards[e][agent_id].keys())
                            current_ids = set(current_found_hazards.keys()) | set(current_reached_hazards.keys())
                            lost_hazard_ids = known_ids - current_ids

                            for hazard_id in lost_hazard_ids:
                                if self.all_known_hazards[e][agent_id][hazard_id]['status'] == 'known':
                                    self.all_known_hazards[e][agent_id][hazard_id]['status'] = 'lost'

                            if not self.all_known_hazards[e][agent_id]:
                                # If there are no known hazards, add a single zero vector
                                obs['hazard_positions'][e][agent_id].append(np.zeros(5, dtype=np.float32))
                            else:
                                for hazard_id, data in self.all_known_hazards[e][agent_id].items():
                                    status = data['status']
                                    pos = data['pos']
                                    abs_pos = (pos[1] - self.base_position[e][0], pos[0] - self.base_position[e][1])
                                    relative_pos = (abs_pos[0] - relative_agent_position[0], abs_pos[1] - relative_agent_position[1])

                                    if self.use_localization_noise:
                                        sensed_pos = relative_pos + noise[:2]
                                    else:
                                        sensed_pos = relative_pos
                                    scaled_pos = np.round(np.array(sensed_pos, dtype=float) / self.map_size, 4)

                                    if status == 'known':
                                        hazard_status_vec = np.array([1, 0, 0], dtype=np.float32)
                                    elif status == 'rescued':
                                        hazard_status_vec = np.array([0, 1, 0], dtype=np.float32)
                                    elif status == 'lost':
                                        hazard_status_vec = np.array([0, 0, 1], dtype=np.float32)
                                        scaled_pos = (0, 0) # Position is unknown for lost targets

                                    
                                    # Add the hazard information to the observation
                                    if obs['hazard_positions'][e][agent_id] is None:
                                        obs['hazard_positions'][e][agent_id] = [np.concatenate([hazard_status_vec, scaled_pos]).astype(np.float32)]
                                    else:
                                        obs['hazard_positions'][e][agent_id].append(np.concatenate([hazard_status_vec, scaled_pos]).astype(np.float32))

                    else:
                        pass
            
        return obs

    def warmup(self):
        # reset env
        dict_obs, infos = self.envs.reset()
        for e in range(self.n_rollout_threads):
            self.agent_pos[e] = infos[e]['agent_pos'] 
            self.agent_dir[e] = infos[e]['agent_direction']
            self.agent_classes_list[e] = np.array(infos[e]['agent_classes_list'])
            self.agent_local_views[e] = np.array(infos[e]['agent_local_views'])
            self.last_active_pos[e] = infos[e]['current_agent_pos']
            self.local_obstacles[e] = infos[e]['agent_local_obstacles']
            if self.algorithm_name == "amat":
                self.occupied_each_map[e] = infos[e]['occupied_each_map']

        # one-hot agent_class_identifier
        self.agent_class_identifier = np.zeros((self.n_rollout_threads, self.num_agents, self.n_agent_types), dtype=np.int32)
        for e in range(self.n_rollout_threads):
            for agent_id in range(self.num_agents):
                for agent_type in range(self.n_agent_types):
                    if self.agent_classes_list[e, agent_id] == agent_type:
                        self.agent_class_identifier[e, agent_id, agent_type] = 1
        
        
        if self.asynch:
            active_agents = self.asynch_control.active_agents()
        else:
            active_agents = None
        
        # used for training with partial comm
        self.connected_agent_groups = []
        for e in range(len(dict_obs)):
            self.connected_agent_groups.append(infos[e]['connected_agent_groups'])
            self.agent_groups[e] = infos[e]['agent_groups']
        self.buffer.agent_groups[0] = self.agent_groups.copy()

        obs = self._convert(dict_obs, infos, 0, active_agents)
        self.obs = obs

        if self.use_action_masking:
            available_actions = self.get_available_actions()
            self.available_actions = available_actions
        else:
            available_actions = None
            self.available_actions = None

        
        for key in obs.keys():
            self.buffer.obs[key][0] = obs[key].copy()
            self.buffer.all_obs[key][0] = obs[key].copy()

        if available_actions is not None:
            self.buffer.available_actions[0] = available_actions.copy()
        
        values, actions, action_log_probs, rnn_states, rnn_states_critic = self.compute_global_goal(0)


        return values, actions, action_log_probs, rnn_states, rnn_states_critic


    def init_hyperparameters(self):
        # Calculating full and local map sizes
        self.map_size = self.all_args.grid_size
        self.max_steps = self.all_args.max_steps
        self.local_step_num = self.all_args.local_step_num
        self.agent_view_size = self.all_args.agent_view_size
        self.full_w, self.full_h = self.map_size + 2*self.agent_view_size, self.map_size + 2*self.agent_view_size
        
        self.use_action_masking = self.all_args.use_action_masking

        self.use_agent_obstacle = self.all_args.use_agent_obstacle

        self.asynch = self.all_args.asynch

        # function_parameters
        self.use_full_comm = self.all_args.use_full_comm
        self.use_partial_comm = self.all_args.use_partial_comm
        self.use_centralized_training = self.all_args.use_centralized_training
        self.use_graph_attention = self.all_args.use_graph_attention
        self.use_graph_attention_eval = self.all_args.use_graph_attention_eval
        self.use_classbased_action = self.all_args.use_classbased_action
        self.use_auxiliary_rewards = self.all_args.use_auxiliary_rewards
        self.use_auxiliary_rewards_decay = self.all_args.use_auxiliary_rewards_decay
    
        # Noise parameters
        self.use_slam_noise = self.all_args.use_slam_noise # SLAM noise for map-based methods
        self.slam_noise_prob = self.all_args.slam_noise_prob
        self.use_perception_noise = self.all_args.use_perception_noise # Perception noise for the local semantic grid
        self.perception_noise_base_value = self.all_args.perception_noise_base_value 
        self.perception_noise_distance_factor = self.all_args.perception_noise_distance_factor
        self.use_localization_noise = self.all_args.use_localization_noise # Localization noise for agent position and orientation
        self.use_localization = self.all_args.use_localization
        self.position_drift_rate = self.all_args.position_drift_rate
        self.orientation_drift_rate = self.all_args.orientation_drift_rate
        self.extended_delays = self.all_args.extended_delays
        
        self.use_rnn = self.all_args.use_rnn
        self.recurrent_hidden_size = self.all_args.recurrent_hidden_size
        self.rnn_type = self.all_args.rnn_type

        self.starting_episode = self.all_args.starting_episode

    @property
    def _rnn_uses_tuple_state(self):
        """Whether the recurrent state is stored as an ``(h, c)`` tuple.

        True for ``LSTM`` and any ``Mixed*`` rnn_type (which carry an LSTM cell
        state alongside the liquid/GRU hidden state), False for the single-array
        types (``GRU`` / ``CfC`` / ``NCP``). Encapsulates the precedence-sensitive
        ``rnn_type == 'LSTM' or 'Mixed' in rnn_type`` test repeated throughout the
        runner so the recurrent-state layout is decided in exactly one place.
        """
        return self.rnn_type == 'LSTM' or 'Mixed' in self.rnn_type

    @staticmethod
    def _rnn_states_to_numpy(rnn_states):
        """Detach recurrent state(s) to NumPy, preserving the tuple-vs-array layout
        (an ``(h, c)`` tuple for LSTM/Mixed types, a single array otherwise)."""
        if rnn_states is None:
            return None
        if isinstance(rnn_states, (tuple, list)):
            return tuple(np.array(_t2n(s)) for s in rnn_states)
        return np.array(_t2n(rnn_states))

    def _goal_to_macro_action(self, goal):
        """Decode a flat discrete action index into a macro-action array.

        macmat indexes a front-camera view (row/col swapped relative to the
        360-degree case, with a trailing primary-action indicator defaulting to
        -1); amat uses ego-relative (row, col) offsets in [-action_size, action_size].
        """
        if self.algorithm_name == "macmat":
            col = goal // (2 * self.action_size + 1)
            row = goal % (2 * self.action_size + 1)
            default_primary_indicator = np.full_like(row, -1, dtype=row.dtype)
            macro_action = np.stack((row, col, default_primary_indicator), axis=-1)
        elif self.algorithm_name == "amat":
            row = goal // (2 * self.action_size + 1) - self.action_size
            col = goal % (2 * self.action_size + 1) - self.action_size
            macro_action = np.stack((row, col), axis=-1)
        return macro_action

    def init_keys(self):
        """Initialize metric tracking keys"""
        
        # Define all possible metrics with their properties
        self.metric_definitions = {
            
            # Agent-level metrics (one value per agent per environment)
            'auxiliary_reward': {'type': 'agent', 'aggregation': 'sum', 'print': False, 'log': True},
            'agent_reward': {'type': 'agent', 'aggregation': 'sum', 'print': True, 'log': True},
            # 'agent_explored_reward': {'type': 'agent', 'aggregation': 'sum', 'print': False, 'log': False},
            # 'n_targets_found': {'type': 'agent', 'aggregation': 'current', 'print': False, 'log': False},
            # 'n_hazards_found': {'type': 'agent', 'aggregation': 'current', 'print': False, 'log': False},

            # Environment-level metrics (one value per environment)
            'merge_explored_ratio': {'type': 'env', 'aggregation': 'mean', 'print': True, 'log': True},
            'fully_explored_step': {'type': 'env', 'aggregation': 'nanmean', 'print': False, 'log': True},
            'mission_completed': {'type': 'env', 'aggregation': 'mean', 'print': False, 'log': True},
            'mission_completed_step': {'type': 'env', 'aggregation': 'nanmean', 'print': True, 'log': True},
            'total_reward': {'type': 'env', 'aggregation': 'mean', 'print': True, 'log': True},
            'training_stage': {'type': 'env', 'aggregation': 'mean', 'print': True, 'log': True},
            
            # Special object metrics (sets that need special handling)
            'gt_target_obj_set': {'type': 'object_set', 'aggregation': 'none', 'print': False, 'log': False},
            'gt_hazard_obj_set': {'type': 'object_set', 'aggregation': 'none', 'print': False, 'log': False},
        }
        
        
        # Create deques for storing metric history
        self.env_infos = {}
        for key in self.metric_definitions:
            if self.metric_definitions[key]['log'] or self.metric_definitions[key]['print']:
                if self.metric_definitions[key]['aggregation'] == 'sum':
                    self.env_infos['sum_' + key] = deque(maxlen=1)
                elif self.metric_definitions[key]['aggregation'] == 'mean' or self.metric_definitions[key]['aggregation'] == 'nanmean':
                    self.env_infos['mean_' + key] = deque(maxlen=1)
                else:
                    self.env_infos[key] = deque(maxlen=1)

    def init_env_info(self):
        """Initialize environment info storage based on metric definitions"""
        self.env_info = {}
        
        for key, props in self.metric_definitions.items():
            if props['type'] == 'env':
                if "step" in key:
                    if props['aggregation'] == 'nanmean':
                        self.env_info['mean_' + key] = np.ones((self.n_rollout_threads,), dtype=np.float32) * self.max_steps
                else:
                    if props['aggregation'] == 'mean':
                        self.env_info['mean_' + key] = np.zeros((self.n_rollout_threads,), dtype=np.float32)
                    
            elif props['type'] == 'agent' and props['aggregation'] == 'sum':
                self.env_info[f'sum_{key}'] = np.zeros((self.n_rollout_threads, self.num_agents), dtype=np.float32)
                    
            elif props['type'] == 'object_set':
                self.env_info[key] = [set() for _ in range(self.n_rollout_threads)]

    def init_eval_env_info(self):
        """Initialize evaluation environment info storage"""
        self.eval_env_info = {}
        
        for key, props in self.metric_definitions.items():
            if props['type'] == 'env':
                if "step" in key:
                    if props['aggregation'] == 'nanmean':
                        self.eval_env_info['mean_' + key] = np.ones((self.n_eval_rollout_threads,), dtype=np.float32) * self.max_steps
                else:
                    if props['aggregation'] == 'mean':
                        self.eval_env_info['mean_' + key] = np.zeros((self.n_eval_rollout_threads,), dtype=np.float32)

            elif props['type'] == 'agent' and props['aggregation'] == 'sum':
                self.eval_env_info[f'sum_{key}'] = np.zeros((self.n_eval_rollout_threads, self.num_agents), dtype=np.float32)

            elif props['type'] == 'object_set':
                self.eval_env_info[key] = [set() for _ in range(self.n_eval_rollout_threads)]


    def init_map_variables(self):
        # action space
        # self.act_dim = self.envs.action_space[0].high - self.envs.action_space[0].low + 1 #for multidiscrete action space
        self.act_dim = self.envs.action_space[0].n #for discrete action space
        self.action_shape = get_shape_from_act_space(self.envs.action_space[0])

        if self.use_action_masking:
            # self.available_actions = np.ones((self.n_rollout_threads, self.num_agents, *self.act_dim), dtype=np.int32)
            self.available_actions = np.ones((self.n_rollout_threads, self.num_agents, self.act_dim), dtype=np.int32)
        else:
            self.available_actions = None
        
        # Initializing agent pos, groups and actions info
        if self.algorithm_name == "macmat":
            self.macro_action = np.full((self.n_rollout_threads, self.num_agents, 3), -1, dtype=np.int32) # Used when macro actions are either coordinates or primary actions
        elif self.algorithm_name == "amat" or self.algorithm_name[:2] == "ft":
            self.macro_action = np.zeros((self.n_rollout_threads, self.num_agents, 2), dtype=np.int32) # Used when macro actions are simply coordinates, nothing else
        self.agent_classes_list = np.zeros((self.n_rollout_threads, self.num_agents), dtype=np.int32)
        self.agent_pos = np.zeros((self.n_rollout_threads, self.num_agents, 2), dtype=np.int32) # based on the actual map
        self.agent_dir = np.zeros((self.n_rollout_threads, self.num_agents), dtype=np.int32) 
        self.base_position = np.zeros((self.n_rollout_threads, 2), dtype=np.int32)
        self.agent_groups = np.ones((self.n_rollout_threads, self.num_agents, self.num_agents), dtype=np.int32)
        self.agent_alive = np.ones((self.n_rollout_threads, self.num_agents), dtype=np.int32)
        self.prev_agent_alive = np.ones((self.n_rollout_threads, self.num_agents), dtype=np.int32)
        self.agent_local_views = np.zeros((self.n_rollout_threads, self.num_agents, self.agent_view_size, self.agent_view_size), dtype=np.int32)

        self.max_num_targets = self.all_args.max_num_targets
        self.n_targets_found = np.zeros((self.n_rollout_threads,self.num_agents), dtype=np.int32)
        self.spawn_hazards = self.all_args.spawn_hazards
        self.max_num_hazards = self.all_args.max_num_hazards
        self.n_hazards_found = np.zeros((self.n_rollout_threads,self.num_agents), dtype=np.int32)
        self.all_known_targets = [[{} for _ in range(self.num_agents)] for _ in range(self.n_rollout_threads)]
        self.all_known_hazards = [[{} for _ in range(self.num_agents)] for _ in range(self.n_rollout_threads)]

        self.exploration_only = self.all_args.exploration_only # The mission is only to explore the whole map.

        self.last_active_step = np.zeros((self.n_rollout_threads, self.num_agents), dtype=np.int32)
        self.last_active_pos = np.zeros((self.n_rollout_threads, self.num_agents, 2), dtype=np.int32)
        self.local_obstacles = np.zeros((self.n_rollout_threads, self.num_agents, self.agent_view_size, self.agent_view_size), dtype=np.int32)
        self.num_targets = np.zeros((self.n_rollout_threads,), dtype=np.int32)
        if self.algorithm_name == "amat":
            self.occupied_each_map = np.zeros((self.n_rollout_threads, self.num_agents, self.full_h, self.full_w), dtype=np.int32)

    def init_eval_map_variables(self):
        # Action Space
        self.act_dim = self.eval_envs.action_space[0].n
        self.action_shape = get_shape_from_act_space(self.eval_envs.action_space[0])
        if self.use_action_masking:
            self.available_actions = np.ones((self.n_eval_rollout_threads, self.num_agents, self.act_dim), dtype=np.int32)
        else:
            self.available_actions = None

        # Initializing full, merge and local map
        if self.algorithm_name == "macmat":
            self.macro_action = np.full((self.n_rollout_threads, self.num_agents, 3), -1, dtype=np.int32) # Used when macro actions are either coordinates or primary actions
        elif self.algorithm_name == "amat" or self.algorithm_name[:2] == "ft":
            self.macro_action = np.zeros((self.n_eval_rollout_threads, self.num_agents, 2), dtype=np.int32) # Used when macro actions are simply coordinates, nothing else
        self.agent_classes_list = np.zeros((self.n_eval_rollout_threads, self.num_agents), dtype=np.int32)
        self.agent_pos = np.zeros((self.n_eval_rollout_threads, self.num_agents, 2), dtype=np.int32) # based on the actual map
        self.agent_dir = np.zeros((self.n_eval_rollout_threads, self.num_agents), dtype=np.int32) 
        self.base_position = np.zeros((self.n_eval_rollout_threads, 2), dtype=np.int32)
        self.eval_agent_groups = np.ones((self.n_eval_rollout_threads, self.num_agents, self.num_agents), dtype=np.float32)
        self.agent_alive = np.ones((self.n_eval_rollout_threads, self.num_agents), dtype=np.int32)
        self.agent_count = np.zeros((self.n_eval_rollout_threads,self.all_args.n_agent_types), dtype=np.int32)
        self.prev_agent_alive = np.ones((self.n_eval_rollout_threads, self.num_agents), dtype=np.int32) # Used to check if an agent was alive in the previous step
        self.agent_local_views = np.zeros((self.n_eval_rollout_threads, self.num_agents, self.agent_view_size, self.agent_view_size), dtype=np.int32)

        
        self.max_num_targets = self.all_args.max_num_targets
        self.n_targets_found = np.zeros((self.n_eval_rollout_threads,self.num_agents), dtype=np.int32)
        self.spawn_hazards = self.all_args.spawn_hazards
        self.max_num_hazards = self.all_args.max_num_hazards
        self.n_hazards_found = np.zeros((self.n_eval_rollout_threads,self.num_agents), dtype=np.int32)

        self.exploration_only = self.all_args.exploration_only # The mission is only to explore the whole map.

        self.last_active_step = np.zeros((self.n_eval_rollout_threads, self.num_agents), dtype=np.int32)
        self.last_active_pos = np.zeros((self.n_eval_rollout_threads, self.num_agents, 2), dtype=np.int32)
        self.local_obstacles = np.zeros((self.n_eval_rollout_threads, self.num_agents, self.agent_view_size, self.agent_view_size), dtype=np.int32)
        self.num_targets = np.zeros((self.n_eval_rollout_threads,), dtype=np.int32)
        if self.algorithm_name == "amat":
            self.occupied_each_map = np.zeros((self.n_eval_rollout_threads, self.num_agents, self.full_h, self.full_w), dtype=np.int32)


    def update_env_info(self, e, infos_e, rewards_e, step):
        """Update environment info for a single environment thread
        
        Args:
            e: Environment thread index
            infos_e: Info dict from environment e
            rewards_e: Rewards array for environment e
        """
        # print(f"Updating environment info for env {e} with rewards: {rewards_e}")

        # Update metrics
        for key, props in self.metric_definitions.items():
            if props['type'] == 'agent' and props['aggregation'] == 'sum': 
                if not self.done_envs[e]:  # Only update if environment wasn't done in the previous step
                    if key == 'auxiliary_reward': 
                        self.env_info['sum_' + key][e] += np.array(infos_e['auxiliary_reward'])
                    elif key == 'agent_reward':
                        self.env_info['sum_' + key][e] += np.array(rewards_e.squeeze())
            else:
                if key in infos_e:
                    # print(f"Updating env info for key {key} in env {e} with value: {infos_e[key]}")
                    if props['aggregation'] == 'mean' or props['aggregation'] == 'nanmean':
                        self.env_info['mean_' + key][e] = infos_e[key]
                    else:
                        self.env_info[key][e] = infos_e[key]
                elif key == 'total_reward':
                    self.env_info['mean_' + key][e] += np.array(rewards_e.sum()/self.num_agents)
                else:
                    print(f"Key {key} not found in infos_e, skipping update for env {e}")

        # # Check mission completion condition
        if (infos_e['agent_count'][0] < 1 or step >= self.max_steps-2 or self.env_info["gt_target_obj_set"][e] == set()) and not self.done_envs[e]:
            self.done_envs[e] = True
            # print(f"Environment {e} done at step {step} with rewards: {rewards_e} and agent count: {infos_e['agent_count'][0]}")
        # elif (infos_e['agent_count'][0] < 1 or step >= self.max_steps) and not self.done_envs[e]:
        #     self.done_envs[e] = True
        return self.done_envs[e], rewards_e
    
    def update_eval_env_info(self, e, infos_e, rewards_e, step):
        """Update environment info for a single environment thread
        
        Args:
            e: Environment thread index
            infos_e: Info dict from environment e
            rewards_e: Rewards array for environment e
        """
        # print(f"Updating environment info for env {e} with rewards: {rewards_e}")

        # Update metrics
        for key, props in self.metric_definitions.items():
            if props['type'] == 'agent' and props['aggregation'] == 'sum': 
                if not self.eval_done_envs[e]:  # Only update if environment wasn't done in the previous step
                    if key == 'auxiliary_reward': 
                        self.eval_env_info['sum_' + key][e] += np.array(infos_e['auxiliary_reward'])
                    elif key == 'agent_reward':
                        self.eval_env_info['sum_' + key][e] += np.array(rewards_e.squeeze())
            else:
                if key in infos_e:
                    # print(f"Updating env info for key {key} in env {e} with value: {infos_e[key]}")
                    if props['aggregation'] == 'mean' or props['aggregation'] == 'nanmean':
                        self.eval_env_info['mean_' + key][e] = infos_e[key]
                    else:
                        self.eval_env_info[key][e] = infos_e[key]
                elif key == 'total_reward':
                    self.eval_env_info['mean_' + key][e] += np.array(rewards_e.sum()/self.num_agents)
                else:
                    print(f"Key {key} not found in infos_e, skipping update for env {e}")

        # # Check mission completion condition
        if (infos_e['agent_count'][0] < 1 or step >= self.max_steps-2 or self.eval_env_info["gt_target_obj_set"][e] == set()) and not self.eval_done_envs[e]:
            self.eval_done_envs[e] = True
            # print(f"Environment {e} done at step {step} with rewards: {rewards_e} and agent count: {infos_e['agent_count'][0]}")
        # elif (infos_e['agent_count'][0] < 1 or step >= self.max_steps) and not self.eval_done_envs[e]:
        #     self.eval_done_envs[e] = True
        return self.eval_done_envs[e], rewards_e

     
    def convert_info(self):
        """Convert and print environment info in an organized way"""
        print_metrics = {
            'mean_merge_explored_ratio': lambda v: f'Mean exploration ratio: {np.mean(v)*100:.1f}%',
            'mean_mission_completed': lambda v: f'Mean mission completed: {np.mean(v)*100:.1f}%',
            'mean_mission_completed_step': lambda v: f'Mean mission completion step: {np.nanmean(v):.1f}',
            'mean_total_reward': lambda v: f'Mean total reward: {np.mean(v):.3f}',
            'sum_agent_reward': lambda v: [f'Agent {i} mean reward: {np.mean(v[:,i]):.3f}' 
                                        for i in range(self.num_agents)],
            'mean_training_stage': lambda v: f'Mean training stage: {np.mean(v):.1f}',
        }
        
        for key, values in self.env_info.items():
            if key in self.env_infos:
                self.env_infos[key].append(values)
                # Print metrics if configured
                if key in print_metrics:
                    result = print_metrics[key](values)
                    if isinstance(result, list):
                        for line in result:
                            print(line)
                    else:
                        print(result)
            # else:
            #     print(f"Key {key} not found in env_infos, skipping logging.")
    def convert_eval_info(self):
        """Convert and print environment info in an organized way"""
        print_metrics = {
            'mean_merge_explored_ratio': lambda v: f'Mean exploration ratio: {np.mean(v)*100:.1f}%',
            'mean_mission_completed': lambda v: f'Mean mission completed: {np.mean(v)*100:.1f}%',
            'mean_mission_completed_step': lambda v: f'Mean mission completion step: {np.nanmean(v):.1f}',
            'mean_total_reward': lambda v: f'Mean total reward: {np.mean(v):.3f}',
            'sum_agent_reward': lambda v: [f'Agent {i} mean reward: {np.mean(v[:,i]):.3f}' 
                                        for i in range(self.num_agents)],
            'mean_training_stage': lambda v: f'Mean training stage: {np.mean(v):.1f}',
        }
        
        for key, values in self.eval_env_info.items():
            if key in self.env_infos:
                self.env_infos[key].append(values)
                # Print metrics if configured
                if key in print_metrics:
                    result = print_metrics[key](values)
                    if isinstance(result, list):
                        for line in result:
                            print(line)
                    else:
                        print(result)

    def log_env(self, env_infos, total_num_steps):
        """Log environment metrics to wandb/tensorboard"""
        for key, values_list in env_infos.items():
            if len(values_list) > 0:
                if key == 'sum_agent_reward':
                    # Log each agent's reward separately
                    for i in range(self.num_agents):
                        self._log_metric(f'agent{i}_sum_reward', np.mean(np.array(values_list)[0][:,i]), total_num_steps)
                elif 'step' in key:
                    # Log step metrics as nanmean
                    self._log_metric(key, np.nanmean(np.array(values_list)), total_num_steps)
                else:
                    # Log other metrics as mean
                    self._log_metric(key, np.mean(np.array(values_list)), total_num_steps)

    def _log_metric(self, name, value, step):
        """Log a single metric to the configured logging system"""
        if self.use_wandb:
            import wandb
            wandb.log({name: value}, step=step)
        else:
            self.writter.add_scalars(name, {name: value}, step)
    
    @torch.no_grad()
    def compute_global_goal(self, step):
        returned_actions = np.zeros((self.n_rollout_threads, self.num_agents, self.action_shape), dtype=np.float32)
        returned_values = np.zeros((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        returned_action_log_probs = np.zeros((self.n_rollout_threads, self.num_agents, self.action_shape), dtype=np.float32)
        if self.use_rnn:
            if self._rnn_uses_tuple_state:
                # returned_rnn_states_actor = (
                #     np.zeros((self.n_rollout_threads, self.num_agents, self.recurrent_hidden_size), dtype=np.float32),
                #     np.zeros((self.n_rollout_threads, self.num_agents, self.recurrent_hidden_size), dtype=np.float32)
                # )
                returned_rnn_states_actor = None
                returned_rnn_states_critic = (
                    np.zeros((self.n_rollout_threads, self.num_agents, self.recurrent_hidden_size), dtype=np.float32),
                    np.zeros((self.n_rollout_threads, self.num_agents, self.recurrent_hidden_size), dtype=np.float32)
                )
            else:
                # returned_rnn_states_actor = np.zeros((self.n_rollout_threads, self.num_agents, self.recurrent_hidden_size), dtype=np.float32)
                returned_rnn_states_actor = None
                returned_rnn_states_critic = np.zeros((self.n_rollout_threads, self.num_agents, self.recurrent_hidden_size), dtype=np.float32)
        else:
            returned_rnn_states_actor = None
            returned_rnn_states_critic = None
        
        n_threads = self.n_rollout_threads
        
        # Get short-term goals.
        def get_short_term_goal(self, concat_obs, concat_share_obs,
                                all_rnn_states_actor, all_rnn_states_critic, n_threads, available_actions, active_agents = None):
            value, action, action_log_prob, rnn_states_actor , rnn_states_critic  = self.trainer.policy.get_actions(concat_share_obs,
                                                                                    concat_obs,
                                                                                    np.concatenate(self.buffer.masks[step]),
                                                                                    all_rnn_states_actor,
                                                                                    all_rnn_states_critic,
                                                                                    available_actions,
                                                                                    active_agents)
            # [self.envs, agents, dim]
            values = np.array(np.split(_t2n(value), n_threads))
            actions = np.array(np.split(_t2n(action), n_threads))
            action_log_probs = np.array(np.split(_t2n(action_log_prob), n_threads))

            rnn_states_actor = self._rnn_states_to_numpy(rnn_states_actor)
            rnn_states_critic = self._rnn_states_to_numpy(rnn_states_critic)

            goal = np.array(np.split(_t2n(action), n_threads)).astype(np.int32)
            macro_action = self._goal_to_macro_action(goal)
            return values, actions, action_log_probs, macro_action, rnn_states_actor, rnn_states_critic

        self.trainer.prep_rollout()

        # only useful for partial comm
        connected_agent_groups = self.connected_agent_groups
        
        # keys = self.buffer.obs.keys()
        # concat_share_obs = dict.fromkeys(keys)
        # concat_obs = dict.fromkeys(keys)
        concat_share_obs = {}
        concat_obs = {}
        all_available_actions = []
        all_rnn_states_actor = [] # actor has no rnn_states in this implementation
        all_rnn_states_critic = []

        obs_shape = get_shape_from_obs_space(self.envs.observation_space[0])
        # print("observation space shape is: ", obs_shape) #would be a dictionary
        if not self.use_graph_attention:
            if self.asynch and self.use_centralized_training:
                active_threads = self.asynch_control.active_agents_threads()
                for e in range(n_threads):
                    if not active_threads[e] or len(active_threads[e]) == 1:
                        continue
                    # In this case, each active agent is an independent training sample
                    for agent_id in active_threads[e][1:]:
                        agent_obs = {}
                        for key in obs_shape:
                            if key == 'target_positions' or key == 'hazard_positions':
                                temp_arr = np.empty((1,), dtype=object)
                                temp_arr[0] = self.buffer.all_obs[key][step][e][agent_id]
                                agent_obs[key] = temp_arr
                            else:
                                agent_obs[key] = np.zeros_like(self.buffer.all_obs[key][step][e][agent_id])
                                agent_obs[key] = np.expand_dims(agent_obs[key], axis=0)
                        if self.use_action_masking:
                            available_actions = np.ones_like(self.available_actions[e][agent_id])
                            available_actions = np.expand_dims(available_actions, axis=0)
                        else:
                            available_actions = None
                        if self.use_rnn:
                            if self._rnn_uses_tuple_state:
                                # rnn_states_actor = (
                                #     np.zeros_like(self.buffer.rnn_states[0][0][0]),
                                #     np.zeros_like(self.buffer.rnn_states[1][0][0])
                                # )
                                # rnn_states_critic = (
                                #     np.zeros_like(self.buffer.rnn_states_critic[0][0][0]),
                                #     np.zeros_like(self.buffer.rnn_states_critic[1][0][0])
                                # )
                                rnn_states_critic = (
                                    np.zeros((1, self.recurrent_hidden_size), dtype=np.float32),
                                    np.zeros((1, self.recurrent_hidden_size), dtype=np.float32)
                                )
                            else:
                                # rnn_states_actor = np.zeros_like(self.buffer.rnn_states[0][0])
                                # rnn_states_critic = np.zeros_like(self.buffer.rnn_states_critic[0][0])
                                rnn_states_critic = np.zeros((1, self.recurrent_hidden_size), dtype=np.float32)
                        for key in obs_shape: #TODO: fix problem with variable target position vectors
                            if key == 'target_positions' or key == 'hazard_positions':
                                agent_obs[key][0] = self.buffer.all_obs[key][step,e,agent_id]
                            else:
                                agent_obs[key][0] = self.buffer.all_obs[key][step,e,agent_id]
                        if self.use_action_masking:
                            available_actions[0] = self.available_actions[e][agent_id]
                        if self.use_rnn:
                            if self._rnn_uses_tuple_state:
                                # rnn_states_actor[0][0] = self.buffer.rnn_states[0][self.asynch_control.cnt[e,agent_id],e,agent_id]
                                # rnn_states_actor[1][0] = self.buffer.rnn_states[1][self.asynch_control.cnt[e,agent_id],e,agent_id]
                                rnn_states_critic[0][0] = self.buffer.rnn_states_critic[0][self.asynch_control.cnt[e,agent_id],e,agent_id]
                                rnn_states_critic[1][0] = self.buffer.rnn_states_critic[1][self.asynch_control.cnt[e,agent_id],e,agent_id]
                            else:
                                # rnn_states_actor[0] = self.buffer.rnn_states[self.asynch_control.cnt[e,agent_id],e,agent_id]
                                rnn_states_critic[0] = self.buffer.rnn_states_critic[self.asynch_control.cnt[e,agent_id],e,agent_id]
                        if not concat_obs:
                            for key in obs_shape:
                                concat_share_obs[key] = np.expand_dims(agent_obs[key], axis=1)
                                concat_obs[key] = np.expand_dims(agent_obs[key], axis=1)
                        else:
                            for key in obs_shape:
                                # if key == 'target_positions' or key == 'hazard_positions':
                                # print("agent_obs[key] is: ", agent_obs[key])
                                # print("key is : ", key)
                                # print("agent obs key shape: ", agent_obs[key].shape)
                                # print("concat obs key shape before concat: ", concat_obs[key].shape)
                                concat_share_obs[key] =  np.concatenate((concat_share_obs[key], np.expand_dims(agent_obs[key], axis=1)))
                                concat_obs[key] =  np.concatenate((concat_obs[key], np.expand_dims(agent_obs[key], axis=1)))
                        all_available_actions.append(available_actions)
                        if self.use_rnn:
                            # all_rnn_states_actor.append(rnn_states_actor)
                            all_rnn_states_critic.append(rnn_states_critic)

                n_stacked_threads = concat_obs['agent_class_identifier'].shape[0]
                # print("Number of active agents considered for global goal computation: ", n_stacked_threads)
                if self.use_action_masking:
                    all_available_actions = np.array(all_available_actions)
                else:
                    all_available_actions = None
                if not all_rnn_states_actor:
                    all_rnn_states_actor = None
                else:
                    all_rnn_states_actor = np.array(all_rnn_states_actor)
                if not all_rnn_states_critic:
                    all_rnn_states_critic = None
                else:
                    all_rnn_states_critic = np.array(all_rnn_states_critic)

                values, actions, action_log_probs, macro_action, rnn_states_actor, rnn_states_critic = \
                    get_short_term_goal(self, concat_obs, concat_share_obs,
                                        all_rnn_states_actor, all_rnn_states_critic, n_stacked_threads, all_available_actions)
                
                values = values.reshape(-1, 1, values.shape[-1])
                actions = actions.reshape(-1, 1, actions.shape[-1])
                action_log_probs = action_log_probs.reshape(-1, 1, action_log_probs.shape[-1])
                macro_action = macro_action.reshape(-1, 1, macro_action.shape[-2], macro_action.shape[-1])
                if self.use_rnn:
                    if self._rnn_uses_tuple_state:
                        rnn_states_critic = (
                            rnn_states_critic[0].reshape(-1, 1, rnn_states_critic[0].shape[-1]),
                            rnn_states_critic[1].reshape(-1, 1, rnn_states_critic[1].shape[-1])
                        )
                    else:
                        rnn_states_critic = rnn_states_critic.reshape(-1, 1, rnn_states_critic.shape[-1])
                
                # Now we need to place the returned actions, values, etc. back to their respective agents
                counter = 0
                for e in range(n_threads):
                    if not active_threads[e] or len(active_threads[e]) == 1:
                        continue
                    for agent_id in active_threads[e][1:]:
                        if self.algorithm_name == "amat":
                            pass
                        elif self.algorithm_name == "macmat":
                             ### front camera view and possibility of primary actions
                            if macro_action[counter][0][0][1] >= 2*self.action_size+1: # A primary action is chosen instead of a navigation goal
                                primary_action_id = macro_action[counter][0][0][0] # row
                                self.macro_action[e][agent_id] = np.array([-1, -1, primary_action_id])
                            else: # A navigation goal is chosen
                                relative_goal = macro_action[counter][0][0][:2] - [self.action_size, 2*self.action_size]
                                if self.agent_dir[e][agent_id] == 0: # facing right
                                    navigation_goal = self.agent_pos[e, agent_id] + [-relative_goal[1], relative_goal[0]]
                                elif self.agent_dir[e][agent_id] == 1: # facing down
                                    navigation_goal = self.agent_pos[e, agent_id] + [-relative_goal[0], -relative_goal[1]]
                                elif self.agent_dir[e][agent_id] == 2: # facing left
                                    navigation_goal = self.agent_pos[e, agent_id] + [relative_goal[1], -relative_goal[0]]
                                elif self.agent_dir[e][agent_id] == 3: # facing up
                                    navigation_goal = self.agent_pos[e, agent_id] + [relative_goal[0], relative_goal[1]]
                                navigation_goal = self.correct_ma_bounds(navigation_goal)
                                self.macro_action[e][agent_id] = np.array([navigation_goal[0], navigation_goal[1], -1])
                        returned_values[e][agent_id] = values[counter][0]
                        returned_actions[e][agent_id] = actions[counter][0]
                        returned_action_log_probs[e][agent_id] = action_log_probs[counter][0]
                        if self.use_rnn:
                            if self._rnn_uses_tuple_state:
                                returned_rnn_states_critic[0][e][agent_id] = rnn_states_critic[0][counter][0]
                                returned_rnn_states_critic[1][e][agent_id] = rnn_states_critic[1][counter][0]
                            else:
                                returned_rnn_states_critic[e][agent_id] = rnn_states_critic[counter][0]
                            

                        counter += 1

            else:
                raise NotImplementedError("Only asynch + centralized training with no graph attention is implemented.")
        else:
        
            if self.asynch:
                if self.use_centralized_training: # used for centralized and asynchronous training with full communication
                    active_threads = self.asynch_control.active_agents_threads()
                    for e in range(n_threads):
                        if len(active_threads[e]) <= 1:
                            continue
                        padded_obs = {}
                        for key in obs_shape:
                            padded_obs[key] = np.zeros_like(self.buffer.all_obs[key][step][e])
                            padded_obs[key] = np.expand_dims(padded_obs[key], axis=0)
                        if self.use_action_masking:
                            available_actions = np.ones_like(self.available_actions[0])
                        else:
                            available_actions = None
                        if self.use_rnn:
                            if self._rnn_uses_tuple_state:
                                # rnn_states_actor = (
                                #     np.zeros_like(self.buffer.rnn_states[0][0][0]),
                                #     np.zeros_like(self.buffer.rnn_states[1][0][0])
                                # )
                                rnn_states_critic = (
                                    np.zeros_like(self.buffer.rnn_states_critic[0][0][0]),
                                    np.zeros_like(self.buffer.rnn_states_critic[1][0][0])
                                )
                            else:
                                # rnn_states_actor = np.zeros_like(self.buffer.rnn_states[0][0])
                                rnn_states_critic = np.zeros_like(self.buffer.rnn_states_critic[0][0])
                        active_cnt = 0
                        inactive_cnt = 0
                        for agent_id in range(self.num_agents):
                            if agent_id in active_threads[e][1:]:
                                for key in obs_shape:
                                    padded_obs[key][0,active_cnt] = self.buffer.all_obs[key][step,e,agent_id]

                                if self.use_action_masking:
                                    available_actions[active_cnt] = self.available_actions[e][agent_id]
                                if self.use_rnn:
                                    if self._rnn_uses_tuple_state:
                                        # rnn_states_actor[0][active_cnt] = self.buffer.rnn_states[0][self.asynch_control.cnt[e,agent_id],e,agent_id]
                                        # rnn_states_actor[1][active_cnt] = self.buffer.rnn_states[1][self.asynch_control.cnt[e,agent_id],e,agent_id]
                                        rnn_states_critic[0][active_cnt] = self.buffer.rnn_states_critic[0][self.asynch_control.cnt[e,agent_id],e,agent_id]
                                        rnn_states_critic[1][active_cnt] = self.buffer.rnn_states_critic[1][self.asynch_control.cnt[e,agent_id],e,agent_id]
                                    else:
                                        # rnn_states_actor[active_cnt] = self.buffer.rnn_states[self.asynch_control.cnt[e,agent_id],e,agent_id]
                                        rnn_states_critic[active_cnt] = self.buffer.rnn_states_critic[self.asynch_control.cnt[e,agent_id],e,agent_id]
                                # print("agent counters are: ", self.asynch_control.cnt[e,agent_id], e, agent_id)
                                active_cnt += 1
                            else:
                                for key in obs_shape:
                                    padded_obs[key][0,-1-inactive_cnt] = self.buffer.all_obs[key][step,e,agent_id]
                                # print("inactive agent counters are: ", self.asynch_control.cnt[e,agent_id], e, agent_id)
                                if self.use_rnn:
                                    if self._rnn_uses_tuple_state:
                                        # rnn_states_actor[0][-1-inactive_cnt] = self.buffer.rnn_states[0][self.asynch_control.cnt[e,agent_id],e,agent_id]
                                        # rnn_states_actor[1][-1-inactive_cnt] = self.buffer.rnn_states[1][self.asynch_control.cnt[e,agent_id],e,agent_id]
                                        rnn_states_critic[0][-1-inactive_cnt] = self.buffer.rnn_states_critic[0][self.asynch_control.cnt[e,agent_id],e,agent_id]
                                        rnn_states_critic[1][-1-inactive_cnt] = self.buffer.rnn_states_critic[1][self.asynch_control.cnt[e,agent_id],e,agent_id]
                                    else:
                                        # rnn_states_actor[-1-inactive_cnt] = self.buffer.rnn_states[self.asynch_control.cnt[e,agent_id],e,agent_id]
                                        rnn_states_critic[-1-inactive_cnt] = self.buffer.rnn_states_critic[self.asynch_control.cnt[e,agent_id],e,agent_id]
                                inactive_cnt += 1
                            
                        if not concat_obs:
                            for key in obs_shape:
                                concat_share_obs[key] = padded_obs[key]
                                concat_obs[key] = padded_obs[key]
                        else:
                            for key in obs_shape:
                                # print("key is: ", key)
                                # print("padded_obs[key] is: ", padded_obs[key])
                                # print("padded_obs key shape: ", padded_obs[key].shape)
                                concat_share_obs[key] = np.concatenate((concat_share_obs[key],padded_obs[key]))
                                concat_obs[key] = np.concatenate((concat_obs[key],padded_obs[key]))
                        all_available_actions.append(available_actions)
                        if self.use_rnn:
                            # all_rnn_states_actor.append(rnn_states_actor)
                            all_rnn_states_critic.append(rnn_states_critic)

                    n_stacked_threads = concat_obs['agent_class_identifier'].shape[0]
                    if self.use_action_masking:
                        all_available_actions = np.array(all_available_actions)
                    else:
                        all_available_actions = None
                    
                    if not all_rnn_states_actor:
                        all_rnn_states_actor = None
                    else:
                        all_rnn_states_actor = np.array(all_rnn_states_actor)

                    if not all_rnn_states_critic:
                        all_rnn_states_critic = None
                    else:
                        all_rnn_states_critic = np.array(all_rnn_states_critic)
                    # print("shape of obs before concatenation is: ", concat_obs['agent_class_identifier'].shape)
                    # for key in obs_shape:
                    #     concat_share_obs[key] = np.concatenate(concat_share_obs[key])
                    #     concat_obs[key] = np.concatenate(concat_obs[key])
                    # print("shape of obs after concatenation is: ", concat_obs['agent_class_identifier'].shape)
                    values, actions, action_log_probs, macro_action, rnn_states_actor, rnn_states_critic \
                            = get_short_term_goal(self, concat_obs, concat_share_obs,
                                                all_rnn_states_actor, all_rnn_states_critic, n_stacked_threads, all_available_actions)
                    
                    counter = 0 # used to move through the concatenated actions resulting from concatenated obs
                    for e in range(n_threads):
                        if len(active_threads[e]) <= 1:
                            continue
                        agent_num = 0
                        for agent_id in active_threads[e][1:]:
                            
                            
                            if self.algorithm_name == "amat":
                                ### ego-relative goals based on 360 degree view around the agent
                                self.macro_action[e, agent_id] = macro_action[counter, agent_num] + self.agent_pos[e, agent_id]
                                self.macro_action[e, agent_id] = self.correct_ma_bounds(self.macro_action[e, agent_id])
                            elif self.algorithm_name == "macmat":
                                ### front camera view and possibility of primary actions
                                if macro_action[counter, agent_num][0][1] >= 2*self.action_size+1: # A primary action is chosen instead of a navigation goal
                                    primary_action_id = macro_action[counter, agent_num][0][0] # row
                                    self.macro_action[e, agent_id] = np.array([-1, -1, primary_action_id])
                                else: # a navigation goal is chosen
                                    relative_goal = macro_action[counter, agent_num][0][:2] - [self.action_size, 2*self.action_size] # because agent is always positioned at the bottom center of its local view
                                    # (0,0) is where the agent is, and it can go from -3 to 3 right or left and from 0 to -6 forward
                                    # The navigation coordinate depends on the direction the agent is facing
                                    if self.agent_dir[e, agent_id] == 0: # agent is facing right
                                        navigation_goal = self.agent_pos[e, agent_id] + [-relative_goal[1], relative_goal[0]]
                                    elif self.agent_dir[e, agent_id] == 1: # agent is facing down
                                        navigation_goal = self.agent_pos[e, agent_id] + [-relative_goal[0], -relative_goal[1]]
                                    elif self.agent_dir[e, agent_id] == 2: # agent is facing left
                                        navigation_goal = self.agent_pos[e, agent_id] + [relative_goal[1], -relative_goal[0]]
                                    elif self.agent_dir[e, agent_id] == 3: # agent is facing up
                                        navigation_goal = self.agent_pos[e, agent_id] + [relative_goal[0], relative_goal[1]]
                                    navigation_goal = self.correct_ma_bounds(navigation_goal)
                                    self.macro_action[e, agent_id] = np.array([navigation_goal[0], navigation_goal[1], -1])
                            
                            returned_values[e, agent_id] = values[counter, agent_num]
                            returned_actions[e, agent_id] = actions[counter, agent_num,:]
                            returned_action_log_probs[e, agent_id] = action_log_probs[counter, agent_num,:]
                            if rnn_states_actor is not None:
                                if self._rnn_uses_tuple_state:
                                    returned_rnn_states_actor[0][e,agent_id] = rnn_states_actor[0][counter, agent_num]
                                    returned_rnn_states_actor[1][e,agent_id] = rnn_states_actor[1][counter, agent_num]
                                else:
                                    returned_rnn_states_actor[e, agent_id] = rnn_states_actor[counter, agent_num]
                            if rnn_states_critic is not None:
                                if self._rnn_uses_tuple_state:
                                    returned_rnn_states_critic[0][e, agent_id] = rnn_states_critic[0][counter, agent_num]
                                    returned_rnn_states_critic[1][e, agent_id] = rnn_states_critic[1][counter, agent_num]  
                                else:
                                    returned_rnn_states_critic[e, agent_id] = rnn_states_critic[counter, agent_num]
                            agent_num += 1
                        counter += 1
                    # print("counter is ", counter)
                else: # used for asynchronous distributed training (Not Used Now)
                    # in each group, puts active agents first, then inactive agents, then zero-paddings
                    active_threads = self.asynch_control.active_agents_threads()
                    # LSTM-friendly implementation pending
                    for e in range(n_threads):
                        if len(active_threads[e]) <= 1:
                            continue
                        for group in connected_agent_groups[e]:
                            active_in_group = []
                            inactive_in_group = []
                            for agent_id in group:
                                if agent_id in active_threads[e][1:]:
                                    active_in_group.append(agent_id)
                                else:
                                    inactive_in_group.append(agent_id)
                            if len(active_in_group) == 0:
                                continue
                            else:
                                padded_obs = {}
                                for key in obs_shape:
                                    padded_obs[key] = np.zeros_like(self.buffer.all_obs[key][step][e])
                                    padded_obs[key] = np.expand_dims(padded_obs[key], axis=0)
                                if self.use_action_masking:
                                    available_actions = np.ones_like(self.available_actions[0])
                                else:
                                    available_actions = None
                                rnn_states_actor = np.zeros_like(self.buffer.rnn_states[0][0])
                                rnn_states_critic = np.zeros_like(self.buffer.rnn_states_critic[0][0])
                                agent_num = 0
                                for agent_id in active_in_group:
                                    for key in obs_shape:
                                        padded_obs[key][0,agent_num] = self.buffer.all_obs[key][step,e,agent_id]
                                    if self.use_action_masking:
                                        available_actions[agent_num] = self.available_actions[e][agent_id]
                                    rnn_states_actor[agent_num] = self.buffer.rnn_states[self.asynch_control.cnt[e,agent_id],e,agent_id]
                                    rnn_states_critic[agent_num] = self.buffer.rnn_states_critic[self.asynch_control.cnt[e,agent_id],e,agent_id]
                                    agent_num += 1
                                for agent_id in inactive_in_group:
                                    for key in obs_shape:
                                        padded_obs[key][0,agent_num] = self.buffer.all_obs[key][step,e,agent_id]
                                    rnn_states_actor[agent_num] = self.buffer.rnn_states[self.asynch_control.cnt[e,agent_id],e,agent_id]
                                    rnn_states_critic[agent_num] = self.buffer.rnn_states_critic[self.asynch_control.cnt[e,agent_id],e,agent_id]
                                    # available actions are not needed for inactive agents
                                    agent_num += 1 

                                if not concat_obs:
                                    for key in obs_shape:
                                        concat_share_obs[key] = padded_obs[key]
                                        concat_obs[key] = padded_obs[key]
                                else:
                                    for key in obs_shape:
                                        concat_share_obs[key] = np.concatenate((concat_share_obs[key],padded_obs[key]))
                                        concat_obs[key] = np.concatenate((concat_obs[key],padded_obs[key]))
                                all_available_actions.append(available_actions)
                                all_rnn_states_actor.append(rnn_states_actor)
                                all_rnn_states_critic.append(rnn_states_critic)

                    n_stacked_threads = concat_obs['agent_class_identifier'].shape[0] # total groups with at least one active agent
                    # print("number of stacked threads are: ", n_stacked_threads)
                    if self.use_action_masking:
                        all_available_actions = np.array(all_available_actions)
                    else:
                        all_available_actions = None
                    all_rnn_states_actor = np.array(all_rnn_states_actor)
                    all_rnn_states_critic = np.array(all_rnn_states_critic)

                    for key in obs_shape:
                        concat_share_obs[key] = np.concatenate(concat_share_obs[key])
                        concat_obs[key] = np.concatenate(concat_obs[key])
                    # print("shape of concat obs is then: ", concat_obs['global_agent_map'].shape)
                    values, actions, action_log_probs, short_term_goal, rnn_states_actor, rnn_states_critic \
                            = get_short_term_goal(self, concat_obs, concat_share_obs,
                                                all_rnn_states_actor, all_rnn_states_critic, n_stacked_threads, all_available_actions)
                    # print("shape of short term goal is: ", short_term_goal.shape)
                    counter = 0 # used to move through the concatenated actions resulting from concatenated obs
                    for e in range(n_threads):
                        if len(active_threads[e]) <= 1:
                            continue
                        for group in connected_agent_groups[e]:
                            agent_num = 0
                            any_active = False
                            for agent_id in group:
                                if agent_id in active_threads[e][1:]:
                                    any_active = True
                                    self.macro_action[e, agent_id] = short_term_goal[counter, agent_num] + self.agent_pos[e, agent_id]
                                    self.macro_action[e, agent_id] = self.correct_ma_bounds(self.macro_action[e, agent_id])
                                    returned_values[e, agent_id] = values[counter, agent_num]
                                    returned_actions[e, agent_id] = actions[counter, agent_num,:]
                                    returned_action_log_probs[e, agent_id] = action_log_probs[counter, agent_num,:]
                                    returned_rnn_states_actor[e, agent_id] = rnn_states_actor[counter, agent_num]
                                    returned_rnn_states_critic[e, agent_id] = rnn_states_critic[counter, agent_num]
                                    agent_num += 1
                            if any_active:
                                counter += 1
                    # print("counter is ", counter)
            else: # synchronous training
                if self.use_centralized_training: # used for centralized and synchronous training
                    for key in obs_shape:
                        concat_share_obs[key] = np.concatenate(self.buffer.share_obs[key][step])
                    for key in obs_shape:
                        concat_obs[key] = np.concatenate(self.buffer.obs[key][step])
                    
                    returned_values, returned_actions, returned_action_log_probs, short_term_goal, returned_rnn_states_actor, returned_rnn_states_critic \
                            = get_short_term_goal(self, concat_obs, concat_share_obs,
                                                self.buffer.rnn_states[step], self.buffer.rnn_states_critic[step], n_threads, self.available_actions)
                    short_term_goal = short_term_goal.squeeze() + self.agent_pos
                    self.macro_action = self.correct_ma_bounds(short_term_goal)
                else: # used for distribued and synchronous training
                    for e in range(self.n_rollout_threads):
                        for group in connected_agent_groups[e]:
                            L = len(group)
                            # print("number of agents in the group are ", len(group))
                            padded_obs = {}
                            for key in obs_shape:
                                padded_obs[key] = np.zeros_like(self.buffer.obs[key][step][e])
                                padded_obs[key] = np.expand_dims(padded_obs[key], axis=0)
                            if self.use_action_masking:
                                available_actions = np.ones_like(self.available_actions[0])
                            else:
                                available_actions = None
                            rnn_states_actor = np.zeros_like(self.buffer.rnn_states[0][0])
                            rnn_states_critic = np.zeros_like(self.buffer.rnn_states_critic[0][0])
                            agent_num = 0
                            for agent_id in group:
                                for key in obs_shape:
                                    padded_obs[key][0,agent_num] = self.buffer.obs[key][step,e,agent_id]
                                if self.use_action_masking:
                                    available_actions[agent_num] = self.available_actions[e][agent_id]
                                rnn_states_actor[agent_num] = self.buffer.rnn_states[step,e,agent_id]
                                rnn_states_critic[agent_num] = self.buffer.rnn_states_critic[step,e,agent_id]
                                agent_num += 1
                            if not concat_obs:
                                for key in obs_shape:
                                    concat_share_obs[key] = padded_obs[key]
                                    concat_obs[key] = padded_obs[key]
                            else:
                                for key in obs_shape:
                                    concat_share_obs[key] = np.concatenate((concat_share_obs[key],padded_obs[key]))
                                    concat_obs[key] = np.concatenate((concat_obs[key],padded_obs[key]))
                            all_available_actions.append(available_actions)
                            all_rnn_states_actor.append(rnn_states_actor)
                            all_rnn_states_critic.append(rnn_states_critic)
                    n_stacked_threads = concat_obs['agent_class_identifier'].shape[0]
                    if self.use_action_masking:
                        all_available_actions = np.array(all_available_actions)
                    else:
                        all_available_actions = None
                    all_rnn_states_actor = np.array(all_rnn_states_actor)
                    all_rnn_states_critic = np.array(all_rnn_states_critic)
                    # print("shape of available actions is: ", all_available_actions.shape)
                    # print("shape of all rnn states actor is: ", all_rnn_states_actor.shape)
                    # print("number of stacked threads are: ", n_stacked_threads)
                    for key in obs_shape:
                        concat_share_obs[key] = np.concatenate(concat_share_obs[key])
                        concat_obs[key] = np.concatenate(concat_obs[key])
                    values, actions, action_log_probs, short_term_goal, rnn_states_actor, rnn_states_critic \
                            = get_short_term_goal(self, concat_obs, concat_share_obs,
                                                all_rnn_states_actor, all_rnn_states_critic,  n_stacked_threads, all_available_actions)
                    
                    counter = 0 # used to move through the concatenated actions resulting from concatenated obs
                    for e in range(self.n_rollout_threads):
                        for group in connected_agent_groups[e]:
                            # group_goals = short_term_goal[counter,:,:]
                            # group_values = values[counter,:,:]
                            # group_actions = actions[counter,:,:]
                            # group_action_log_probs = action_log_probs[counter,:,:]
                            agent_num = 0
                            for agent_id in group:
                                self.macro_action[e, agent_id] = short_term_goal[counter,agent_num] + self.agent_pos[e, agent_id]
                                self.macro_action[e, agent_id] = self.correct_ma_bounds(self.macro_action[e, agent_id])
                                returned_values[e, agent_id] = values[counter,agent_num]
                                returned_actions[e, agent_id] = actions[counter,agent_num,:]
                                returned_action_log_probs[e, agent_id] = action_log_probs[counter,agent_num,:]
                                returned_rnn_states_actor[e, agent_id] = rnn_states_actor[counter, agent_num]
                                returned_rnn_states_critic[e, agent_id] = rnn_states_critic[counter, agent_num]
                                agent_num += 1
                            counter += 1

        # concat_share_obs = np.concatenate(self.buffer.share_obs[step])
        # concat_obs = np.concatenate(self.buffer.obs[step])
        # print("shape of concat obs is: ", concat_obs.shape)
        return returned_values, returned_actions, returned_action_log_probs, returned_rnn_states_actor, returned_rnn_states_critic
    
    def eval_compute_global_goal(self, step, infos, use_ft):
        if self.use_render:
            n_threads = self.n_rollout_threads
        elif self.use_eval:
            n_threads = self.n_eval_rollout_threads
        returned_actions = np.zeros((n_threads, self.num_agents, self.action_shape), dtype=np.float32)
        connected_agent_groups = []
        for e in range(n_threads):
            connected_agent_groups.append(infos[e]['connected_agent_groups'])
        if self._rnn_uses_tuple_state:
            returned_rnn_states_actor = None
            returned_rnn_states_critic = (
                np.zeros((n_threads, self.num_agents, self.recurrent_hidden_size), dtype=np.float32),
                np.zeros((n_threads, self.num_agents, self.recurrent_hidden_size), dtype=np.float32)
            )
        else:
            returned_rnn_states_actor = None
            returned_rnn_states_critic = np.zeros((n_threads, self.num_agents, self.recurrent_hidden_size), dtype=np.float32)
        
        # function to return short-term goals and rnn_states (optional) using the trained policy
        def get_short_term_goal(self, concat_obs, concat_share_obs, rnn_states_actor, rnn_states_critic, n_threads, available_actions = None):
            if self.use_render:
                action, rnn_states_actor, rnn_states_critic = self.trainer.policy.act(
                    concat_share_obs,
                    concat_obs,
                    np.concatenate(self.masks),
                    rnn_states_actor,
                    rnn_states_critic,
                    available_actions = available_actions,
                    deterministic=True
                )
                # self.rnn_states = np.array(np.split(_t2n(rnn_states), n_threads))

            elif self.use_eval:
                action, rnn_states_actor, rnn_states_critic = self.trainer.policy.act(
                    concat_share_obs,
                    concat_obs,
                    np.concatenate(self.eval_masks),
                    rnn_states_actor,
                    rnn_states_critic,
                    available_actions = available_actions,
                    deterministic=True
                )
                # self.eval_rnn_states = np.array(np.split(_t2n(rnn_states), n_threads))
            
            
            rnn_states_actor = self._rnn_states_to_numpy(rnn_states_actor)
            rnn_states_critic = self._rnn_states_to_numpy(rnn_states_critic)
            goal = np.array(np.split(_t2n(action), n_threads)).astype(np.int32)
            macro_action = self._goal_to_macro_action(goal)
            return goal, macro_action, rnn_states_actor, rnn_states_critic
    
        if use_ft:
            if self.use_render:
                self.ft_short_term_goals = self.envs.ft_get_short_term_goals(self.all_args, mode=self.all_args.algorithm_name[3:])
            elif self.use_eval:
                self.ft_short_term_goals = self.eval_envs.ft_get_short_term_goals(self.all_args, mode=self.all_args.algorithm_name[3:])
            # Used to render for ft methods.
            # if (not self.asynch):
            #     self.macro_action = np.array([
            #         [
            #             # (x, y) ---> (y, x) in minigrid
            #             (goal[1] - self.agent_view_size, goal[0] - self.agent_view_size)
            #             for goal in env_goals
            #         ]
            #         for env_goals in self.ft_short_term_goals
            #     ])
            # else:
            #     short_term_goals = [
            #         [
            #             # (x, y) ---> (y, x) in minigrid
            #             (goal[1] - self.agent_view_size, goal[0] - self.agent_view_size)
            #             for goal in env_goals
            #         ]
            #         for env_goals in self.ft_short_term_goals
            #     ]
            #     self.macro_action[:,:,:2] = (short_term_goals * self.asynch_control.active.reshape(self.n_rollout_threads, self.num_agents, 1)).astype(int) \
            #         + (self.macro_action[:,:,:2] * (1-self.asynch_control.active.reshape(self.n_rollout_threads, self.num_agents, 1))).astype(int)
            if (not self.asynch):
                self.macro_action = np.array([
                    [
                        # (x, y) ---> (y, x) in minigrid
                        (goal[1] - self.agent_view_size, goal[0] - self.agent_view_size) if goal is not None else self.macro_action[e_i, g_i, :2]
                        for g_i, goal in enumerate(env_goals)
                    ]
                    for e_i, env_goals in enumerate(self.ft_short_term_goals)
                ])
            else:
                short_term_goals = [
                    [
                        # (x, y) ---> (y, x) in minigrid
                        (goal[1] - self.agent_view_size, goal[0] - self.agent_view_size) if goal is not None else self.macro_action[e_i, g_i, :2]
                        for g_i, goal in enumerate(env_goals)
                    ]
                    for e_i, env_goals in enumerate(self.ft_short_term_goals)
                ]
                self.macro_action[:,:,:2] = (short_term_goals * self.asynch_control.active.reshape(self.n_rollout_threads, self.num_agents, 1)).astype(int) \
                    + (self.macro_action[:,:,:2] * (1-self.asynch_control.active.reshape(self.n_rollout_threads, self.num_agents, 1))).astype(int)
            
            self.macro_action[self.macro_action>=self.map_size]=self.map_size-1
            self.macro_action[self.macro_action<0]=0
        
        else:
            self.trainer.prep_rollout()

            concat_obs = {}
            concat_share_obs = {}
            all_available_actions = []
            all_rnn_states_actor = []
            all_rnn_states_critic = []

            if self.use_render:
                obs = self.obs
                # old_rnn_states_actor = self.rnn_states_actor
                old_rnn_states_critic = self.rnn_states_critic
            elif self.use_eval:
                obs = self.eval_obs
                # old_rnn_states_actor = self.eval_rnn_states_actor
                old_rnn_states_critic = self.eval_rnn_states_critic

            if not self.use_graph_attention_eval:
                active_threads = self.asynch_control.active_agents_threads()
                # Each agent finds its macro-action individually by padding observations
                for e in range(n_threads):
                    if not active_threads[e] or len(active_threads[e]) == 1:
                        continue
                    
                    for agent_id in active_threads[e][1:]:
                        padded_obs = {}
                        for key in obs.keys():
                            if key == 'target_positions' or key == 'hazard_positions':
                                padded_obs[key] = np.empty_like(obs[key][e], dtype=object)
                                padded_obs[key] = np.expand_dims(padded_obs[key], axis=0)
                            else:
                                padded_obs[key] = np.zeros_like(obs[key][e])
                                padded_obs[key] = np.expand_dims(padded_obs[key], axis=0)

                        available_actions = np.zeros_like(self.available_actions[0])

                        if self._rnn_uses_tuple_state:
                                rnn_states_critic = (
                                    np.zeros_like(old_rnn_states_critic[0][0]),
                                    np.zeros_like(old_rnn_states_critic[1][0])
                                )
                        else:
                            # rnn_states_actor = np.zeros_like(old_rnn_states_actor[0])
                            rnn_states_critic = np.zeros_like(old_rnn_states_critic[0])

                        # Fill agent data
                        for key in obs.keys():
                            padded_obs[key][0,0] = obs[key][e,agent_id]
                        available_actions[0] = self.available_actions[e][agent_id]
                        if self._rnn_uses_tuple_state:
                            rnn_states_critic[0][0] = old_rnn_states_critic[0][e, agent_id]
                            rnn_states_critic[1][0] = old_rnn_states_critic[1][e, agent_id]
                        else:
                            # rnn_states_actor[0] = old_rnn_states_actor[e,agent_id]
                            rnn_states_critic[0] = old_rnn_states_critic[e,agent_id]
                        
                        if not concat_obs:
                            for key in obs.keys():
                                concat_share_obs[key] = padded_obs[key]
                                concat_obs[key] = padded_obs[key]
                        else:
                            for key in obs.keys():
                                concat_share_obs[key] = np.concatenate((concat_share_obs[key],padded_obs[key]))
                                concat_obs[key] = np.concatenate((concat_obs[key],padded_obs[key]))
                        
                        all_available_actions.append(available_actions)
                        all_rnn_states_critic.append(rnn_states_critic)


                n_stacked_threads = concat_obs['agent_class_identifier'].shape[0]
                # print("Active agents are : ", active_threads)
                # print("number of stacked threads are: ", n_stacked_threads)
                if self.use_action_masking:
                    all_available_actions = np.array(all_available_actions)
                else:
                    all_available_actions = None
                    
                if not all_rnn_states_actor:
                    all_rnn_states_actor = None
                else:
                    all_rnn_states_actor = np.array(all_rnn_states_actor)
                
                if not all_rnn_states_critic:
                    all_rnn_states_critic = None
                else:
                    all_rnn_states_critic = np.array(all_rnn_states_critic)

                actions, macro_action, rnn_states_actor, rnn_states_critic = get_short_term_goal(self, concat_obs, concat_share_obs,
                                              all_rnn_states_actor, all_rnn_states_critic, n_stacked_threads, all_available_actions)
                
                counter = 0 # used to move through the concatenated actions resulting from concatenated obs
                for e in range(n_threads):
                    if not active_threads[e] or len(active_threads[e]) == 1:
                        continue
                    for agent_id in active_threads[e][1:]:
                        # print("Agent ", agent_id, " in env ", e, " selected macro action: ", macro_action[counter, 0])
                        if self.algorithm_name == "amat":
                            ### ego-relative goals based on 360 degree view around the agent
                            self.macro_action[e, agent_id] = macro_action[counter, 0] + self.agent_pos[e, agent_id]
                            self.macro_action[e, agent_id] = self.correct_ma_bounds(self.macro_action[e, agent_id])
                        elif self.algorithm_name == "macmat":
                            ### front camera view and possibility of primary actions
                            if macro_action[counter][0][0][1] >= 2*self.action_size+1: # A primary action is chosen instead of a navigation goal
                                primary_action_id = macro_action[counter][0][0][0] # row
                                self.macro_action[e, agent_id] = np.array([-1, -1, primary_action_id])
                            else: # a navigation goal is chosen
                                relative_goal = macro_action[counter][0][0][:2] - [self.action_size, 2*self.action_size] # because agent is always positioned at the bottom center of its local view
                                # (0,0) is where the agent is, and it can go from -3 to 3 right or left and from 0 to -6 forward
                                # The navigation coordinate depends on the direction the agent is facing
                                if self.agent_dir[e, agent_id] == 0: # agent is facing right
                                    navigation_goal = self.agent_pos[e, agent_id] + [-relative_goal[1], relative_goal[0]]
                                elif self.agent_dir[e, agent_id] == 1: # agent is facing down
                                    navigation_goal = self.agent_pos[e, agent_id] + [-relative_goal[0], -relative_goal[1]]
                                elif self.agent_dir[e, agent_id] == 2: # agent is facing left
                                    navigation_goal = self.agent_pos[e, agent_id] + [relative_goal[1], -relative_goal[0]]
                                elif self.agent_dir[e, agent_id] == 3: # agent is facing up
                                    navigation_goal = self.agent_pos[e, agent_id] + [relative_goal[0], relative_goal[1]]
                                navigation_goal = self.correct_ma_bounds(navigation_goal)
                                self.macro_action[e, agent_id] = np.array([navigation_goal[0], navigation_goal[1], -1])
                                
                        returned_actions[e, agent_id] = actions[counter][0]
                        if rnn_states_actor is not None:
                            if self._rnn_uses_tuple_state:
                                returned_rnn_states_actor[0][e,agent_id] = rnn_states_actor[0][counter][0]
                                returned_rnn_states_actor[1][e,agent_id] = rnn_states_actor[1][counter][0]
                            else:
                                returned_rnn_states_actor[e, agent_id] = rnn_states_actor[counter][0]
                        if rnn_states_critic is not None:
                            if self._rnn_uses_tuple_state:
                                returned_rnn_states_critic[0][e, agent_id] = rnn_states_critic[0][counter][0]
                                returned_rnn_states_critic[1][e, agent_id] = rnn_states_critic[1][counter][0]  
                            else:
                                returned_rnn_states_critic[e, agent_id] = rnn_states_critic[counter][0]
                        counter += 1


                return returned_actions, returned_rnn_states_actor, returned_rnn_states_critic

            else: # normal: graph attention is used
            
            
                # # Ensure target_positions and hazard_positions have object dtype for eval
                # obs['target_positions'] = np.array(obs['target_positions'], dtype=object)
                # if self.spawn_hazards:
                #     obs['hazard_positions'] = np.array(obs['hazard_positions'], dtype=object)

                if self.asynch:
                    if self.use_full_comm:
                        active_threads = self.asynch_control.active_agents_threads()
                        # print("shape of available actions in eval compute is: ", available_actions.shape)
                        # print("active_threads are ", active_threads)
                        for e in range(n_threads):
                            if len(active_threads[e]) <= 1:
                                continue
                            padded_obs = {}
                            for key in obs.keys():
                                if key == 'target_positions' or key == 'hazard_positions':
                                    padded_obs[key] = np.empty_like(obs[key][e], dtype=object)
                                    padded_obs[key] = np.expand_dims(padded_obs[key], axis=0)
                                else:
                                    padded_obs[key] = np.zeros_like(obs[key][e])
                                    padded_obs[key] = np.expand_dims(padded_obs[key], axis=0)
                            if self.use_action_masking:
                                available_actions = np.zeros_like(self.available_actions[0])
                            else:
                                available_actions = None
                            if self._rnn_uses_tuple_state:
                                rnn_states_critic = (
                                    np.zeros_like(old_rnn_states_critic[0][0]),
                                    np.zeros_like(old_rnn_states_critic[1][0])
                                )
                            else:
                                # rnn_states_actor = np.zeros_like(old_rnn_states_actor[0])
                                rnn_states_critic = np.zeros_like(old_rnn_states_critic[0])

                            active_cnt = 0
                            inactive_cnt = 0
                            for agent_id in range(self.num_agents):
                                if agent_id in active_threads[e][1:]:
                                    for key in obs.keys():
                                        padded_obs[key][0,active_cnt] = obs[key][e][agent_id]
                                    
                                    if self.use_action_masking:
                                        available_actions[active_cnt] = self.available_actions[e][agent_id]
                                    if self._rnn_uses_tuple_state:
                                        rnn_states_critic[0][active_cnt] = old_rnn_states_critic[0][e][agent_id]
                                        rnn_states_critic[1][active_cnt] = old_rnn_states_critic[1][e][agent_id]
                                    else:
                                        # rnn_states_actor[active_cnt] = old_rnn_states_actor[e,agent_id]
                                        rnn_states_critic[active_cnt] = old_rnn_states_critic[e][agent_id]
                                    active_cnt += 1
                                else:
                                    for key in obs.keys():
                                        padded_obs[key][0,-1-inactive_cnt] = obs[key][e][agent_id]
                                    if self._rnn_uses_tuple_state:
                                        rnn_states_critic[0][-1-inactive_cnt] = old_rnn_states_critic[0][e][agent_id]
                                        rnn_states_critic[1][-1-inactive_cnt] = old_rnn_states_critic[1][e][agent_id]
                                    else:
                                        # rnn_states_actor[-1-inactive_cnt] = old_rnn_states_actor[e,agent_id]
                                        rnn_states_critic[-1-inactive_cnt] = old_rnn_states_critic[e][agent_id]
                                    inactive_cnt += 1
                                
                            if not concat_obs:
                                for key in obs.keys():
                                    concat_share_obs[key] = padded_obs[key]
                                    concat_obs[key] = padded_obs[key]
                            else:
                                for key in obs.keys():
                                    concat_share_obs[key] = np.concatenate((concat_share_obs[key],padded_obs[key]))
                                    concat_obs[key] = np.concatenate((concat_obs[key],padded_obs[key]))
                            all_available_actions.append(available_actions)
                            # all_rnn_states_actor.append(rnn_states_actor)
                            all_rnn_states_critic.append(rnn_states_critic)
                        n_stacked_threads = concat_obs['agent_class_identifier'].shape[0]
                        if self.use_action_masking:
                            all_available_actions = np.array(all_available_actions)
                        else:
                            all_available_actions = None
                        
                        if not all_rnn_states_actor:
                            all_rnn_states_actor = None
                        else:
                            all_rnn_states_actor = np.array(all_rnn_states_actor)
                        
                        if not all_rnn_states_critic:
                            all_rnn_states_critic = None
                        else:
                            all_rnn_states_critic = np.array(all_rnn_states_critic)

                        # for key in obs.keys():
                        #     concat_share_obs[key] = np.concatenate(concat_share_obs[key])
                        #     concat_obs[key] = np.concatenate(concat_obs[key])
                        actions, macro_action, rnn_states_actor, rnn_states_critic = get_short_term_goal(self, concat_obs, concat_share_obs, all_rnn_states_actor, all_rnn_states_critic, n_stacked_threads, all_available_actions)
                        counter = 0 # used to move through the concatenated actions resulting from concatenated obs
                        for e in range(n_threads):
                            if len(active_threads[e]) <= 1:
                                continue
                            agent_num = 0
                            for agent_id in active_threads[e][1:]:
                                if self.algorithm_name == "amat":
                                    ### ego-relative goals based on 360 degree view around the agent
                                    self.macro_action[e, agent_id] = macro_action[counter, agent_num] + self.agent_pos[e, agent_id]
                                    self.macro_action[e, agent_id] = self.correct_ma_bounds(self.macro_action[e, agent_id])
                                else:
                                    if self.algorithm_name == "macmat":
                                        ### front camera view and possibility of primary actions
                                        if macro_action[counter, agent_num][0][1] >= 2*self.action_size+1: # A primary action is chosen instead of a navigation goal
                                            primary_action_id = macro_action[counter, agent_num][0][1] - (2*self.action_size+1) + macro_action[counter, agent_num][0][0] #col - (2*self.action_size+1) + row
                                            self.macro_action[e, agent_id] = np.array([-1, -1, primary_action_id])
                                        else: # a navigation goal is chosen
                                            relative_goal = macro_action[counter, agent_num][0][:2] - [self.action_size, 2*self.action_size] # because agent is always positioned at the bottom center of its local view
                                            # The navigation coordinate depends on the direction the agent is facing
                                            if self.agent_dir[e, agent_id] == 0: # agent is facing right
                                                navigation_goal = self.agent_pos[e, agent_id] + [-relative_goal[1], relative_goal[0]]
                                            elif self.agent_dir[e, agent_id] == 1: # agent is facing down
                                                navigation_goal = self.agent_pos[e, agent_id] + [-relative_goal[0], -relative_goal[1]]
                                            elif self.agent_dir[e, agent_id] == 2: # agent is facing left
                                                navigation_goal = self.agent_pos[e, agent_id] + [relative_goal[1], -relative_goal[0]]
                                            elif self.agent_dir[e, agent_id] == 3: # agent is facing up
                                                navigation_goal = self.agent_pos[e, agent_id] + [relative_goal[0], relative_goal[1]]
                                            navigation_goal = self.correct_ma_bounds(navigation_goal)
                                            self.macro_action[e, agent_id] = np.array([navigation_goal[0], navigation_goal[1], -1])
                                
                                returned_actions[e, agent_id] = actions[counter, agent_num,:]
                                if rnn_states_actor is not None:
                                    if self._rnn_uses_tuple_state:
                                        returned_rnn_states_actor[0][e,agent_id] = rnn_states_actor[0][counter, agent_num]
                                        returned_rnn_states_actor[1][e,agent_id] = rnn_states_actor[1][counter, agent_num]
                                    else:
                                        returned_rnn_states_actor[e, agent_id] = rnn_states_actor[counter, agent_num]
                                if rnn_states_critic is not None:
                                    if self._rnn_uses_tuple_state:
                                        returned_rnn_states_critic[0][e, agent_id] = rnn_states_critic[0][counter, agent_num]
                                        returned_rnn_states_critic[1][e, agent_id] = rnn_states_critic[1][counter, agent_num]
                                    else:
                                        returned_rnn_states_critic[e, agent_id] = rnn_states_critic[counter, agent_num]
                                agent_num += 1
                            counter += 1
                        
                    elif self.use_partial_comm: 
                        # in each group, puts active agents first, then inactive agents, then zero-paddings
                        active_threads = self.asynch_control.active_agents_threads()
                        for e in range(n_threads):
                            if len(active_threads[e]) <= 1:
                                    continue
                            for group in connected_agent_groups[e]:
                                
                                padded_obs = {}
                                for key in obs.keys():
                                    if key == 'target_positions' or key == 'hazard_positions':
                                        padded_obs[key] = np.empty_like(obs[key][e], dtype=object)
                                        padded_obs[key] = np.expand_dims(padded_obs[key], axis=0)
                                    else:
                                        padded_obs[key] = np.zeros_like(obs[key][e])
                                        padded_obs[key] = np.expand_dims(padded_obs[key], axis=0)
                                available_actions = np.zeros_like(self.available_actions[0])
                                if self._rnn_uses_tuple_state:
                                    rnn_states_critic = (
                                        np.zeros_like(old_rnn_states_critic[0][0]),
                                        np.zeros_like(old_rnn_states_critic[1][0])
                                    )
                                else:
                                    # rnn_states_actor = np.zeros_like(old_rnn_states_actor[0])
                                    rnn_states_critic = np.zeros_like(old_rnn_states_critic[0])
                                active_in_group = []
                                inactive_in_group = []
                                for agent_id in group:
                                    if agent_id in active_threads[e][1:]:
                                        active_in_group.append(agent_id)
                                    else:
                                        inactive_in_group.append(agent_id)
                                if len(active_in_group) == 0:
                                    continue
                                else:
                                    agent_num = 0
                                    for agent_id in active_in_group:
                                        for key in obs.keys():
                                            padded_obs[key][0,agent_num] = obs[key][e,agent_id]
                                        available_actions[agent_num] = self.available_actions[e][agent_id]
                                        if self._rnn_uses_tuple_state:
                                            rnn_states_critic[0][agent_num] = old_rnn_states_critic[0][e,agent_id]
                                            rnn_states_critic[1][agent_num] = old_rnn_states_critic[1][e,agent_id]
                                        else:
                                            # rnn_states_actor[agent_num] = old_rnn_states_actor[e,agent_id]
                                            rnn_states_critic[agent_num] = old_rnn_states_critic[e,agent_id]
                                        agent_num += 1
                                    for agent_id in inactive_in_group:
                                        for key in obs.keys():
                                            padded_obs[key][0,agent_num] = obs[key][e,agent_id]
                                        if self._rnn_uses_tuple_state:
                                            rnn_states_critic[0][agent_num] = old_rnn_states_critic[0][e,agent_id]
                                            rnn_states_critic[1][agent_num] = old_rnn_states_critic[1][e,agent_id]
                                        else:
                                            # rnn_states_actor[agent_num] = old_rnn_states_actor[e,agent_id]
                                            rnn_states_critic[agent_num] = old_rnn_states_critic[e,agent_id]
                                        agent_num += 1 
                                if not concat_obs:
                                    for key in obs.keys():
                                        concat_share_obs[key] = padded_obs[key]
                                        concat_obs[key] = padded_obs[key]
                                else:
                                    for key in obs.keys():
                                        concat_share_obs[key] = np.concatenate((concat_share_obs[key],padded_obs[key]))
                                        concat_obs[key] = np.concatenate((concat_obs[key],padded_obs[key]))
                                all_available_actions.append(available_actions)
                                # all_rnn_states_actor.append(rnn_states_actor)
                                all_rnn_states_critic.append(rnn_states_critic)

                        n_stacked_threads = concat_obs['agent_class_identifier'].shape[0]
                        # print("Active agents are : ", active_threads)
                        # print("number of stacked threads are: ", n_stacked_threads)
                        # for key in obs.keys():
                        #     concat_share_obs[key] = np.concatenate(concat_share_obs[key])
                        #     concat_obs[key] = np.concatenate(concat_obs[key])
                        
                        if self.use_action_masking:
                            all_available_actions = np.array(all_available_actions)
                        else:
                            all_available_actions = None

                        if not all_rnn_states_actor:
                            all_rnn_states_actor = None
                        else:
                            all_rnn_states_actor = np.array(all_rnn_states_actor)
                        
                        if not all_rnn_states_critic:
                            all_rnn_states_critic = None
                        else:
                            all_rnn_states_critic = np.array(all_rnn_states_critic)

                        actions, macro_action, rnn_states_actor, rnn_states_critic = get_short_term_goal(self, concat_obs, concat_share_obs, all_rnn_states_actor, all_rnn_states_critic, n_stacked_threads, all_available_actions)
                        counter = 0 # used to move through the concatenated actions resulting from concatenated obs
                        for e in range(n_threads):
                            if len(active_threads[e]) <= 1:
                                continue
                            for group in connected_agent_groups[e]:
                                active_in_group = []
                                inactive_in_group = []
                                for agent_id in group:
                                    if agent_id in active_threads[e][1:]:
                                        active_in_group.append(agent_id)
                                    else:
                                        inactive_in_group.append(agent_id)
                                if len(active_in_group) == 0:
                                    continue
                                else:
                                    group_goals = macro_action[counter,:,:]
                                    agent_num = 0
                                    for agent_id in active_in_group: # only active agents get new macro_actions
                                        if self.algorithm_name == "amat":
                                            ### 360 degree vision and ego-relative goals
                                            self.macro_action[e, agent_id] = group_goals[agent_num,:] + self.agent_pos[e, agent_id]
                                            self.macro_action[e, agent_id] = self.correct_ma_bounds(self.macro_action[e, agent_id])
                                        elif self.algorithm_name == "macmat":
                                            ### these lines are for front camera view and possibility of primary actions
                                            if group_goals[agent_num, :][0][1] >= 2*self.action_size+1: # A primary action is chosen instead of a navigation goal
                                                primary_action_id = group_goals[agent_num, :][0][0] # row
                                                self.macro_action[e, agent_id] = np.array([-1, -1, primary_action_id])
                                            else: # a navigation goal is chosen
                                                relative_goal = group_goals[agent_num, :][0][:2] - [self.action_size, 2*self.action_size] # because agent is always positioned at the bottom center of its local view
                                                # The navigation coordinate depends on the direction the agent is facing
                                                if self.agent_dir[e, agent_id] == 0: # agent is facing right
                                                    navigation_goal = self.agent_pos[e, agent_id] + [-relative_goal[1], relative_goal[0]]
                                                elif self.agent_dir[e, agent_id] == 1: # agent is facing down
                                                    navigation_goal = self.agent_pos[e, agent_id] + [-relative_goal[0], -relative_goal[1]]
                                                elif self.agent_dir[e, agent_id] == 2: # agent is facing left
                                                    navigation_goal = self.agent_pos[e, agent_id] + [relative_goal[1], -relative_goal[0]]
                                                elif self.agent_dir[e, agent_id] == 3: # agent is facing up
                                                    navigation_goal = self.agent_pos[e, agent_id] + [relative_goal[0], relative_goal[1]]
                                                navigation_goal = self.correct_ma_bounds(navigation_goal)
                                                self.macro_action[e, agent_id] = np.array([navigation_goal[0], navigation_goal[1], -1])



                                        returned_actions[e, agent_id] = actions[counter, agent_num,:]
                                        if rnn_states_actor is not None:
                                            if self._rnn_uses_tuple_state:
                                                returned_rnn_states_actor[0][e, agent_id] = rnn_states_actor[0][counter, agent_num]
                                                returned_rnn_states_actor[1][e, agent_id] = rnn_states_actor[1][counter, agent_num]
                                            else:
                                                returned_rnn_states_actor[e, agent_id] = rnn_states_actor[counter, agent_num]
                                        if rnn_states_critic is not None:
                                            if self._rnn_uses_tuple_state:
                                                returned_rnn_states_critic[0][e, agent_id] = rnn_states_critic[0][counter, agent_num]
                                                returned_rnn_states_critic[1][e, agent_id] = rnn_states_critic[1][counter, agent_num]
                                            else:
                                                returned_rnn_states_critic[e, agent_id] = rnn_states_critic[counter, agent_num]
                                        agent_num += 1
                                    counter += 1
                else: # synchronous (not supported right now)
                    if self.use_full_comm:
                        for key in obs.keys():
                            concat_obs[key] = np.concatenate(obs[key])
                            concat_share_obs[key] = np.concatenate(obs[key])
                        # print("shape of concat obs is: ", concat_obs['global_agent_map'].shape)
                        available_actions = np.array(self.available_actions)
                        short_term_goal, returned_rnn_states_actor, returned_rnn_states_critic = get_short_term_goal(self, concat_obs, concat_share_obs, old_rnn_states_actor, old_rnn_states_critic, n_threads, available_actions)
                        short_term_goal = short_term_goal.squeeze() + self.agent_pos
                        # print("short term goals are ", short_term_goal)
                        self.macro_action = self.correct_ma_bounds(short_term_goal)
                        # print("macro actions are ", self.macro_action)

                    elif self.use_partial_comm:
                        for e in range(n_threads):
                            for group in connected_agent_groups[e]:
                                L = len(group)
                                padded_obs = {}
                                for key in obs.keys():
                                    padded_obs[key] = np.zeros_like(obs[key][e])
                                    padded_obs[key] = np.expand_dims(padded_obs[key], axis=0)
                                available_actions = np.zeros_like(self.available_actions[0])
                                rnn_states_actor = np.zeros_like(old_rnn_states_actor[0])
                                rnn_states_critic = np.zeros_like(old_rnn_states_critic[0])
                                agent_num = 0
                                for agent_id in group:
                                    for key in obs.keys():
                                        padded_obs[key][0,agent_num] = obs[key][e,agent_id]
                                    available_actions[agent_num] = self.available_actions[e][agent_id]
                                    rnn_states_actor[agent_num] = old_rnn_states_actor[e,agent_id]
                                    rnn_states_critic[agent_num] = old_rnn_states_critic[e,agent_id]
                                    agent_num += 1
                
                                if not concat_obs:
                                    for key in obs.keys():
                                        concat_share_obs[key] = padded_obs[key]
                                        concat_obs[key] = padded_obs[key]
                                else:
                                    for key in obs.keys():
                                        concat_share_obs[key] = np.concatenate((concat_share_obs[key],padded_obs[key]))
                                        concat_obs[key] = np.concatenate((concat_obs[key],padded_obs[key]))
                                all_available_actions.append(available_actions)
                                all_rnn_states_actor.append(rnn_states_actor)
                                all_rnn_states_critic.append(rnn_states_critic)
                        n_stacked_threads = concat_obs['agent_class_identifier'].shape[0]
                        for key in obs.keys():
                            concat_share_obs[key] = np.concatenate(concat_share_obs[key])
                            concat_obs[key] = np.concatenate(concat_obs[key])
                        all_available_actions = np.array(all_available_actions)
                        all_rnn_states_actor= np.array(all_rnn_states_actor)
                        all_rnn_states_critic = np.array(all_rnn_states_critic)
                        short_term_goal, rnn_states_actor, rnn_states_critic = get_short_term_goal(self, concat_obs, concat_share_obs, all_rnn_states_actor, all_rnn_states_critic, n_stacked_threads, all_available_actions)
                        counter = 0 # used to move through the concatenated groups
                        for e in range(n_threads):
                            for group in connected_agent_groups[e]:
                                group_goals = short_term_goal[counter,:,:]
                                agent_num = 0
                                for agent_id in group:
                                    self.macro_action[e, agent_id] = group_goals[agent_num,:] + self.agent_pos[e, agent_id]
                                    self.macro_action[e, agent_id] = self.correct_ma_bounds(self.macro_action[e, agent_id])
                                    if rnn_states_actor is not None:
                                        returned_rnn_states_actor[e, agent_id] = rnn_states_actor[counter, agent_num]
                                    if rnn_states_critic is not None:
                                        returned_rnn_states_critic[e, agent_id] = rnn_states_critic[counter, agent_num]
                                    agent_num += 1
                                counter += 1
            

        return returned_actions, returned_rnn_states_actor, returned_rnn_states_critic
        
    @torch.no_grad()
    def compute(self, next_values):
        self.trainer.prep_rollout()
        
        """
        # This is migrated to the run() function
        # we have to find each agent (e,a) last active timstep, and concatenate the obs from that step over all the agents        
        
        concat_obs = {}
        concat_share_obs = {}
        
        if self.asynch:
            for e in range(self.n_rollout_threads):
                for a in range(self.num_agents):
                    last_index = self.buffer.active_steps[self.buffer.update_step[e,a], e, a]
                    if not concat_obs:
                        for key in self.buffer.obs.keys():
                            concat_share_obs[key] = self.buffer.all_obs[key][last_index, e]
                            concat_obs[key] = self.buffer.all_obs[key][last_index, e]
                    else:
                        for key in self.buffer.obs.keys():
                            concat_share_obs[key] = np.concatenate((concat_share_obs[key],self.buffer.all_obs[key][last_index, e]))
                            concat_obs[key] = np.concatenate((concat_obs[key],self.buffer.all_obs[key][last_index, e]))
            # The resulting number of threads would be n_rollout_threads*num_agents
            for key in self.buffer.share_obs.keys(): 
                concat_share_obs[key] = np.squeeze(concat_share_obs[key])
                concat_obs[key] = np.squeeze(concat_obs[key])
                concat_share_obs[key] = np.concatenate(concat_share_obs[key])
                concat_obs[key] = np.concatenate(concat_obs[key])
            # print("shape of concat_obs['global_agent_map'] is now: ", concat_obs['global_agent_map'].shape)
            next_values = self.trainer.policy.get_values(concat_share_obs,
                                                        concat_obs,
                                                        np.concatenate(
                                                            self.buffer.rnn_states_critic[-1]),
                                                        np.concatenate(self.buffer.masks[-1]))
            next_values = np.array(np.split(_t2n(next_values), self.n_rollout_threads*self.num_agents))
            # print("shape of next_values is: ", next_values.shape)
            counter = 0
            actual_next_values = np.zeros((self.n_rollout_threads, self.num_agents), dtype=np.float32)
            for e in range(self.n_rollout_threads):
                for a in range(self.num_agents):
                    actual_next_values[e, a] = next_values[counter][a]
                    counter += 1
            # print("actual_next_values are: ", actual_next_values)
            
            self.buffer.async_compute_returns(actual_next_values, self.trainer.value_normalizer)
        else:
            #concat is done to merge threads and agents
            for key in self.buffer.obs.keys():
                concat_obs[key] = np.concatenate(self.buffer.obs[key][-1])
            for key in self.buffer.share_obs.keys():
                concat_share_obs[key] = np.concatenate(self.buffer.share_obs[key][-1])
            # print("shape of concat_obs['global_agent_map'] is: ", concat_obs['global_agent_map'].shape)

            
            # The value of observations when making the final decision of the episode
            next_values = self.trainer.policy.get_values(concat_share_obs,
                                                        concat_obs,
                                                        np.concatenate(
                                                            self.buffer.rnn_states_critic[-1]),
                                                        np.concatenate(self.buffer.masks[-1]))
            next_values = np.array(np.split(_t2n(next_values), self.n_rollout_threads)) 
            
            # print("next values are: ", next_values)
            # print("shape of next_values is: ", next_values.shape)
            self.buffer.compute_returns(next_values, self.trainer.value_normalizer) """
        if self.asynch:
            self.buffer.async_compute_returns(next_values, self.trainer.value_normalizer)
        else:
            self.buffer.compute_returns(next_values, self.trainer.value_normalizer)

    def insert(self, data, step, active_agents=None):
        dict_obs, rewards, dones, infos, values, actions, action_log_probs, rnn_states, rnn_states_critic, agent_groups, available_actions = data
        # print("dones are: ", dones) 
        dones_env = np.all(dones, axis=-1)
        # print("dones_env are: ", dones_env)
         
        # rnn_states[dones_env == True] = np.zeros(
        #     ((dones_env == True).sum(), self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
        # rnn_states_critic[dones_env == True] = np.zeros(((dones_env == True).sum(
        # ), self.num_agents, *self.buffer.rnn_states_critic.shape[3:]), dtype=np.float32)
        
        masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        masks[dones_env == True] = np.zeros(
            ((dones_env == True).sum(), self.num_agents, 1), dtype=np.float32)

        # print("current step is ", step)
        # print("masks in insert are: ", masks)
        if active_agents is None:
            obs = self._convert(dict_obs, infos, step)
        else:
            obs = self._convert(dict_obs, infos, step, active_agents)
        self.obs = obs
        share_obs = obs.copy()

        # print("shape of available actions is: ", available_actions.shape)              

        if active_agents is None:
            self.buffer.insert(share_obs, obs, rnn_states, rnn_states_critic, actions, action_log_probs, values, rewards, masks, agent_groups, available_actions=available_actions)
        else:
            self.buffer.async_insert(share_obs, obs, rnn_states, rnn_states_critic, actions, action_log_probs, values, rewards, masks, agent_groups, available_actions=available_actions, active_agents=active_agents)

    @torch.no_grad()
    def eval(self):
        if self.extended_delays:
            rest_time = 50
            min_wait = 0
            max_wait = 60
        else:
            rest_time = self.all_args.max_ma_duration
            min_wait = self.all_args.async_min_wait
            max_wait = self.all_args.async_max_wait
        if self.asynch:
            def generate_random_period(min_t,max_t):
                if self.extended_delays:
                    return np.random.randint(min_t,max_t)
                else: # by default, no artificial delays in decision-making
                    return 0
            self.asynch_control = AsynchControl(num_envs=self.n_eval_rollout_threads, num_agents=self.num_agents,
                                                limit=self.episode_length, random_fn=generate_random_period, min_wait=min_wait, max_wait=max_wait, rest_time = rest_time)
        eval_envs = self.eval_envs
        self.eval_env_infos = defaultdict(list)
        use_ft = self.all_args.algorithm_name[:2] == "ft"


        # # Logging 
        # # Temporarily commenting this to evaluate the MixedNCP-NoAttn and LSTM-NoAttn models
        # if self.use_graph_attention_eval:
        #     if self.use_full_comm:
        #         comm_status = "full_comm"
        #     elif self.all_args.comm_radius == 0:
        #         comm_status = "no_comm"
        #     else:
        #         comm_status = "comm_sigma_" + str(self.all_args.comm_sigma)
        # else:
        #     if self.use_full_comm:
        #         comm_status = "full_comm_no_attn"
        #     elif self.all_args.comm_radius == 0:
        #         comm_status = "no_comm"
        #     else:
        #         comm_status = "comm_sigma_" + str(self.all_args.comm_sigma) + "_no_attn"
        if self.use_full_comm:
            comm_status = "full_comm"
        elif self.all_args.comm_radius == 0:
            comm_status = "no_comm"
        else:
            comm_status = "comm_sigma_" + str(self.all_args.comm_sigma)



        if self.all_args.dynamic_targets:
            if self.all_args.target_behavior == "default":
                target_status = "dynamicT"
            elif self.all_args.target_behavior == "evadeT":
                target_status = "evadeT"
            elif self.all_args.target_behavior == "hyperT":
                target_status = "hyperT"
        else:
            target_status = "staticT"
        
        if self.all_args.dynamic_obstacles:
            obstacle_status = "dynamicO" if self.all_args.obstacle_behavior == "random" else "oscillateO"
        else:
            obstacle_status = "staticO"

        if not self.use_localization:
            if self.use_perception_noise:
                if self.use_perception_noise == 'rescuers_only':
                    noise_status = f"perception_noise_rescuers_range{self.perception_noise_base_value}"
                else:
                    noise_status = f"perception_noise_base{self.perception_noise_base_value}_distance_factor{self.perception_noise_distance_factor}_no_localization"
            elif self.use_localization_noise:
                noise_status = f"no_perception_noise_localization_noise_position{self.position_drift_rate}_orientation{self.orientation_drift_rate}"
            else:
                noise_status = "no_localization"
        else:
            if self.use_perception_noise:
                if self.use_perception_noise == 'rescuers_only':
                    if self.use_localization_noise:
                        noise_status = f"perception_noise_rescuers_range{self.perception_noise_base_value}_localization_noise_position{self.position_drift_rate}_orientation{self.orientation_drift_rate}"
                    else:
                        noise_status = f"perception_noise_rescuers_range{self.perception_noise_base_value}_no_localization_noise"
                else:
                    if self.use_localization_noise:
                        noise_status = f"perception_noise_base{self.perception_noise_base_value}_distance_factor{self.perception_noise_distance_factor}_localization_noise_position{self.position_drift_rate}_orientation{self.orientation_drift_rate}"
                    else:
                        noise_status = f"perception_noise_base{self.perception_noise_base_value}_distance_factor{self.perception_noise_distance_factor}_no_localization_noise"
            elif self.use_localization_noise:
                noise_status = f"no_perception_noise_localization_noise_position{self.position_drift_rate}_orientation{self.orientation_drift_rate}"
            else:
                noise_status = "no_noise"
        
        if self.algorithm_name == "macmat":
            if self.use_rnn:
                model_type = self.rnn_type
            else:
                model_type = "NoRNN"
            if not self.use_graph_attention:
                model_type += "_no_attn"
            if not self.use_classbased_action:
                model_type += "_no_class"
        else:
            model_type = self.algorithm_name
        
        # Agent Composition string
        if self.all_args.n_agent_types == 2:
            num_rescuers = 0
            num_scouts = 0
            num_randoms = 0
            for class_id in self.all_args.agent_classes_list:
                if class_id == 0:
                    num_rescuers += 1
                elif class_id == 1:
                    num_scouts += 1
                elif class_id == 10:
                    num_randoms += 1
                else:
                    raise ValueError("Unknown agent class id found in agent_classes_list.")
            if num_randoms > 0:
                agent_composition = f"{num_rescuers}r{num_scouts}s{num_randoms}rand"
            else:
                agent_composition = f"{num_rescuers}r{num_scouts}s"
        elif self.all_args.n_agent_types == 3:
            pass

        if self.extended_delays:
            delay_status = "-extended_delays"
        else:
            delay_status = ""
        
        run_name = f"{self.map_size}x{self.map_size}-{model_type}-{agent_composition}-{comm_status}-{target_status}-{obstacle_status}-{noise_status}{delay_status}"
        print(f"Eval Run Name: {run_name}")

        
        # Visualization of the results
        mission_completion_data = []
        target_rescue_data = []
        eval_start_time = time.time()
        eval_inference_time = []
        rewards_data = []

        for episode in range(self.all_args.eval_episodes):
            self.init_eval_env_info()
            self.init_eval_map_variables()
            self.timespan_list = []
            reset_choose = np.ones(self.n_eval_rollout_threads) == 1.0
            self.eval_done_envs = np.zeros((self.n_eval_rollout_threads,)).astype(bool)
            eval_dict_obs, eval_infos = eval_envs.reset(reset_choose)

            for e in range(self.n_eval_rollout_threads):
                self.agent_pos[e] = eval_infos[e]['agent_pos'] #initial agent positions
                self.agent_dir[e] = eval_infos[e]['agent_direction']
                self.agent_classes_list[e] = np.array(eval_infos[e]['agent_classes_list'])
                self.agent_local_views[e] = eval_infos[e]['agent_local_views']
                self.local_obstacles[e] = eval_infos[e]['agent_local_obstacles']
                self.num_targets[e] = eval_infos[e]['num_targets']
                if self.algorithm_name == "amat":
                    self.occupied_each_map[e] = eval_infos[e]['occupied_each_map']

            # Initialize RNN states
            if self._rnn_uses_tuple_state:
                # self.eval_rnn_states_actor = (
                #     np.zeros((self.n_eval_rollout_threads, self.num_agents, self.recurrent_hidden_size),dtype=np.float32),
                #     np.zeros((self.n_eval_rollout_threads, self.num_agents, self.recurrent_hidden_size),dtype=np.float32)
                # )
                self.eval_rnn_states_critic = (
                    np.zeros((self.n_eval_rollout_threads, self.num_agents, self.recurrent_hidden_size),dtype=np.float32),
                    np.zeros((self.n_eval_rollout_threads, self.num_agents, self.recurrent_hidden_size),dtype=np.float32)
                )
            else:
                # self.rnn_states_actor = np.zeros((self.n_rollout_threads, self.num_agents, self.recurrent_hidden_size),dtype=np.float32)
                self.eval_rnn_states_critic = np.zeros((self.n_eval_rollout_threads, self.num_agents, self.recurrent_hidden_size),dtype=np.float32)
            self.eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)
            is_last_step = np.full((self.n_eval_rollout_threads, self.num_agents), False)

            # one-hot agent_class_identifier
            self.agent_class_identifier = np.zeros((self.n_eval_rollout_threads, self.num_agents, self.n_agent_types), dtype=np.float32)
            for e in range(self.n_eval_rollout_threads):
                for agent_id in range(self.num_agents):
                    for agent_type in range(self.n_agent_types):
                        if self.agent_classes_list[e, agent_id] == agent_type:
                            self.agent_class_identifier[e, agent_id, agent_type] = 1

            if self.asynch:
                self.asynch_control.reset()
                active_agents = self.asynch_control.active_agents()
            else:
                active_agents = None
            
            if self.use_action_masking:
                available_actions = self.get_available_actions()
                self.available_actions[0] = available_actions[0]
            else:
                available_actions = None
                self.available_actions = None

            active_mask = (self.asynch_control.active == 1)      
            eval_obs = self._convert(eval_dict_obs, eval_infos, 0, active_agents)
            self.eval_obs = eval_obs
            actions, eval_rnn_states_actor, eval_rnn_states_critic = self.eval_compute_global_goal(0, eval_infos, use_ft)



            self.eval_rnn_states_actor = eval_rnn_states_actor
            self.eval_rnn_states_critic = eval_rnn_states_critic

            eval_full_episode_rewards = np.zeros((self.n_eval_rollout_threads, self.num_agents), dtype=np.float32)
                        
            for step in range(self.max_steps):

                local_step = step % self.local_step_num

                if use_ft:
                    eval_actions_env = eval_envs.ft_get_short_term_actions(self.macro_action)
                else:
                    eval_actions_env = eval_envs.get_short_term_action(self.macro_action)
                eval_actions_env = np.array(eval_actions_env)

                eval_dict_obs, eval_rewards, eval_dones, eval_infos = eval_envs.step(eval_actions_env)
    
                # Adding information to their respective dictionary keys.
                for e in range(self.n_eval_rollout_threads):
                    env_done, eval_rewards[e] = self.update_eval_env_info(e, eval_infos[e], eval_rewards[e], step)
                    if env_done:
                        eval_dones[e] = True
                    self.agent_pos[e] = eval_infos[e]['agent_pos']
                    self.agent_dir[e] = eval_infos[e]['agent_direction']
                    self.agent_alive[e] = eval_infos[e]['agent_alive'] #agent alive status
                    self.agent_count[e] = eval_infos[e]['agent_count'] #agent counts
                    self.agent_local_views[e] = eval_infos[e]['agent_local_views']
                    self.local_obstacles[e] = eval_infos[e]['agent_local_obstacles']
                    if self.algorithm_name == "amat":
                        self.occupied_each_map[e] = eval_infos[e]['occupied_each_map']

                if self.asynch:
                    self.asynch_control.step()
                    for e in range(self.n_eval_rollout_threads):
                        for a in range(self.num_agents):
                            if self.agent_alive[e][a] == 0:
                                if self.prev_agent_alive[e][a] == 1:
                                    self.asynch_control.activate(e, a)
                                    is_last_step[e,a] = True
                                else:
                                    self.asynch_control.standby[e, a] = 0
                                    self.asynch_control.active[e, a] = 0
                                    is_last_step[e,a] = True
                                    continue
                            if self.eval_env_info["gt_target_obj_set"][e] == set() and not is_last_step[e,a]:
                                    is_last_step[e,a] = True
                                    # self.asynch_control.standby[e, a] = 1
                                    self.asynch_control.activate(e, a)
                            if self.agent_classes_list[e, a] == 2 and eval_infos[e]['agent_cleaning_hazard'][a] is not None:
                                # If the agent is a cleaner and cleaning a hazard, don't end macro-action and stay there
                                continue
                            if not self.asynch_control.active[e, a] and not self.asynch_control.standby[e, a]:
                                # Checks if agent has reached its short term goal / stopped and puts it on standby if so
                                #TODO: If the agent has reached its ultimate objective, instead of putting it on standby, it should be completely deactivated
                                # Under the assumption of full communication in training, this is added to ensure the final MA of all agents finish at the same time
                                if eval_actions_env[e, a] == 3: # If agent stops
                                    # self.asynch_control.standby[e, a] = 1
                                    self.asynch_control.activate(e, a) # Simply activate during eval
                                    

                                elif not use_ft and self.algorithm_name == "macmat" and self.macro_action[e, a][2] in {0, 1, 2}:  # If the macro-action is to turn left or right or toggles
                                    # self.asynch_control.standby[e, a] = 1
                                    self.asynch_control.activate(e, a) # Simply activate during eval
                                    self.macro_action[e,a] = [self.agent_pos[e,a][0], self.agent_pos[e,a][1], -1] # Rewriting the macro action to take the stop action
                                    # if self.asynch_control.wait[e, a] <= 0:
                                    #     self.asynch_control.activate(e, a)
                                    
                                if (eval_infos[e]['n_targets_found'][a] > self.n_targets_found[e,a]) and self.use_localization:
                                    # print(f"agent {a} knew {self.n_targets_found[e,a]} targets and now knows {infos[e]['n_targets_found'][a]} targets.")
                                    self.n_targets_found[e,a] = eval_infos[e]['n_targets_found'][a]
                                    # self.asynch_control.standby[e, a] = 1
                                    self.asynch_control.activate(e, a)

                    # if not self.extended_delays:             
                    #     if np.any(self.asynch_control.standby) and np.any(self.asynch_control.active): #Activates on-standby agents based on communication model
                    #         for thread in self.asynch_control.active_agents_threads():
                    #             if len(thread) > 1:
                    #                 if self.use_partial_comm:
                    #                     connected_agents = eval_infos[thread[0]]['connected_agent_groups']
                    #                     # print("connected agent groups are" , connected_agents)
                    #                     for agent_id in thread[1:]:
                    #                         for group in connected_agents:
                    #                             if agent_id in group:
                    #                                 for i in group:
                    #                                     if self.asynch_control.standby[thread[0], i]:
                    #                                     # if self.asynch_control.standby[thread[0], i] and self.asynch_control.wait[thread[0], i] <= generate_random_period(2, 4):
                    #                                         self.asynch_control.activate(thread[0], i)
                    #                 elif self.use_full_comm:
                    #                     for i in range(self.num_agents):
                    #                         if self.asynch_control.standby[thread[0], i]:
                    #                         # if self.asynch_control.standby[thread[0], i] and self.asynch_control.wait[thread[0], i] <= generate_random_period(2, 4):
                    #                             self.asynch_control.activate(thread[0], i)
                else:
                    for e in range(self.n_rollout_threads):
                        for a in range(self.num_agents):
                            if self.agent_alive[e][a] == 0:
                                is_last_step[e,a] = True
                                continue
                        if self.eval_env_info["gt_target_obj_set"][e] == set():
                            is_last_step[e,:] = True
                            # rewards[e, a, 0] = 0
                
                eval_full_episode_rewards += eval_rewards[:,:,0]
                
                if (not self.asynch and local_step == self.local_step_num - 1) or (self.asynch and np.any(self.asynch_control.active)):
                    if self.asynch:
                        active_agents = self.asynch_control.active_agents()
                    else:
                        active_agents = None
                    eval_obs = self._convert(eval_dict_obs, eval_infos, step, active_agents)
                    self.eval_obs = eval_obs

                    eval_dones_env = np.all(eval_dones, axis=-1)
                    # self.eval_rnn_states[eval_dones_env == True] = np.zeros(
                    #     ((eval_dones_env == True).sum(), self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
                    self.eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)
                    self.eval_masks[eval_dones_env == True] = np.zeros(
                        ((eval_dones_env == True).sum(), self.num_agents, 1), dtype=np.float32)
                    
                    if self.use_action_masking:
                        available_actions = self.get_available_actions()
                        self.available_actions[0] = available_actions[0]
                    else:
                        available_actions = None
                        self.available_actions = None

                    active_mask = (self.asynch_control.active == 1)                  
                    inference_start_time = time.time()
                    actions, eval_rnn_states_actor, eval_rnn_states_critic = self.eval_compute_global_goal(step, eval_infos, use_ft)
                    inference_end_time = time.time()
                    eval_inference_time.append(inference_end_time - inference_start_time)



                    if not self.asynch:
                        self.eval_rnn_states_actor = eval_rnn_states_actor
                        self.eval_rnn_states_critic = eval_rnn_states_critic
                    else:
                        for (e, a, s) in self.asynch_control.active_agents():
                            if eval_rnn_states_actor is not None:
                                if self._rnn_uses_tuple_state:
                                    self.eval_rnn_states_actor[0][e, a] = eval_rnn_states_actor[0][e, a]
                                    self.eval_rnn_states_actor[1][e, a] = eval_rnn_states_actor[1][e, a]
                                else:
                                    self.eval_rnn_states_actor[e, a] = eval_rnn_states_actor[e, a]
                            if eval_rnn_states_critic is not None:
                                if self._rnn_uses_tuple_state:
                                    self.eval_rnn_states_critic[0][e, a] = eval_rnn_states_critic[0][e, a]
                                    self.eval_rnn_states_critic[1][e, a] = eval_rnn_states_critic[1][e, a]
                                else:
                                    self.eval_rnn_states_critic[e, a] = eval_rnn_states_critic[e, a]
                            # print("e a s are: ", e, a, s)
                self.prev_agent_alive = np.copy(self.agent_alive)
                if is_last_step.all():
                    break    
            
            completion_time_data = np.where(np.isnan(self.eval_env_info['mean_mission_completed_step']), -1, self.eval_env_info['mean_mission_completed_step'])
            target_rescue_times = list(eval_infos[0]['target_rescue_times']) # this only works with one eval env thread
            if len(target_rescue_times) < self.num_targets[0]:
                # Some targets were never rescued, fill with -1
                num_missing = self.num_targets[0] - len(target_rescue_times)
                for _ in range(num_missing):
                    target_rescue_times.append(-1)
            print("completion time data is: ", completion_time_data)
            print("target rescue times are: ", np.array(target_rescue_times))
            target_rescue_data.append(target_rescue_times)
            mission_completion_data.extend(completion_time_data)
            print("episode reward is:", eval_full_episode_rewards)
            rewards_data.append(eval_full_episode_rewards)
            self.convert_eval_info()
        # print("completion info is ", mission_completion_data)
        
        eval_end_time = time.time()
        
        
        # Save data to file
        
        
        # Visualize the results
        success_counts_array = np.zeros(self.max_steps + 1)
        success_counts = 0
        sum_steps_for_success = 0
        for value in mission_completion_data:
            if value != -1:
                success_counts_array[int(value)] += 1
                success_counts += 1
                sum_steps_for_success += int(value)
                    
        print("Success Rate is: ", success_counts / len(mission_completion_data))
        cumulative_success_counts = np.cumsum(success_counts_array)
        total_episodes = len(mission_completion_data)
        success_rate_array = cumulative_success_counts / total_episodes
        # average_steps_to_success measure:
        average_steps_to_success = sum_steps_for_success / success_counts if success_counts > 0 else float('inf')
        print("Average Steps to Success is: ", average_steps_to_success)
        # average_timesteps_per_episode measure:
        sum_steps = sum_steps_for_success + (total_episodes - success_counts) * self.max_steps
        average_timesteps_per_episode = sum_steps / total_episodes if total_episodes > 0 else float('inf')
        print("Average Timesteps per Episode is: ", average_timesteps_per_episode)
        print(f"Algorithm was: {self.algorithm_name}, RNN type was: {self.rnn_type}, Map size was: {self.map_size}x{self.map_size}, Communication: {comm_status}, Targets: {target_status}, Obstacles: {obstacle_status}, Noise: {noise_status}, Agent Composition: {agent_composition}")
                
        # Eval time
        print("Total Evaluation time: ", eval_end_time - eval_start_time)
        print("Average Inference time per decision step: ", np.mean(eval_inference_time))
        # Save mission_completion data, timespan data, evaluation inference time data to a single file
        eval_data_dict = {
            'mission_completion_data': mission_completion_data,
            'target_rescue_data': target_rescue_data,
            'timespan_data': self.timespan_list,
            'eval_inference_time': eval_inference_time,
            'total_eval_time': eval_end_time - eval_start_time,
            'rewards_data': rewards_data

        }
        eval_data_dir = os.path.join(str(self.run_dir), 'eval_data')
        os.makedirs(eval_data_dir, exist_ok=True)
        eval_data_path = os.path.join(eval_data_dir, f'{agent_composition}-{self.map_size}x{self.map_size}map-{target_status}-{obstacle_status}-{comm_status}-{noise_status}-{model_type}{delay_status}-data.npy')
        np.save(eval_data_path, eval_data_dict)
        print("Saved eval data to", eval_data_path)

        # # Plot the cumulative success probability
        # plt.figure(figsize=(10, 6))
        # plt.plot(range(self.max_steps + 1), success_rate_array, label='Cumulative Success Probability')
        # plt.xlabel('Time Step')
        # plt.ylabel('Probability')
        # plt.title('Probability of Mission Success Before Time Step t')
        # plt.legend()
        # plt.show()

        # Visualize the distribution of timespan between actions (delta t)
        timespan_array = np.array(self.timespan_list)
        mean_timespan = np.mean(timespan_array)
        std_timespan = np.std(timespan_array)

        # # Plot the bell curve on top of the histogram
        # plt.figure(figsize=(10, 6))
        # count, bins, ignored = plt.hist(timespan_array, bins=30, density=True, alpha=0.6, color='g', label='Timespan Distribution')
        # plt.xlabel('Timespan between Actions (delta t)')
        # plt.ylabel('Density')
        # plt.title(f'Distribution of Timespan between Actions for the {model_type} Model')
        # # Plotting the normal distribution curve
        # from scipy.stats import norm
        # mu, sigma = mean_timespan, std_timespan
        # best_fit_line = norm.pdf(bins, mu, sigma)
        # plt.plot(bins, best_fit_line, 'r--', label='Normal Distribution Fit')
        # plt.legend()
        # plt.show()
        
    @torch.no_grad()
    def render(self):
        if self.extended_delays:
            rest_time = 50
            min_wait = 0
            max_wait = 60
        else:   
            rest_time = self.all_args.max_ma_duration
            min_wait = self.all_args.async_min_wait
            max_wait = self.all_args.async_max_wait
        if self.asynch:
            def generate_random_period(min_t,max_t):
                if self.extended_delays:
                    return np.random.randint(min_t,max_t)
                else:
                    return 0
            self.asynch_control = AsynchControl(num_envs=self.n_rollout_threads, num_agents=self.num_agents,
                                                limit=self.episode_length, random_fn=generate_random_period, min_wait=min_wait, max_wait=max_wait, rest_time = rest_time)
        
        envs = self.envs
        # Init env infos.
        # self.eval_infos = defaultdict(list)
        use_ft = self.all_args.algorithm_name[:2] == "ft"
        all_frames = []
        all_local_frames = []

        for episode in range(self.all_args.render_episodes):
            self.init_env_info()
            self.init_map_variables()
            reset_choose = np.ones(self.n_rollout_threads) == 1.0
            self.done_envs = np.zeros((self.n_rollout_threads,)).astype(bool)
            dict_obs, infos = envs.reset(reset_choose)
            for e in range(self.n_rollout_threads):
                self.agent_pos[e] = infos[e]['agent_pos'] #initial agent positions
                self.agent_dir[e] = infos[e]['agent_direction']
                self.agent_classes_list[e] = np.array(infos[e]['agent_classes_list'])
                self.agent_local_views[e] = infos[e]['agent_local_views']
                self.local_obstacles[e] = infos[e]['agent_local_obstacles']
                self.num_targets[e] = infos[e]['num_targets']
                if self.algorithm_name == "amat":
                    self.occupied_each_map[e] = infos[e]['occupied_each_map']
            print("Agent types are: ", self.agent_classes_list)

            if self._rnn_uses_tuple_state:
                # self.rnn_states_actor = (
                #     np.zeros((self.n_rollout_threads, self.num_agents, self.recurrent_hidden_size),dtype=np.float32),
                #     np.zeros((self.n_rollout_threads, self.num_agents, self.recurrent_hidden_size),dtype=np.float32)
                # )
                self.rnn_states_critic = (
                    np.zeros((self.n_rollout_threads, self.num_agents, self.recurrent_hidden_size),dtype=np.float32),
                    np.zeros((self.n_rollout_threads, self.num_agents, self.recurrent_hidden_size),dtype=np.float32)
                )
            else:
                # self.rnn_states_actor = np.zeros((self.n_rollout_threads, self.num_agents, self.recurrent_hidden_size),dtype=np.float32)
                self.rnn_states_critic = np.zeros((self.n_rollout_threads, self.num_agents, self.recurrent_hidden_size),dtype=np.float32)
            self.masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
            is_last_step = np.full((self.n_rollout_threads, self.num_agents), False)
            
            # one-hot agent_class_identifier
            self.agent_class_identifier = np.zeros((self.n_rollout_threads, self.num_agents, self.n_agent_types), dtype=np.int32)
            for e in range(self.n_rollout_threads):
                for agent_id in range(self.num_agents):
                    for agent_type in range(self.n_agent_types):
                        if self.agent_classes_list[e, agent_id] == agent_type:
                            self.agent_class_identifier[e, agent_id, agent_type] = 1

                
            if self.asynch:
                self.asynch_control.reset()
                active_agents=self.asynch_control.active_agents()

            if self.use_action_masking:
                available_actions = self.get_available_actions()
                self.available_actions[0] = available_actions[0]
            else:
                available_actions = None
                self.available_actions = None

            if self.asynch:
                active_agents=self.asynch_control.active_agents()
            else:
                active_agents = None

            obs = self._convert(dict_obs, infos, 0, active_agents)
            self.obs = obs
            actions, rnn_states_actor, rnn_states_critic = self.eval_compute_global_goal(0, infos, use_ft)

            
            self.rnn_states_actor = rnn_states_actor
            self.rnn_states_critic = rnn_states_critic
            period_rewards = np.zeros((self.n_rollout_threads, self.num_agents, 1))
            
            # if self.use_render: #and episode == 0:
            #     if self.all_args.save_gifs:
            #         # if self.env_info["target_rescued"][e] == 1:
            #         #     pass
            #         # else:
            #         image, local_image = envs.render('rgb_array', self.macro_action)[0]
            #         all_frames.append(image)
            #         all_local_frames.append(local_image)
            #         calc_end = time.time()
            #         elapsed = calc_end - calc_start
            #         if elapsed < self.all_args.ifi:
            #             time.sleep(self.all_args.ifi - elapsed)
            #     else:
            #         # print(self.macro_action)
            #         envs.render('human', self.macro_action)
            # time.sleep(5)

            full_episode_rewards = np.zeros((self.n_rollout_threads, self.num_agents), dtype=np.float32)
            
            self.current_episode = episode 
            for step in range(self.max_steps):
                calc_start = time.time()
                local_step = step % self.local_step_num                    
                # print(f"Macro actions to be taken are: {self.macro_action}")
                # print("target rescue times are: ", infos[e]["target_rescue_times"])
                if use_ft:
                    actions_env = envs.ft_get_short_term_actions(self.macro_action)
                else:
                    actions_env = envs.get_short_term_action(self.macro_action)
                actions_env = np.array(actions_env)

                dict_obs, rewards, dones, infos = envs.step(actions_env)
                # print("connected agents are: ", infos[0]['connected_agent_groups'])
                # Adding information to their respective dictionary keys.
                for e in range(self.n_rollout_threads):
                    env_done, rewards[e] = self.update_env_info(e, infos[e], rewards[e], step)
                    if env_done:
                        dones[e] = True
                    self.agent_pos[e] = infos[e]['agent_pos']
                    self.agent_dir[e] = infos[e]['agent_direction']
                    self.agent_alive[e] = infos[e]['agent_alive'] #agent alive status
                    self.agent_local_views[e] = infos[e]['agent_local_views']
                    self.local_obstacles[e] = infos[e]['agent_local_obstacles']
                    if self.algorithm_name == "amat":
                        self.occupied_each_map[e] = infos[e]['occupied_each_map']
                    # print(f"At step {step}, agents alive are: {self.agent_alive[e]}")
                
                if self.asynch:
                    self.asynch_control.step()
                    # print(f"Active agents, standby agents and the rest times at step {step} are: {self.asynch_control.active}, {self.asynch_control.standby} and {self.asynch_control.rest}")
                    for e in range(self.n_rollout_threads):
                        for a in range(self.num_agents):
                            if (self.env_info["gt_target_obj_set"][e] == set() or step == self.max_steps - 1 or np.all(dones[e]) == True):
                                # is_last_step[e,a] = True
                                # self.asynch_control.standby[e, a] = 1
                                self.asynch_control.activate(e, a)
                                # print(f"agent {a} is activated because last step is reached.")
                            elif self.agent_alive[e][a] == 0:
                                self.asynch_control.active[e, a] = 0
                                self.asynch_control.standby[e, a] = 0
                                continue

                            if self.agent_classes_list[e, a] == 2 and infos[e]['agent_cleaning_hazard'][a] is not None:
                                # If the agent is a cleaner and cleaning a hazard, don't end macro-action and stay there
                                # print(f"agent {a} is a cleaner and cleaning a hazard, so it continues its macro-action.")
                                self.asynch_control.active[e, a] = 0
                                self.asynch_control.standby[e, a] = 0
                                continue
                            if not self.asynch_control.active[e, a] and not self.asynch_control.standby[e, a]:
                                # Checks if agent has reached its short term goal / stopped and puts it on standby if so
                                #TODO: If the agent has reached its ultimate objective, instead of putting it on standby, it should be completely deactivated
                                # Under the assumption of full communication in training, this is added to ensure the final MA of all agents finish at the same time
                  
                                if actions_env[e, a] == 3: # If agent stops
                                    # self.asynch_control.standby[e, a] = 1
                                    # if self.asynch_control.wait[e, a] <= 0:
                                    #     self.asynch_control.activate(e, a)
                                    self.asynch_control.activate(e, a) # Simply activate during eval
                                
                                elif not use_ft and self.algorithm_name == "macmat" and self.macro_action[e, a][2] in {0, 1, 2}:  # If the macro-action is to turn left or right or toggles
                                    # self.asynch_control.standby[e, a] = 1
                                    self.asynch_control.activate(e, a) # Simply activate during eval
                                    self.macro_action[e,a] = [self.agent_pos[e,a][0], self.agent_pos[e,a][1], -1] # Rewriting the macro action to take the stop action
                                    # if self.asynch_control.wait[e, a] <= 0:
                                    #     self.asynch_control.activate(e, a)

                                    
                                if (infos[e]['n_targets_found'][a] > self.n_targets_found[e,a]) and self.use_localization:
                                    # print(f"agent {a} knew {self.n_targets_found[e,a]} targets and now knows {infos[e]['n_targets_found'][a]} targets.")
                                    self.n_targets_found[e,a] = infos[e]['n_targets_found'][a]
                                    # self.asynch_control.standby[e, a] = 1
                                    # if self.asynch_control.wait[e, a] <= 0:
                                    #     self.asynch_control.activate(e, a)
                                    self.asynch_control.activate(e, a)
                                    # print(f"agent {a} is standby because a new target is found.")
                                

                    # if not self.extended_delays:             
                    #     if np.any(self.asynch_control.standby) and np.any(self.asynch_control.active): #Activates on-standby agents based on communication model
                    #         for thread in self.asynch_control.active_agents_threads():
                    #             if len(thread) > 1:
                    #                 if self.use_partial_comm:
                    #                     connected_agents = infos[thread[0]]['connected_agent_groups']
                    #                     # print("connected agent groups are" , connected_agents)
                    #                     for agent_id in thread[1:]:
                    #                         for group in connected_agents:
                    #                             if agent_id in group:
                    #                                 for i in group:
                    #                                     if self.asynch_control.standby[thread[0], i]:
                    #                                     # if self.asynch_control.standby[thread[0], i] and self.asynch_control.wait[thread[0], i] <= generate_random_period(2, 4):
                    #                                         self.asynch_control.activate(thread[0], i)
                    #                 elif self.use_full_comm:
                    #                     for i in range(self.num_agents):
                    #                         if self.asynch_control.standby[thread[0], i]:
                    #                         # if self.asynch_control.standby[thread[0], i] and self.asynch_control.wait[thread[0], i] <= generate_random_period(2, 4):
                    #                             self.asynch_control.activate(thread[0], i)
                    #                             # print(f"standby agent {i} is activated at step {step} with another agent.")
                    # # print(f"Active agents, standby agents and the rest times at step {step} after manipulation are: {self.asynch_control.active}, {self.asynch_control.standby} and {self.asynch_control.rest}")
                else:
                    for e in range(self.n_rollout_threads):
                        # for a in range(self.num_agents):
                        #     if self.agent_alive[e][a] == 0:
                        #         is_last_step[e,a] = True
                        #         continue
                        if self.env_info["gt_target_obj_set"][e] == set() or step == self.max_steps - 1 or np.all(dones[e]) == True:
                            is_last_step[e,:] = True
                            # rewards[e, a, 0] = 0
                
                # print(f"at step {step}, Active agents are {self.asynch_control.active_agents()} and standby agents are {self.asynch_control.standby_agents()}")
                period_rewards += rewards
                
                if (not self.asynch and local_step == self.local_step_num - 1) or (self.asynch and np.any(self.asynch_control.active)):
                    
                    if self.asynch:
                        for e, a, s in self.asynch_control.active_agents():
                            if period_rewards[e, a, 0] != 0:
                                # if a == 0:
                                # print(f"period reward for agent {a} at step {step} is: ", period_rewards[e, a, 0])
                            # if a == 0:                                # print("agent {}: {}".format(a, period_rewards[e, a, 0]))
                                # print("agent's current pos is ", (infos[e]['current_agent_pos'][a]-self.agent_view_size).T)
                                pass
                            full_episode_rewards[e, a] += period_rewards[e, a, 0]
                            period_rewards[e, a, 0] = 0
                    else:
                        # ic(period_rewards) #for monitoring the rewards
                        full_episode_rewards += period_rewards[:,:,0]
                        period_rewards = np.zeros((self.n_rollout_threads, self.num_agents, 1))
                    
                    if self.asynch:
                        active_agents=self.asynch_control.active_agents()
                    else:
                        active_agents = None
                    obs = self._convert(dict_obs, infos, step, active_agents)
                    self.obs = obs #to be used in eval_compute_global_goal
                    
                    dones_env = np.all(dones, axis=-1)
                    # self.rnn_states[dones_env == True] = np.zeros(
                    #     ((dones_env == True).sum(), self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
                    self.masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
                    self.masks[dones_env == True] = np.zeros(
                        ((dones_env == True).sum(), self.num_agents, 1), dtype=np.float32)
                    
                    
                    if self.use_action_masking:
                        available_actions = self.get_available_actions()
                        # print("available actions are: ", available_actions)
                        self.available_actions[0] = available_actions[0]
                    else:
                        available_actions = None
                        self.available_actions = None
                        
                    actions, rnn_states_actor, rnn_states_critic = self.eval_compute_global_goal(step, infos, use_ft)
                    
                
                    if not self.asynch:
                        self.rnn_states_actor = rnn_states_actor
                        self.rnn_states_critic = rnn_states_critic
                    else:
                        for (e, a, s) in self.asynch_control.active_agents():
                            if rnn_states_actor is not None:
                                if self._rnn_uses_tuple_state:
                                    self.rnn_states_actor[0][e, a] = rnn_states_actor[0][e, a]
                                    self.rnn_states_actor[1][e, a] = rnn_states_actor[1][e, a]
                                else:
                                    self.rnn_states_actor[e, a] = rnn_states_actor[e, a]
                            if rnn_states_critic is not None:
                                if self._rnn_uses_tuple_state:
                                    self.rnn_states_critic[0][e, a] = rnn_states_critic[0][e, a]
                                    self.rnn_states_critic[1][e, a] = rnn_states_critic[1][e, a]
                                else:
                                    self.rnn_states_critic[e, a] = rnn_states_critic[e, a]
                            # print("e a s are: ", e, a, s)
                    # print("updated rnn_states_critics are: ", self.rnn_states_critic)
                    
                    
                if self.use_render: #and episode == 0:
                    if self.all_args.save_gifs:
                        # if self.env_info["target_rescued"][e] == 1:
                        #     pass
                        # else:
                        image, local_image = envs.render('rgb_array', self.macro_action)[0]
                        all_frames.append(image)
                        all_local_frames.append(local_image)
                        calc_end = time.time()
                        elapsed = calc_end - calc_start
                        if elapsed < self.all_args.ifi:
                            time.sleep(self.all_args.ifi - elapsed)
                    else:
                        # print(self.macro_action)
                        envs.render('human', self.macro_action)
                        # pass
                # print("step is: ", step)
                # if (step > 10) and (step % 1) == 0:
                #     time.sleep(5) # pause for debugging 
                # if episode > 0 and step == 0:
                #     time.sleep(5) # pause for debugging for 5 secs
                # if step == 20:
                #     time.sleep(10)
                
                self.prev_agent_alive = np.copy(self.agent_alive)
                
                # if is_last_step.all():
                #     break
                if np.all(dones):
                    print("The env is done at step ", step)
                    break
            
            print("episode reward is: ", full_episode_rewards)
            self.convert_info()
            total_num_steps = (episode + 1) * self.max_steps * self.n_rollout_threads
            if not self.use_render :
                self.log_env(self.env_infos, total_num_steps)
                self.log_agent(self.env_infos, total_num_steps)
            
        for k, v in self.env_infos.items():
            print("eval average {}: {}".format(k, np.nanmean(v) if k == 'merge_explored_ratio_step' else np.mean(v)))

        if self.all_args.save_gifs:
            print("Saving GIFs...")
            imageio.mimsave(str(self.gif_dir) + '/merge.gif',
                            all_frames, duration=self.all_args.ifi)
            imageio.mimsave(str(self.gif_dir) + '/local.gif',
                            all_local_frames, duration=self.all_args.ifi)
            print("Saved GIFs to", str(self.gif_dir))

    