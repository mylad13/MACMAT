import os
import torch
import torch.nn as nn
from torch.nn import functional as F
import math
import numpy as np
from torch.distributions import Categorical
from hetmarl.algorithms.utils.util import check, init
from hetmarl.algorithms.utils.transformer_act import discrete_autoregreesive_act
from hetmarl.algorithms.utils.transformer_act import discrete_parallel_act
from hetmarl.algorithms.utils.transformer_act import continuous_autoregreesive_act
from hetmarl.algorithms.utils.transformer_act import continuous_parallel_act
from hetmarl.algorithms.utils.transformer_act import multidiscrete_autoregreesive_act
from hetmarl.algorithms.utils.transformer_act import multidiscrete_parallel_act
from functools import partial
from hetmarl.algorithms.utils.torchncp import CfCCell, WiredCfCCell
from hetmarl.algorithms.utils.torchncp.wirings import AutoNCP, NCP_No_Motor
from hetmarl.algorithms.utils.set_transformer import SetTransformer

def init_(m, gain=0.01, activate=False):
    if activate:
        gain = nn.init.calculate_gain('relu')
    return init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), gain=gain)


class SelfAttention(nn.Module):

    def __init__(self, n_embd, n_head, masked=False):
        super(SelfAttention, self).__init__()

        assert n_embd % n_head == 0
        self.masked = masked
        self.n_head = n_head
        # key, query, value projections for all heads
        self.key = init_(nn.Linear(n_embd, n_embd))
        self.query = init_(nn.Linear(n_embd, n_embd))
        self.value = init_(nn.Linear(n_embd, n_embd))
        # output projection
        self.proj = init_(nn.Linear(n_embd, n_embd))
        
        # causal mask to ensure that attention is only applied to the left in the input sequence
        # self.register_buffer("mask", torch.tril(torch.ones(n_agent + 1, n_agent + 1))
        #                      .view(1, 1, n_agent + 1, n_agent + 1))
        self.att_bp = None

        self.dropout_rate = 0.1
        # self.dropout = nn.Dropout(self.dropout_rate)

    def forward(self, key, value, query):
        B, L, D = query.size()
        # print("B, L, D is: ", B, L, D)
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(key).view(B, L, self.n_head, D // self.n_head).transpose(1, 2)  # (B, nh, L, hs)
        # print("k shape is: ", k.shape)
        q = self.query(query).view(B, L, self.n_head, D // self.n_head).transpose(1, 2)  # (B, nh, L, hs)
        v = self.value(value).view(B, L, self.n_head, D // self.n_head).transpose(1, 2)  # (B, nh, L, hs)

        # causal attention: (B, nh, L, hs) x (B, nh, hs, L) -> (B, nh, L, L)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))

        # self.att_bp = F.softmax(att, dim=-1)
        
        n_agents = L
        
        
        if self.masked: #adaptive masked attention
            mask = torch.tril(torch.ones(n_agents + 1, n_agents + 1, device=att.device)).view(1, 1, n_agents + 1, n_agents + 1)
            att = att.masked_fill(mask[:, :, :L, :L] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        # att = self.dropout(att)
        self.att_bp = att # for visualization

        y = att @ v  # (B, nh, L, L) x (B, nh, L, hs) -> (B, nh, L, hs)
        y = y.transpose(1, 2).contiguous().view(B, L, D)  # re-assemble all head outputs side by side

        # output projection
        y = self.proj(y)
        # y = self.dropout(y)
        return y



class EncodeBlock(nn.Module):
    """ an unassuming Transformer block """

    def __init__(self, n_embd, n_head):
        super(EncodeBlock, self).__init__()

        self.dropout_rate = 0.1

        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        # self.attn = SelfAttention(n_embd, n_head, n_agent, masked=True)
        self.attn = SelfAttention(n_embd, n_head, masked=False)
        self.mlp = nn.Sequential(
            init_(nn.Linear(n_embd, 1 * n_embd), activate=True),
            nn.GELU(),
            # nn.Dropout(self.dropout_rate),
            init_(nn.Linear(1 * n_embd, n_embd)),
            # nn.Dropout(self.dropout_rate)
        )

    def forward(self, x):
        x = self.ln1(x + self.attn(x, x, x))
        x = self.ln2(x + self.mlp(x))
        return x


class DecodeBlock(nn.Module):
    """ an unassuming Transformer block """

    def __init__(self, n_embd, n_head):
        super(DecodeBlock, self).__init__()

        self.dropout_rate = 0.1
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.ln3 = nn.LayerNorm(n_embd)
        self.attn1 = SelfAttention(n_embd, n_head, masked=True)
        self.attn2 = SelfAttention(n_embd, n_head, masked=True)
        self.mlp = nn.Sequential(
            init_(nn.Linear(n_embd, 1 * n_embd), activate=True),
            nn.GELU(),
            # nn.Dropout(self.dropout_rate),
            init_(nn.Linear(1 * n_embd, n_embd)),
            # nn.Dropout(self.dropout_rate)
        )

    def forward(self, x, rep_enc):
        x = self.ln1(x + self.attn1(x, x, x))
        x = self.ln2(rep_enc + self.attn2(key=x, value=x, query=rep_enc))
        x = self.ln3(x + self.mlp(x))
        return x


class Encoder(nn.Module):

    def __init__(self, state_dim, obs_dim, action_dim, n_block, n_embd, n_hidden, n_hidden_r, n_embd_vit, n_head, n_agent,
                  n_agent_types, max_num_targets, spawn_hazards, max_num_hazards, use_rnn, rnn_type, encode_state, use_classbased_action, use_graph_attention):
        super(Encoder, self).__init__()

        self.state_dim = state_dim
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.n_embd = n_embd
        self.n_agent = n_agent
        self.rnn_type = rnn_type
        self.encode_state = encode_state
        self.n_hidden = n_hidden
        self.n_hidden_r = n_hidden_r
        self.use_rnn = use_rnn
        self.max_num_targets = max_num_targets
        self.n_agent_types = n_agent_types
        self.spawn_hazards = spawn_hazards
        self.max_num_hazards = max_num_hazards
        self.use_classbased_action = use_classbased_action
        self.use_graph_attention = use_graph_attention

        self.dropout_rate = 0.1

        self.relu = nn.ReLU()
        self.gelu = nn.GELU()

            
        local_input_channel = obs_dim['local_agent_view'].shape[0]
        local_input_width = obs_dim['local_agent_view'].shape[1]
        local_input_height = obs_dim['local_agent_view'].shape[2]

        hidden_size1 = n_hidden
        hidden_size2 = n_embd//2

        local_out_channel1 = 32
        local_kernel_size1 = 3
        local_stride1 = 1
        local_padding_size1 = 1

        local_out_channel2 = 64
        local_kernel_size2 = 3
        local_stride2 = 1
        local_padding_size2 = 1

        local_out_channel3 = 64
        local_kernel_size3 = 3
        local_stride3 = 1
        local_padding_size3 = 1
        
        local_output_width1 = (local_input_width-local_kernel_size1+local_padding_size1*2)//local_stride1 + 1
        local_output_height1 = (local_input_height-local_kernel_size1+local_padding_size1*2)//local_stride1 + 1
        local_out_image_size1 = local_output_width1*local_output_height1
        local_output_width2 = (local_output_width1-local_kernel_size2+local_padding_size2*2)//local_stride2 + 1
        local_output_height2 = (local_output_height1-local_kernel_size2+local_padding_size2*2)//local_stride2 + 1
        local_out_image_size2 = local_output_width2*local_output_height2
        local_output_width3 = (local_output_width2-local_kernel_size3+local_padding_size3*2)//local_stride3 + 1
        local_output_height3 = (local_output_height2-local_kernel_size3+local_padding_size3*2)//local_stride3 + 1
        local_out_image_size3 = local_output_width3*local_output_height3

        local_flattened_size1 = local_out_channel1*local_out_image_size1
        local_flattened_size2 = local_out_channel2*local_out_image_size2
        local_flattened_size3 = local_out_channel3*local_out_image_size3
        
        # print("local_flattened_size1 is: ", local_flattened_size1)

        # self.local_view_encoder = nn.Sequential(
        #     init_(nn.Conv2d(in_channels=local_input_channel, out_channels=local_out_channel1,
        #                      kernel_size=local_kernel_size1, stride=local_stride1, padding=local_padding_size1), activate=True), 
        #     nn.LayerNorm([local_out_channel1, local_output_width1, local_output_height1]), self.relu,
        #     init_(nn.Conv2d(in_channels=local_out_channel1, out_channels=local_out_channel2,
        #                      kernel_size=local_kernel_size2, stride=local_stride2, padding=local_padding_size2), activate=True), 
        #     nn.LayerNorm([local_out_channel2, local_output_width2, local_output_height2]), self.relu,
        #     init_(nn.Conv2d(in_channels=local_out_channel2, out_channels=local_out_channel3,
        #                      kernel_size=local_kernel_size3, stride=local_stride3, padding=local_padding_size3), activate=True), 
        #     nn.LayerNorm([local_out_channel3, local_output_width3, local_output_height3]), self.gelu, nn.Flatten(),
        #     init_(nn.Linear(local_flattened_size3, n_embd), activate=True), self.gelu)

        self.local_view_encoder = nn.Sequential(
            init_(nn.Conv2d(in_channels=local_input_channel, out_channels=local_out_channel1,
                             kernel_size=local_kernel_size1, stride=local_stride1, padding=local_padding_size1), activate=True), 
            nn.LayerNorm([local_out_channel1, local_output_width1, local_output_height1]), self.gelu,
            nn.Flatten(),
            init_(nn.Linear(local_flattened_size1, n_embd), activate=True), self.gelu)
        

        self.agent_pose_encoder = nn.Sequential(
            init_(nn.Linear(4, n_embd), activate=True), self.gelu)
        # self.agent_pose_encoder = nn.Sequential(
        #     init_(nn.Linear(4, hidden_size1), activate=True), self.gelu, nn.LayerNorm(hidden_size1),
        #     init_(nn.Linear(hidden_size1, n_embd), activate=True), self.gelu)
        
        
        self.target_positions_encoder = SetTransformer(dim_input=5, num_outputs=1, dim_output=n_embd, dim_hidden=hidden_size1, num_heads=1, ln=True)
        # self.target_positions_encoder = DeepSetBlock(in_dim=5, hidden_dim=hidden_size1, out_dim=n_embd)
        # self.other_agents_positions_encoder = SetTransformer(dim_input=n_agent_types+2, num_outputs=1, dim_output=n_embd, dim_hidden=hidden_size1, num_heads=1, ln=True)
        # self.other_agents_positions_encoder = DeepSetBlock(in_dim=n_agent_types+2, hidden_dim=hidden_size1, out_dim=n_embd)
        
        if self.spawn_hazards:
            self.hazard_positions_encoder = SetTransformer(dim_input=5, num_outputs=1, dim_output=n_embd, dim_hidden=hidden_size1, num_heads=1, ln=True)
            # self.hazard_positions_encoder = DeepSetBlock(in_dim=5, hidden_dim=hidden_size1, out_dim=n_embd)
            # # Option 1: Concatenate the class encoding with observation embeddings
                       
            # self.global_info_encoder = nn.Sequential(
            #     nn.LayerNorm(2*n_embd),
            #     init_(nn.Linear(2*n_embd, hidden_size1), activate=True), self.gelu, nn.LayerNorm(hidden_size1),
            #     init_(nn.Linear(hidden_size1, n_embd), activate=True), self.gelu)
            # self.local_info_encoder = nn.Sequential(
            #     nn.LayerNorm(2*n_embd),
            #     init_(nn.Linear(2*n_embd, hidden_size1), activate=True), self.gelu, nn.LayerNorm(hidden_size1),
            #     init_(nn.Linear(hidden_size1, n_embd), activate=True), self.gelu)
            # self.obs_encoder = nn.Sequential(
            #     nn.LayerNorm(3*n_embd),
            #     init_(nn.Linear(3*n_embd, hidden_size1), activate=True), self.gelu, nn.LayerNorm(hidden_size1),
            #     init_(nn.Linear(hidden_size1, n_embd), activate=True), self.gelu)
            # Option 2: add the class encoding to the observation embeddings
            # self.obs_encoder = nn.Sequential(
            #     nn.LayerNorm(2*n_embd),
            #     init_(nn.Linear(2*n_embd, hidden_size1), activate=True), self.gelu, nn.LayerNorm(hidden_size1),
            #     init_(nn.Linear(hidden_size1, n_embd), activate=True), self.gelu)
            if self.use_classbased_action:
                self.single_obs_encoder = nn.Sequential(
                    nn.LayerNorm(4*n_embd),
                    init_(nn.Linear(4*n_embd, n_embd), activate=True), self.gelu)
            else:
                self.single_obs_encoder = nn.Sequential(
                    nn.LayerNorm(5*n_embd),
                    init_(nn.Linear(5*n_embd, n_embd), activate=True), self.gelu)

        else:
            pass
            if self.use_classbased_action:
                self.single_obs_encoder = nn.Sequential(
                    nn.LayerNorm(3*n_embd),
                    init_(nn.Linear(3*n_embd, n_embd), activate=True), self.gelu)
            else:
                self.single_obs_encoder = nn.Sequential(
                    nn.LayerNorm(4*n_embd),
                    init_(nn.Linear(4*n_embd, n_embd), activate=True), self.gelu)
               

        if self.use_rnn:
            if rnn_type == 'LSTM':
                self.rnn = nn.LSTMCell(input_size=n_embd, hidden_size=n_hidden_r) #experiment with projection
                self.fusion_layer = nn.Sequential(
                    nn.LayerNorm(n_embd+n_hidden_r),
                    init_(nn.Linear(n_embd+n_hidden_r, hidden_size1), activate=True), self.gelu, nn.LayerNorm(hidden_size1),
                    init_(nn.Linear(hidden_size1, n_embd), activate=True), self.gelu)
            elif rnn_type == 'GRU':
                self.rnn = nn.GRUCell(input_size=n_embd, hidden_size=n_hidden_r)
            elif rnn_type == 'CfC':
                self.rnn = CfCCell(input_size=n_embd, hidden_size=n_hidden_r)
                self.fusion_layer = nn.Sequential(
                    nn.LayerNorm(n_embd+n_hidden_r),
                    init_(nn.Linear(n_embd+n_hidden_r, hidden_size1), activate=True), self.gelu, nn.LayerNorm(hidden_size1),
                    init_(nn.Linear(hidden_size1, n_embd), activate=True), self.gelu)
            elif rnn_type == 'MixedCfC':
                self.rnn = CfCCell(input_size=n_embd, hidden_size=n_hidden_r)
                # self.rnn = WiredCfCCell(input_size=n_embd, wiring=AutoNCP(n_hidden_r, 1))
                self.lstm = nn.LSTMCell(input_size=n_embd, hidden_size=n_hidden_r)
                self.fusion_layer = nn.Sequential(
                    nn.LayerNorm(n_embd+n_hidden_r),
                    init_(nn.Linear(n_embd+n_hidden_r, hidden_size1), activate=True), self.gelu, nn.LayerNorm(hidden_size1),
                    init_(nn.Linear(hidden_size1, n_embd), activate=True), self.gelu)
            elif rnn_type == 'NCP':
                # wiring = NCP_No_Motor(inter_neurons = 20,
                #                       command_neurons = 12, # ~1:2 ratio of inter_neurons to command_neurons
                #                       sensory_fanout = 5, # ~1:4 ratio of inter_neurons to sensory_fanout
                #                       inter_fanout = 3, # ~1:4 ratio of inter_fanout to command_neurons
                #                       recurrent_command_synapses = 12 # ~1 per command
                #                       ) # this is for n_hidden_r = 32
                # wiring = NCP_No_Motor(inter_neurons = 40,
                #                       command_neurons = 24, # ~1:2 ratio of inter_neurons to command_neurons
                #                       sensory_fanout = 10, # ~1:4 ratio of inter_neurons to sensory_fanout
                #                       inter_fanout = 6, # ~1:4 ratio of inter_fanout to command_neurons
                #                       recurrent_command_synapses = 24 # ~1 per command
                #                       ) # this is for n_hidden_r = 64
                # wiring = NCP_No_Motor(inter_neurons = 64,
                #                       command_neurons = 32, # ~1:2 ratio of inter_neurons to command_neurons
                #                       sensory_fanout = 16, # ~1:4 ratio of sensory_fanout to inter_neurons
                #                       inter_fanout = 8, # ~1:4 ratio of inter_fanout to command_neurons
                #                       recurrent_command_synapses = 32 # ~1 per command
                #                       ) # this is for n_hidden_r = 96
                wiring = NCP_No_Motor(inter_neurons = 80,
                                      command_neurons = 48, # ~1:2 ratio of inter_neurons to command_neurons
                                      sensory_fanout = 20, # ~1:4 ratio of inter_neurons to sensory_fanout
                                      inter_fanout = 12, # ~1:4 ratio of inter_fanout to command_neurons
                                      recurrent_command_synapses = 48 # ~1 per command
                                      ) # this is for n_hidden_r = 128
                # wiring = NCP_No_Motor(inter_neurons = 96,
                #                       command_neurons = 48, # ~1:2 ratio of inter_neurons to command_neurons
                #                       sensory_fanout = 24, # ~1:4 ratio of sensory_fanout to inter_neurons
                #                       inter_fanout = 12, # ~1:4 ratio of inter_fanout to command_neurons
                #                       recurrent_command_synapses = 48 # ~1 per command
                #                       ) # this is for n_hidden_r = 144
                # wiring = NCP_No_Motor(inter_neurons = 128,
                #                       command_neurons = 64, # ~1:2 ratio of inter_neurons to command_neurons
                #                       sensory_fanout = 32, # ~1:4 ratio of sensory_fanout to inter_neurons
                #                       inter_fanout = 16, # ~1:4 ratio of inter_fanout to command_neurons
                #                       recurrent_command_synapses = 64 # ~1 per command
                #                       ) # this is for n_hidden_r = 192
                self.rnn = WiredCfCCell(input_size=n_embd, wiring=wiring, mode="default")
                self.fusion_layer = nn.Sequential(
                    nn.LayerNorm(n_embd+n_hidden_r),
                    init_(nn.Linear(n_embd+n_hidden_r, hidden_size1), activate=True), self.gelu, nn.LayerNorm(hidden_size1),
                    init_(nn.Linear(hidden_size1, n_embd), activate=True), self.gelu)
            elif rnn_type == 'MixedNCP':
                wiring = NCP_No_Motor(inter_neurons = 80,
                                      command_neurons = 48, # ~1:2 ratio of inter_neurons to command_neurons
                                      sensory_fanout = 20, # ~1:4 ratio of inter_neurons to sensory_fanout
                                      inter_fanout = 12, # ~1:4 ratio of inter_fanout to command_neurons
                                      recurrent_command_synapses = 48 # ~1 per command
                                      ) # this is for n_hidden_r = 128
                self.rnn = WiredCfCCell(input_size=n_embd, wiring=wiring, mode="default")
                self.lstm = nn.LSTMCell(input_size=n_embd, hidden_size=n_hidden_r)
                self.fusion_layer = nn.Sequential(
                    nn.LayerNorm(n_embd+n_hidden_r),
                    init_(nn.Linear(n_embd+n_hidden_r, hidden_size1), activate=True), self.gelu, nn.LayerNorm(hidden_size1),
                    init_(nn.Linear(hidden_size1, n_embd), activate=True), self.gelu)
            else:
                pass
        
        if not self.use_classbased_action:
            self.agent_class_encoder = nn.Sequential(
                init_(nn.Linear(n_agent_types, n_embd), activate=True), self.gelu)
        
        self.ln = nn.LayerNorm(n_embd)
        
        if self.use_graph_attention:
            self.blocks = nn.Sequential(*[EncodeBlock(n_embd, n_head) for _ in range(n_block)])
        self.head = nn.Sequential(init_(nn.Linear(n_embd, n_embd), activate=True), nn.GELU(), nn.LayerNorm(n_embd),
                                  init_(nn.Linear(n_embd, 1)))

    def forward(self, state, obs, rnn_states, active_agents=None):
        # Shape of obs is (batch*n_agent, **)

        for key in obs.keys():
            shape = obs[key].shape
            # print("shape of obs[{}] is: ".format(key), shape)
            # if obs[key].dtype != np.object_:
            #     obs[key] = obs[key].reshape(-1, *shape[2:])  # Reshape to (batch*n_agent, *)
            # else:
            #     obs[key] = obs[key].reshape(-1,)  # Reshape to (batch*n_agent,)
            obs[key] = obs[key].reshape(-1, *shape[2:])  # Reshape to (batch*n_agent, *)
            # print("shape of obs[{}] after reshape is: ".format(key), obs[key].shape)
        

        # Encoding of local agent view, agent pose and agent class
        local_view_encoding = self.local_view_encoder(obs['local_agent_view']) # (batch*n_agent, n_embd)
        pose_encoding = self.agent_pose_encoder(obs['agent_pose']) 
        # print("target position obs are: ", obs['target_positions'])
        # Handle variable-length sequences for targets
        # target_tensors = [torch.from_numpy(np.array(item_list, dtype=np.float32)).to(local_view_encoding.device) for item_list in obs['target_positions']]
        target_tensors = [torch.from_numpy(np.array(item_list, dtype=np.float32)).to(local_view_encoding.device) if item_list is not None else torch.empty(0, 5).to(local_view_encoding.device) for item_list in obs['target_positions']]

        # print("target_tensors are: ", target_tensors)
        padded_targets = torch.nn.utils.rnn.pad_sequence(target_tensors, batch_first=True, padding_value=0)
        # print("padded_targets are: ", padded_targets)
        aggregated_target_encoding = self.target_positions_encoder(padded_targets).squeeze(1) # (batch*n_agent, n_embd)
        # print("shape of aggregated_target_encoding is: ", aggregated_target_encoding.shape)


        # aggregated_other_agents_encoding = self.other_agents_positions_encoder(obs['other_agents_positions']).squeeze(1) # (batch*n_agent, n_embd)
        if not self.use_classbased_action:
            agent_class_encoding = self.agent_class_encoder(obs['agent_class_identifier'])
        if self.spawn_hazards:
            # Handle variable-length sequences for hazards
            hazard_tensors = [torch.from_numpy(np.array(item_list, dtype=np.float32)).to(local_view_encoding.device) for item_list in obs['hazard_positions']]
            padded_hazards = torch.nn.utils.rnn.pad_sequence(hazard_tensors, batch_first=True, padding_value=0)
            aggregated_hazard_encoding = self.hazard_positions_encoder(padded_hazards).squeeze(1)
            
            # Concatenation of all the above encodings
            if not self.use_classbased_action:
                combined_obs = torch.cat([
                    agent_class_encoding,
                    local_view_encoding,
                    pose_encoding,
                    # aggregated_other_agents_encoding,
                    aggregated_target_encoding,
                    aggregated_hazard_encoding
                ], dim=1)
            else:
                combined_obs = torch.cat([
                    local_view_encoding,
                    pose_encoding,
                    # aggregated_other_agents_encoding,
                    aggregated_target_encoding,
                    aggregated_hazard_encoding
                ], dim=1)
            obs_embeddings = self.single_obs_encoder(combined_obs) # (n_rollout_threads*n_agent, n_embd)

        else:
            if not self.use_classbased_action:
                combined_obs = torch.cat([
                    agent_class_encoding,
                    local_view_encoding,
                    pose_encoding,
                    # aggregated_other_agents_encoding,
                    aggregated_target_encoding,
                ], dim=1)
            else:
                combined_obs = torch.cat([
                    local_view_encoding,
                    pose_encoding,
                    # aggregated_other_agents_encoding,
                    aggregated_target_encoding,
                ], dim=1)
            obs_embeddings = self.single_obs_encoder(combined_obs) # (n_rollout_threads*n_agent, n_embd)

        if self.use_rnn:
            # rep = rep.reshape(-1, self.n_embd)  # reshape to (batch*n_agent, n_embd)
            if self.rnn_type == 'CfC' or self.rnn_type == 'NCP':
                rnn_states = rnn_states.reshape(-1, self.n_hidden_r)
                ts = obs['timespan']
                _, rnn_states = self.rnn(obs_embeddings, rnn_states, ts)
                obs_embeddings = obs_embeddings.reshape(-1, self.n_agent, self.n_embd)
                rnn_states = rnn_states.reshape(-1, self.n_agent, self.n_hidden_r)
                
                # Option A: Context-aware memory updates
                # rnn_states = self.blocks(self.ln(rnn_states)) # This is the updated context-aware memory of each agent, passed to be used for the next macro-step
                # rep = self.fusion_layer(torch.cat((obs_embeddings, rnn_states), dim=2))
                
                # Option B: Context-aware memory, no updates
                # context_aware_memory = self.blocks(self.ln(rnn_states)) # This is the updated context-aware memory of each agent, only used for the current step
                # rep = self.fusion_layer(torch.cat((obs_embeddings, context_aware_memory), dim=2))

                # Option C: memory and observation merge, then become context aware
                fused_embedding = self.fusion_layer(torch.cat((obs_embeddings, rnn_states), dim=2))
                state_representation = fused_embedding

            elif 'Mixed' in self.rnn_type:
                ts = obs['timespan']
                hidden_states = rnn_states[:,0,:] # rnn_states is a tuple of (rnn_states, cell_states)
                cell_states = rnn_states[:,1,:]
                hidden_states = hidden_states.reshape(-1, self.n_hidden_r)
                cell_states = cell_states.reshape(-1, self.n_hidden_r)
                hidden_states, cell_states = self.lstm(obs_embeddings, (hidden_states, cell_states))
                _ , hidden_states = self.rnn(obs_embeddings, hidden_states, ts)
                obs_embeddings = obs_embeddings.reshape(-1, self.n_agent, self.n_embd)
                hidden_states = hidden_states.reshape(-1, self.n_agent, self.n_hidden_r)
                cell_states = cell_states.reshape(-1, self.n_agent, self.n_hidden_r)
                rnn_states = (hidden_states, cell_states)
                fused_embedding = self.fusion_layer(torch.cat((obs_embeddings, hidden_states), dim=2))
                state_representation = fused_embedding
                
            elif self.rnn_type == 'LSTM':
                hidden_states = rnn_states[:,0,:] # rnn_states is a tuple of (rnn_states, cell_states)
                cell_states = rnn_states[:,1,:]
                hidden_states = hidden_states.reshape(-1, self.n_hidden_r)
                cell_states = cell_states.reshape(-1, self.n_hidden_r)
                hidden_states, cell_states = self.rnn(obs_embeddings, (hidden_states, cell_states)) 
                obs_embeddings = obs_embeddings.reshape(-1, self.n_agent, self.n_embd)
                hidden_states = hidden_states.reshape(-1, self.n_agent, self.n_hidden_r)
                cell_states = cell_states.reshape(-1, self.n_agent, self.n_hidden_r)
                rnn_states = (hidden_states, cell_states) 
                fused_embedding = self.fusion_layer(torch.cat((obs_embeddings, hidden_states), dim=2))
                state_representation = fused_embedding
            else: # GRU
                rnn_states = rnn_states.reshape(-1, self.n_hidden_r)
                # rep = rep.unsqueeze(1)
                # rnn_states = rnn_states.unsqueeze(0)
                rnn_states = self.rnn(obs_embeddings, rnn_states)
                # rep = rnn_states
                obs_embeddings = obs_embeddings.reshape(-1, self.n_agent, self.n_embd)
                rnn_states = rnn_states.reshape(-1, self.n_agent, self.n_hidden_r)
                fused_embedding = self.fusion_layer(torch.cat((obs_embeddings, rnn_states), dim=2))
                state_representation = fused_embedding
        else:
            obs_embeddings = obs_embeddings.reshape(-1, self.n_agent, self.n_embd)
            state_representation = obs_embeddings
        
        if self.use_graph_attention:
            rep = self.blocks(self.ln(state_representation)) # Graph attention among agents leading to the context-aware representation
        else:
            rep = self.ln(state_representation)
        
        
        v_loc = self.head(rep)
        return v_loc, rep, rnn_states


class Decoder(nn.Module):

    def __init__(self, obs_dim, action_dim, n_block, n_embd, n_embd_vit, n_head, n_agent, n_agent_types, spawn_hazards = False,
                 action_type='Discrete', dec_actor=False, share_actor=False, use_classbased_action=False, use_graph_attention=True):
        super(Decoder, self).__init__()

        self.action_dim = action_dim
        self.n_embd = n_embd
        self.dec_actor = dec_actor
        self.share_actor = share_actor
        self.action_type = action_type
        self.n_agent = n_agent
        self.use_classbased_action = use_classbased_action
        self.use_graph_attention = use_graph_attention
        self.spawn_hazards = spawn_hazards

        self.relu = nn.ReLU()
        self.gelu = nn.GELU()

        self.dropout_rate = 0.1

        if action_type == 'Discrete':
            # self.action_encoder = nn.Sequential(
            #     init_(nn.Linear(action_dim + 1, n_embd, bias=False), activate=True), nn.GELU())
            
            self.action_embedder = nn.Sequential(
                init_(nn.Linear(action_dim + 1, n_embd, bias=False), activate=True), nn.GELU())
            
            if self.use_classbased_action: # This is implemented for 3 agent classes only (
                self.class_0_decision_layer = nn.Sequential(
                                        init_(nn.Linear(n_embd, n_embd), activate=True), nn.GELU(), nn.LayerNorm(n_embd),
                                        # nn.Dropout(self.dropout_rate),
                                        init_(nn.Linear(n_embd, action_dim))) # For agent class 0, rescuers
                self.class_1_decision_layer = nn.Sequential(
                                        init_(nn.Linear(n_embd, n_embd), activate=True), nn.GELU(), nn.LayerNorm(n_embd),
                                        # nn.Dropout(self.dropout_rate),
                                        init_(nn.Linear(n_embd, action_dim))) # For agent class 1, scouts
                if self.spawn_hazards:
                    self.class_2_decision_layer = nn.Sequential(
                                            init_(nn.Linear(n_embd, n_embd), activate=True), nn.GELU(), nn.LayerNorm(n_embd),
                                            # nn.Dropout(self.dropout_rate),
                                            init_(nn.Linear(n_embd, action_dim))) # For agent class 2, cleaners
            else:
                self.agent_class_encoder = nn.Sequential(init_(nn.Linear(n_agent_types, n_embd), activate=True), self.gelu)
                self.head = nn.Sequential(init_(nn.Linear(n_embd, n_embd), activate=True), nn.GELU(), nn.LayerNorm(n_embd),
                                        # nn.Dropout(self.dropout_rate),
                                        init_(nn.Linear(n_embd, action_dim)))


        elif action_type == 'MultiDiscrete': # not used in this implementation
            # self.action_encoder = nn.Sequential(nn.Flatten(), init_(nn.Linear(1*(action_dim[0]+1)*(action_dim[1]+1), n_embd-n_agent_types)), self.gelu)
            self.action_embedder = nn.Sequential(init_(nn.Linear(1 + action_dim[0]+action_dim[1], n_embd, bias=False), activate=True), self.gelu)
            # print("shape of action encoder is: ", self.action_encoder)
            self.agent_class_encoder = nn.Sequential(init_(nn.Linear(n_agent_types, n_embd), activate=True), self.gelu)
            self.heads = []
            for act_dim in self.action_dim:
                self.heads.append(nn.Sequential(init_(nn.Linear(n_embd, n_embd), activate=True), nn.GELU(), nn.LayerNorm(n_embd),
                                                init_(nn.Linear(n_embd, act_dim))))
            self.heads = nn.ModuleList(self.heads)
            
        else:
            self.action_embedder = nn.Sequential(init_(nn.Linear(action_dim, n_embd), activate=True), nn.GELU())
        
        self.ln = nn.LayerNorm(n_embd)

        if self.use_graph_attention:
            self.blocks = nn.Sequential(*[DecodeBlock(n_embd, n_head) for _ in range(n_block)])
            

    def zero_std(self, device):
        if self.action_type != 'Discrete':
            log_std = torch.zeros(self.action_dim).to(device)
            self.log_std.data = log_std

    # state, action, and return
    def forward(self, action, obs_rep, obs, rnn_states=None):
        # action: (batch, n_agent, action_dim), one-hot/logits?
        # obs_rep: (batch, n_agent, n_embd)
        if self.action_type == 'Discrete':
            if self.use_graph_attention:
                
                action = action.reshape(-1, 1 + self.action_dim) # Actions of previous agents in the sequence
                action_embeddings = self.action_embedder(action)

                if not self.use_classbased_action:
                    agent_class_encoding = self.agent_class_encoder(obs['agent_class_identifier'])
                
                    # # Option 1: concat and pass through another linear layer
                    # act_emdb = torch.cat((agent_class_encoding, action_embeddings), dim=1)
                    # action_embeddings = self.action_encoder(act_emdb) # (batch_size * n_agent, n_embd)

                    # Option 2: Add the agent class encoding to the action embeddings
                    action_embeddings = action_embeddings + agent_class_encoding # (batch_size * n_agent, n_embd)

                action_embeddings = action_embeddings.reshape(-1, self.n_agent, self.n_embd) # (batch_size, n_agent, n_embd)
                x = self.ln(action_embeddings)
                for block in self.blocks:
                    x = block(x, obs_rep)
            else: # no graph attention, directly use obs_rep
                x = self.ln(obs_rep)

            if self.use_classbased_action:
                # agent_classes = obs['agent_class_identifier'].reshape(-1, self.n_agent, obs['agent_class_identifier'].shape[-1]).argmax(dim=-1) # (batch_size, n_agent)
                agent_classes_one_hot = obs['agent_class_identifier'].reshape(-1, self.n_agent, obs['agent_class_identifier'].shape[-1])

                # Get logits from each head
                logits_0 = self.class_0_decision_layer(x)
                logits_1 = self.class_1_decision_layer(x)
                if self.spawn_hazards:
                    logits_2 = self.class_2_decision_layer(x)

                # Stack logits along the last dimension: (batch_size, n_agent, action_dim, C)
                if self.spawn_hazards:
                    all_logits = torch.stack([logits_0, logits_1, logits_2], dim=-1)
                else:
                    all_logits = torch.stack([logits_0, logits_1], dim=-1)

                # Reshape one-hot vector for broadcasting: (batch_size, n_agent, 1, C)
                selector = agent_classes_one_hot.unsqueeze(-2)

                # Multiply and sum to select the correct logits
                # (batch, n_agent, action_dim, 3) * (batch, n_agent, 1, 3) -> (batch, n_agent, action_dim, 3)
                # sum along the last dim -> (batch, n_agent, action_dim)
                logit = (all_logits * selector).sum(dim=-1)
            else:
                logit = self.head(x)
        else: # for MultiDiscrete actions, not used in this implementation
            action = action.reshape(-1, 1, 1 + self.action_dim[0] + self.action_dim[1]) # (batch_size * n_agent, 1, act_dim[0]+1,act_dim[1]+1)
            action = torch.squeeze(action)
            action_embeddings = self.action_encoder(action)
            action_embeddings = action_embeddings.reshape(-1, self.n_agent, self.n_embd) # (batch_size, n_agent, n_embd)
            x = self.ln(action_embeddings)
            for block in self.blocks:
                x = block(x, obs_rep)
            logit = [] # logits
            for head in self.heads:
                logit.append(head(x))    

        return logit, rnn_states


class AsynchronousClassBasedLiquidMultiAgentTransformer(nn.Module):

    def __init__(self, state_dim, obs_dim, action_dim, n_agent, n_agent_types, max_num_targets, spawn_hazards, max_num_hazards,
                 n_block, n_embd, n_hidden, n_hidden_r, n_embd_vit, n_head, use_rnn=False, rnn_type='GRU', encode_state=False, use_classbased_action=False,
                 use_graph_attention=True, device=torch.device("cpu"), action_type='Discrete', dec_actor=False, share_actor=False):
        super(AsynchronousClassBasedLiquidMultiAgentTransformer, self).__init__()

        self.n_agent = n_agent
        self.action_dim = action_dim
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.action_type = action_type
        self.device = device
        self.rnn_type = rnn_type
        self.n_hidden_r = n_hidden_r
        self.use_rnn = use_rnn
        self.use_graph_attention = use_graph_attention

        # state unused
        # state_dim = 37
        if 'Dict' in obs_dim.__class__.__name__:
            self._mixed_obs = True
        else:
            self._mixed_obs = False


        self.encoder = Encoder(state_dim, obs_dim, action_dim, n_block, n_embd, n_hidden, n_hidden_r, n_embd_vit, n_head, n_agent, 
                               n_agent_types, max_num_targets, spawn_hazards, max_num_hazards, use_rnn, rnn_type, encode_state, use_classbased_action, use_graph_attention)
        self.decoder = Decoder(obs_dim, action_dim, n_block, n_embd, n_embd_vit, n_head, n_agent, n_agent_types, spawn_hazards,
                               self.action_type, dec_actor=dec_actor, share_actor=share_actor, use_classbased_action=use_classbased_action, use_graph_attention=use_graph_attention)
        self.to(device)

    def zero_std(self):
        if self.action_type != 'Discrete':
            self.decoder.zero_std(self.device)

    def forward(self, state, obs, action, available_actions=None, active_agents=None, rnn_states_actor=None, rnn_states_critic=None):
        # state: (batch, n_agent, state_dim)
        # obs: (batch, n_agent, obs_dim)
        # action: (batch, n_agent, 1)
        # available_actions: (batch, n_agent, act_dim)

        # state unused
        ori_shape = np.shape(state)
        state = np.zeros((*ori_shape[:-1], 37), dtype=np.float32)

        if self._mixed_obs:
            for key in obs.keys():
                if obs[key].dtype == np.object_:
                    # print("Found object type in obs[{}], skipping tensor conversion.".format(key))
                    continue
                if key != 'agent_prompt':
                    obs[key] = check(obs[key]).to(**self.tpdv)
                # state[key] = check(state[key]).to(**self.tpdv)
        else:
            obs = check(obs).to(**self.tpdv)
        state = check(state).to(**self.tpdv)
        action = check(action).to(**self.tpdv)
        if rnn_states_actor is not None:
            if (self.rnn_type == 'LSTM' or 'Mixed' in self.rnn_type):
                rnn_states_actor = (check(rnn_states_actor[0]).to(**self.tpdv), check(rnn_states_actor[1]).to(**self.tpdv))
                rnn_states_actor = torch.stack(rnn_states_actor, dim=1)
            else:
                rnn_states_actor = check(rnn_states_actor).to(**self.tpdv)
        if rnn_states_critic is not None:
            if (self.rnn_type == 'LSTM' or 'Mixed' in self.rnn_type):
                rnn_states_critic = (check(rnn_states_critic[0]).to(**self.tpdv), check(rnn_states_critic[1]).to(**self.tpdv))
                rnn_states_critic = torch.stack(rnn_states_critic, dim=1)
            else:
                rnn_states_critic = check(rnn_states_critic).to(**self.tpdv)
        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)
        if active_agents is not None:
            active_agents = check(active_agents).to(**self.tpdv)

        batch_size = np.shape(obs['agent_class_identifier'])[0]
        v_loc, obs_rep, rnn_states_critic  = self.encoder(state, obs, rnn_states_critic, active_agents)

        if self.use_graph_attention:
            if self.action_type == 'Discrete':
                action = action.long()
                action_log, entropy = discrete_parallel_act(self.decoder, obs_rep, obs, action, batch_size,
                                                            self.n_agent, self.action_dim, self.tpdv, available_actions, rnn_states_actor, active_agents)
            elif self.action_type == "MultiDiscrete":
                action = action.long()
                action_log, entropy = multidiscrete_parallel_act(self.decoder, obs_rep, obs, action, batch_size,
                                                            self.n_agent, self.action_dim, self.tpdv, available_actions, rnn_states_actor, active_agents) #action_log and entropy are lists
            else:
                action_log, entropy = continuous_parallel_act(self.decoder, obs_rep, obs, action, batch_size,
                                                            self.n_agent, self.action_dim, self.tpdv)
        else:  # No sequential decision-making
            logits, _ = self.decoder(None, obs_rep, obs)
            if self.action_type == 'Discrete':
                # 1. Masking
                if available_actions is not None:
                    logits[available_actions == 0] = torch.finfo(logits.dtype).min

                # 2. Distribution
                distri = Categorical(logits=logits)
                
                # 3. Log Prob of the taken actions
                action_log = distri.log_prob(action.squeeze(-1)).unsqueeze(-1)
                entropy = distri.entropy().unsqueeze(-1)

                # 4. Handle Active Agents Masking
                if active_agents is not None:
                    entropy = (entropy * active_agents).sum() / active_agents.sum()
                else:
                    entropy = entropy.mean()
            elif self.action_type == 'MultiDiscrete':
                pass # not implemented yet

        return action_log, v_loc, entropy

    def get_actions(self, state, obs, available_actions=None, deterministic=False, active_agents=None, rnn_states_actor=None, rnn_states_critic=None):
        # state unused
        ori_shape = np.shape(obs)
        state = np.zeros((*ori_shape[:-1], 37), dtype=np.float32)


        if self._mixed_obs:
            for key in obs.keys():
                if obs[key].dtype == np.object_:
                    # print("Found object type in obs[{}], skipping tensor conversion.".format(key))
                    continue
                if key != 'agent_prompt':
                    obs[key] = check(obs[key]).to(**self.tpdv)
                # state[key] = check(state[key]).to(**self.tpdv)
        else:
            obs = check(obs).to(**self.tpdv)
        state = check(state).to(**self.tpdv)

        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)
        if rnn_states_critic is not None:
            if self.use_rnn and (self.rnn_type == 'LSTM' or 'Mixed' in self.rnn_type):
                rnn_states_critic = rnn_states_critic.transpose(1, 2)  # Now shape is (batch, n_agents, 2, n_hidden_r)
                rnn_states_critic = rnn_states_critic.reshape(-1, 2, self.n_hidden_r)  # Now shape is (batch*n_agents, 2, n_hidden_r)


        batch_size = np.shape(obs['agent_class_identifier'])[0]
        # print("batch_size: ", batch_size)
        # print("obs shape: ", obs['local_agent_view'].shape)
        v_loc, obs_rep, rnn_states_critic = self.encoder(state, obs, rnn_states_critic, active_agents)

        # # Visualizing attention maps
        # attention_block = self.encoder.blocks[0].attn 
        # attention_weights = attention_block.att_bp.detach().cpu().numpy()

        # # The shape of attention_weights will be (batch_size, num_heads, sequence_length, sequence_length)
        # # print(attention_weights.shape)
        # np.save(os.path.expanduser(f'~/Dissertation Results Visualization/attention_weights_step.npy'), attention_weights)

        if self.use_graph_attention:

            if self.action_type == "Discrete":
                output_action, output_action_log, rnn_states_actor = discrete_autoregreesive_act(self.decoder, obs_rep, obs, batch_size,
                                                                            self.n_agent, self.action_dim, self.tpdv,
                                                                            available_actions, rnn_states_actor, deterministic, active_agents)
            elif self.action_type == "MultiDiscrete":
                output_action, output_action_log, rnn_states_actor = multidiscrete_autoregreesive_act(self.decoder, obs_rep, obs, batch_size,
                                                                            self.n_agent, self.action_dim, self.tpdv,
                                                                            available_actions, rnn_states_actor, deterministic, active_agents)
            else:
                output_action, output_action_log, rnn_states_actor = continuous_autoregreesive_act(self.decoder, obs_rep, obs, batch_size,
                                                                                self.n_agent, self.action_dim, self.tpdv,
                                                                                deterministic)
            
        else:  # No sequential decision-making
            # 1. Pass through Decoder (Acting as MLP)
            logits, _ = self.decoder(None, obs_rep, obs)
            
            output_action_log = torch.zeros_like(logits[..., 0:1]) # Placeholder for shape
            
            if self.action_type == "Discrete":
                # Masking
                if available_actions is not None:
                    available_actions = available_actions.reshape(logits.shape)
                    logits[available_actions == 0] = torch.finfo(logits.dtype).min
                
                distri = Categorical(logits=logits)
                action = distri.probs.argmax(dim=-1) if deterministic else distri.sample()
                action_log = distri.log_prob(action)
                
                # Fix Shapes: (Batch, Agent) -> (Batch, Agent, 1)
                output_action = action.unsqueeze(-1)
                output_action_log = action_log.unsqueeze(-1)

            elif self.action_type == "Continuous":
                pass # not implemented yet

        return output_action, output_action_log, v_loc, rnn_states_actor, rnn_states_critic


    def get_values(self, state, obs, available_actions=None):
        # NOTE: Value estimates for MACMAT are produced inside the encoder during
        # `get_actions`/`forward` (the encoder returns `v_loc`), and bootstrap returns are
        # handled by `GridWorldRunner.compute(next_values)`, which overrides
        # `Runner.compute()`. This standalone path (used by the base `Runner.compute()`)
        # is therefore not part of the MACMAT pipeline; it also cannot work as written
        # because the recurrent critic requires `rnn_states` that are not threaded here.
        raise NotImplementedError(
            "AsynchronousClassBasedLiquidMultiAgentTransformer.get_values is unused: "
            "values are computed via get_actions/forward and bootstrapped in "
            "GridWorldRunner.compute()."
        )



