from hetmarl.algorithms.utils.torchncp.wirings import AutoNCP, NCP_No_Motor
from hetmarl.algorithms.utils.torchncp import CfCCell, WiredCfCCell

wiring = NCP_No_Motor(inter_neurons = 80,
                                      command_neurons = 48, # ~1:2 ratio of inter_neurons to command_neurons
                                      sensory_fanout = 20, # ~1:4 ratio of inter_neurons to sensory_fanout
                                      inter_fanout = 12, # ~1:4 ratio of inter_fanout to command_neurons
                                      recurrent_command_synapses = 48 # ~1 per command
                                      ) # this is for n_hidden_r = 128
# wiring = AutoNCP(units=128, output_size=1)
ncp_cfc = WiredCfCCell(input_size=128, wiring=wiring, mode="default")
cfc = CfCCell(input_size=128, hidden_size=128, mode="default")

n_params_ncp = 0
for name, param in ncp_cfc.named_parameters():
    print(name, param.size())
    n_params_ncp += param.numel()
print(f"Total number of NCP parameters: {n_params_ncp}")

n_params_cfc = 0
for name, param in cfc.named_parameters():
    print(name, param.size())
    n_params_cfc += param.numel()
print(f"Total number of CfC parameters: {n_params_cfc}")

# import matplotlib.pyplot as plt
# plt.figure(figsize=(6, 4))
# legend_handles = wiring.draw_graph(draw_labels=True)
# plt.legend(handles=legend_handles, loc="upper center", bbox_to_anchor=(1, 1))
# plt.tight_layout()
# plt.show()