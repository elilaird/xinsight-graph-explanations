import torch
import argparse
import os 
import networkx as nx
from torch_geometric.utils import to_networkx
from matplotlib import pyplot as plt

from gflownet.environment import MutagEnvironment, class_prob_reward
from gflownet.proxy import GCN_Proxy
from gflownet.backbone import GCNBackbone
from gflownet.gflownet_wrapper import MUTAG_GFlowNet

def generate_nx_plot(graph):
    atoms = {0: 'C', 1: 'N', 2: 'O', 3: 'F', 4: 'I', 5: 'Cl', 6: 'Br'}
    colors = {0: 'green', 1: 'blue', 2: 'red', 3: 'yellow', 4: 'purple', 5: 'orange', 6: 'pink'}
    labelsdict = {i:atoms[int(graph.y[i].item())] for i in range(graph.num_nodes)}
    
    fig = plt.figure(figsize=(4, 4))
    _colors = [colors[int(graph.y[i].item())] for i in range(graph.num_nodes)]
    G_nx = to_networkx(graph, to_undirected=True)
    nx.draw_networkx(G_nx, pos=nx.spring_layout(G_nx, seed=42), with_labels=True, node_color=_colors, labels=labelsdict)
    
    return fig

parser = argparse.ArgumentParser()
parser.add_argument('-g', '--num_graphs', type=int, default=1, help='Number of graphs to generate')
parser.add_argument('-m', '--max_actions', type=int, default=30, help='Maximum number of actions to take')
parser.add_argument('-n', '--num_nodes', type=int, default=10, help='Minimum number of nodes in a graph')
parser.add_argument('-f', '--model_file', type=str, default='models/gflownet_backbone/mutag_gflownet_backbone.pt', help='Path to model file')
parser.add_argument('-p', '--proxy_file', type=str, default='models/proxy/mutag_proxy_86.pt', help='Path to proxy model file')
parser.add_argument('-o', '--output_dir', type=str, default='./graphs', help='Path to output directory')
parser.add_argument('-v', '--verbose', action='store_true', help='Print verbose output')

args = parser.parse_args()

ACTIONS_LIMIT = args.max_actions
MIN_NODE_COUNT = args.num_nodes
OUTPUT_DIR = args.output_dir
VERBOSE = args.verbose

# check if output directory exists
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

cfg = {
    'node_feature_size': 7,
    'alpha': 1.0,
    'threshold': 0.5,
    'target': 1,
}

proxy = GCN_Proxy(num_node_features=7, num_gcn_hidden=[32, 48, 64], num_mlp_hidden=[64, 32], num_classes=2, dropout=0.1)
proxy.load_state_dict(torch.load('models/proxy/mutag_proxy.pt'))
proxy.eval()

backbone = GCNBackbone(num_features=7, num_gcn_hidden=[32, 64, 128], num_mlp_hidden=[128, 512])
env = MutagEnvironment("mutag", reward_fn=class_prob_reward, proxy=proxy, config=cfg)

gflownet = MUTAG_GFlowNet(backbone, env)
gflownet.load('models/gflownet_backbone/mutag_gflownet_backbone.pt')
gflownet.eval()

# generate graphs
graphs = []
for i in range(args.num_graphs):
    graph = gflownet.generate(MIN_NODE_COUNT, ACTIONS_LIMIT)
    graphs.append(graph)
    torch.save(graph, os.path.join(OUTPUT_DIR, f'graph_{i}.pt'))
    fig = generate_nx_plot(graph)
    fig.savefig(os.path.join(OUTPUT_DIR, f'graph_{i}.png'))
    
    if VERBOSE:
        print('graph_{}:'.format(i))
        print(graph.x)
    
        
# save graphs
torch.save(graphs, os.path.join(OUTPUT_DIR, 'all_generated_graphs.pt'))
    
    