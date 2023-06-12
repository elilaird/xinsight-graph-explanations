import torch
from torch_geometric.utils import degree
from torch_geometric.data import Data
import numpy as np

### GRAPH UTILS ###
def check_edge(edge_index, new_edge):
    # Check if an edge is in the graph
    return bool(torch.all(torch.eq(edge_index, new_edge), dim=0).any())

def append_edge(edge_index, new_edge):
    return torch.cat((edge_index, new_edge), dim=1)

def get_init_state_graph(num_node_features, start_idx=-1):
    # Create initial graph with only the starting node
    g = Data(x=torch.zeros((1, num_node_features)), 
             edge_index=torch.zeros((2, 0), dtype=torch.long), 
             y=torch.torch.zeros((1, 1), dtype=torch.long),
        )
    
    # Assign starting node feature
    start_idx = np.random.randint(0, num_node_features)
    g.x[0, start_idx] = 1
    g.y[0] = start_idx

    return g

### MUTAG UTILS ###

def check_valency_violation(G):
    valency_dict = {0: 4, 1: 3, 2: 2, 3: 1, 4: 1, 5: 1, 6: 1}
    degree_values = degree(G.edge_index[0], num_nodes=G.num_nodes) + degree(G.edge_index[1], num_nodes=G.num_nodes)

    for i in range(G.num_nodes):
        atom_type = G.x[i].nonzero().item()
        if degree_values[i] > valency_dict[atom_type]:
            return True
    return False

def take_action_mutag(G, action):
    # Takes an action in the form (starting_node, ending_node) and 
    # returns the new graph, whether the action is valid, and whether the action is a stop action
    start, end = action

    G_new = G.clone()

    # If end node is stop action, return graph
    if end == G.x.size(0):
        return G, True, True

    # If end node is new candidate, add it to the graph
    if end > G.x.size(0): # changed from end > G.x.size(0) - 1 because now stop action is G.x.size(0)
        # Create new node
        candidate_idx = end - G.x.size(0) - 1
        new_feature = torch.zeros(1, G_new.x.size(1))
        new_feature[0, candidate_idx] = 1
        G_new.x = torch.cat([G_new.x, new_feature], dim=0)
        G_new.y = torch.cat([G_new.y, torch.zeros((1, 1))], dim=0)
        G_new.y[G_new.x.size(0)-1] = candidate_idx 
        end = G_new.x.size(0) - 1
    
    # Check if edge already exists
    if check_edge(G_new.edge_index, torch.tensor([[start], [end]])):
        # If edge exists, return original G 
        return G, False, False
    else:
        # Add edge from start to end
        G_new.edge_index = append_edge(G_new.edge_index, torch.tensor([[start], [end]]))
    
    # Check if valency is violated
    if check_valency_violation(G_new):
        return G, False, False
    else:
        return G_new, True, False
    
    
### LOGISTICS UTILS ###

def save_gflownet(gflownet, path, name):
    import os
    from datetime import datetime
    
    dt_string = datetime.now().strftime("%d_%m_%Y_%H_%M_%S")

    if not os.path.exists(path):
        os.makedirs(path)
    
    torch.save(gflownet.backbone.state_dict(), os.path.join(path, name + '_backbone_' + dt_string + '.pt'))
    torch.save(gflownet.proxy.state_dict(), os.path.join(path, name + '_proxy_' + dt_string + '.pt'))
    
def load_gflownet(gflownet, path, name):
    import os
    
    backbone_path = os.path.join(path, name + '_backbone.pt')
    proxy_path = os.path.join(path, name + '_proxy.pt')
    
    gflownet.backbone.load_state_dict(torch.load(backbone_path))
    gflownet.proxy.load_state_dict(torch.load(proxy_path))
    
    return gflownet