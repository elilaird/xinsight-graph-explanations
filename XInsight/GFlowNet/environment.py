import utils
import torch
from torch_geometric.data import Data
from torch_geometric.utils import degree

from proxy import Proxy


### ACTION CLASSES ###

class Action():
    def __init__(self, action_type):
        self.type = action_type

class MutagAction(Action):
    def __init__(self, action_type, start, end):
        super(MutagAction, self).__init__(action_type)
        
        self.start = start
        self.end = end        
        
### STATE CLASSES ###
class State():
    def __init__(self, state_type, value):
        self.type = state_type
        self.value = value
        
class MutagState(State):
    def __init__(self, state_type, value):
        super(MutagState, self).__init__(state_type, value)
        

### REWARD FUNCTIONS ###
def class_prob_reward(proxy: Proxy, state: State, target: int, alpha: float=1.0,  threshold: float=0.5):
    
    pred = torch.softmax(proxy(state.value.x, state.value.edge_index, state.value.batch), dim=1)[0]
    if pred[target] < threshold:
        return torch.tensor(0.0)
    else:
        return alpha * pred[target]
    
        
        
### ENVIRONMENT CLASSES ###
 
class Environment():
    def __init__(self, env_name, reward_fn, proxy, config):
        self.env_name = env_name
        self.reward_fn = reward_fn
        self.proxy = proxy
        self.config = config
    
    def new(self):
        pass
    
    def step(self, state: State, action: Action):
        pass
    
    
class MutagEnvironment(Environment):
    def __init__(self, env_name, reward_fn, proxy, config):
        super(MutagEnvironment, self).__init__(env_name, reward_fn, proxy, config)
            
        self.node_feature_size = config['node_feature_size']
        self.alpha = config['alpha']
        self.threshold = config['threshold']
        self.target = config['target']
        
        self.action_space = self._init_action_space()

        
        
    def new(self, start_idx=-1):
        return utils.get_init_state_graph(self.node_feature_size, start_idx)
    
    def step(self, state: MutagState, action: MutagAction):
        new_state, valid, stop = self.take_action_mutag(state.value, (action.start, action.end))
        reward = self.calculate_reward(new_state)
        
        return MutagState('mutag', new_state), reward, valid, stop
    
    def calculate_reward(self, state: MutagState):
        return self.reward_fn(self.proxy, state, self.target, self.alpha, self.threshold)
        
    def _init_action_space(self):
        # Stop action is the first node in candidate list
        x = Data(
            x=torch.zeros(self.num_features + 1, self.num_features), # +1 adds stop action
            edge_index=torch.zeros((2, 0), dtype=torch.long),
        )

        for i in range(1, self.num_features + 1):
            x.x[i, i-1] = 1
        
        return x

    def _take_action_mutag(self, G, action):
        # Takes an action in the form (starting_node, ending_node) and 
        # returns the new graph, whether the action is valid, and whether the action is a stop action
        start, end = action

        G_new = G.clone()
        
        if start == end:
            return G, False, False

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
        if utils.check_edge(G_new.edge_index, torch.tensor([[start], [end]])):
            # If edge exists, return original G 
            return G, False, False
        else:
            # Add edge from start to end
            G_new.edge_index = utils.append_edge(G_new.edge_index, torch.tensor([[start], [end]]))
        
        # Check if valency is violated
        if self._check_valency_violation(G_new):
            return G, False, False
        else:
            return G_new, True, False
        
    def _check_valency_violation(self, G):
        valency_dict = {0: 4, 1: 3, 2: 2, 3: 1, 4: 1, 5: 1, 6: 1}
        degree_values = degree(G.edge_index[0], num_nodes=G.num_nodes) + degree(G.edge_index[1], num_nodes=G.num_nodes)

        for i in range(G.num_nodes):
            atom_type = G.x[i].nonzero().item()
            if degree_values[i] > valency_dict[atom_type]:
                return True
        return False