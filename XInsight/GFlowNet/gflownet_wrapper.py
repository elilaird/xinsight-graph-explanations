import torch
from torch.distributions import Categorical

from proxy import Proxy, GCN_Proxy
from backbone import GFlowNetBackbone
from environment import *


class GFlowNet(object):
    def __init__(self, backbone: GFlowNetBackbone, environment: Environment):
        
        self.backbone = backbone
        self.environment = environment
        self.proxy = environment.proxy
        
        self.curr_flows = None
        self.prev_flows = None
        self.total_forward_flow = torch.tensor(0.0)
        self.total_backward_flow = torch.tensor(0.0)
    
    def new():
        pass
    
    def forward(self, **kwargs):
        return self.backbone.forward(**kwargs)

    def sample_action(self, state, edge_index, action):
        pass
    
    def take_action(self, state, edge_index, action):
        pass
    
    def calculate_flows(self, state):
        pass
    
    def update_flows(self, forward_flow, backward_flow):
        pass
    
class MUTAG_GFlowNet(GFlowNet):
    def __init__(self, backbone: GFlowNetBackbone, environment: MutagEnvironment):
        super(MUTAG_GFlowNet, self).__init__(backbone, environment)

    def sample_action(self, state, P_Forward):
        P_F, _ = self.calculate_flows(state)
        
        # Sample start node
        start_dist = Categorical(logits=P_F[0])
        start = start_dist.sample()
        
        # Mask out starting node 
        mask = torch.ones_like(P_F[1])
        mask[start] = 0
        
        P_F[1] = P_F[1] * mask - 100 * (1 - mask)
        
        # Sample end node
        end_dist = Categorical(logits=P_F[1])
        end = end_dist.sample()
        
        return MutagAction("add_node", start, end), P_F
    
    def take_action(self, state, action):
        return self.environment.step(state, action)
    
    def new(self):
        return self.environment.new()   
    
    def calculate_reward(self, state: MutagState):
        return self.environment.calculate_reward(state)
    
    def calculate_flow_dist(self, state):
        return self.backbone(state.value.x, state.value.edge_index)
    
    def calculate_forward_flow(self, action, p_forward):
        return Categorical(logits=p_forward[0]).log_prob(action.start) + Categorical(logits=p_forward[1]).log_prob(action.end)
    
    def calculate_backward_flow(self, action, p_backward):
        return Categorical(logits=p_backward[0]).log_prob(action.start) + Categorical(logits=p_backward[1]).log_prob(action.end)
    
    def update_flows(self, action, forward_flow, backward_flow):
        self.total_forward_flow += forward_flow
        self.total_backward_flow += backward_flow
    
    def __call__(self, state):
        return self.calculate_flows(state)