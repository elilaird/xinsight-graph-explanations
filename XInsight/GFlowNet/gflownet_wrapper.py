from proxy import Proxy
from backbone import GFlowNetBackbone


class GFlowNet(object):
    def __init__(self, backbone: GFlowNetBackbone, proxy: Proxy):
        
        self.backbone = backbone
        self.proxy = proxy
    
    def forward(self, **kwargs):
        return self.backbone.forward(**kwargs)
    
    def train(self, epochs, target, lr, optimizer, loss_fn):
        pass
    
    def sample_action(self, state, edge_index, action):
        pass
    
    def take_action(self, state, edge_index, action):
        pass
    
    def calculate_reward(self, state, edge_index, action):
        pass
    
    
class GCN_GFlowNet(GFlowNet):
    def __init__(self, backbone: GFlowNetBackbone, proxy: Proxy):
        super(GCN_GFlowNet, self).__init__(backbone, proxy)
    
    def train(self, epochs, target, lr, optimizer, loss_fn):
        pass

    def sample_action(self, state, edge_index, action):
        pass
    
    def take_action(self, state, edge_index, action):
        pass
    
    def calculate_reward(self, state, edge_index, action):
        pass
    
    