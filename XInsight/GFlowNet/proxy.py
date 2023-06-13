
from torch.nn import Linear, LeakyReLU, Dropout, Module
from torch_geometric.nn import GCNConv, global_mean_pool, Sequential

class Proxy(object):
    def __init__(self):
        pass
    
    def forward(self, **kwargs):
        pass
    
    
class GCN_Proxy(Module):
    def __init__(self, num_node_features, num_gcn_hidden, num_mlp_hidden, num_classes, dropout=0.1):
        super(GCN_Proxy, self).__init__()
        
        self.num_node_features = num_node_features
        self.num_gcn_hidden = num_gcn_hidden
        self.num_mlp_hidden = num_mlp_hidden
        self.num_classes = num_classes
        self.dropout = dropout
        
        self._init_proxy()
        
        
    def _init_proxy(self):
        
        layers = []
        for i in range(len(self.num_gcn_hidden)):
            if i == 0:
                layers.append((GCNConv(self.num_node_features, self.num_hidden[i]), 'x, edge_index -> x'))
            else:
                layers.append((GCNConv(self.num_hidden[i-1], self.num_hidden[i]), 'x, edge_index -> x'))
            layers.append(LeakyReLU(inplace=True))
            layers.append((Dropout(p=self.dropout), 'x -> x'))
            
        layers.append((global_mean_pool, 'x, batch -> x'))
        # MLP
        for i in range(len(self.num_mlp_hidden)):
            if i == 0:
                layers.append((Linear(self.num_gcn_hidden[-1], self.num_mlp_hidden[i]), 'x -> x'))
            else:
                layers.append((Linear(self.num_mlp_hidden[i-1], self.num_mlp_hidden[i]), 'x -> x'))
            layers.append(LeakyReLU(inplace=True))
            layers.append((Dropout(p=self.dropout), 'x -> x'))        
            

        self.model = Sequential(
            'x, edge_index', layers
        )
        
    def forward(self, x, edge_index):
        
        return self.model(x, edge_index)
