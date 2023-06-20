import torch
import tqdm
import copy
import numpy as np

from gflownet.environment import MutagEnvironment, class_prob_reward
from gflownet.proxy import GCN_Proxy
from gflownet.backbone import GCNBackbone
from gflownet.gflownet_wrapper import MUTAG_GFlowNet

EPOCHS = 100000
ACTIONS_LIMIT = 30
MIN_NODE_COUNT = 10
CLIP = -1
UPDATE_FREQ = 2
LR_1 = 1e-3
LR_2 = 1e-4

cfg = {
    'node_feature_size': 7,
    'alpha': 1.0,
    'threshold': 0.5,
    'target': 1,
}

proxy = GCN_Proxy(num_node_features=7, num_gcn_hidden=[32, 48, 64], num_mlp_hidden=[64, 32], num_classes=2, dropout=0.1)
proxy.load_state_dict(torch.load('models/proxy/mutag_proxy.pt'))
proxy.eval()

print("Proxy")
print(proxy)

backbone = GCNBackbone(num_features=7, num_gcn_hidden=[32, 64, 128], num_mlp_hidden=[128, 512])
env = MutagEnvironment("mutag", reward_fn=class_prob_reward, proxy=proxy, config=cfg)

print("Backbone")
print(backbone)

gflownet = MUTAG_GFlowNet(backbone, env)

opt = torch.optim.Adam(gflownet.backbone.parameters(), lr=LR_1)

# training loop 
actions_taken = minibatch_loss = 0
best_loss = np.inf
best_model = None
pbar = tqdm.tqdm(range(EPOCHS), desc=f"Epoch: {0}, Loss: {0:.4f}", unit="episode")
for episode in pbar:
    
    # reset gflownet flows
    gflownet.zero_flows()
    
    # initialize state
    state = gflownet.new(start_idx=0)
    
    while actions_taken < ACTIONS_LIMIT:
        
        # sample action
        action, P_Forward = gflownet.sample_action(state)
        
        # take action 
        new_state, _, valid, stop = gflownet.take_action(state, action)
        
        # calculate forward flow of action
        forward_flow = gflownet.calculate_forward_flow(action, P_Forward)
        
        # calculate backward flow
        _, P_Backward = gflownet.calculate_flow_dist(new_state)
        backward_flow = gflownet.calculate_backward_flow(action, P_Backward)
        
        # if valid action, update flows
        if valid:
            gflownet.update_flows(forward_flow, backward_flow)
        else:
            gflownet.update_flows(torch.tensor(0.0), torch.tensor(0.0))
        
        state = new_state
        actions_taken += 1
        
        if stop and state.size() >= MIN_NODE_COUNT:
            break
        
    # calculate reward for completed graph
    reward = gflownet.calculate_reward(state)

    # calculate loss
    loss = (gflownet.logZ + gflownet.total_forward_flow - torch.log(reward).clip(CLIP) - gflownet.total_backward_flow).pow(2)
    minibatch_loss += loss
    
    if minibatch_loss < best_loss:
        best_loss = minibatch_loss
        best_model = copy.deepcopy(gflownet.backbone)
    
    pbar.set_description(f"Epoch: {episode}, Loss: {minibatch_loss.item():.4f}")
    
    if episode % UPDATE_FREQ == 0:
        minibatch_loss.backward()
        opt.step()
        opt.zero_grad()
        minibatch_loss = 0
        

print("Best Loss: ", best_loss.item())
# save best model
torch.save(best_model.state_dict(), 'models/gflownet_backbone/mutag_gflownet_backbone.pt')