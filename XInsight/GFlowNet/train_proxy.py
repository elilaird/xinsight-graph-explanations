import tqdm
import numpy as np
import copy
import torch
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
import torch.optim.lr_scheduler as lr_scheduler

from gflownet.proxy import GCN_Proxy

USE_MPS = False
DATA_ROOT = './data'
EPOCHS = 4000

if USE_MPS:
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
else:
    device = "cpu"

torch.manual_seed(1)

dataset = TUDataset(root=DATA_ROOT, name='MUTAG')
dataset = dataset.shuffle()

train_dataset = dataset[:150]
test_dataset = dataset[150:]

print(f'Number of training graphs: {len(train_dataset)}')
print(f'Number of test graphs: {len(test_dataset)}')

train_loader = DataLoader(train_dataset, batch_size=48, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=True)


# Load Model
model = GCN_Proxy(num_node_features=7, num_gcn_hidden=[32, 48, 64], num_mlp_hidden=[64, 32], num_classes=2, dropout=0.1).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.02)
scheduler = lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.5, total_iters=EPOCHS)
criterion = torch.nn.CrossEntropyLoss()

print("Proxy Model")
print(model)

def train():
    model.train()

    for data in train_loader:  # Iterate in batches over the training dataset.
        data = data.to(device)
        out = model(data.x, data.edge_index, data.batch)  # Perform a single forward pass.
        loss = criterion(out, data.y)  # Compute the loss.
        loss.backward()  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.
        optimizer.zero_grad()  # Clear gradients.

def test(loader):
     model.eval()

     correct = 0
     for data in loader:  # Iterate in batches over the training/test dataset.
        data = data.to(device)
        out = model(data.x, data.edge_index, data.batch)  
        pred = out.argmax(dim=1)  # Use the class with highest probability.
        correct += int((pred == data.y).sum())  # Check against ground-truth labels.
     return correct / len(loader.dataset)  # Derive ratio of correct predictions.

best_model = None
best_val_acc = 0
best_acc_epoch = 0
train_acc = test_acc = epoch = 0

progress_bar = tqdm.tqdm(range(1, EPOCHS), desc=f"Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}, lr: {scheduler.get_last_lr()[0]:.4f}", unit="epoch")

for epoch in progress_bar:
#for epoch in range(1, EPOCHS):
    train()
    train_acc = test(train_loader)
    test_acc = test(test_loader)
    if test_acc > best_val_acc:
        best_val_acc = test_acc
        best_model = model
        best_acc_epoch = epoch
    scheduler.step()
    progress_bar.set_description(f"Epoch: {epoch:03d}, Train Acc: {train_acc:.2f}, Test Acc: {test_acc:.2f}, lr: {scheduler.get_last_lr()[0]:.4f}")

print(f'Best Test Accuracy: {best_val_acc:.4f} at epoch: {best_acc_epoch}')

#save model
torch.save(best_model.state_dict(), 'models/proxy/mutag_proxy.pt')