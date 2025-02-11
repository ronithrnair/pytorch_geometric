import os.path as osp
import torch
import torch.nn.functional as F
from torch_geometric.datasets import IMDB
from torch_geometric.nn import SAGEConv, HeteroConv, Linear
from torch_geometric.loader import HeteroGraphSAINTRandomWalkSampler

# Load the IMDB dataset
path = osp.join(osp.dirname(osp.realpath(__file__)), 'data', 'IMDB')
dataset = IMDB(path)
data = dataset[0]  # First (and only) graph
# Get the actual input feature dimensions for each node type
in_channels_dict = {node_type: data.x_dict[node_type].shape[1] if data.x_dict[node_type] is not None else 0 for node_type in data.x_dict}

# If a node type has no features, use a default dimension (you can adjust this)
default_dim = 128
for node_type in in_channels_dict:
    if in_channels_dict[node_type] == 0:
        data.x_dict[node_type] = torch.zeros((data[node_type].num_nodes, default_dim))
        in_channels_dict[node_type] = default_dim

class HeteroGNN(torch.nn.Module):
    def __init__(self, metadata, in_channels_dict, hidden_channels, out_channels):
        super().__init__()

        self.conv1 = HeteroConv({
            (stype, 'to', dtype): SAGEConv(in_channels_dict[stype], hidden_channels) # Use in_channels_dict
            for stype in in_channels_dict for dtype in in_channels_dict if (stype, 'to', dtype) in data.edge_index_dict # Handle missing edge types
        }, aggr='sum')

        self.conv2 = HeteroConv({
            (stype, 'to', dtype): SAGEConv(hidden_channels, out_channels)
            for stype in in_channels_dict for dtype in in_channels_dict if (stype, 'to', dtype) in data.edge_index_dict
        }, aggr='sum')

    def forward(self, x_dict, edge_index_dict):
        x_dict = self.conv1(x_dict, edge_index_dict)
        x_dict = {key: F.relu(x) for key, x in x_dict.items()}
        x_dict = self.conv2(x_dict, edge_index_dict)
        return x_dict

# Initialize model and training components.  Use in_channels_dict!
model = HeteroGNN(metadata=data.metadata(), in_channels_dict = in_channels_dict, hidden_channels=64, out_channels=3)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.CrossEntropyLoss()

# Initialize the HeteroGraphSAINTRandomWalkSampler
train_loader = HeteroGraphSAINTRandomWalkSampler(
    data, batch_size=200, walk_length=20,
    num_steps=1, sample_coverage=100,
    num_workers=0  # num_workers=4 may cause issues
)

test_loader = HeteroGraphSAINTRandomWalkSampler(
    data, batch_size=200, walk_length=20,
    num_steps=1, sample_coverage=100,
    num_workers=0  # num_workers=4 may cause issues
)

# Training function
def train():
    model.train()
    total_loss = 0

    for batch in train_loader:        
        optimizer.zero_grad()
        out = model(batch.x_dict, batch.edge_index_dict)

        loss = criterion(out['movie'], batch['movie'].y.squeeze())
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(train_loader)

# Testing function
def test():
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in test_loader:
            out = model(batch.x_dict, batch.edge_index_dict)
            pred = out['movie'].argmax(dim=1)
            correct += (pred == batch['movie'].y.squeeze()).sum().item()
            total += batch['movie'].y.size(0)
    return correct / total if total > 0 else 0  # Prevent division by zero

# Train and evaluate the model
for epoch in range(1, 100):
    loss = train()
    acc = test()
    print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}, Accuracy: {acc:.4f}')