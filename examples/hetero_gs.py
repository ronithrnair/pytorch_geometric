import os.path as osp
import torch
import torch.nn.functional as F
from torch_geometric.datasets import IMDB
from torch_geometric.nn import GraphConv, HeteroConv, Linear
from torch_geometric.loader import HeteroGraphSAINTRandomWalkSampler
from sklearn.metrics import precision_score, recall_score, f1_score
from torch.utils.data import DataLoader, TensorDataset
from torch_geometric.utils import degree


path = osp.join(osp.dirname(osp.realpath(__file__)), 'data', 'IMDB')
save_path = osp.join(osp.dirname(osp.realpath(__file__)), 'processed', 'IMDB')

dataset = IMDB(path)
data = dataset[0]  # First (and only) graph

device = 'cuda' if torch.cuda.is_available() else 'cpu'

in_channels_dict = {node_type: data.x_dict[node_type].shape[1] if data.x_dict[node_type] is not None else 0 for node_type in data.x_dict}

default_dim = 128
for node_type in in_channels_dict:
    if in_channels_dict[node_type] == 0:
        data.x_dict[node_type] = torch.zeros((data[node_type].num_nodes, default_dim))
        in_channels_dict[node_type] = default_dim



for edge_type, edge_index in data.edge_index_dict.items():
    _, dst = edge_index  
    num_nodes = data[edge_type[2]].num_nodes  

    in_degree = degree(dst, num_nodes=num_nodes)  
    norm_weight = 1. / in_degree[dst] 

    data[edge_type].edge_weight = norm_weight  

print(data)

test_mask = data['movie'].test_mask.nonzero(as_tuple=True)[0]
test_dataset = TensorDataset(test_mask)
test_loader = DataLoader(test_dataset, batch_size=1024, shuffle=False) 

class SimpleHeteroGNN(torch.nn.Module):
    def __init__(self, metadata, in_channels_dict, hidden_channels, out_channels):
        super().__init__()

        self.conv = HeteroConv({
            (stype, 'to', dtype): GraphConv(in_channels_dict[stype], out_channels)
            for stype in in_channels_dict for dtype in in_channels_dict if (stype, 'to', dtype) in metadata[1]
        }, aggr='sum')

    def forward(self, x_dict, edge_index_dict):
        x_dict = self.conv(x_dict, edge_index_dict)
        x_dict = {key: F.relu(x) for key, x in x_dict.items()}  # Apply ReLU activation
        return x_dict

class HeteroGNN(torch.nn.Module):
    def __init__(self, metadata, in_channels_dict, hidden_channels, out_channels):
        super().__init__()

        self.conv1 = HeteroConv({
            (stype, 'to', dtype): GraphConv(in_channels_dict[stype], hidden_channels) # Use in_channels_dict
            for stype in in_channels_dict for dtype in in_channels_dict if (stype, 'to', dtype) in data.edge_index_dict # Handle missing edge types
        }, aggr='sum')

        self.conv2 = HeteroConv({
            (stype, 'to', dtype): GraphConv(hidden_channels, out_channels)
            for stype in in_channels_dict for dtype in in_channels_dict if (stype, 'to', dtype) in data.edge_index_dict
        }, aggr='sum')

    def forward(self, x_dict, edge_index_dict, edge_weight_dict = None):
        x_dict = self.conv1(x_dict, edge_index_dict, edge_weight_dict)
        x_dict = {key: F.relu(x) for key, x in x_dict.items()}
        x_dict = self.conv2(x_dict, edge_index_dict, edge_weight_dict)
        # else : 
        #     x_dict = self.conv1(x_dict, edge_index_dict)
        #     x_dict = {key: F.relu(x) for key, x in x_dict.items()}
        #     x_dict = self.conv2(x_dict, edge_index_dict)
        return x_dict

# Initialize model and training components.  Use in_channels_dict!
model = HeteroGNN(metadata=data.metadata(), in_channels_dict = in_channels_dict, hidden_channels=64, out_channels=3).to(device)
# model = SimpleHeteroGNN(metadata=data.metadata(), in_channels_dict = in_channels_dict, hidden_channels=64, out_channels=3).to(device)
print(model)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Initialize the HeteroGraphSAINTRandomWalkSampler
train_loader = HeteroGraphSAINTRandomWalkSampler(
    data, batch_size=200, walk_length=20,
    num_steps=1, sample_coverage=10,
    num_workers=0, save_dir= save_path, training= True
)

test_loader = HeteroGraphSAINTRandomWalkSampler(
    data, batch_size=3000, walk_length=20,
    num_steps=1, sample_coverage=10,
    num_workers=0, save_dir= save_path, testing= True
)


# Training function
def train():
    model.train()
    total_loss = 0
    i = 0
    for batch in train_loader:

        if hasattr(batch, 'edge_weight_dict') and hasattr(batch, 'edge_norm_dict'):
            for k in batch.edge_weight_dict:
                batch[k].edge_weight = batch[k].edge_weight * batch[k].edge_norm

        batch.to(device)     
        optimizer.zero_grad()
        out = model(batch.x_dict, batch.edge_index_dict, batch.edge_weight_dict)

        loss = F.cross_entropy(out['movie'], batch['movie'].y, reduction='none')
        loss = (loss * batch['movie'].node_norm).sum()

        # loss = criterion(out['movie'], batch['movie'].y.squeeze())
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        break

    return total_loss / len(train_loader)



@torch.no_grad()
def test():
    model.eval()
    all_preds = []
    all_labels = []

    # for batch_idx in test_loader:
        # batch_idx = batch_idx[0].to(device)  # Move batch index to GPU
        # batch_x = {k: v.to(device) for k, v in data.x_dict.items()}  # Move features to GPU
        # batch_edges = {k: v.to(device) for k, v in data.edge_index_dict.items()}  # Move edges to GPU
        # out = model(batch_x, batch_edges)
        # pred = out['movie'][batch_idx].argmax(dim=1).cpu().numpy()
        # labels = data['movie'].y.to(device)[batch_idx].cpu().numpy()  

    for batch in test_loader:
        
        if hasattr(batch, 'edge_weight_dict') and hasattr(batch, 'edge_norm_dict'):
            for k in batch.edge_weight_dict:
                batch[k].edge_weight = batch[k].edge_weight * batch[k].edge_norm

        batch.to(device)
        out = model(batch.x_dict, batch.edge_index_dict, batch.edge_weight_dict)
        pred = out['movie'].argmax(dim=1).cpu().numpy()
        labels = batch['movie'].y.squeeze().cpu().numpy()

        all_preds.extend(pred)
        all_labels.extend(labels)

    if len(all_labels) == 0:
        return 0  # Prevent division by zero

    accuracy = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)
    precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    f1_macro = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    f1_micro = f1_score(all_labels, all_preds, average='micro', zero_division=0)

    return accuracy, precision, recall, f1_macro, f1_micro


# Train and evaluate the model
for epoch in range(1, 100):
    loss = train()
    acc, precision, recall, f1_macro, f1_micro = test()
    if epoch%10 == 9 :
        print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}, Accuracy: {acc:.4f}, '
            f'Precision: {precision:.4f}, Recall: {recall:.4f}, '
            f'F1 (Macro): {f1_macro:.4f}, F1 (Micro): {f1_micro:.4f}')