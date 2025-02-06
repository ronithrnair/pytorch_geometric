import time
import os.path as osp

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.nn import Sequential, Linear, ReLU, Dropout, ModuleList

from torch_geometric.datasets import IMDB
from torch_geometric.nn import NARS
from sklearn.metrics import f1_score

# Load the IMDB dataset
root = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'IMDB')
dataset = IMDB(root)
data = dataset[0]  # Get the graph data object
print(data)

# Set device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using device: {device}')
data = data.to(device)

# Ensure edge_index_dict is moved to GPU
for key in data.edge_index_dict:
    data.edge_index_dict[key] = data.edge_index_dict[key].to(device)

# Initialize NARS
nars = NARS(num_hops=3, num_sampled_subsets=4, num_features=data.x_dict['movie'].size(-1)).to(device)

# Pre-processing features
t = time.perf_counter()
print('Pre-processing features...', end=' ', flush=True)

aggr_dict = nars(data.x_dict, data.edge_index_dict)
print(f'Done! [{time.perf_counter() - t:.2f}s]')

# Extract 'movie' node embeddings and labels
x = aggr_dict['movie']  # [num_hops, num_subsets, num_nodes, num_features]
y = data.y_dict['movie'].view(-1)  # Labels for movies
del aggr_dict
del data

# Define train/val/test splits
num_nodes = x.size(2)
train_idx = torch.randperm(num_nodes)[:int(0.8 * num_nodes)]
val_idx = torch.randperm(num_nodes)[int(0.8 * num_nodes):int(0.9 * num_nodes)]
test_idx = torch.randperm(num_nodes)[int(0.9 * num_nodes):]

class SIGN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_hops, dropout=0.5):
        super(SIGN, self).__init__()

        self.mlps = ModuleList()
        for _ in range(num_hops):
            mlp = Sequential(
                Linear(in_channels, hidden_channels),
                ReLU(),
                Linear(hidden_channels, hidden_channels),
                ReLU(),
            )
            self.mlps.append(mlp)

        self.mlp = Sequential(
            Dropout(dropout),
            Linear(num_hops * hidden_channels, hidden_channels),
            ReLU(),
            Dropout(dropout),
            Linear(hidden_channels, out_channels),
        )

    def forward(self, xs):
        outs = [mlp(x) for x, mlp in zip(xs, self.mlps)]
        return self.mlp(torch.cat(outs, dim=-1))

num_classes = y.max().item() + 1  # Get the number of unique classes from y
sign = SIGN(x.size(-1), 256, num_classes, num_hops=3).to(device)

optimizer = torch.optim.Adam(
    list(nars.parameters()) + list(sign.parameters()), lr=0.001)


def train(x, y, idx):
    nars.train()
    sign.train()

    total_loss = num_examples = 0
    for idx in DataLoader(idx, batch_size=10000, shuffle=True):
        x_mini = x[:, :, idx].to(device)
        y_mini = y[idx].to(device)

        optimizer.zero_grad()
        x_mini = nars.weighted_aggregation(x_mini)
        out = sign(x_mini.unbind(0))
        loss = F.cross_entropy(out, y_mini)
        loss.backward()
        optimizer.step()

        total_loss += float(loss) * idx.numel()
        num_examples += idx.numel()

    return total_loss / num_examples



@torch.no_grad()
def test(x, y, idx):
    total_correct = num_examples = 0
    y_true, y_pred = [], []

    for idx in DataLoader(idx, batch_size=20000):
        x_mini = x[:, :, idx].to(device)
        y_mini = y[idx].to(device)

        x_mini = nars.weighted_aggregation(x_mini)
        pred = sign(x_mini.unbind(0)).argmax(dim=-1)

        total_correct += int((pred == y_mini).sum())
        num_examples += idx.numel()

        # Collect labels for F1-score computation
        y_true.extend(y_mini.cpu().numpy())
        y_pred.extend(pred.cpu().numpy())

    acc = total_correct / num_examples
    micro_f1 = f1_score(y_true, y_pred, average='micro')
    macro_f1 = f1_score(y_true, y_pred, average='macro')

    return acc, micro_f1, macro_f1

for epoch in range(1, 101):
    loss = train(x, y, train_idx)
    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')
    
    if epoch % 10 == 0:
        train_acc, train_micro_f1, train_macro_f1 = test(x, y, train_idx)
        val_acc, val_micro_f1, val_macro_f1 = test(x, y, val_idx)
        test_acc, test_micro_f1, test_macro_f1 = test(x, y, test_idx)
        
        print(f'Train: Acc={train_acc:.4f}, Micro-F1={train_micro_f1:.4f}, Macro-F1={train_macro_f1:.4f}')
        print(f'Val: Acc={val_acc:.4f}, Micro-F1={val_micro_f1:.4f}, Macro-F1={val_macro_f1:.4f}')
        print(f'Test: Acc={test_acc:.4f}, Micro-F1={test_micro_f1:.4f}, Macro-F1={test_macro_f1:.4f}')
