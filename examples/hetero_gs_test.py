import torch
from torch_geometric.data import HeteroData
from torch_geometric.loader import HeteroGraphSAINTNodeSampler

# Define a small heterogeneous graph
data = HeteroData()
# Add user nodes (3 users)
data["user"].num_nodes = 3
# Add item nodes (4 items)
data["item"].num_nodes = 4
data["user"].x = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=torch.float)

# Add item nodes (4 items) with 2-dimensional features
data["item"].x = torch.tensor([[7.0, 8.0], [9.0, 10.0], [11.0, 12.0], [13.0, 14.0]], dtype=torch.float)
# Define edges: users buying items
edge_index = torch.tensor([
    [0, 0,0,0,1,1,1,1,2,2,2,2],  # User IDs (source)
    [0, 1,2,3,0,1,2,3,0,1,2,3]   # Item IDs (destination)
], dtype=torch.long)

data["user", "buys", "item"].edge_index = edge_index

# Step 2: Instantiate the sampler
sampler = HeteroGraphSAINTNodeSampler(data, batch_size=2, num_steps=1, sample_coverage=1)

# Step 3: Run the sampler and print the output
for batch in sampler:
    print(batch.node_norm_dict)