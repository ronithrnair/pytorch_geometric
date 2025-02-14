import torch
from torch_geometric.datasets import IMDB
import matplotlib.pyplot as plt

# Load dataset
dataset = IMDB(root="data/IMDB")
data = dataset[0]  # Single graph dataset

print("==== Dataset Overview ====")
print(f"Node types: {data.node_types}")
print(f"Edge types: {data.edge_types}")

# Extract movie node labels and train/val/test masks
labels = data['movie'].y
train_mask = data['movie'].train_mask
val_mask = data['movie'].val_mask
test_mask = data['movie'].test_mask

# Label Distribution for Train, Val, and Test sets
def label_distribution(labels, mask, set_name):
    subset_labels = labels[mask]
    unique, counts = subset_labels.unique(return_counts=True)
    label_counts = dict(zip(unique.tolist(), counts.tolist()))
    print(f"\n==== {set_name} Label Distribution ====")
    for label, count in label_counts.items():
        genre = ["Action", "Comedy", "Drama"][label]
        print(f"{genre} ({label}): {count}")

# Compute label distributions
label_distribution(labels, train_mask, "Training Set")
label_distribution(labels, val_mask, "Validation Set")
label_distribution(labels, test_mask, "Test Set")

# # Plot Label Distribution
# def plot_label_distribution():
#     all_labels = labels.cpu().numpy()
#     plt.hist(all_labels, bins=[-0.5, 0.5, 1.5, 2.5], edgecolor='black', alpha=0.7)
#     plt.xticks([0, 1, 2], ["Action", "Comedy", "Drama"])
#     plt.xlabel("Movie Genre")
#     plt.ylabel("Count")
#     plt.title("Overall Label Distribution in IMDB Dataset")
#     plt.show()

# plot_label_distribution()

print("\n==== Node Features ====")
for node_type in data.x_dict:
    if data.x_dict[node_type] is not None:
        print(f"{node_type} feature shape: {data.x_dict[node_type].shape}")
    else:
        print(f"{node_type} has no features.")

print("\n==== Average No. of Non-Zero Feature Indices Per Entity ====")

for node_type, features in data.x_dict.items():
    if features is not None:
        non_zero_counts = features.count_nonzero(dim=1)  # Count non-zero features per node
        avg_non_zero = non_zero_counts.float().mean().item()  # Compute average
        print(f"Node Type: {node_type}, Average Non-Zero Features: {avg_non_zero:.2f}")
    else:
        print(f"Node Type: {node_type} has no features.")

print("\n==== Edge Information ====")
for edge_type, edge_index in data.edge_index_dict.items():
    print(f"Edge type: {edge_type}, Num edges: {edge_index.shape[1]}")

# Sample Edge Indices for a Chosen Edge Type
print("\n==== Sample Edge Indices for an Edge Type ====")
chosen_edge_type = ('director', 'to', 'movie')  # Example edge type
if chosen_edge_type in data.edge_index_dict:
    sample_edges = data.edge_index_dict[chosen_edge_type][:, :5]  # First 5 edges
    print(f"Edge Type: {chosen_edge_type}, Sample Edges:\n{sample_edges}")
else:
    print(f"Edge type {chosen_edge_type} not found in dataset.")
