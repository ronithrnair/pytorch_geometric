import os.path as osp
from typing import Optional, Dict, List, Tuple

import torch
from tqdm import tqdm
from torch_sparse import SparseTensor
from torch_geometric.io import fs
from torch_geometric.typing import SparseTensor
from torch_geometric.data import HeteroData

def hetero_saint_subgraph(
    adj_dict: Dict[Tuple[str, str, str], SparseTensor],  # Heterogeneous adjacency dict
    node_idx_dict: Dict[str, torch.Tensor]  # Sampled node indices per node type
) -> Dict[Tuple[str, str, str], Tuple[SparseTensor, torch.Tensor]]:
    """
    Samples a subgraph from a heterogeneous graph using node indices.

    Args:
        adj_dict: A dictionary where keys are (source_type, relation, target_type) 
                  and values are SparseTensors (adjacency matrices).
        node_idx_dict: A dictionary where keys are node types and values are 
                       Tensors of sampled node indices.

    Returns:
        A dictionary with the same keys but sub-sampled adjacency matrices.
    """
    subgraph_dict = {}

    for (src_type, rel, dst_type), adj in adj_dict.items():
        if src_type not in node_idx_dict or dst_type not in node_idx_dict:
            continue  # Skip relations where we don't have sampled nodes

        src_node_idx = node_idx_dict[src_type]
        dst_node_idx = node_idx_dict[dst_type]

        # Convert adjacency matrix to COO format
        row, col, value = adj.coo()
        rowptr = adj.storage.rowptr()

        # Sampled subgraph using SAINT
        data = torch.ops.torch_sparse.saint_subgraph(src_node_idx, rowptr, row, col)
        sampled_row, sampled_col, edge_index = data

        # Update edge values if present
        if value is not None:
            value = value[edge_index]

        # Create new SparseTensor for the sampled subgraph
        sub_adj = SparseTensor(
            row=sampled_row, col=sampled_col, value=value,
            sparse_sizes=(src_node_idx.size(0), dst_node_idx.size(0)), is_sorted=True
        )

        subgraph_dict[(src_type, rel, dst_type)] = (sub_adj, edge_index)

    return subgraph_dict

class HeteroGraphSAINTSampler(torch.utils.data.DataLoader):
    r"""The HeteroGraphSAINT sampler base class from the `"HeteroGraphSAINT: Graph
    Sampling Based Inductive Learning Method"
    <https://arxiv.org/abs/1907.04931>`_ paper.
    Given a graph in a :obj:`data` object, this class samples nodes and
    constructs subgraphs that can be processed in a mini-batch fashion.
    Normalization coefficients for each mini-batch are given via
    :obj:`node_norm` and :obj:`edge_norm` data attributes.

    .. note::

        See :class:`~torch_geometric.loader.HeteroGraphSAINTNodeSampler`,
        :class:`~torch_geometric.loader.HeteroGraphSAINTEdgeSampler` and
        :class:`~torch_geometric.loader.HeteroGraphSAINTRandomWalkSampler` for
        currently supported samplers.
        For an example of using HeteroGraphSAINT sampling, see
        `examples/graph_saint.py <https://github.com/pyg-team/
        pytorch_geometric/blob/master/examples/graph_saint.py>`_.

    Args:
        data (HeteroData): The heterogeneous graph data object.
        batch_size (int): The approximate number of samples per batch.
        num_steps (int, optional): The number of iterations per epoch.
            (default: :obj:`1`)
        sample_coverage (int): How many samples per node should be used to
            compute normalization statistics. (default: :obj:`0`)
        save_dir (str, optional): If set, will save normalization statistics to
            the :obj:`save_dir` directory for faster re-use.
            (default: :obj:`None`)
        log (bool, optional): If set to :obj:`False`, will not log any
            pre-processing progress. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch.utils.data.DataLoader`, such as :obj:`batch_size` or
            :obj:`num_workers`.
    """
    def __init__(self, data: HeteroData, batch_size: int, num_steps: int = 1,
                 sample_coverage: int = 0, save_dir: Optional[str] = None,
                 log: bool = True, **kwargs):

        # Remove for PyTorch Lightning:
        kwargs.pop('dataset', None)
        kwargs.pop('collate_fn', None)

        self.num_steps = num_steps
        self._batch_size = batch_size
        self.sample_coverage = sample_coverage
        self.save_dir = save_dir
        self.log = log

        self.node_types = data.node_types
        self.edge_types = data.edge_types

        self.adj_dict: Dict[Tuple[str, str, str], SparseTensor] = {}
        for edge_type in self.edge_types:
            edge_index = data[edge_type].edge_index
            assert edge_index is not None, f"Edge index for {edge_type} is None."
            assert not edge_index.is_cuda, f"Edge index for {edge_type} is on GPU."

            N_src = data[edge_type[0]].num_nodes
            N_dst = data[edge_type[2]].num_nodes
            E = edge_index.size(1)

            self.adj_dict[edge_type] = SparseTensor(
                row=edge_index[0], col=edge_index[1],
                value=torch.arange(E, device=edge_index.device),
                sparse_sizes=(N_src, N_dst))

        self.data = data

        super().__init__(self, batch_size=1, collate_fn=self._collate,
                         **kwargs)

        if self.sample_coverage > 0:
            path = osp.join(save_dir or '', self._filename)
            if save_dir is not None and osp.exists(path):  # pragma: no cover
                self.node_norm_dict, self.edge_norm_dict = fs.torch_load(path)
            else:
                self.node_norm_dict, self.edge_norm_dict = self._compute_norm()
                if save_dir is not None:  # pragma: no cover
                    torch.save((self.node_norm_dict, self.edge_norm_dict), path)

    @property
    def _filename(self):
        return f'{self.__class__.__name__.lower()}_{self.sample_coverage}.pt'

    def __len__(self):
        return self.num_steps

    def _sample_nodes(self, batch_size) -> Dict[str, torch.Tensor]:
        raise NotImplementedError

    def create_node_index_map(self, node_index_dict):
        """
        Given a dictionary where key is the node type and value is a tensor of nodes, 
        return a mapping where the key is the node type, and the value is a dictionary 
        mapping each node to its index in the tensor.
        
        Args:
            node_index_dict (dict): {node_type (str): tensor of nodes}
        
        Returns:
            dict: {node_type (str): {node: index in tensor}}
        """
        node_map = {}
        for node_type, nodes in node_index_dict.items():
            node_map[node_type] = {node.item(): idx for idx, node in enumerate(nodes)}
        
        return node_map

    def extract_subgraph(self,node_map, edge_type, src_type, source_nodes,dest_type, dest_nodes, row, column):
        source_nodes_set = set(source_nodes.tolist())
        dest_nodes_set = set(dest_nodes.tolist())
        new_row = []
        new_col = []
        self.selected_edges = {edge_type: []
                          for edge_type in self.edge_types}
        i = 0
        for i in range(0,len(row)):
            if row[i].item() in source_nodes_set and column[i].item() in dest_nodes_set:
                new_row.append(node_map[src_type][row[i].item()])
                new_col.append(node_map[dest_type][column[i].item()])
                self.selected_edges[edge_type].append(i)
        subgraph = SparseTensor(row=torch.tensor(new_row), col=torch.tensor(new_col), sparse_sizes=(max(new_row) + 1, max(new_col) + 1))
        return subgraph

    def __getitem__(self, idx) -> Tuple[Dict[str, torch.Tensor], Dict[Tuple[str, str, str], SparseTensor]]:
        node_idx_dict = self._sample_nodes(self._batch_size)
        node_index_map = self.create_node_index_map(node_idx_dict)
        adj_dict = {}
        for edge_type, adj in self.adj_dict.items():
            src_type, _, dst_type = edge_type
            src_node_idx = node_idx_dict[src_type]
            dst_node_idx = node_idx_dict[dst_type]


            if src_node_idx.numel() == 0 or dst_node_idx.numel() == 0:
                continue

            row, col, value = adj.coo()
            rowptr = adj.storage.rowptr()

            sub_adj = self.extract_subgraph(node_index_map, edge_type, src_type, src_node_idx, dst_type, dst_node_idx, row, col)
            adj_dict[edge_type] = sub_adj

        return node_idx_dict, adj_dict

    def _collate(self, data_list) -> HeteroData:
        assert len(data_list) == 1
        node_idx_dict, adj_dict = data_list[0]
        data = self.data.__class__()
        for node_type, node_idx in node_idx_dict.items():
            data[node_type].num_nodes = node_idx.size(0)

        for edge_type, adj in adj_dict.items():
            row, col, edge_idx = adj.coo()
            data[edge_type].edge_index = torch.stack([row, col], dim=0)

        for key, items in self.data.to_dict().items():
            for k,item in items.items():
                if isinstance(item,torch.Tensor):
                    if key in self.node_types and item.size(0) == self.data[key].num_nodes:
                        data[key][k] = item[node_idx_dict[key]]
                    elif key in self.edge_types and item.size(0) == self.data[key].edge_index.size(1):
                        data[key][k] = item[edge_idx]
                else :
                    data[key][k] = item

        if self.sample_coverage > 0:
            for node_type, node_idx in node_idx_dict.items():
                data[node_type].node_norm = self.node_norm_dict[node_type][node_idx]
            for edge_type in adj_dict:
                data[edge_type].edge_norm = self.edge_norm_dict[edge_type][self.selected_edges[edge_type]]

        return data

    def _compute_norm(self) -> Tuple[Dict[str, torch.Tensor], Dict[Tuple[str, str, str], torch.Tensor]]:
        node_count_dict = {node_type: torch.zeros(self.data[node_type].num_nodes, dtype=torch.float)
                          for node_type in self.node_types}
        edge_count_dict = {edge_type: torch.zeros(self.data[edge_type].edge_index.size(1), dtype=torch.float)
                          for edge_type in self.edge_types}

        loader = torch.utils.data.DataLoader(self, batch_size=200,
                                             collate_fn=lambda x: x,
                                             num_workers=self.num_workers)

        if self.log:  # pragma: no cover
            pbar = tqdm(total=sum(self.data[node_type].num_nodes for node_type in self.node_types) * self.sample_coverage)
            pbar.set_description('Compute HeteroGraphSAINT normalization')

        num_samples = total_sampled_nodes = 0
        while total_sampled_nodes < sum(self.data[node_type].num_nodes for node_type in self.node_types) * self.sample_coverage:
            for data in loader:
                for node_idx_dict, adj_dict in data:
                    for node_type, node_idx in node_idx_dict.items() :
                        node_count_dict[node_type][node_idx] += 1
                    for edge_type, adj in adj_dict.items():
                        edge_idx = adj.storage.value()
                        src_type, _, dst_type = edge_type
                        edge_count_dict[edge_type][edge_idx] += 1
                        total_sampled_nodes += node_idx_dict[src_type].size(0) + node_idx_dict[dst_type].size(0)

                    if self.log:  # pragma: no cover
                        pbar.update(node_idx_dict[src_type].size(0) + node_idx_dict[dst_type].size(0))
            num_samples += self.num_steps
        if self.log:  # pragma: no cover
            pbar.close()
        node_norm_dict = {}
        for node_type, node_count in node_count_dict.items():
            node_count[node_count == 0] = 0.1
            node_norm_dict[node_type] = num_samples / node_count / self.data[node_type].num_nodes
        edge_norm_dict = {}
        for edge_type, edge_count in edge_count_dict.items():
            row, _, edge_idx = self.adj_dict[edge_type].coo()
            src_type, _, dst_type = edge_type
            t = torch.empty_like(edge_count).scatter_(0, edge_idx, node_count_dict[src_type][row])
            edge_norm = (t / edge_count).clamp_(0, 1e4)
            edge_norm[torch.isnan(edge_norm)] = 0.1
            edge_norm_dict[edge_type] = edge_norm
        return node_norm_dict, edge_norm_dict


class HeteroGraphSAINTNodeSampler(HeteroGraphSAINTSampler):
    r"""The HeteroGraphSAINT node sampler class (see
    :class:`~torch_geometric.loader.HeteroGraphSAINTSampler`).
    """
    def _sample_nodes(self, batch_size) -> Dict[str, torch.Tensor]:
        node_idx_dict = {}
        for node_type in self.node_types:
            node_idx = torch.randint(0, self.data[node_type].num_nodes, (batch_size,), dtype=torch.long)
            node_idx = torch.unique(node_idx)
            node_idx_dict[node_type] = node_idx
        return node_idx_dict


class HeteroGraphSAINTEdgeSampler(HeteroGraphSAINTSampler):
    r"""The HeteroGraphSAINT edge sampler class (see
    :class:`~torch_geometric.loader.HeteroGraphSAINTSampler`).
    """
    def _sample_nodes(self, batch_size) -> Dict[str, torch.Tensor]:
        node_idx_dict = {node_type: torch.tensor([], dtype=torch.long) for node_type in self.node_types}
        for edge_type in self.edge_types:
            row, col, _ = self.adj_dict[edge_type].coo()
            src_type, _, dst_type = edge_type

            deg_in = 1. / self.adj_dict[edge_type].storage.colcount()
            deg_out = 1. / self.adj_dict[edge_type].storage.rowcount()
            prob = (1. / deg_in[row]) + (1. / deg_out[col])

            # Parallel multinomial sampling (without replacement)
            rand = torch.rand(batch_size, row.size(0)).log() / (prob + 1e-10)
            edge_sample = rand.topk(self.batch_size, dim=-1).indices

            source_node_sample = col[edge_sample]
            target_node_sample = row[edge_sample]

            node_idx_dict[src_type] = torch.cat([node_idx_dict[src_type], source_node_sample])
            node_idx_dict[dst_type] = torch.cat([node_idx_dict[dst_type], target_node_sample])

        return node_idx_dict


class HeteroGraphSAINTRandomWalkSampler(HeteroGraphSAINTSampler):
    r"""The HeteroGraphSAINT random walk sampler class (see
    :class:`~torch_geometric.loader.HeteroGraphSAINTSampler`).

    Args:
        walk_length (int): The length of each random walk.
    """
    def __init__(self, data: HeteroData, batch_size: int, walk_length: int,
                 num_steps: int = 1, sample_coverage: int = 0,
                 save_dir: Optional[str] = None, log: bool = True, **kwargs):
        self.walk_length = walk_length
        super().__init__(data, batch_size, num_steps, sample_coverage,
                         save_dir, log, **kwargs)

    @property
    def _filename(self):
        return (f'{self.__class__.__name__.lower()}_{self.walk_length}_'
                f'{self.sample_coverage}.pt')

    def _sample_nodes(self, batch_size) -> Dict[str, torch.Tensor]:
        node_idx_dict = {}
        for node_type in self.node_types:
            # Check if a self-loop edge type exists for this node type
            self_loop_edge_type = (node_type, 'to', node_type)
            if self_loop_edge_type in self.adj_dict:
                start = torch.randint(0, self.data[node_type].num_nodes, (batch_size,), dtype=torch.long)
                node_idx = self.adj_dict[self_loop_edge_type].random_walk(start.flatten(), self.walk_length)
                node_idx_dict[node_type] = node_idx.view(-1)
            else:
                # If no self-loop edge type exists, sample nodes uniformly
                # Ensure that node_idx is sampled only from nodes in the train_mask
                if "train_mask" in self.data[node_type]:
                    train_mask = self.data[node_type].train_mask.nonzero(as_tuple=True)[0] # Get indices of train_mask
                    node_idx = train_mask[torch.randint(0, train_mask.size(0), (batch_size,), dtype=torch.long)]
                    # print(node_idx)
                else :
                    node_idx = torch.randint(0, self.data[node_type].num_nodes, (batch_size,), dtype=torch.long)
                node_idx_dict[node_type] = node_idx
        return node_idx_dict