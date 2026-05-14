import logging
import os
import pathlib
import pickle

import networkx as nx
import torch
import torch_geometric.utils
from torch_geometric.data import InMemoryDataset, Data

from datasets.abstract_dataset import AbstractDataModule, AbstractDatasetInfos

logger = logging.getLogger(__name__)


class MyTreeGraphDataset(InMemoryDataset):
    def __init__(
        self,
        root,
        split='test',  # 'train', 'val', or 'test'
        transform=None,
        pre_transform=None,
        pre_filter=None,
    ):
        self.split = split
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return [f"{self.split}.pt"]

    @property
    def processed_file_names(self):
        return [f"data_{self.split}.pt"]

    def download(self):
        # No download needed; we use locally generated data
        pass

    def process(self):
        # Load data from the raw file
        raw_path = os.path.join(self.raw_dir, f"{self.split}.pt")
        if not os.path.exists(raw_path):
            raise FileNotFoundError(f"Data file not found: {raw_path}. Please run the dataset generation script first.")

        loaded_data = torch.load(raw_path)

        # Check data format
        if isinstance(loaded_data, tuple) and len(loaded_data) == 2:
            data, slices = loaded_data

            # If slices is None, the data is in Batch format and needs conversion
            if slices is None and hasattr(data, 'to_data_list'):
                logger.info("Converting Batch data with %d graphs...", data.num_graphs)
                # data is a Batch object; split it into individual graphs
                data_list = data.to_data_list()

                # Re-batch using the standard collate method
                data, slices = self.collate(data_list)
                logger.info("Conversion complete. Number of graphs: %d", len(slices['x']) - 1)
            elif slices is not None:
                logger.info("Data already in correct format with %d graphs", len(slices['x']) - 1)
        else:
            # If it is a single Batch object
            if hasattr(loaded_data, 'to_data_list'):
                logger.info("Converting single Batch data with %d graphs...", loaded_data.num_graphs)
                data_list = loaded_data.to_data_list()
                data, slices = self.collate(data_list)
                logger.info("Conversion complete. Number of graphs: %d", len(slices['x']) - 1)
            else:
                logger.warning("Unknown data format, using as-is")
                data = loaded_data
                slices = None

        # Save data in PyG format
        torch.save((data, slices), self.processed_paths[0])


class MyTreeGraphDataModule(AbstractDataModule):
    def __init__(self, cfg):
        self.cfg = cfg
        self.datadir = cfg.dataset.datadir
        base_path = pathlib.Path(os.path.realpath(__file__)).parents[2]
        root_path = os.path.join(base_path, self.datadir)

        # Load train, validation, and test datasets
        train_dataset = MyTreeGraphDataset(root=root_path, split='train')
        val_dataset = MyTreeGraphDataset(root=root_path, split='val')
        test_dataset = MyTreeGraphDataset(root=root_path, split='test')

        datasets = {
            "train": train_dataset,
            "val": val_dataset,
            "test": test_dataset
        }

        super().__init__(cfg, datasets)
        self.inner = self.train_dataset  # Use the training set as the inner dataset


class MyTreeDatasetInfos(AbstractDatasetInfos):
    def __init__(self, datamodule, dataset_config):
        self.datamodule = datamodule
        self.dataset_name = "my_tree"
        self.n_nodes = self.datamodule.node_counts()
        self.node_types = self.datamodule.node_types()
        self.edge_types = self.datamodule.edge_counts()
        super().complete_infos(self.n_nodes, self.node_types)


# Utility function for converting nx.Graph to PyG Data externally
def convert_nx_to_pyg_data(graph: nx.Graph) -> Data:
    """
    Convert a single networkx graph object to a PyTorch Geometric Data object.
    The logic is consistent with generate_tree_testsets.py to ensure format compatibility.
    """
    # Get adjacency matrix from the networkx graph
    adj = torch.Tensor(nx.to_numpy_array(graph))

    n = adj.shape[-1]

    # Node features: all ones, shape [N, 1]
    x = torch.ones(n, 1, dtype=torch.float)

    # Labels: empty, shape [1, 0]
    y = torch.zeros([1, 0]).float()

    # Get sparse edge index from the adjacency matrix
    edge_index, _ = torch_geometric.utils.dense_to_sparse(adj)

    # Edge features: for existing edges, value is [0, 1], shape [num_edges, 2]
    edge_attr = torch.zeros(edge_index.shape[-1], 2, dtype=torch.float)
    edge_attr[:, 1] = 1

    # Number of nodes
    num_nodes = torch.tensor(n, dtype=torch.long).view(1)

    data = Data(
        x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, n_nodes=num_nodes
    )
    return data
