# my_planar_dataset.py
import logging
import os
import pathlib
import torch
import torch_geometric.utils
from torch_geometric.data import InMemoryDataset, Data
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
import networkx as nx

from datasets.abstract_dataset import AbstractDataModule, AbstractDatasetInfos

logger = logging.getLogger(__name__)

# --- Follows the structure and logic of spectre_dataset.py ---

class MyPlanarGraphDataset(InMemoryDataset):
    def __init__(
        self,
        root,
        split='test',
        transform=None,
        pre_transform=None,
        pre_filter=None,
    ):
        self.split = split
        super().__init__(root, transform, pre_transform, pre_filter)
        # Load the final standard data produced by the process() method
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        # The process() method reads this file from the raw folder
        return [f"{self.split}.pt"]

    @property
    def processed_file_names(self):
        # The process() method saves the processed result with this filename
        return [f"data_{self.split}.pt"]

    def download(self):
        # No download needed; we use locally generated data
        pass

    def process(self):
        """
        Process data following the approach in spectre_dataset.py.
        Reads raw data, creates PyG Data objects, then saves using the collate method.
        """
        # Load data from the raw file
        raw_path = os.path.join(self.raw_dir, f"{self.split}.pt")
        if not os.path.exists(raw_path):
            raise FileNotFoundError(f"Data file not found: {raw_path}. Please run the dataset generation script first.")

        loaded_data = torch.load(raw_path)

        # Handle different data formats
        if isinstance(loaded_data, list):
            # If it is a list of adjacency matrices (official dataset format),
            # process consistently with spectre_dataset.py
            data_list = []
            for adj in loaded_data:
                n = adj.shape[-1]
                X = torch.ones(n, 1, dtype=torch.float)
                y = torch.zeros([1, 0]).float()
                edge_index, _ = torch_geometric.utils.dense_to_sparse(adj)
                edge_attr = torch.zeros(edge_index.shape[-1], 2, dtype=torch.float)
                edge_attr[:, 1] = 1
                num_nodes = n * torch.ones(1, dtype=torch.long)

                data_obj = Data(
                    x=X, edge_index=edge_index, edge_attr=edge_attr, y=y, n_nodes=num_nodes
                )

                if self.pre_filter is not None and not self.pre_filter(data_obj):
                    continue
                if self.pre_transform is not None:
                    data_obj = self.pre_transform(data_obj)

                data_list.append(data_obj)
        elif isinstance(loaded_data, tuple) and len(loaded_data) == 2:
            # If it is in (data, slices) format, convert back to data_list
            data, slices = loaded_data
            if slices is not None:
                # Extract individual graph data from collated data
                data_list = []
                num_graphs = len(slices['x']) - 1
                for i in range(num_graphs):
                    start_idx = slices['x'][i]
                    end_idx = slices['x'][i + 1]

                    # Extract node features
                    x = data.x[start_idx:end_idx]
                    n = x.shape[0]

                    # Reconstruct adjacency matrix
                    edge_start = slices['edge_index'][i]
                    edge_end = slices['edge_index'][i + 1]
                    edges = data.edge_index[:, edge_start:edge_end] - start_idx

                    # Create adjacency matrix
                    adj = torch.zeros(n, n)
                    adj[edges[0], edges[1]] = 1

                    # Re-create Data object following spectre_dataset conventions
                    X = torch.ones(n, 1, dtype=torch.float)
                    y = torch.zeros([1, 0]).float()
                    edge_index, _ = torch_geometric.utils.dense_to_sparse(adj)
                    edge_attr = torch.zeros(edge_index.shape[-1], 2, dtype=torch.float)
                    edge_attr[:, 1] = 1
                    num_nodes = n * torch.ones(1, dtype=torch.long)

                    data_obj = Data(
                        x=X, edge_index=edge_index, edge_attr=edge_attr, y=y, n_nodes=num_nodes
                    )

                    if self.pre_filter is not None and not self.pre_filter(data_obj):
                        continue
                    if self.pre_transform is not None:
                        data_obj = self.pre_transform(data_obj)

                    data_list.append(data_obj)
            else:
                # If it is in Batch format, convert to data_list
                if hasattr(data, 'to_data_list'):
                    original_data_list = data.to_data_list()
                    data_list = []

                    for data_obj in original_data_list:
                        # Re-create Data object following spectre_dataset conventions
                        n = data_obj.x.shape[0]

                        # Reconstruct adjacency matrix
                        adj = torch.zeros(n, n)
                        adj[data_obj.edge_index[0], data_obj.edge_index[1]] = 1

                        X = torch.ones(n, 1, dtype=torch.float)
                        y = torch.zeros([1, 0]).float()
                        edge_index, _ = torch_geometric.utils.dense_to_sparse(adj)
                        edge_attr = torch.zeros(edge_index.shape[-1], 2, dtype=torch.float)
                        edge_attr[:, 1] = 1
                        num_nodes = n * torch.ones(1, dtype=torch.long)

                        new_data_obj = Data(
                            x=X, edge_index=edge_index, edge_attr=edge_attr, y=y, n_nodes=num_nodes
                        )

                        if self.pre_filter is not None and not self.pre_filter(new_data_obj):
                            continue
                        if self.pre_transform is not None:
                            new_data_obj = self.pre_transform(new_data_obj)

                        data_list.append(new_data_obj)
                else:
                    raise ValueError(f"Unknown data format: {type(data)}")
        else:
            # If it is a single Batch object
            if hasattr(loaded_data, 'to_data_list'):
                original_data_list = loaded_data.to_data_list()
                data_list = []

                for data_obj in original_data_list:
                    # Re-create Data object following spectre_dataset conventions
                    n = data_obj.x.shape[0]

                    # Reconstruct adjacency matrix
                    adj = torch.zeros(n, n)
                    adj[data_obj.edge_index[0], data_obj.edge_index[1]] = 1

                    X = torch.ones(n, 1, dtype=torch.float)
                    y = torch.zeros([1, 0]).float()
                    edge_index, _ = torch_geometric.utils.dense_to_sparse(adj)
                    edge_attr = torch.zeros(edge_index.shape[-1], 2, dtype=torch.float)
                    edge_attr[:, 1] = 1
                    num_nodes = n * torch.ones(1, dtype=torch.long)

                    new_data_obj = Data(
                        x=X, edge_index=edge_index, edge_attr=edge_attr, y=y, n_nodes=num_nodes
                    )

                    if self.pre_filter is not None and not self.pre_filter(new_data_obj):
                        continue
                    if self.pre_transform is not None:
                        new_data_obj = self.pre_transform(new_data_obj)

                    data_list.append(new_data_obj)
            else:
                raise ValueError(f"Unknown data format: {type(loaded_data)}")

        # Save data using the collate method (consistent with spectre_dataset.py)
        torch.save(self.collate(data_list), self.processed_paths[0])


class MyPlanarGraphDataModule(AbstractDataModule):
    def __init__(self, cfg):
        self.cfg = cfg
        self.datadir = cfg.dataset.datadir
        base_path = pathlib.Path(os.path.realpath(__file__)).parents[2]
        root_path = os.path.join(base_path, self.datadir)

        # Load train, validation, and test datasets
        train_dataset = MyPlanarGraphDataset(root=root_path, split='train')
        val_dataset = MyPlanarGraphDataset(root=root_path, split='val')
        test_dataset = MyPlanarGraphDataset(root=root_path, split='test')

        datasets = {
            "train": train_dataset,
            "val": val_dataset,
            "test": test_dataset
        }

        # Print dataset size info (consistent with spectre_dataset.py)
        train_len = len(datasets["train"].data.n_nodes)
        val_len = len(datasets["val"].data.n_nodes)
        test_len = len(datasets["test"].data.n_nodes)
        logger.info("Dataset sizes: train %d, val %d, test %d", train_len, val_len, test_len)

        super().__init__(cfg, datasets)
        self.inner = self.train_dataset  # Use the training set as the inner dataset

    def __getitem__(self, item):
        return self.inner[item]


class MyPlanarDatasetInfos(AbstractDatasetInfos):
    def __init__(self, datamodule, dataset_config):
        self.datamodule = datamodule
        self.dataset_name = "my_planar"
        self.n_nodes = self.datamodule.node_counts()
        self.node_types = self.datamodule.node_types()
        self.edge_types = self.datamodule.edge_counts()
        super().complete_infos(self.n_nodes, self.node_types)


# Utility function for converting nx.Graph to PyG Data externally
def convert_nx_to_pyg_data(graph: nx.Graph) -> Data:
    """
    Convert a single networkx graph object to a PyTorch Geometric Data object.
    The logic is consistent with the processing in spectre_dataset.py to ensure format compatibility.
    """
    # Get adjacency matrix from the networkx graph
    adj = torch.Tensor(nx.to_numpy_array(graph))

    n = adj.shape[-1]

    # Create Data object following the standard format of spectre_dataset.py
    X = torch.ones(n, 1, dtype=torch.float)
    y = torch.zeros([1, 0]).float()
    edge_index, _ = torch_geometric.utils.dense_to_sparse(adj)
    edge_attr = torch.zeros(edge_index.shape[-1], 2, dtype=torch.float)
    edge_attr[:, 1] = 1
    num_nodes = n * torch.ones(1, dtype=torch.long)

    data = Data(
        x=X, edge_index=edge_index, edge_attr=edge_attr, y=y, n_nodes=num_nodes
    )
    return data
