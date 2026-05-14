import logging
import torch
import torch_geometric
import networkx as nx
import os
import sys
from tqdm import tqdm

logger = logging.getLogger(__name__)

# ==================== User Configuration ====================
# 1. Relative path to the project 'src' directory.
#    This allows the script to find and import the SpectreGraphDataset class.
#    Assumes this script is run from src/datasets/; ".." resolves to src/.
PROJECT_SRC_PATH = ".."

# 2. Root directory for the dataset, where the 'processed' subdirectory will be created.
#    Example here uses the 'tree' dataset.
DATASET_ROOT_DIR = "../../data/my_tree"

# 3. Dataset size configuration
TRAIN_GRAPHS = 128  # Number of graphs in the training set
VAL_GRAPHS = 32     # Number of graphs in the validation set
TEST_GRAPHS = 100   # Number of graphs in the test set
# ==============================================================

# --- Add project src directory to Python path ---
try:
    if not os.path.isdir(PROJECT_SRC_PATH):
        raise FileNotFoundError
    sys.path.append(PROJECT_SRC_PATH)
    from datasets.spectre_dataset import SpectreGraphDataset

    logger.info("Successfully imported SpectreGraphDataset from the project.")
except (ImportError, FileNotFoundError):
    logger.error("Failed to find 'datasets.spectre_dataset' at '%s'.", PROJECT_SRC_PATH)
    logger.error("Please ensure PROJECT_SRC_PATH points to your project's 'src' directory.")
    exit()


def convert_nx_to_pyg_data(graph: nx.Graph) -> torch_geometric.data.Data:
    """
    Convert a single networkx graph object to a PyTorch Geometric Data object.
    The logic is consistent with the processing in spectre_dataset.py to ensure format compatibility.
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

    data = torch_geometric.data.Data(
        x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, n_nodes=num_nodes
    )
    return data


def generate_and_process_datasets(
    node_counts,
    train_size=TRAIN_GRAPHS,
    val_size=VAL_GRAPHS,
    test_size=TEST_GRAPHS,
    root_dir=DATASET_ROOT_DIR
):
    """
    Generate train, validation, and test sets for the specified node count list,
    and process them into the format required by the project.

    Args:
        node_counts: List of node counts for each graph.
        train_size: Number of graphs per node count in the training set.
        val_size: Number of graphs per node count in the validation set.
        test_size: Number of graphs per node count in the test set.
        root_dir: Root directory for saving the dataset.
    """
    raw_dir = os.path.join(root_dir, "raw")
    os.makedirs(raw_dir, exist_ok=True)
    logger.info("All processed dataset files will be saved in '%s/'.", raw_dir)

    # Create a temporary Dataset instance to call its collate method
    class DummyDataset:
        def collate(self, data_list):
            return torch_geometric.data.Batch.from_data_list(data_list), None

    dummy_dataset = DummyDataset()

    # Generate data for each dataset split (train, val, test)
    datasets = {
        'train': train_size,
        'val': val_size,
        'test': test_size
    }

    for dataset_name, dataset_size in datasets.items():
        logger.info("Generating %s set...", dataset_name)
        all_data_list = []

        for n_nodes in node_counts:
            logger.info("  Generating %s set for graphs with %d nodes...", dataset_name, n_nodes)

            for _ in tqdm(range(dataset_size), desc=f"  Generating {n_nodes}-node graphs"):
                # 1. Generate a random tree
                if n_nodes > 1:
                    nx_graph = nx.random_tree(n=n_nodes, seed=None)
                else:
                    nx_graph = nx.empty_graph(n=1)

                # 2. Convert to PyG Data object
                pyg_data = convert_nx_to_pyg_data(nx_graph)
                all_data_list.append(pyg_data)

        # 3. Use the project's collate method to batch the data list
        collated_data, slices = dummy_dataset.collate(all_data_list)

        # 4. Define the filename and save the batched data
        file_path = os.path.join(raw_dir, f"{dataset_name}.pt")
        torch.save((collated_data, slices), file_path)

        logger.info("  Successfully generated and batched %d graphs, saved to '%s'", len(all_data_list), file_path)


if __name__ == "__main__":
    # Set the node counts for the graphs to generate.
    # For simplicity, only a single node count of 80 is used here;
    # more node counts can be added as needed.
    node_list = [80]

    generate_and_process_datasets(
        node_counts=node_list,
        train_size=TRAIN_GRAPHS,
        val_size=VAL_GRAPHS,
        test_size=TEST_GRAPHS
    )

    logger.info("Train, validation, and test sets have been successfully generated!")
    logger.info("Dataset statistics:")
    logger.info("   - Training set: %d graphs", len(node_list) * TRAIN_GRAPHS)
    logger.info("   - Validation set: %d graphs", len(node_list) * VAL_GRAPHS)
    logger.info("   - Test set: %d graphs", len(node_list) * TEST_GRAPHS)
    logger.info("Please verify that the data has been generated correctly before training.")
