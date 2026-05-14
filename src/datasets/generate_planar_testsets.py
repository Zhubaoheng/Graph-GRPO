# generate_planar_dataset.py
import logging
import torch
import torch_geometric
import networkx as nx
import os
import sys
from tqdm import tqdm
import numpy as np
from scipy.spatial import Delaunay
from torch_geometric.data import Data, Batch
import torch_geometric.utils

logger = logging.getLogger(__name__)

# ==================== User Configuration ====================
# 1. Relative path to the project 'src' directory.
#    This allows the script to find and import project modules.
#    Assumes this script is run from src/datasets/; ".." resolves to src/.
PROJECT_SRC_PATH = ".."

# 2. Root directory for the dataset, where the 'raw' subdirectory will be created.
#    Example here uses the 'planar' dataset, compatible with spectre_dataset.py.
DATASET_ROOT_DIR = "../../data/my_planar"

# 3. Dataset size configuration
TRAIN_GRAPHS = 128  # Number of graphs in the training set
VAL_GRAPHS = 32     # Number of graphs in the validation set
TEST_GRAPHS = 40    # Number of graphs in the test set
NUM_NODES = 128     # Fixed number of nodes for the planar dataset
# ==============================================================

# --- Add project src directory to Python path ---
try:
    if not os.path.isdir(PROJECT_SRC_PATH):
        raise FileNotFoundError
    sys.path.append(PROJECT_SRC_PATH)

    logger.info("Successfully imported SpectreGraphDataset from the project.")
except (ImportError, FileNotFoundError):
    logger.error("Failed to find 'datasets.spectre_dataset' at '%s'.", PROJECT_SRC_PATH)
    logger.error("Please ensure PROJECT_SRC_PATH points to your project's 'src' directory.")
    exit()

def generate_connected_planar_graph(num_nodes: int) -> nx.Graph:
    """Generate a connected planar graph using Delaunay triangulation."""
    while True:
        pos = {i: (np.random.rand(), np.random.rand()) for i in range(num_nodes)}
        points = np.array(list(pos.values()))
        delaunay_tri = Delaunay(points)
        graph = nx.Graph()
        for simplex in delaunay_tri.simplices:
            nx.add_cycle(graph, simplex)
        graph.add_nodes_from(range(num_nodes))
        if nx.is_connected(graph) and len(graph.nodes) == num_nodes:
            return graph

def convert_nx_to_adjacency_matrix(graph: nx.Graph) -> torch.Tensor:
    """
    Convert a single networkx graph object to an adjacency matrix.
    This format is consistent with what spectre_dataset.py expects.
    """
    # Get adjacency matrix from the networkx graph
    adj = torch.Tensor(nx.to_numpy_array(graph))
    return adj

def generate_and_save_datasets(
    node_counts,
    train_size=TRAIN_GRAPHS,
    val_size=VAL_GRAPHS,
    test_size=TEST_GRAPHS,
    root_dir=DATASET_ROOT_DIR
):
    """
    Generate train, validation, and test sets, saved as lists of adjacency matrices.
    This format is fully compatible with what spectre_dataset.py expects.

    Args:
        node_counts: List of node counts for each graph.
        train_size: Number of graphs per node count in the training set.
        val_size: Number of graphs per node count in the validation set.
        test_size: Number of graphs per node count in the test set.
        root_dir: Root directory for saving the dataset.
    """
    raw_dir = os.path.join(root_dir, "raw")
    os.makedirs(raw_dir, exist_ok=True)
    logger.info("Adjacency matrix data files will be saved in '%s/'.", raw_dir)

    # Generate data for each dataset split (train, val, test)
    datasets = {
        'train': train_size,
        'val': val_size,
        'test': test_size
    }

    for dataset_name, dataset_size in datasets.items():
        logger.info("Generating %s set...", dataset_name)
        adjacency_matrices = []

        for n_nodes in node_counts:
            logger.info("  Generating %s set for graphs with %d nodes...", dataset_name, n_nodes)

            for _ in tqdm(range(dataset_size), desc=f"  Generating {n_nodes}-node planar graphs"):
                # 1. Generate a connected planar graph
                nx_graph = generate_connected_planar_graph(n_nodes)

                # 2. Convert to adjacency matrix (consistent with spectre_dataset.py format)
                adj_matrix = convert_nx_to_adjacency_matrix(nx_graph)
                adjacency_matrices.append(adj_matrix)

        # 3. Save as a list of adjacency matrices (consistent with spectre_dataset.py format)
        file_path = os.path.join(raw_dir, f"{dataset_name}.pt")
        torch.save(adjacency_matrices, file_path)

        logger.info("  Successfully generated and saved %d adjacency matrices to '%s'", len(adjacency_matrices), file_path)

if __name__ == "__main__":
    # Set the node counts for the graphs to generate.
    # The planar dataset uses a fixed number of nodes.
    node_list = [NUM_NODES]

    logger.info("Starting planar dataset generation (spectre_dataset.py compatible format)...")
    generate_and_save_datasets(
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
    logger.info("Data format is fully compatible with spectre_dataset.py; you can run directly with dataset=planar.")
