import logging
import torch
import os.path as osp
import pathlib
import os
from collections import Counter
from rdkit import Chem, RDLogger
from rdkit.Chem.rdchem import BondType as BT
import torch.nn.functional as F
from tqdm import tqdm
from torch_geometric.data import Data, InMemoryDataset, download_url
from src.analysis.rdkit_functions import (
    mol2smiles,
    build_molecule_with_partial_charges,
)
from src.datasets.abstract_dataset import AbstractDatasetInfos, MolecularDataModule
import numpy as np

logger = logging.getLogger(__name__)


ALL_HASH = "677b757ccec4809febd83850b43e1616"

# Optional: set this environment variable to point to an existing copy of the
# raw SMILES file so the dataset can be loaded without downloading.
# Example: export GUACAMOL_RAW_DIR="/data/guacamol/guacamol_pyg/raw"
_EXTERNAL_RAW_DIR = os.environ.get("GUACAMOL_RAW_DIR", None)


def compare_hash(output_file: str, correct_hash: str) -> bool:
    """
    Computes the md5 hash of a SMILES file and check it against a given one
    Returns false if hashes are different
    """
    import hashlib
    output_hash = hashlib.md5(open(output_file, "rb").read()).hexdigest()
    if output_hash != correct_hash:
        logger.warning(
            "%s file has different hash, %s, than expected, %s!",
            output_file, output_hash, correct_hash,
        )
        return False

    return True


class GuacamolDataset(InMemoryDataset):
    all_url = "https://figshare.com/ndownloader/files/13612745"

    def __init__(
        self,
        stage,
        root,
        filter_dataset: bool,
        transform=None,
        pre_transform=None,
        pre_filter=None,
    ):
        self.stage = stage
        self.filter_dataset = filter_dataset
        if self.stage == "train":
            self.file_idx = 0
        elif self.stage == "val":
            self.file_idx = 1
        else:
            self.file_idx = 2
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[self.file_idx])

    @property
    def raw_file_names(self):
        return ["guacamol_v1_all.smiles"]

    @property
    def split_file_name(self):
        return ["guacamol_v1_all.smiles"]

    @property
    def split_paths(self):
        files = self.split_file_name
        return [osp.join(self.raw_dir, f) for f in files]

    @property
    def processed_file_names(self):
        if self.filter_dataset:
            return ["new_ood_proc_tr.pt", "new_ood_proc_val.pt", "new_ood_proc_test.pt"]
        else:
            return ["old_ood_proc_tr.pt", "old_ood_proc_val.pt", "old_ood_proc_test.pt"]

    def download(self):
        import rdkit  # noqa

        # Check if the file already exists locally
        all_path = osp.join(self.raw_dir, "guacamol_v1_all.smiles")
        external_path = (
            osp.join(_EXTERNAL_RAW_DIR, "guacamol_v1_all.smiles")
            if _EXTERNAL_RAW_DIR
            else None
        )

        if os.path.exists(all_path):
            logger.info("File already exists at %s, skipping download.", all_path)
            return
        elif external_path and os.path.exists(external_path):
            # If the file exists at the external path, copy it to the target location
            import shutil
            os.makedirs(self.raw_dir, exist_ok=True)
            shutil.copy2(external_path, all_path)
            logger.info("File copied from %s to %s", external_path, all_path)
            logger.info("Using existing file, skipping download.")
            return

        # Download the file
        all_path = download_url(self.all_url, self.raw_dir)
        os.rename(all_path, osp.join(self.raw_dir, "guacamol_v1_all.smiles"))
        all_path = osp.join(self.raw_dir, "guacamol_v1_all.smiles")

        # check the hashes
        valid_hash = compare_hash(all_path, ALL_HASH)
        if not valid_hash:
            raise SystemExit("Invalid hash for the dataset file")

        logger.info("Dataset download successful. Hash is correct.")

    def process(self):
        RDLogger.DisableLog("rdApp.*")
        types = {
            "C": 0,
            "N": 1,
            "O": 2,
            "F": 3,
            "B": 4,
            "Br": 5,
            "Cl": 6,
            "I": 7,
            "P": 8,
            "S": 9,
            "Se": 10,
            "Si": 11,
        }
        bonds = {BT.SINGLE: 0, BT.DOUBLE: 1, BT.TRIPLE: 2, BT.AROMATIC: 3}

        # Read all SMILES
        smile_list = open(self.split_paths[0]).readlines()
        logger.info("Total molecules: %d", len(smile_list))

        # Step 1: Compute node count for each molecule
        node_counts = []
        valid_smiles = []

        logger.info("Calculating node counts for all molecules...")
        for i, smile in enumerate(tqdm(smile_list)):
            try:
                mol = Chem.MolFromSmiles(smile)
                if mol is not None:
                    node_counts.append(mol.GetNumAtoms())
                    valid_smiles.append(smile)
            except Exception:
                continue

        logger.info("Valid molecules: %d", len(valid_smiles))

        # Create indices and sort by node count
        indices = list(range(len(valid_smiles)))
        sorted_indices = sorted(indices, key=lambda i: node_counts[i])

        # Split the dataset
        total_count = len(sorted_indices)
        test_count = int(total_count * 0.15)  # 15% for test set
        val_count = int(total_count * 0.05)   # 5% for validation set
        train_count = total_count - test_count - val_count  # 80% for training set

        # Test set: top 15% by node count
        test_indices = sorted_indices[-test_count:]

        # Remaining data: bottom 85% by node count
        remaining_indices = sorted_indices[:-test_count]

        # Randomly select 5% of the remaining data for validation; rest for training
        import random
        random.shuffle(remaining_indices)
        val_indices = remaining_indices[:val_count]
        train_indices = remaining_indices[val_count:]

        # Process data by stage
        if self.stage == "test":
            stage_indices = test_indices
            logger.info("Processing test set with %d molecules (top 15%% by node count)", len(stage_indices))
        elif self.stage == "val":
            stage_indices = val_indices
            logger.info("Processing validation set with %d molecules (5%% of total)", len(stage_indices))
        else:  # train
            stage_indices = train_indices
            logger.info("Processing training set with %d molecules (80%% of total)", len(stage_indices))

        # Process selected molecules
        data_list = []
        smiles_kept = []

        for i in tqdm(stage_indices, desc=f"Processing {self.stage} set"):
            smile = valid_smiles[i]
            mol = Chem.MolFromSmiles(smile)
            N = mol.GetNumAtoms()

            type_idx = []
            for atom in mol.GetAtoms():
                type_idx.append(types[atom.GetSymbol()])

            row, col, edge_type = [], [], []
            for bond in mol.GetBonds():
                start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                row += [start, end]
                col += [end, start]
                edge_type += 2 * [bonds[bond.GetBondType()] + 1]

            if len(row) == 0:
                continue

            edge_index = torch.tensor([row, col], dtype=torch.long)
            edge_type = torch.tensor(edge_type, dtype=torch.long)
            edge_attr = F.one_hot(edge_type, num_classes=len(bonds) + 1).to(torch.float)

            perm = (edge_index[0] * N + edge_index[1]).argsort()
            edge_index = edge_index[:, perm]
            edge_attr = edge_attr[perm]

            x = F.one_hot(torch.tensor(type_idx), num_classes=len(types)).float()
            y = torch.zeros(size=(1, 0), dtype=torch.float)

            data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, idx=i)

            if self.filter_dataset:
                # Try to build the molecule again from the graph. If it fails, do not add it to the training set
                try:
                    from src import utils
                    dense_data, node_mask = utils.to_dense(
                        data.x, data.edge_index, data.edge_attr, data.batch
                    )
                    dense_data = dense_data.mask(node_mask, collapse=True)
                    X, E = dense_data.X, dense_data.E

                    assert X.size(0) == 1
                    atom_types = X[0]
                    edge_types = E[0]
                    atom_decoder = [
                        "C",
                        "N",
                        "O",
                        "F",
                        "B",
                        "Br",
                        "Cl",
                        "I",
                        "P",
                        "S",
                        "Se",
                        "Si",
                    ]
                    mol = build_molecule_with_partial_charges(
                        atom_types, edge_types, atom_decoder
                    )
                    smiles = mol2smiles(mol)
                    if smiles is not None:
                        try:
                            mol_frags = Chem.rdmolops.GetMolFrags(
                                mol, asMols=True, sanitizeFrags=True
                            )
                            if len(mol_frags) == 1:
                                data_list.append(data)
                                smiles_kept.append(smiles)
                        except Chem.rdchem.AtomValenceException:
                            logger.warning("Valence error in GetmolFrags")
                        except Chem.rdchem.KekulizeException:
                            logger.warning("Can't kekulize molecule")
                except Exception as e:
                    logger.warning("Error processing molecule: %s", e)
            else:
                if self.pre_filter is not None and not self.pre_filter(data):
                    continue
                if self.pre_transform is not None:
                    data = self.pre_transform(data)
                data_list.append(data)

        torch.save(self.collate(data_list), self.processed_paths[self.file_idx])
        if self.filter_dataset:
            smiles_save_path = osp.join(
                pathlib.Path(self.raw_paths[0]).parent, f"new_ood_{self.stage}.smiles"
            )
            logger.info("Saving SMILES to %s", smiles_save_path)
            with open(smiles_save_path, "w") as f:
                f.writelines("%s\n" % s for s in smiles_kept)
            logger.info("Number of molecules kept: %d / %d", len(smiles_kept), len(stage_indices))


class GuacamolDataModule(MolecularDataModule):
    def __init__(self, cfg):
        self.remove_h = True
        self.datadir = cfg.dataset.datadir
        self.filter = cfg.dataset.filter
        self.train_smiles = []
        base_path = pathlib.Path(os.path.realpath(__file__)).parents[2]
        root_path = os.path.join(base_path, self.datadir)
        datasets = {
            "train": GuacamolDataset(
                stage="train", root=root_path, filter_dataset=self.filter
            ),
            "val": GuacamolDataset(
                stage="val", root=root_path, filter_dataset=self.filter
            ),
            "test": GuacamolDataset(
                stage="test", root=root_path, filter_dataset=self.filter
            ),
        }
        super().__init__(cfg, datasets)


class Guacamolinfos(AbstractDatasetInfos):
    atom_encoder = {
        "C": 0,
        "N": 1,
        "O": 2,
        "F": 3,
        "B": 4,
        "Br": 5,
        "Cl": 6,
        "I": 7,
        "P": 8,
        "S": 9,
        "Se": 10,
        "Si": 11,
    }
    atom_decoder = ["C", "N", "O", "F", "B", "Br", "Cl", "I", "P", "S", "Se", "Si"]

    def __init__(self, datamodule, cfg, recompute_statistics: bool = True):
        test_only_mode = bool(getattr(cfg.general, "test_only", None))
        recompute_statistics = recompute_statistics or test_only_mode
        self.name = "Guacamol"
        self.input_dims = None
        self.output_dims = None
        self.remove_h = True
        self.compute_fcd = cfg.dataset.compute_fcd
        self.num_atom_types = 12
        self.max_weight = 1000

        self.valencies = [4, 3, 2, 1, 3, 1, 1, 1, 3, 2, 2, 4]

        self.atom_weights = {
            1: 12,
            2: 14,
            3: 16,
            4: 19,
            5: 10.81,
            6: 79.9,
            7: 35.45,
            8: 126.9,
            9: 30.97,
            10: 30.07,
            11: 78.97,
            12: 28.09,
        }

        # Initialized to None; will be recomputed later
        self.node_types = None
        self.edge_types = None
        self.n_nodes = None

        stats_tag = "test" if test_only_mode else "trainval"
        stats_path = pathlib.Path(datamodule.train_dataset.root) / f"guacamol_stats_{stats_tag}.pt"
        stats_cache = None
        if stats_path.exists():
            try:
                stats_cache = torch.load(stats_path)
                logger.info("Loaded cached Guacamol statistics from %s", stats_path)
            except Exception as exc:
                logger.warning("Failed to load cached Guacamol statistics: %s", exc)
                stats_cache = None

        if recompute_statistics:
            if stats_cache is None:
                stat_splits = ["test"] if test_only_mode else ["train", "val"]
                logger.info("Recomputing Guacamol statistics from splits: %s", stat_splits)

                self.n_nodes = _compute_guacamol_node_counts(datamodule, stat_splits)
                logger.info("Distribution of number of nodes: %s", self.n_nodes)

                self.node_types = _compute_guacamol_node_types(datamodule, stat_splits)
                logger.info("Distribution of node types: %s", self.node_types)

                self.edge_types = _compute_guacamol_edge_types(datamodule, stat_splits)
                logger.info("Distribution of edge types: %s", self.edge_types)

                valencies = _compute_guacamol_valencies(
                    datamodule, stat_splits, max_n_nodes=len(self.n_nodes) - 1
                )
                logger.info("Distribution of the valencies: %s", valencies)

                stats_cache = {
                    "n_nodes": self.n_nodes.cpu(),
                    "node_types": self.node_types.cpu(),
                    "edge_types": self.edge_types.cpu(),
                    "valencies": valencies.cpu(),
                }
                try:
                    torch.save(stats_cache, stats_path)
                    logger.info("Saved Guacamol statistics to %s", stats_path)
                except Exception as exc:
                    logger.warning("Failed to save Guacamol statistics: %s", exc)
            else:
                self.n_nodes = stats_cache["n_nodes"]
                self.node_types = stats_cache["node_types"]
                self.edge_types = stats_cache["edge_types"]
                valencies = stats_cache["valencies"]
                logger.info("Using cached Guacamol statistics (%s).", stats_tag)

            self.complete_infos(n_nodes=self.n_nodes, node_types=self.node_types)
            self.valency_distribution = valencies
        else:
            # Use default distributions (same as original Guacamol)
            self.node_types = torch.tensor([
                7.4090e-01,
                1.0693e-01,
                1.1220e-01,
                1.4213e-02,
                6.0579e-05,
                1.7171e-03,
                8.4113e-03,
                2.2902e-04,
                5.6947e-04,
                1.4673e-02,
                4.1532e-05,
                5.3416e-05,
            ])

            self.edge_types = torch.tensor([
                9.2526e-01,
                3.6241e-02,
                4.8489e-03,
                1.6513e-04,
                3.3489e-02,
            ])

            self.n_nodes = torch.tensor([
                0,
                0,
                3.5760e-06,
                2.7893e-05,
                6.9374e-05,
                1.6020e-04,
                2.8036e-04,
                4.3484e-04,
                7.3022e-04,
                1.1722e-03,
                1.7830e-03,
                2.8129e-03,
                4.0981e-03,
                5.5421e-03,
                7.9645e-03,
                1.0824e-02,
                1.4459e-02,
                1.8818e-02,
                2.3961e-02,
                2.9558e-02,
                3.6324e-02,
                4.1931e-02,
                4.8105e-02,
                5.2316e-02,
                5.6601e-02,
                5.7483e-02,
                5.6685e-02,
                5.2317e-02,
                5.2107e-02,
                4.9651e-02,
                4.8100e-02,
                4.4363e-02,
                4.0704e-02,
                3.5719e-02,
                3.1685e-02,
                2.6821e-02,
                2.2542e-02,
                1.8591e-02,
                1.6114e-02,
                1.3399e-02,
                1.1543e-02,
                9.6116e-03,
                8.4744e-03,
                6.9532e-03,
                6.2001e-03,
                4.9921e-03,
                4.4378e-03,
                3.5803e-03,
                3.3078e-03,
                2.7085e-03,
                2.6784e-03,
                2.2050e-03,
                2.0533e-03,
                1.5598e-03,
                1.5177e-03,
                9.8626e-04,
                8.6396e-04,
                5.6429e-04,
                5.0422e-04,
                2.9323e-04,
                2.2243e-04,
                9.8697e-05,
                9.9413e-05,
                6.0077e-05,
                6.9374e-05,
                3.0754e-05,
                3.5045e-05,
                1.6450e-05,
                2.1456e-05,
                1.2874e-05,
                1.2158e-05,
                5.7216e-06,
                7.1520e-06,
                2.8608e-06,
                2.8608e-06,
                7.1520e-07,
                2.8608e-06,
                1.4304e-06,
                7.1520e-07,
                0.0000e00,
                0.0000e00,
                0.0000e00,
                7.1520e-07,
                0.0000e00,
                1.4304e-06,
                7.1520e-07,
                7.1520e-07,
                0.0000e00,
                1.4304e-06,
            ])

            # Call complete_infos first to set max_n_nodes
            self.complete_infos(n_nodes=self.n_nodes, node_types=self.node_types)

            # Initialize valency_distribution with a zero tensor of the same dimension
            # as the training data; this will be updated when reference metrics are first computed
            self.valency_distribution = torch.zeros(self.max_n_nodes * 3 - 2)
            # Provide reasonable default values
            default_valency_values = torch.tensor([0.0000, 0.1105, 0.2645, 0.3599, 0.2552, 0.0046, 0.0053])
            self.valency_distribution[:len(default_valency_values)] = default_valency_values


def _get_guacamol_loaders(datamodule, splits):
    loader_map = {
        "train": datamodule.train_dataloader,
        "val": datamodule.val_dataloader,
        "test": datamodule.test_dataloader,
    }
    loaders = []
    for split in splits:
        loader_fn = loader_map.get(split)
        if loader_fn is None:
            continue
        loader = loader_fn()
        if loader is not None:
            loaders.append(loader)
    if not loaders:
        raise RuntimeError(f"No dataloaders available for splits: {splits}")
    return loaders


def _compute_guacamol_node_counts(datamodule, splits):
    loaders = _get_guacamol_loaders(datamodule, splits)
    counter = Counter()
    for loader in loaders:
        for data in loader:
            _, counts = torch.unique(data.batch, return_counts=True)
            for count in counts:
                counter[int(count.item())] += 1
    if not counter:
        raise RuntimeError("Unable to compute node counts; received no graphs.")
    max_nodes = max(counter.keys())
    counts = torch.zeros(max_nodes + 1, dtype=torch.float)
    for nodes, freq in counter.items():
        counts[nodes] = freq
    return counts / counts.sum()


def _compute_guacamol_node_types(datamodule, splits):
    loaders = _get_guacamol_loaders(datamodule, splits)
    counts = None
    for loader in loaders:
        for data in loader:
            if counts is None:
                counts = torch.zeros(data.x.shape[1], dtype=torch.float)
            counts += data.x.sum(dim=0)
    if counts is None:
        raise RuntimeError("Unable to compute node type distribution.")
    return counts / counts.sum()


def _compute_guacamol_edge_types(datamodule, splits):
    loaders = _get_guacamol_loaders(datamodule, splits)
    counts = None
    for loader in loaders:
        for data in loader:
            if counts is None:
                counts = torch.zeros(data.edge_attr.shape[1], dtype=torch.float)
            unique, batch_counts = torch.unique(data.batch, return_counts=True)
            all_pairs = torch.sum(batch_counts * (batch_counts - 1)).item()
            num_edges = int(data.edge_index.shape[1])
            num_non_edges = all_pairs - num_edges
            edge_types = data.edge_attr.sum(dim=0)
            counts[0] += float(num_non_edges)
            counts[1:] += edge_types[1:]
    if counts is None:
        raise RuntimeError("Unable to compute edge type distribution.")
    return counts / counts.sum()


_VALENCY_MULTIPLIER = torch.tensor([0.0, 1.0, 2.0, 3.0, 1.5])


def _compute_guacamol_valencies(datamodule, splits, max_n_nodes):
    loaders = _get_guacamol_loaders(datamodule, splits)
    valencies = torch.zeros(3 * max_n_nodes - 2, dtype=torch.float)
    for loader in loaders:
        for data in loader:
            multiplier = _VALENCY_MULTIPLIER.to(data.edge_attr)
            n = data.x.shape[0]
            for atom in range(n):
                edges = data.edge_attr[data.edge_index[0] == atom]
                if edges.numel() == 0:
                    valency = 0
                else:
                    edges_total = edges.sum(dim=0)
                    valency = (edges_total * multiplier).sum()
                idx = int(valency.long().item())
                idx = min(idx, valencies.size(0) - 1)
                valencies[idx] += 1
    total = valencies.sum()
    if total == 0:
        return valencies
    return valencies / total


def get_smiles(raw_dir, filter_dataset):
    if filter_dataset:
        smiles_save_paths = {
            "train": osp.join(raw_dir, "new_ood_train.smiles"),
            "val": osp.join(raw_dir, "new_ood_val.smiles"),
            "test": osp.join(raw_dir, "new_ood_test.smiles"),
        }
    else:
        # If dataset is not filtered, use the default filename
        smiles_save_paths = {
            "train": osp.join(raw_dir, "guacamol_v1_all.smiles"),
            "val": osp.join(raw_dir, "guacamol_v1_all.smiles"),
            "test": osp.join(raw_dir, "guacamol_v1_all.smiles"),
        }

    def extract_smiles_from_file(file_path):
        with open(file_path, "r") as f:
            lines = [line.strip() for line in f.readlines()]
        return lines

    return {
        "train": extract_smiles_from_file(smiles_save_paths["train"]),
        "val": extract_smiles_from_file(smiles_save_paths["val"]),
        "test": extract_smiles_from_file(smiles_save_paths["test"]),
    }
