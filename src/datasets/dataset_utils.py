import os.path as osp
import pickle
from typing import Any, Sequence

from rdkit import Chem
import torch
from torch_geometric.data import Data
from torch_geometric.utils import subgraph


def mol_to_torch_geometric(mol, atom_encoder, smiles):
    adj = torch.from_numpy(Chem.rdmolops.GetAdjacencyMatrix(mol, useBO=True))
    edge_index = adj.nonzero().contiguous().T
    bond_types = adj[edge_index[0], edge_index[1]]
    bond_types[bond_types == 1.5] = 4
    edge_attr = bond_types.long()

    node_types = []
    all_charge = []
    for atom in mol.GetAtoms():
        node_types.append(atom_encoder[atom.GetSymbol()])
        all_charge.append(atom.GetFormalCharge())

    node_types = torch.Tensor(node_types).long()
    all_charge = torch.Tensor(all_charge).long()

    data = Data(
        x=node_types,
        edge_index=edge_index,
        edge_attr=edge_attr,
        charge=all_charge,
        smiles=smiles,
    )
    return data


def remove_hydrogens(data: Data):
    to_keep = data.x > 0
    new_edge_index, new_edge_attr = subgraph(
        to_keep,
        data.edge_index,
        data.edge_attr,
        relabel_nodes=True,
        num_nodes=len(to_keep),
    )
    return Data(
        x=data.x[to_keep] - 1,  # Shift onehot encoding to match atom decoder
        charge=data.charge[to_keep],
        edge_index=new_edge_index,
        edge_attr=new_edge_attr,
    )


def save_pickle(array, path):
    with open(path, "wb") as f:
        pickle.dump(array, f)


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def files_exist(files) -> bool:
    return len(files) != 0 and all([osp.exists(f) for f in files])


def to_list(value: Any) -> Sequence:
    if isinstance(value, Sequence) and not isinstance(value, str):
        return value
    else:
        return [value]


class Statistics:
    def __init__(
        self, num_nodes, node_types, bond_types, charge_types=None, valencies=None
    ):
        self.num_nodes = num_nodes
        self.node_types = node_types
        self.bond_types = bond_types
        self.charge_types = charge_types
        self.valencies = valencies


class RemoveYTransform:
    def __call__(self, data):
        data.y = torch.zeros((1, 0), dtype=torch.float)
        return data


class DistributionNodes:
    def __init__(self, histogram):
        """Compute the distribution of the number of nodes in the dataset, and sample from this distribution.
        historgram: dict. The keys are num_nodes, the values are counts
        """

        if type(histogram) == dict:
            max_n_nodes = max(histogram.keys())
            prob = torch.zeros(max_n_nodes + 1)
            for num_nodes, count in histogram.items():
                prob[num_nodes] = count
        else:
            prob = histogram

        self.prob = prob / prob.sum()
        self.m = torch.distributions.Categorical(prob)

    def sample_n(self, n_samples, device):
        idx = self.m.sample((n_samples,))
        return idx.to(device)

    def update_prob(self, prob: torch.Tensor) -> None:
        """
        Update the node-count distribution online.

        Note: must rebuild the underlying Categorical distribution.
        """
        if prob is None:
            return
        p = torch.as_tensor(prob, dtype=torch.float32).detach().cpu()
        if p.dim() != 1 or p.numel() == 0:
            raise ValueError(f"DistributionNodes.update_prob expects 1D prob, got shape={tuple(p.shape)}")
        # Be strict: Categorical/multinomial will assert on NaN/Inf/negative.
        p = torch.nan_to_num(p, nan=0.0, posinf=0.0, neginf=0.0)
        p = torch.clamp(p, min=0.0)
        total = float(p.sum().item())
        if total <= 0:
            raise ValueError("DistributionNodes.update_prob: prob sum must be > 0")
        p = p / total
        self.prob = p
        self.m = torch.distributions.Categorical(p)

    def log_prob(self, batch_n_nodes):
        assert len(batch_n_nodes.size()) == 1
        p = self.prob.to(batch_n_nodes.device)

        probas = p[batch_n_nodes]
        log_p = torch.log(probas + 1e-30)
        return log_p
