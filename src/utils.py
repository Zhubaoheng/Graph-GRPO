import functools
import logging
import os
import warnings
from collections.abc import Iterable
from typing import Optional

import numpy as np
import torch
import torch_geometric.utils
import omegaconf
import swanlab
from omegaconf import OmegaConf, open_dict
from torch_geometric.utils import to_dense_adj, to_dense_batch

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------

def setup_logging(level: int = logging.INFO) -> None:
    """Configure the root logger with a concise format."""
    logging.basicConfig(
        level=level,
        format="[%(asctime)s][%(name)s][%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def get_logger(name: str) -> logging.Logger:
    """Return a named logger (convenience wrapper)."""
    return logging.getLogger(name)


# ---------------------------------------------------------------------------
# Warning suppression (call once from entry points)
# ---------------------------------------------------------------------------

def suppress_noisy_warnings() -> None:
    """Suppress known noisy warnings from third-party libraries."""
    warnings.filterwarnings("ignore", message=".*pkg_resources is deprecated.*")
    warnings.filterwarnings("ignore", message=".*Boto3 will no longer support Python 3.9.*")
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", message=".*PossibleUserWarning.*")


# ---------------------------------------------------------------------------
# torch.load patch (call once from entry points)
# ---------------------------------------------------------------------------

def patch_torch_load_weights_only_default() -> None:
    """Globally set ``torch.load`` default to ``weights_only=False``.

    PyTorch >= 2.6 changed the default to ``True``, which blocks loading
    checkpoints containing full Python objects (e.g. OmegaConf configs,
    Lightning state).  All checkpoints in this project are locally produced
    and trusted, so reverting the default is safe.
    """
    if getattr(torch.load, "_rl_graph_patched", False):
        return

    original_load = torch.load

    @functools.wraps(original_load)
    def _patched_load(*args, **kwargs):
        if "weights_only" not in kwargs:
            kwargs["weights_only"] = False
        return original_load(*args, **kwargs)

    _patched_load._rl_graph_patched = True
    torch.load = _patched_load
    logger.debug("Patched torch.load default: weights_only=False")


# ---------------------------------------------------------------------------
# NumPy legacy alias compatibility
# ---------------------------------------------------------------------------

_NUMPY_ALIAS_MAP = {
    "float": np.float64,
    "float_": np.float64,
    "complex": np.complex128,
    "complex_": np.complex128,
    "bool": np.bool_,
    "bool_": np.bool_,
    "int": np.int64,
    "object": object,
    "Inf": np.inf,
    "Infinity": np.inf,
    "infty": np.inf,
    "NaN": np.nan,
    "NAN": np.nan,
}


def ensure_legacy_aliases():
    """Re-create deprecated NumPy aliases removed in NumPy 2.0."""
    for alias, target in _NUMPY_ALIAS_MAP.items():
        if alias not in np.__dict__:
            setattr(np, alias, target)


def create_folders(args):
    try:
        os.makedirs("graphs")
        os.makedirs("chains")
    except OSError:
        pass

    try:
        os.makedirs("graphs/" + args.general.name)
        os.makedirs("chains/" + args.general.name)
    except OSError:
        pass


def normalize(X, E, y, norm_values, norm_biases, node_mask):
    X = (X - norm_biases[0]) / norm_values[0]
    E = (E - norm_biases[1]) / norm_values[1]
    y = (y - norm_biases[2]) / norm_values[2]

    diag = (
        torch.eye(E.shape[1], dtype=torch.bool).unsqueeze(0).expand(E.shape[0], -1, -1)
    )
    E[diag] = 0

    return PlaceHolder(X=X, E=E, y=y).mask(node_mask)


def unnormalize(X, E, y, norm_values, norm_biases, node_mask, collapse=False):
    """
    X : node features
    E : edge features
    y : global features`
    norm_values : [norm value X, norm value E, norm value y]
    norm_biases : same order
    node_mask
    """
    X = X * norm_values[0] + norm_biases[0]
    E = E * norm_values[1] + norm_biases[1]
    y = y * norm_values[2] + norm_biases[2]

    return PlaceHolder(X=X, E=E, y=y).mask(node_mask, collapse)


def symmetrize_and_mask_diag(E):
    # symmetrize the edge matrix
    upper_triangular_mask = torch.zeros_like(E)
    indices = torch.triu_indices(row=E.size(1), col=E.size(2), offset=1)
    if len(E.shape) == 4:
        upper_triangular_mask[:, indices[0], indices[1], :] = 1
    else:
        upper_triangular_mask[:, indices[0], indices[1]] = 1
    E = E * upper_triangular_mask
    E = E + torch.transpose(E, 1, 2)
    # mask the diagonal
    diag = (
        torch.eye(E.shape[1], dtype=torch.bool).unsqueeze(0).expand(E.shape[0], -1, -1)
    )
    E[diag] = 0

    return E


def to_dense(x, edge_index, edge_attr, batch):
    X, node_mask = to_dense_batch(x=x, batch=batch)
    edge_index, edge_attr = torch_geometric.utils.remove_self_loops(
        edge_index, edge_attr
    )
    max_num_nodes = X.size(1)
    E = to_dense_adj(
        edge_index=edge_index,
        batch=batch,
        edge_attr=edge_attr,
        max_num_nodes=max_num_nodes,
    )
    E = encode_no_edge(E)

    return PlaceHolder(X=X, E=E, y=None), node_mask


def encode_no_edge(E):
    assert len(E.shape) == 4
    if E.shape[-1] == 0:
        return E
    no_edge = torch.sum(E, dim=3) == 0
    first_elt = E[:, :, :, 0]
    first_elt[no_edge] = 1
    E[:, :, :, 0] = first_elt
    diag = (
        torch.eye(E.shape[1], dtype=torch.bool).unsqueeze(0).expand(E.shape[0], -1, -1)
    )
    E[diag] = 0
    return E


def update_config_with_new_keys(cfg, saved_cfg):
    saved_general = saved_cfg.general
    saved_train = saved_cfg.train
    saved_model = saved_cfg.model

    for key, val in saved_general.items():
        OmegaConf.set_struct(cfg.general, True)
        with open_dict(cfg.general):
            if key not in cfg.general.keys():
                setattr(cfg.general, key, val)

    OmegaConf.set_struct(cfg.train, True)
    with open_dict(cfg.train):
        for key, val in saved_train.items():
            if key not in cfg.train.keys():
                setattr(cfg.train, key, val)

    OmegaConf.set_struct(cfg.model, True)
    with open_dict(cfg.model):
        for key, val in saved_model.items():
            if key not in cfg.model.keys():
                setattr(cfg.model, key, val)
    return cfg


class PlaceHolder:
    def __init__(self, X, E, y):
        self.X = X
        self.E = E
        self.y = y
        
    def to(self, *args, **kwargs):
        """Move all internal tensors to the given device / dtype.

        Returns a new PlaceHolder with the moved tensors.
        """
        moved_X = self.X.to(*args, **kwargs) if self.X is not None else None
        moved_E = self.E.to(*args, **kwargs) if self.E is not None else None
        moved_y = self.y.to(*args, **kwargs) if self.y is not None else None
        return self.__class__(X=moved_X, E=moved_E, y=moved_y)

    def type_as(self, x: torch.Tensor):
        """Changes the device and dtype of X, E, y."""
        self.X = self.X.type_as(x)
        self.E = self.E.type_as(x)
        self.y = self.y.type_as(x)
        return self

    def to_device(self, device):
        """Changes the device and dtype of X, E, y."""
        self.X = self.X.to(device)
        self.E = self.E.to(device)
        self.y = self.y.to(device) if self.y is not None else None
        return self

    def mask(self, node_mask, collapse=False):
        x_mask = node_mask.unsqueeze(-1)  # bs, n, 1
        e_mask1 = x_mask.unsqueeze(2)  # bs, n, 1, 1
        e_mask2 = x_mask.unsqueeze(1)  # bs, 1, n, 1

        if collapse:
            self.X = torch.argmax(self.X, dim=-1)
            self.E = torch.argmax(self.E, dim=-1)

            self.X[node_mask == 0] = -1
            self.E[(e_mask1 * e_mask2).squeeze(-1) == 0] = -1
        else:
            self.X = self.X * x_mask
            self.E = self.E * e_mask1 * e_mask2
            assert torch.allclose(self.E, torch.transpose(self.E, 1, 2))
        return self

    def __repr__(self):
        return (
            f"X: {self.X.shape if type(self.X) == torch.Tensor else self.X} -- "
            + f"E: {self.E.shape if type(self.E) == torch.Tensor else self.E} -- "
            + f"y: {self.y.shape if type(self.y) == torch.Tensor else self.y}"
        )

    def split(self, node_mask):
        """Split a PlaceHolder representing a batch into a list of placeholders representing individual graphs."""
        graph_list = []
        batch_size = self.X.shape[0]
        for i in range(batch_size):
            n = torch.sum(node_mask[i], dim=0)
            x = self.X[i, :n]
            e = self.E[i, :n, :n]
            y = self.y[i] if self.y is not None else None
            graph_list.append(PlaceHolder(X=x, E=e, y=y))
        return graph_list


def setup_swanlab(cfg):
    """Initialize SwanLab experiment tracking."""
    if cfg.general.test_only is not None:
        logger.info("Skipping SwanLab initialization in test mode")
        return

    config_dict = omegaconf.OmegaConf.to_container(
        cfg, resolve=True, throw_on_missing=True
    )
    if cfg.general.test_only is None:
        name = f"{cfg.general.name}"
    else:
        if cfg.sample.search:
            name = f"{cfg.general.name}_search_{cfg.sample.search}"
        else:
            name = f"{cfg.general.name}_{cfg.sample.time_distortion}"
    kwargs = {
        "name": name,
        "project": f"graph_dfm_{cfg.dataset.name}",
        "config": config_dict,
        "settings": swanlab.Settings(_disable_stats=True),
        "reinit": True,
        "mode": cfg.general.swanlab,
    }
    config_dict["general"]["local_dir"] = os.getcwd()

    try:
        swanlab.init(**kwargs)
    except Exception as e:
        logger.warning("Failed to initialize SwanLab: %s", e)
        if cfg.general.swanlab != "offline":
            try:
                kwargs["mode"] = "offline"
                swanlab.init(**kwargs)
                logger.info("SwanLab initialized in offline mode")
            except Exception:
                pass


