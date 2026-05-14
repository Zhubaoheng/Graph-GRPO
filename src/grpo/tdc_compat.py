"""Compatibility patches for PyTDC oracle model files.

PyTDC 1.1.15 ships/downloads some sklearn 0.23 random-forest pickles
(`jnk3_current.pkl`, `gsk3b_current.pkl`).  sklearn >= 1.3 added a
`missing_go_to_left` field to the low-level tree node dtype, so the normal
pickle loader raises before the oracle can be constructed.  The project runs
on Python 3.12, where pinning sklearn 0.23 is not a viable option.

This module only takes over when the normal PyTDC loader hits that exact tree
dtype error.  It unpickles the old Tree objects into placeholders, converts the
saved state to the current sklearn Tree representation, and returns a regular
sklearn model object.
"""
from __future__ import annotations

import pickle
import warnings
from typing import Any, Dict, Optional, Set

import numpy as np


_PATCHED = False
_TREE_DTYPE_ERROR = "node array from the pickle has an incompatible dtype"


class _LegacyTreePlaceholder:
    def __init__(self, *args: Any):
        self.args = args
        self.state: Optional[Dict[str, Any]] = None

    def __setstate__(self, state: Dict[str, Any]) -> None:
        self.state = state


class _LegacySklearnTreeUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str) -> Any:
        if module == "sklearn.tree._tree" and name == "Tree":
            return _LegacyTreePlaceholder
        return super().find_class(module, name)


def _is_tree_dtype_error(exc: BaseException) -> bool:
    return _TREE_DTYPE_ERROR in str(exc)


def _convert_nodes_dtype(nodes: np.ndarray) -> np.ndarray:
    from sklearn.tree import _tree

    target_dtype = _tree.NODE_DTYPE
    if nodes.dtype == target_dtype:
        return nodes

    converted = np.zeros(nodes.shape, dtype=target_dtype)
    for field in nodes.dtype.names or ():
        if field in converted.dtype.names:
            converted[field] = nodes[field]
    return converted


def _normalize_tree_values(values: np.ndarray) -> np.ndarray:
    """Convert sklearn 0.23 class-count leaves to sklearn 1.x probabilities."""
    values = np.asarray(values, dtype=np.float64).copy()
    denom = values.sum(axis=2, keepdims=True)
    return np.divide(values, denom, out=np.zeros_like(values), where=denom != 0)


def _convert_tree(legacy_tree: _LegacyTreePlaceholder) -> Any:
    from sklearn.tree import _tree

    if legacy_tree.state is None:
        raise ValueError("Legacy sklearn Tree pickle did not contain tree state")

    tree = _tree.Tree(*legacy_tree.args)
    state = dict(legacy_tree.state)
    state["nodes"] = _convert_nodes_dtype(state["nodes"])
    state["values"] = _normalize_tree_values(state["values"])
    tree.__setstate__(state)
    return tree


def _patch_estimator_attrs(obj: Any) -> None:
    """Fill attributes renamed/added between sklearn 0.23 and modern sklearn."""
    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.tree import DecisionTreeClassifier
    except Exception:
        return

    attrs = getattr(obj, "__dict__", None)
    if attrs is None:
        return

    if isinstance(obj, DecisionTreeClassifier):
        attrs.setdefault("ccp_alpha", 0.0)
        attrs.setdefault("monotonic_cst", None)
        if attrs.get("max_features") == "auto":
            attrs["max_features"] = "sqrt"
        if "n_features_in_" not in attrs and "n_features_" in attrs:
            attrs["n_features_in_"] = attrs["n_features_"]

    if isinstance(obj, RandomForestClassifier):
        attrs.setdefault("ccp_alpha", 0.0)
        attrs.setdefault("monotonic_cst", None)
        if attrs.get("max_features") == "auto":
            attrs["max_features"] = "sqrt"

        if "estimator" not in attrs:
            attrs["estimator"] = attrs.get("base_estimator") or DecisionTreeClassifier()
        if "estimator_" not in attrs:
            attrs["estimator_"] = attrs.get("base_estimator_") or attrs["estimator"]

        attrs["estimator_params"] = (
            "criterion",
            "max_depth",
            "min_samples_split",
            "min_samples_leaf",
            "min_weight_fraction_leaf",
            "max_features",
            "max_leaf_nodes",
            "min_impurity_decrease",
            "random_state",
            "ccp_alpha",
            "monotonic_cst",
        )

        if "n_features_in_" not in attrs:
            estimators = attrs.get("estimators_") or []
            if estimators:
                first = estimators[0]
                n_features = getattr(first, "n_features_in_", None)
                if n_features is None:
                    n_features = getattr(first, "n_features_", None)
                if n_features is not None:
                    attrs["n_features_in_"] = n_features


def _walk_and_convert(obj: Any, seen: Optional[Set[int]] = None) -> Any:
    if seen is None:
        seen = set()

    oid = id(obj)
    if oid in seen:
        return obj
    seen.add(oid)

    if isinstance(obj, _LegacyTreePlaceholder):
        return _convert_tree(obj)

    if isinstance(obj, dict):
        for key, value in list(obj.items()):
            obj[key] = _walk_and_convert(value, seen)
        return obj

    if isinstance(obj, list):
        for idx, value in enumerate(obj):
            obj[idx] = _walk_and_convert(value, seen)
        return obj

    if isinstance(obj, tuple):
        return tuple(_walk_and_convert(value, seen) for value in obj)

    attrs = getattr(obj, "__dict__", None)
    if attrs is not None:
        for key, value in list(attrs.items()):
            setattr(obj, key, _walk_and_convert(value, seen))
        _patch_estimator_attrs(obj)

    return obj


def load_legacy_sklearn_tree_pickle(path: str) -> Any:
    with open(path, "rb") as f:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Trying to unpickle estimator .* from version")
            model = _LegacySklearnTreeUnpickler(f).load()
    return _walk_and_convert(model)


def patch_tdc_legacy_sklearn_pickles() -> None:
    """Patch PyTDC's pickle loader for old sklearn tree oracle files."""
    global _PATCHED
    if _PATCHED:
        return

    try:
        import tdc.chem_utils.oracle.oracle as oracle_module
    except Exception:
        return

    original_loader = oracle_module.load_pickled_model

    def compatible_loader(name: str) -> Any:
        try:
            return original_loader(name)
        except ValueError as exc:
            if not _is_tree_dtype_error(exc):
                raise
            return load_legacy_sklearn_tree_pickle(name)

    oracle_module.load_pickled_model = compatible_loader
    _PATCHED = True
