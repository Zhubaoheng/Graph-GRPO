"""Utility container for flowing GRPO trajectory batches.

TrajectoryData keeps every tensor/list column that belongs to the same
trajectory batch aligned so we can slice/concatenate/move it as a single
object.  It mirrors the behaviour of a mini dataset living entirely in
memory, which makes passing data between the sampling, reward and training
stages significantly simpler – especially in distributed setups.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Union

import torch

from utils import PlaceHolder


IndexType = Union[int, slice, Sequence[int], torch.Tensor]


@dataclass(frozen=True)
class _SliceResult:
    tensor_index: IndexType
    list_indices: List[int]


class TrajectoryData:
    """Container that stores trajectory-aligned tensors and metadata.

    The first dimension of every stored tensor/list represents the batch
    dimension.  The class exposes slicing (``__getitem__``), concatenation
    (``union``) and device transfer (``to``) so the training loop can treat a
    large trajectory dump as a stream of on-demand mini-batches.
    """

    def __init__(
        self,
        tensor_data: Optional[Dict[str, torch.Tensor]] = None,
        list_data: Optional[Dict[str, Sequence[Any]]] = None,
    ) -> None:
        self.tensor_data: Dict[str, torch.Tensor] = tensor_data or {}
        self.list_data: Dict[str, Sequence[Any]] = list_data or {}
        self._length = self._infer_batch_size()

    # ------------------------------------------------------------------
    # basic properties
    def __len__(self) -> int:  # pragma: no cover - trivial
        return self._length

    def is_empty(self) -> bool:
        return self._length == 0

    # ------------------------------------------------------------------
    # slicing utilities
    def __getitem__(self, idx: IndexType) -> "TrajectoryData":
        slice_result = self._build_slice(idx)

        tensor_view = {
            key: self._slice_tensor(val, slice_result.tensor_index)
            for key, val in self.tensor_data.items()
        }

        list_view = {
            key: self._slice_sequence(val, slice_result.list_indices)
            for key, val in self.list_data.items()
        }

        return TrajectoryData(tensor_view, list_view)

    def _build_slice(self, idx: IndexType) -> _SliceResult:
        if isinstance(idx, slice):
            start, stop, step = idx.indices(self._length)
            list_indices = list(range(start, stop, step))
            tensor_index: IndexType = slice(start, stop, step)
        elif isinstance(idx, torch.Tensor):
            idx = idx.flatten().long()
            list_indices = idx.cpu().tolist()
            tensor_index = idx
        elif isinstance(idx, Iterable) and not isinstance(idx, (str, bytes)):
            idx = list(idx)
            list_indices = [self._normalize_single_index(i) for i in idx]
            tensor_index = torch.tensor(list_indices, dtype=torch.long)
        else:
            normalized = self._normalize_single_index(idx)  # type: ignore[arg-type]
            list_indices = [normalized]
            tensor_index = slice(normalized, normalized + 1, 1)

        return _SliceResult(tensor_index=tensor_index, list_indices=list_indices)

    def _normalize_single_index(self, idx: int) -> int:
        if idx < 0:
            idx += self._length
        if idx < 0 or idx >= self._length:
            raise IndexError("TrajectoryData index out of range")
        return idx

    def _slice_tensor(self, tensor: torch.Tensor, idx: IndexType) -> torch.Tensor:
        if isinstance(idx, torch.Tensor):
            return tensor.index_select(0, idx.to(tensor.device))
        return tensor[idx]

    def _slice_sequence(self, seq: Sequence[Any], indices: List[int]) -> List[Any]:
        if seq is None:
            return []
        return [seq[i] for i in indices]

    # ------------------------------------------------------------------
    # merging utilities
    def union(self, other: "TrajectoryData") -> "TrajectoryData":
        if self.is_empty():
            return other
        if other.is_empty():
            return self

        tensor_keys = set(self.tensor_data.keys()) | set(other.tensor_data.keys())
        merged_tensors: Dict[str, torch.Tensor] = {}
        for key in tensor_keys:
            left = self.tensor_data.get(key)
            right = other.tensor_data.get(key)
            if left is None:
                merged_tensors[key] = right
            elif right is None:
                merged_tensors[key] = left
            else:
                merged_tensors[key] = torch.cat([left, right], dim=0)

        list_keys = set(self.list_data.keys()) | set(other.list_data.keys())
        merged_lists: Dict[str, Sequence[Any]] = {}
        for key in list_keys:
            left_list = self.list_data.get(key)
            right_list = other.list_data.get(key)
            if left_list is None:
                merged_lists[key] = right_list
            elif right_list is None:
                merged_lists[key] = left_list
            else:
                merged_lists[key] = list(left_list) + list(right_list)

        return TrajectoryData(merged_tensors, merged_lists)

    # ------------------------------------------------------------------
    # device helpers
    def to(self, *args, **kwargs) -> "TrajectoryData":
        tensor_view = {
            key: value.to(*args, **kwargs)
            for key, value in self.tensor_data.items()
        }

        list_view = {
            key: self._move_nested(value, *args, **kwargs)
            for key, value in self.list_data.items()
        }

        return TrajectoryData(tensor_view, list_view)

    def _move_nested(self, value: Any, *args, **kwargs) -> Any:
        if isinstance(value, torch.Tensor):
            return value.to(*args, **kwargs)
        if isinstance(value, PlaceHolder):
            return value.to(*args, **kwargs)
        if isinstance(value, list):
            return [self._move_nested(v, *args, **kwargs) for v in value]
        if isinstance(value, tuple):
            return tuple(self._move_nested(v, *args, **kwargs) for v in value)
        return value

    # ------------------------------------------------------------------
    def as_dict(self) -> Dict[str, Any]:
        data = {}
        data.update(self.tensor_data)
        data.update(self.list_data)
        return data

    def with_tensor(self, key: str, value: torch.Tensor) -> "TrajectoryData":
        tensor_data = dict(self.tensor_data)
        tensor_data[key] = value
        return TrajectoryData(tensor_data, dict(self.list_data))

    @staticmethod
    def concatenate(batches: Sequence["TrajectoryData"]) -> "TrajectoryData":
        """Efficiently concatenate many batches without repeated torch.cat."""
        valid_batches: List["TrajectoryData"] = [
            batch for batch in batches if batch is not None and not batch.is_empty()
        ]
        if not valid_batches:
            return TrajectoryData()

        tensor_keys = set()
        list_keys = set()
        for batch in valid_batches:
            tensor_keys.update(batch.tensor_data.keys())
            list_keys.update(batch.list_data.keys())

        merged_tensors: Dict[str, torch.Tensor] = {}
        for key in tensor_keys:
            parts: List[torch.Tensor] = []
            for batch in valid_batches:
                tensor = batch.tensor_data.get(key)
                if tensor is None:
                    continue
                parts.append(tensor)
            if not parts:
                continue
            if len(parts) != len(valid_batches):
                raise ValueError(f"Inconsistent presence of tensor column '{key}' across batches")
            merged_tensors[key] = torch.cat(parts, dim=0)

        merged_lists: Dict[str, Sequence[Any]] = {}
        for key in list_keys:
            concatenated: List[Any] = []
            for batch in valid_batches:
                seq = batch.list_data.get(key)
                if seq is None:
                    continue
                concatenated.extend(seq)
            if concatenated:
                merged_lists[key] = concatenated

        return TrajectoryData(merged_tensors, merged_lists)

    # ------------------------------------------------------------------
    def _infer_batch_size(self) -> int:
        lengths: List[int] = []
        for tensor in self.tensor_data.values():
            if tensor is None:
                continue
            lengths.append(tensor.shape[0])
        for seq in self.list_data.values():
            if seq is None:
                continue
            lengths.append(len(seq))

        if not lengths:
            return 0

        first = lengths[0]
        if not all(length == first for length in lengths):
            raise ValueError("Inconsistent batch sizes between stored tensors/lists")
        return first

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        tensor_desc = {k: tuple(v.shape) for k, v in self.tensor_data.items()}
        list_desc = {k: len(v) if v is not None else 0 for k, v in self.list_data.items()}
        return f"TrajectoryData(len={self._length}, tensors={tensor_desc}, lists={list_desc})"
