"""Tests for grpo.trajectory_data.TrajectoryData."""

import pytest
import torch

from grpo.trajectory_data import TrajectoryData


class TestTrajectoryDataConstruction:
    """TrajectoryData should accept tensor_data and list_data dicts."""

    def test_empty_construction(self):
        td = TrajectoryData()
        assert len(td) == 0
        assert td.is_empty()

    def test_tensor_only(self):
        td = TrajectoryData(tensor_data={"x": torch.randn(5, 3)})
        assert len(td) == 5
        assert not td.is_empty()

    def test_list_only(self):
        td = TrajectoryData(list_data={"names": ["a", "b", "c"]})
        assert len(td) == 3

    def test_mixed(self):
        td = TrajectoryData(
            tensor_data={"x": torch.randn(4, 2)},
            list_data={"labels": ["l0", "l1", "l2", "l3"]},
        )
        assert len(td) == 4

    def test_mismatched_batch_size_raises(self):
        with pytest.raises(ValueError, match="Inconsistent batch sizes"):
            TrajectoryData(
                tensor_data={"x": torch.randn(3, 2)},
                list_data={"labels": ["a", "b"]},
            )


class TestTrajectoryDataSlicing:
    """Slicing via __getitem__ returns a new TrajectoryData."""

    @pytest.fixture()
    def sample_td(self):
        return TrajectoryData(
            tensor_data={
                "reward": torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0]),
                "state": torch.randn(5, 8),
            },
            list_data={"smiles": ["A", "B", "C", "D", "E"]},
        )

    def test_slice_range(self, sample_td):
        sub = sample_td[1:3]
        assert len(sub) == 2
        assert sub.list_data["smiles"] == ["B", "C"]
        assert torch.allclose(sub.tensor_data["reward"], torch.tensor([2.0, 3.0]))

    def test_slice_single_int(self, sample_td):
        sub = sample_td[0]
        assert len(sub) == 1
        assert sub.list_data["smiles"] == ["A"]

    def test_slice_negative_index(self, sample_td):
        sub = sample_td[-1]
        assert len(sub) == 1
        assert sub.list_data["smiles"] == ["E"]

    def test_slice_list_of_indices(self, sample_td):
        sub = sample_td[[0, 2, 4]]
        assert len(sub) == 3
        assert sub.list_data["smiles"] == ["A", "C", "E"]

    def test_slice_tensor_index(self, sample_td):
        idx = torch.tensor([1, 3])
        sub = sample_td[idx]
        assert len(sub) == 2
        assert sub.list_data["smiles"] == ["B", "D"]

    def test_index_out_of_range_raises(self, sample_td):
        with pytest.raises(IndexError, match="out of range"):
            sample_td[10]


class TestTrajectoryDataUnion:
    """union() concatenates two TrajectoryData objects."""

    def test_basic_union(self):
        a = TrajectoryData(
            tensor_data={"r": torch.tensor([1.0, 2.0])},
            list_data={"s": ["a", "b"]},
        )
        b = TrajectoryData(
            tensor_data={"r": torch.tensor([3.0])},
            list_data={"s": ["c"]},
        )
        merged = a.union(b)
        assert len(merged) == 3
        assert torch.allclose(merged.tensor_data["r"], torch.tensor([1.0, 2.0, 3.0]))
        assert list(merged.list_data["s"]) == ["a", "b", "c"]

    def test_union_with_empty(self):
        a = TrajectoryData(tensor_data={"r": torch.tensor([1.0])})
        empty = TrajectoryData()
        assert a.union(empty) is a
        assert empty.union(a) is a


class TestTrajectoryDataConcatenate:
    """Static concatenate() method merges a list of batches."""

    def test_concatenate_multiple(self):
        batches = [
            TrajectoryData(tensor_data={"v": torch.tensor([1.0])}),
            TrajectoryData(tensor_data={"v": torch.tensor([2.0])}),
            TrajectoryData(tensor_data={"v": torch.tensor([3.0])}),
        ]
        merged = TrajectoryData.concatenate(batches)
        assert len(merged) == 3

    def test_concatenate_empty_list(self):
        merged = TrajectoryData.concatenate([])
        assert merged.is_empty()

    def test_concatenate_skips_none_and_empty(self):
        batches = [
            None,
            TrajectoryData(),
            TrajectoryData(tensor_data={"v": torch.tensor([1.0])}),
        ]
        merged = TrajectoryData.concatenate(batches)
        assert len(merged) == 1


class TestTrajectoryDataDevice:
    """to() returns a new object with tensors moved to the target device."""

    def test_to_cpu_is_noop(self):
        td = TrajectoryData(tensor_data={"x": torch.randn(3, 2)})
        moved = td.to("cpu")
        assert moved.tensor_data["x"].device.type == "cpu"
        assert len(moved) == 3

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="No GPU available")
    def test_to_cuda(self):
        td = TrajectoryData(tensor_data={"x": torch.randn(3, 2)})
        moved = td.to("cuda")
        assert moved.tensor_data["x"].device.type == "cuda"


class TestTrajectoryDataMisc:
    """Miscellaneous helpers."""

    def test_as_dict(self):
        td = TrajectoryData(
            tensor_data={"t": torch.tensor([1.0])},
            list_data={"l": ["x"]},
        )
        d = td.as_dict()
        assert "t" in d
        assert "l" in d

    def test_with_tensor(self):
        td = TrajectoryData(tensor_data={"a": torch.tensor([1.0])})
        td2 = td.with_tensor("b", torch.tensor([2.0]))
        assert "b" in td2.tensor_data
        # Original unchanged
        assert "b" not in td.tensor_data

    def test_repr(self):
        td = TrajectoryData(tensor_data={"x": torch.randn(2, 3)})
        r = repr(td)
        assert "TrajectoryData" in r
        assert "len=2" in r
