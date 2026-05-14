"""Tests for grpo.graph_conversion -- GraphConversionMixin."""

import pytest

_SKIP_REASON = "rdkit or other heavy dependencies not available"

try:
    from grpo.graph_conversion import GraphConversionMixin
    _HAS_DEPS = True
except ImportError:
    _HAS_DEPS = False

pytestmark = pytest.mark.skipif(not _HAS_DEPS, reason=_SKIP_REASON)


class TestGraphConversionMixinStructure:
    """GraphConversionMixin should expose the expected interface."""

    def test_class_exists(self):
        assert isinstance(GraphConversionMixin, type)

    def test_has_smiles_to_graph(self):
        assert hasattr(GraphConversionMixin, "_smiles_to_graph")
        assert callable(GraphConversionMixin._smiles_to_graph)

    def test_has_graph_to_smiles(self):
        assert hasattr(GraphConversionMixin, "_graph_to_smiles")
        assert callable(GraphConversionMixin._graph_to_smiles)

    def test_has_normalize_smiles_list(self):
        assert hasattr(GraphConversionMixin, "_normalize_smiles_list")
        assert callable(GraphConversionMixin._normalize_smiles_list)


class TestNormalizeSmilesList:
    """_normalize_smiles_list is a static method; test without instantiation."""

    def test_none_returns_empty(self):
        result = GraphConversionMixin._normalize_smiles_list(None)
        assert result == []

    def test_empty_string_returns_empty(self):
        result = GraphConversionMixin._normalize_smiles_list("")
        assert result == []

    def test_single_smiles_string(self):
        result = GraphConversionMixin._normalize_smiles_list("CCO")
        assert result == ["CCO"]

    def test_comma_separated(self):
        result = GraphConversionMixin._normalize_smiles_list("CCO, CC, C")
        assert result == ["CCO", "CC", "C"]

    def test_list_passthrough(self):
        result = GraphConversionMixin._normalize_smiles_list(["CCO", "CC"])
        assert result == ["CCO", "CC"]

    def test_strips_whitespace(self):
        result = GraphConversionMixin._normalize_smiles_list(["  CCO  ", " CC "])
        assert result == ["CCO", "CC"]

    def test_filters_blank_entries(self):
        result = GraphConversionMixin._normalize_smiles_list(["CCO", "", "  ", "CC"])
        assert result == ["CCO", "CC"]
