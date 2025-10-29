"""Tests for unirec.plugins.candidate_merger module."""

import pytest
from unirec.plugins.candidate_merger.union import WeightedUnion
from unirec.core.state import Candidate


def test_weighted_union_creation():
    """Test WeightedUnion component creation."""
    merger = WeightedUnion()
    assert merger.component_kind == "candidate_merger"


def test_weighted_union_merge_basic():
    """Test basic merge functionality."""
    merger = WeightedUnion()

    pools = {
        "retriever1": [
            Candidate(item_id=1, score=0.9),
            Candidate(item_id=2, score=0.8),
        ],
        "retriever2": [
            Candidate(item_id=3, score=0.7),
            Candidate(item_id=4, score=0.6),
        ],
    }

    result = merger.merge(pools, user_id=100, topk=10)

    assert result.user_id == 100
    assert len(result.candidates) == 4
    # Should be sorted by score descending
    assert result.candidates[0].score >= result.candidates[1].score


def test_weighted_union_merge_with_overlap():
    """Test merge with overlapping items across pools."""
    merger = WeightedUnion()

    pools = {
        "retriever1": [
            Candidate(item_id=1, score=0.9),
            Candidate(item_id=2, score=0.8),
        ],
        "retriever2": [
            Candidate(item_id=1, score=0.7),  # Same item_id as in retriever1
            Candidate(item_id=3, score=0.6),
        ],
    }

    result = merger.merge(pools, user_id=100, topk=10)

    # Should have 3 unique items
    assert len(result.candidates) == 3
    # Item 1 should have combined score (0.9 + 0.7 = 1.6)
    item1 = next(c for c in result.candidates if c.item_id == 1)
    assert item1.score == pytest.approx(1.6)
    assert "retriever1" in item1.source
    assert "retriever2" in item1.source


def test_weighted_union_merge_with_weights():
    """Test merge with custom weights."""
    merger = WeightedUnion(weights={"retriever1": 2.0, "retriever2": 0.5})

    pools = {
        "retriever1": [Candidate(item_id=1, score=1.0)],
        "retriever2": [Candidate(item_id=2, score=1.0)],
    }

    result = merger.merge(pools, user_id=100, topk=10)

    # Item 1 should have score 2.0 * 1.0 = 2.0
    # Item 2 should have score 0.5 * 1.0 = 0.5
    item1 = next(c for c in result.candidates if c.item_id == 1)
    item2 = next(c for c in result.candidates if c.item_id == 2)

    assert item1.score == pytest.approx(2.0)
    assert item2.score == pytest.approx(0.5)
    # Item 1 should be ranked higher
    assert result.candidates[0].item_id == 1


def test_weighted_union_merge_respects_topk():
    """Test that merge respects topk parameter."""
    merger = WeightedUnion()

    pools = {
        "retriever1": [
            Candidate(item_id=i, score=1.0 - i * 0.1) for i in range(10)
        ],
    }

    result = merger.merge(pools, user_id=100, topk=5)

    assert len(result.candidates) == 5
    # Should have top 5 items by score
    assert result.candidates[0].item_id == 0  # Highest score


def test_weighted_union_merge_empty_pools():
    """Test merge with empty pools."""
    merger = WeightedUnion()

    result = merger.merge({}, user_id=100, topk=10)

    assert result.user_id == 100
    assert len(result.candidates) == 0


def test_weighted_union_merge_default_weight():
    """Test that missing weights default to 1.0."""
    merger = WeightedUnion(weights={"retriever1": 2.0})

    pools = {
        "retriever1": [Candidate(item_id=1, score=1.0)],
        "retriever2": [Candidate(item_id=2, score=1.0)],  # No weight specified
    }

    result = merger.merge(pools, user_id=100, topk=10)

    item1 = next(c for c in result.candidates if c.item_id == 1)
    item2 = next(c for c in result.candidates if c.item_id == 2)

    assert item1.score == pytest.approx(2.0)  # 2.0 * 1.0
    assert item2.score == pytest.approx(1.0)  # 1.0 * 1.0 (default weight)


def test_weighted_union_source_tracking():
    """Test that source is properly tracked for merged candidates."""
    merger = WeightedUnion()

    pools = {
        "retriever_a": [Candidate(item_id=1, score=0.5)],
        "retriever_b": [Candidate(item_id=1, score=0.5)],
        "retriever_c": [Candidate(item_id=1, score=0.5)],
    }

    result = merger.merge(pools, user_id=100, topk=10)

    assert len(result.candidates) == 1
    item = result.candidates[0]
    # All three sources should be in the source string
    assert "retriever_a" in item.source
    assert "retriever_b" in item.source
    assert "retriever_c" in item.source
    # Sources should be comma-separated
    assert "," in item.source
