"""Tests for unirec.plugins.eval.metrics module."""

import pytest
import numpy as np
from numpy.typing import NDArray
from unirec.plugins.eval.metrics import (
    dcg_at_k,
    ndcg_at_k,
    recall_at_k,
    ild_at_k,
    entropy_at_k,
    coverage_at_k,
)


def test_dcg_at_k_basic():
    """Test basic DCG calculation."""
    rels = [1, 1, 0, 1, 0]
    dcg = dcg_at_k(rels, k=3)
    # DCG@3 = 1 + 1/log2(2) + 0/log2(3) = 1 + 1 = 2.0
    assert dcg == pytest.approx(2.0)


def test_dcg_at_k_empty():
    """Test DCG with empty relevance list."""
    dcg = dcg_at_k([], k=5)
    assert dcg == 0.0


def test_dcg_at_k_with_k_larger_than_list():
    """Test DCG when k is larger than list size."""
    rels = [1, 1]
    dcg = dcg_at_k(rels, k=10)
    assert dcg == pytest.approx(2.0)


def test_ndcg_at_k_perfect():
    """Test NDCG with perfect ranking."""
    gt_items = {1, 2, 3}
    pred_items = [1, 2, 3, 4, 5]
    ndcg = ndcg_at_k(gt_items, pred_items, k=5)
    # All relevant items at the top, should be 1.0
    assert ndcg == pytest.approx(1.0)


def test_ndcg_at_k_no_relevant():
    """Test NDCG when no relevant items are predicted."""
    gt_items = {1, 2, 3}
    pred_items = [4, 5, 6, 7, 8]
    ndcg = ndcg_at_k(gt_items, pred_items, k=5)
    assert ndcg == 0.0


def test_ndcg_at_k_empty_ground_truth():
    """Test NDCG with empty ground truth."""
    gt_items = set()
    pred_items = [1, 2, 3]
    ndcg = ndcg_at_k(gt_items, pred_items, k=3)
    assert ndcg == 0.0


def test_ndcg_at_k_partial_match():
    """Test NDCG with partial match."""
    gt_items = {1, 2, 3}
    pred_items = [1, 4, 2, 5, 3]
    ndcg = ndcg_at_k(gt_items, pred_items, k=5)
    # Should be between 0 and 1
    assert 0.0 < ndcg < 1.0


def test_recall_at_k_perfect():
    """Test recall with all items found."""
    gt_items = {1, 2, 3}
    pred_items = [1, 2, 3, 4, 5]
    recall = recall_at_k(gt_items, pred_items, k=5)
    assert recall == 1.0


def test_recall_at_k_partial():
    """Test recall with partial match."""
    gt_items = {1, 2, 3, 4}
    pred_items = [1, 2, 5, 6, 7]
    recall = recall_at_k(gt_items, pred_items, k=5)
    assert recall == 0.5  # 2 out of 4


def test_recall_at_k_none():
    """Test recall with no matches."""
    gt_items = {1, 2, 3}
    pred_items = [4, 5, 6, 7, 8]
    recall = recall_at_k(gt_items, pred_items, k=5)
    assert recall == 0.0


def test_recall_at_k_empty_ground_truth():
    """Test recall with empty ground truth."""
    gt_items = set()
    pred_items = [1, 2, 3]
    recall = recall_at_k(gt_items, pred_items, k=3)
    assert recall == 0.0


def test_recall_at_k_respects_k():
    """Test that recall only considers top-k items."""
    gt_items = {1, 2, 3}
    pred_items = [4, 5, 1, 2, 3]  # Relevant items are after position k
    recall = recall_at_k(gt_items, pred_items, k=2)
    assert recall == 0.0  # No relevant items in top 2


def test_ild_at_k_identical_items():
    """Test ILD with identical embeddings (minimum diversity)."""
    pred_items = [0, 1, 2]
    item_emb: NDArray[np.float32] = np.array(
        [[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]], dtype=np.float32
    )
    ild = ild_at_k(pred_items, item_emb, k=3)
    # Cosine similarity is 1.0, so diversity = 1 - 1 = 0
    assert ild == pytest.approx(0.0, abs=1e-5)


def test_ild_at_k_orthogonal_items():
    """Test ILD with orthogonal embeddings (maximum diversity)."""
    pred_items = [0, 1]
    item_emb: NDArray[np.float32] = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    ild = ild_at_k(pred_items, item_emb, k=2)
    # Cosine similarity is 0.0, so diversity = 1 - 0 = 1
    assert ild == pytest.approx(1.0, abs=1e-5)


def test_ild_at_k_single_item():
    """Test ILD with single item."""
    pred_items = [0]
    item_emb: NDArray[np.float32] = np.array([[1.0, 0.0]], dtype=np.float32)
    ild = ild_at_k(pred_items, item_emb, k=1)
    assert ild == 0.0


def test_ild_at_k_empty_items():
    """Test ILD with no items."""
    pred_items = []
    item_emb: NDArray[np.float32] = np.array([], dtype=np.float32).reshape(0, 2)
    ild = ild_at_k(pred_items, item_emb, k=5)
    assert ild == 0.0


def test_entropy_at_k_uniform():
    """Test entropy with uniform distribution."""
    pred_items = [1, 2, 3, 4]
    categories = {1: "A", 2: "B", 3: "C", 4: "D"}
    entropy = entropy_at_k(pred_items, categories, k=4)
    # With 4 different categories, entropy should be log(4)
    expected = -4 * (0.25 * np.log(0.25))
    assert entropy == pytest.approx(expected, abs=1e-5)


def test_entropy_at_k_single_category():
    """Test entropy with all items in same category."""
    pred_items = [1, 2, 3, 4]
    categories = {1: "A", 2: "A", 3: "A", 4: "A"}
    entropy = entropy_at_k(pred_items, categories, k=4)
    # All same category, entropy should be 0
    assert entropy == pytest.approx(0.0, abs=1e-5)


def test_entropy_at_k_with_unknown_category():
    """Test entropy with items not in category map."""
    pred_items = [1, 2, 3]
    categories = {1: "A"}  # 2 and 3 will be "_unk"
    entropy = entropy_at_k(pred_items, categories, k=3)
    # Should handle unknown categories
    assert entropy > 0


def test_entropy_at_k_empty():
    """Test entropy with empty prediction list."""
    pred_items = []
    categories = {}
    entropy = entropy_at_k(pred_items, categories, k=5)
    assert entropy == 0.0


def test_coverage_at_k_full():
    """Test coverage with all items covered."""
    all_pred_items = [[1, 2, 3], [4, 5, 6], [7, 8, 9, 10]]
    catalog_size = 10
    coverage = coverage_at_k(all_pred_items, catalog_size, k=10)
    assert coverage == 1.0


def test_coverage_at_k_partial():
    """Test coverage with partial catalog coverage."""
    all_pred_items = [[1, 2, 3], [1, 2, 4], [1, 2, 5]]
    catalog_size = 10
    coverage = coverage_at_k(all_pred_items, catalog_size, k=3)
    # Covers items 1, 2, 3, 4, 5 = 5 out of 10
    assert coverage == 0.5


def test_coverage_at_k_respects_k():
    """Test that coverage respects k parameter."""
    all_pred_items = [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]]
    catalog_size = 10
    coverage = coverage_at_k(all_pred_items, catalog_size, k=2)
    # Only considers first 2 items from each slate
    # Items: 1, 2, 6, 7 = 4 out of 10
    assert coverage == 0.4


def test_coverage_at_k_empty():
    """Test coverage with no predictions."""
    all_pred_items = []
    catalog_size = 10
    coverage = coverage_at_k(all_pred_items, catalog_size, k=5)
    assert coverage == 0.0


def test_coverage_at_k_zero_catalog():
    """Test coverage with zero catalog size."""
    all_pred_items = [[1, 2, 3]]
    catalog_size = 0
    coverage = coverage_at_k(all_pred_items, catalog_size, k=3)
    # Should handle division by zero gracefully
    assert coverage >= 0.0
