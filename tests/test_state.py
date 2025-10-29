"""Tests for unirec.core.state module."""

import pytest
from unirec.core.state import Candidate, CandidateSet, Slate, PipelineState
from unirec.data.encodable import UserEncodable
import unirec.data.context.user_context as user_ctx

UserProfileContext = user_ctx.UserProfileContext
UserSessionContext = user_ctx.UserSessionContext


def test_candidate_creation():
    """Test Candidate dataclass creation."""
    cand = Candidate(item_id=1, score=0.9)
    assert cand.item_id == 1
    assert cand.score == 0.9
    assert cand.source == "retrieval"
    assert cand.features == {}


def test_candidate_with_features():
    """Test Candidate with custom features."""
    cand = Candidate(item_id=2, score=0.8, source="ranker", features={"quality": 0.95})
    assert cand.item_id == 2
    assert cand.score == 0.8
    assert cand.source == "ranker"
    assert cand.features["quality"] == 0.95


def test_candidateset_creation():
    """Test CandidateSet creation."""
    candidates = [
        Candidate(item_id=1, score=0.9),
        Candidate(item_id=2, score=0.8),
        Candidate(item_id=3, score=0.7),
    ]
    candset = CandidateSet(user_id=100, candidates=candidates)
    assert candset.user_id == 100
    assert len(candset.candidates) == 3


def test_candidateset_topk():
    """Test CandidateSet topk method."""
    candidates = [
        Candidate(item_id=1, score=0.5),
        Candidate(item_id=2, score=0.9),
        Candidate(item_id=3, score=0.7),
        Candidate(item_id=4, score=0.3),
    ]
    candset = CandidateSet(user_id=100, candidates=candidates)
    top2 = candset.topk(2)

    assert len(top2.candidates) == 2
    assert top2.candidates[0].item_id == 2
    assert top2.candidates[0].score == 0.9
    assert top2.candidates[1].item_id == 3
    assert top2.candidates[1].score == 0.7
    assert top2.user_id == 100


def test_candidateset_topk_with_k_larger_than_size():
    """Test topk when k is larger than candidate list."""
    candidates = [
        Candidate(item_id=1, score=0.9),
        Candidate(item_id=2, score=0.8),
    ]
    candset = CandidateSet(user_id=100, candidates=candidates)
    top10 = candset.topk(10)

    assert len(top10.candidates) == 2


def test_slate_creation():
    """Test Slate creation."""
    slate = Slate(user_id=100, items=[1, 2, 3])
    assert slate.user_id == 100
    assert slate.items == [1, 2, 3]
    assert slate.aux == {}


def test_slate_with_aux():
    """Test Slate with auxiliary data."""
    slate = Slate(user_id=100, items=[1, 2, 3], aux={"algo": "ucb"})
    assert slate.aux["algo"] == "ucb"


def test_pipeline_state_creation():
    """Test PipelineState creation."""
    profile = UserProfileContext(_id=100, _meta={})
    session = UserSessionContext(_meta={})
    user = UserEncodable(profile=profile, session=session)

    state = PipelineState(user=user, context={})
    assert state.user == user
    assert state.context == {}
    assert state.request is None
    assert state.candset is None
    assert state.slate is None
    assert state.logs == {}


def test_pipeline_state_with_all_fields():
    """Test PipelineState with all fields populated."""
    profile = UserProfileContext(_id=100, _meta={})
    session = UserSessionContext(_meta={})
    user = UserEncodable(profile=profile, session=session)

    candidates = [Candidate(item_id=1, score=0.9)]
    candset = CandidateSet(user_id=100, candidates=candidates)
    slate = Slate(user_id=100, items=[1])

    state = PipelineState(
        user=user,
        context={"timestamp": 12345},
        candset=candset,
        slate=slate,
        logs={"retrieval": {}},
    )

    assert state.user == user
    assert state.context["timestamp"] == 12345
    assert state.candset == candset
    assert state.slate == slate
    assert "retrieval" in state.logs
