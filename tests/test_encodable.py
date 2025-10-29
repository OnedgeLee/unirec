"""Tests for unirec.data.encodable and encoded modules."""

import pytest
import torch
from packaging.version import Version
from unirec.data.encodable import UserEncodable, ItemEncodable
from unirec.data.encoded import UserEncoded, ItemEncoded
import unirec.data.context.user_context as user_ctx
import unirec.data.context.item_context as item_ctx

UserProfileContext = user_ctx.UserProfileContext
UserSessionContext = user_ctx.UserSessionContext
ItemProfileContext = item_ctx.ItemProfileContext
ItemSessionContext = item_ctx.ItemSessionContext


def test_user_encodable_creation():
    """Test UserEncodable creation."""
    profile = UserProfileContext(_id=123, _meta={"name": "Alice"})
    session = UserSessionContext(_meta={"session_id": "abc"})
    encodable = UserEncodable(profile=profile, session=session)

    assert encodable.profile == profile
    assert encodable.session == session
    assert encodable.profile.id == 123


def test_user_encodable_with_meta():
    """Test UserEncodable with custom meta."""
    profile = UserProfileContext(_id=123, _meta={})
    session = UserSessionContext(_meta={})
    encodable = UserEncodable(profile=profile, session=session, meta={"key": "value"})

    assert encodable.meta["key"] == "value"


def test_user_encodable_version():
    """Test UserEncodable has VERSION."""
    assert hasattr(UserEncodable, "VERSION")
    assert isinstance(UserEncodable.VERSION, Version)


def test_item_encodable_creation():
    """Test ItemEncodable creation."""
    profile = ItemProfileContext(_id=456, _meta={"title": "Product"})
    session = ItemSessionContext(_meta={})
    encodable = ItemEncodable(profile=profile, session=session)

    assert encodable.profile == profile
    assert encodable.session == session
    assert encodable.profile.id == 456


def test_item_encodable_version():
    """Test ItemEncodable has VERSION."""
    assert hasattr(ItemEncodable, "VERSION")
    assert isinstance(ItemEncodable.VERSION, Version)


def test_user_encoded_creation():
    """Test UserEncoded creation."""
    profile = UserProfileContext(_id=123, _meta={})
    session = UserSessionContext(_meta={})
    encodable = UserEncodable(profile=profile, session=session)

    vector = torch.tensor([1.0, 2.0, 3.0])
    encoded = UserEncoded(vector=vector, origin=encodable)

    assert torch.equal(encoded.vector, vector)
    assert encoded.profile == profile
    assert encoded.session == session


def test_user_encoded_with_request():
    """Test UserEncoded with request context."""
    import unirec.data.context.request_context as req_ctx

    RequestContext = req_ctx.RequestContext

    profile = UserProfileContext(_id=123, _meta={})
    session = UserSessionContext(_meta={})
    encodable = UserEncodable(profile=profile, session=session)
    request = RequestContext(_meta={"device": "mobile"})

    vector = torch.tensor([1.0, 2.0, 3.0])
    encoded = UserEncoded(vector=vector, origin=encodable, request=request)

    assert encoded.request == request
    assert encoded.request.meta["device"] == "mobile"


def test_user_encoded_delegates_to_origin():
    """Test that UserEncoded delegates properties to origin."""
    profile = UserProfileContext(_id=123, _meta={"name": "Alice"})
    session = UserSessionContext(_meta={"session_id": "abc"})
    encodable = UserEncodable(profile=profile, session=session, meta={"key": "value"})

    vector = torch.tensor([1.0, 2.0, 3.0])
    encoded = UserEncoded(vector=vector, origin=encodable)

    assert encoded.profile.id == 123
    assert encoded.meta["key"] == "value"


def test_item_encoded_creation():
    """Test ItemEncoded creation."""
    profile = ItemProfileContext(_id=456, _meta={"title": "Product"})
    session = ItemSessionContext(_meta={})
    encodable = ItemEncodable(profile=profile, session=session)

    vector = torch.tensor([4.0, 5.0, 6.0])
    encoded = ItemEncoded(vector=vector, origin=encodable)

    assert torch.equal(encoded.vector, vector)
    assert encoded.profile == profile
    assert encoded.profile.id == 456


def test_item_encoded_version():
    """Test ItemEncoded has VERSION."""
    assert hasattr(ItemEncoded, "VERSION")
    assert isinstance(ItemEncoded.VERSION, Version)


def test_encoded_vector_is_tensor():
    """Test that encoded vector is a torch.Tensor."""
    profile = UserProfileContext(_id=123, _meta={})
    session = UserSessionContext(_meta={})
    encodable = UserEncodable(profile=profile, session=session)

    vector = torch.tensor([1.0, 2.0, 3.0])
    encoded = UserEncoded(vector=vector, origin=encodable)

    assert isinstance(encoded.vector, torch.Tensor)
    assert encoded.vector.shape == (3,)
