"""Tests for unirec.data.context modules."""

from packaging.version import Version
import unirec.data.context.user_context as user_ctx
import unirec.data.context.item_context as item_ctx
import unirec.data.context.request_context as req_ctx

UserContext = user_ctx.UserContext
UserProfileContext = user_ctx.UserProfileContext
UserSessionContext = user_ctx.UserSessionContext
ItemContext = item_ctx.ItemContext
ItemProfileContext = item_ctx.ItemProfileContext
ItemSessionContext = item_ctx.ItemSessionContext
RequestContext = req_ctx.RequestContext


def test_user_context_version():
    """Test UserContext has VERSION."""
    assert hasattr(UserContext, "VERSION")
    assert isinstance(UserContext.VERSION, Version)


def test_user_profile_context_creation():
    """Test UserProfileContext creation."""
    profile = UserProfileContext(_id=123, _meta={"name": "Alice"})
    assert profile.id == 123
    assert profile.meta["name"] == "Alice"


def test_user_profile_context_id_property():
    """Test UserProfileContext id property is read-only."""
    profile = UserProfileContext(_id=123, _meta={})
    assert profile.id == 123


def test_user_profile_context_meta_property():
    """Test UserProfileContext meta property."""
    meta = {"age": 30, "country": "US"}
    profile = UserProfileContext(_id=123, _meta=meta)
    assert profile.meta["age"] == 30
    assert profile.meta["country"] == "US"


def test_user_session_context_creation():
    """Test UserSessionContext creation."""
    session = UserSessionContext(_meta={"session_id": "abc123"})
    assert session.meta["session_id"] == "abc123"


def test_user_session_context_empty_meta():
    """Test UserSessionContext with empty meta."""
    session = UserSessionContext(_meta={})
    assert session.meta == {}


def test_item_context_version():
    """Test ItemContext has VERSION."""
    assert hasattr(ItemContext, "VERSION")
    assert isinstance(ItemContext.VERSION, Version)


def test_item_profile_context_creation():
    """Test ItemProfileContext creation."""
    profile = ItemProfileContext(_id=456, _meta={"title": "Product A"})
    assert profile.id == 456
    assert profile.meta["title"] == "Product A"


def test_item_profile_context_id_property():
    """Test ItemProfileContext id property."""
    profile = ItemProfileContext(_id=456, _meta={})
    assert profile.id == 456


def test_item_profile_context_meta_property():
    """Test ItemProfileContext meta property."""
    meta = {"category": "electronics", "price": 99.99}
    profile = ItemProfileContext(_id=456, _meta=meta)
    assert profile.meta["category"] == "electronics"
    assert profile.meta["price"] == 99.99


def test_item_session_context_creation():
    """Test ItemSessionContext creation."""
    session = ItemSessionContext(_meta={"views": 100})
    assert session.meta["views"] == 100


def test_item_session_context_empty_meta():
    """Test ItemSessionContext with empty meta."""
    session = ItemSessionContext(_meta={})
    assert session.meta == {}


def test_request_context_creation():
    """Test RequestContext creation."""
    request = RequestContext(_meta={"device": "mobile"})
    assert request.meta["device"] == "mobile"


def test_request_context_empty_meta():
    """Test RequestContext with empty meta."""
    request = RequestContext(_meta={})
    assert request.meta == {}


def test_request_context_is_fingerprintable():
    """Test that RequestContext is Fingerprintable."""
    request = RequestContext(_meta={"timestamp": 12345})
    # Should have key property from Fingerprintable
    assert hasattr(request, "key")
    # Key should be a string
    key = request.key
    assert isinstance(key, str)
    assert ":" in key  # Should have algo:hash_len:digest format
