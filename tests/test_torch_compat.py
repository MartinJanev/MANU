import sys
import types

from common.torch_compat import alias_transform_get_item_to_index


def _fake_torch(monkeypatch, new_home_has_class):
    """Stand-in torch modules laid out like the NGC 24.12 image (class only in the old module) or a release."""
    torch = types.ModuleType("torch")
    dynamo = types.ModuleType("torch._dynamo")
    new_home = types.ModuleType("torch._dynamo._trace_wrapped_higher_order_op")
    hops = types.ModuleType("torch._higher_order_ops")
    old_home = types.ModuleType("torch._higher_order_ops.flex_attention")

    class TransformGetItemToIndex:
        pass

    old_home.TransformGetItemToIndex = TransformGetItemToIndex
    if new_home_has_class:
        new_home.TransformGetItemToIndex = TransformGetItemToIndex
    torch._dynamo, dynamo._trace_wrapped_higher_order_op = dynamo, new_home
    torch._higher_order_ops, hops.flex_attention = hops, old_home
    for module in (torch, dynamo, new_home, hops, old_home):
        monkeypatch.setitem(sys.modules, module.__name__, module)
    return new_home, TransformGetItemToIndex


def test_aliases_the_old_class_in_the_image_layout(monkeypatch):
    new_home, cls = _fake_torch(monkeypatch, new_home_has_class=False)
    assert alias_transform_get_item_to_index() is True
    assert new_home.TransformGetItemToIndex is cls
    assert alias_transform_get_item_to_index() is False  # second call: already in place


def test_leaves_a_release_torch_alone(monkeypatch):
    new_home, cls = _fake_torch(monkeypatch, new_home_has_class=True)
    assert alias_transform_get_item_to_index() is False
    assert new_home.TransformGetItemToIndex is cls


def test_no_torch(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", None)  # makes `import torch` raise ImportError
    assert alias_transform_get_item_to_index() is False
