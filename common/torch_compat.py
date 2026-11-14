"""Work-around for the PyTorch in containers/ngc2412.sif (NGC 24.12: 2.6.0a0+df5bbc0, a pre-release of 2.6).

transformers >= 4.53 (masking_utils.py) runs
    from torch._dynamo._trace_wrapped_higher_order_op import TransformGetItemToIndex
whenever torch reports 2.6 or newer, pre-releases included. The class moved to that module in the 2.6.0
release; the image's snapshot still defines the same class in torch._higher_order_ops.flex_attention. So in
the image `from transformers import AutoModel` fails with ImportError, and open_clip reports that failure as
"Please `pip install transformers`". Pins affected: hlip (4.57.6), braingemma3d (4.57.3), healthgptpro27b (5.6.0).

alias_transform_get_item_to_index() publishes the image's own class under the new name. It has to run before
transformers is imported: common.runner calls it on import, and every adapter imports common.runner first.
It changes nothing where the class is already in place (torch 2.6.0 and later) or where torch is missing.
"""


def alias_transform_get_item_to_index() -> bool:
    """Add the alias if this torch needs it; True if it was added."""
    try:
        from torch._dynamo import _trace_wrapped_higher_order_op as new_home
    except ImportError:  # no torch here (local tests), or a torch without this module
        return False
    if hasattr(new_home, "TransformGetItemToIndex"):
        return False
    try:
        from torch._higher_order_ops.flex_attention import TransformGetItemToIndex
    except ImportError:
        return False
    new_home.TransformGetItemToIndex = TransformGetItemToIndex
    return True
