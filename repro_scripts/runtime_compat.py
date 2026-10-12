"""Expose the same DTensor class under its newer public import path on Torch 2.2."""
import torch.distributed.tensor as public_tensor

def ensure_dtensor_alias():
    if not hasattr(public_tensor, 'DTensor'):
        from torch.distributed._tensor import DTensor
        public_tensor.DTensor = DTensor
    import transformers.modeling_utils as modeling_utils
    if not hasattr(modeling_utils, 'DTensor'):
        modeling_utils.DTensor = public_tensor.DTensor
