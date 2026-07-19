"""
Optional dependency helper.
Centralizes imports for Torch and snnTorch to avoid crash on non-DL systems.
"""

# First check pytorch availability
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, random_split, TensorDataset, Dataset
    torch_available = True
except ImportError:
    torch_available = False
    class Dummy:
        """
        Placeholder object used when an optional dependency is unavailable.
        """
        pass
    torch = Dummy()
    nn = Dummy()
    nn.Module = object
    DataLoader = object
    random_split = object
    TensorDataset = object
    Dataset = object

# Then check snntorch availability
if torch_available:
    try:
        import snntorch as snn
        from snntorch import surrogate, utils
        snntorch_available = True
    except ImportError:
        snntorch_available = False
        snn = object
        surrogate = object
        utils = object
else:
    snntorch_available = False
    snn = object
    surrogate = object
    utils = object

__all__ = [
    "torch", "nn", "DataLoader", "random_split", "TensorDataset", "Dataset",
    "torch_available",
    "snn", "surrogate", "utils", "snntorch_available"
]