"""Keeps loaded models in memory between separations.

Loading a model (reading a ~1 GB checkpoint, building the network, or creating
an ONNX Runtime session) used to happen for every song and every pass. The
cache keeps the most recent models in RAM, but only the model in use stays on
the GPU, so peak VRAM is the same as reloading each time.
"""
import gc
import os
from collections import OrderedDict

import torch

MAX_CACHED_MODELS = 2

_entries = OrderedDict()


def file_key(path):
    """Identity of a model file on disk, so a replaced file is reloaded."""
    stat = os.stat(path)
    return os.path.abspath(path), stat.st_mtime_ns, stat.st_size


def _on_accelerator(model):
    return any(p.device.type != 'cpu' for p in model.parameters())


def _park(key):
    """Move a cached model off the GPU. Returns True if GPU memory was released."""
    model = _entries[key]
    if isinstance(model, torch.nn.Module):
        if _on_accelerator(model):
            model.to('cpu')
            return True
        return False
    # ONNX Runtime sessions can't move off the GPU, so drop them.
    del _entries[key]
    return True


def _empty_device_caches():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


def get(key, loader, device=None):
    """Return the cached model for key, loading it with loader() if needed.

    torch modules are moved to device; any other cached model is moved off
    the GPU first.
    """
    released = False
    for other_key in list(_entries):
        if other_key != key:
            released |= _park(other_key)
    if released:
        _empty_device_caches()

    if key in _entries:
        _entries.move_to_end(key)
        model = _entries[key]
    else:
        model = loader()
        _entries[key] = model
        while len(_entries) > MAX_CACHED_MODELS:
            _entries.popitem(last=False)

    if device is not None and isinstance(model, torch.nn.Module):
        model.to(device)
    return model


def release_gpu():
    """Moves every cached model off the GPU, keeping it in RAM for the next run.

    Call only when no separation is running.
    """
    released = False
    for key in list(_entries):
        released |= _park(key)
    if released:
        _empty_device_caches()


def clear():
    _entries.clear()
    _empty_device_caches()
