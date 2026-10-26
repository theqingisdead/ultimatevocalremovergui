"""Lets NumPy 1.x load files pickled with NumPy 2.

NumPy 2 renamed numpy.core to numpy._core, so checkpoints saved with NumPy 2
(common for newer community models) reference numpy._core.* and fail under
the NumPy 1.x UVR uses with "No module named 'numpy._core'". Import this after
other libraries, since some check for numpy._core to detect NumPy 2.
"""
import importlib
import sys

try:
    import numpy._core  # noqa: F401  (NumPy 2, nothing to do)
except ImportError:
    import numpy.core
    sys.modules['numpy._core'] = numpy.core
    for name in ('multiarray', 'numeric', '_multiarray_umath', 'umath', 'fromnumeric',
                 '_internal', 'records', 'defchararray', '_dtype', 'numerictypes'):
        try:
            sys.modules[f'numpy._core.{name}'] = importlib.import_module(f'numpy.core.{name}')
        except ImportError:
            pass
