# compat.py
"""
Compatibility shim to provide the small subset of `tensorflow.contrib` used
by older GQ-CNN code. Designed to be imported *before* code that does
`import tensorflow.contrib.*` so that we avoid ModuleNotFoundError.


This is intentionally small and only implements what GQ-CNN typically expects:
- framework.arg_scope (via tf_slim.arg_scope)
- layers (tf_slim)
- get_or_create_global_step (via tf.compat.v1)


Requirements: install tf_slim (pip install tf_slim) and use TF2 with v1
compatibility (or TF1).
"""
import types
import sys


# Note: import tensorflow.compat.v1 as tf and disable v2 behavior in package init
import tensorflow.compat.v1 as tf


# Small helper to build a contrib-like namespace
_contrib = types.SimpleNamespace()


# framework shim
class _FrameworkShim:
    @staticmethod
    def arg_scope(*args, **kwargs):
        try:
            import tf_slim as slim
        except Exception as e:
            raise ImportError(
            "tf_slim is required for arg_scope compatibility. Install with: pip install tf_slim"
            ) from e
        return slim.arg_scope(*args, **kwargs)


    @staticmethod
    def get_or_create_global_step():
        return tf.train.get_or_create_global_step()


_contrib.framework = _FrameworkShim()


# layers shim (tf_slim if available, else fallback to tf.layers)
try:
    import tf_slim as slim
    _contrib.layers = slim
except Exception:
# fallback: provide minimal layers namespace mapping to tf.compat.v1.layers
    class _LayersShim:
        dense = tf.layers.dense
        conv2d = tf.layers.conv2d
        max_pool2d = tf.layers.max_pooling2d
        batch_norm = tf.layers.batch_normalization


    _contrib.layers = _LayersShim()


# expose training as empty namespace (if code expects it)
_contrib.training = types.SimpleNamespace()


# Insert into a fake `tensorflow.contrib` module if not present
if 'tensorflow' in sys.modules and not hasattr(sys.modules['tensorflow'], 'contrib'):
    # attach contrib to the tensorflow package object
    setattr(sys.modules['tensorflow'], 'contrib', _contrib)


# Also export convenient names for direct imports
tcf = _contrib.framework
layers = _contrib.layers
training = _contrib.training