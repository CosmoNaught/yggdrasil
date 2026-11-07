"""Yggdrasil: a linear recurrence whose transition is a stored expert picked by a tree router."""
import os

# fp32 matmuls (no TF32) so GPU runs match CPU; set before jax reads them
os.environ.setdefault("JAX_DEFAULT_MATMUL_PRECISION", "highest")
os.environ.setdefault("TRITON_F32_DEFAULT", "ieee")

from .inference import Yggdrasil  # noqa: E402

__all__ = ["Yggdrasil"]
