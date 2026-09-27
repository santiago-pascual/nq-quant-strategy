"""Compatibility export for the canonical directional feature engine.

The implementation lives in ``src.research.direction.direction_features``.
Keeping this import path preserves the frozen research modules that use the
original public location without duplicating any feature calculations.
"""

from src.research.direction.direction_features import add_directional_features

__all__ = ["add_directional_features"]
