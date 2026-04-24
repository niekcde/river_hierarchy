"""Phase 1 compatibility exports for shared synthetic network objects."""

from __future__ import annotations

import sys
from pathlib import Path


_LEGACY_DIR = Path(__file__).resolve().parents[3]
if str(_LEGACY_DIR) not in sys.path:
    sys.path.insert(0, str(_LEGACY_DIR))

from synthetic_admissable_networkx_part_save import Params, RiverNetworkNX, canonical_signature


__all__ = ["Params", "RiverNetworkNX", "canonical_signature"]
