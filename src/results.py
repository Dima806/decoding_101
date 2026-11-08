"""Write the numbers and text behind every figure to JSON, next to the PNGs.

Each file is ``outputs/results/<name>.json`` with the seed that produced it::

    {"name": "...", "seed": 42, "data": ...}

``data`` may contain DataFrames, Series, tensors, numpy values, and paths; they are converted to
plain JSON. NaN and infinities become ``null``.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from torch import Tensor

from src.config import get_settings


def to_jsonable(obj: Any) -> Any:
    """Recursively convert ``obj`` into JSON-serialisable Python values."""
    if isinstance(obj, pd.DataFrame):
        default_index = isinstance(obj.index, pd.RangeIndex) and obj.index.name is None
        frame = obj if default_index else obj.reset_index()
        return [to_jsonable(row) for row in frame.to_dict(orient="records")]
    if isinstance(obj, pd.Series):
        return to_jsonable(obj.to_dict())
    if isinstance(obj, Tensor | np.ndarray):
        return to_jsonable(obj.tolist())
    if isinstance(obj, np.generic):
        return to_jsonable(obj.item())
    if isinstance(obj, Mapping):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple | set | range):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, bool | int | str) or obj is None:
        return obj
    if isinstance(obj, Path):
        return str(obj)
    return str(obj)


def save_json(name: str, data: Any, directory: Path | None = None) -> Path:
    """Write ``data`` to ``<results dir>/<name>.json`` with the configured seed."""
    settings = get_settings()
    out_dir = directory or settings.results_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.json"
    payload = {"name": name, "seed": settings.seed, "data": to_jsonable(data)}
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    return path
