"""I/O helpers: run directory creation, image loading, JSON dumping."""
from __future__ import annotations

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from PIL import Image

from .errors import LowQualityImageError


MIN_IMAGE_DIM = 256  # below this on the shortest side -> warning


def new_run_id(prefix: str = "run") -> str:
    """Generate a timestamped unique run ID."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    short = uuid.uuid4().hex[:6]
    return f"{prefix}_{ts}_{short}"


def make_run_dir(outputs_root: Path, run_id: str) -> Path:
    run_dir = outputs_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def load_image(path: str | Path) -> Image.Image:
    """Load an image, raising LowQualityImageError if it can't be decoded."""
    p = Path(path)
    if not p.exists():
        raise LowQualityImageError(f"Image not found: {p}")
    try:
        img = Image.open(p).convert("RGB")
        img.load()
        return img
    except Exception as exc:
        raise LowQualityImageError(f"Could not decode image {p}: {exc}") from exc


def is_low_quality(img: Image.Image) -> tuple[bool, str]:
    """Return (is_low_quality, reason). Not fatal — just a warning flag."""
    w, h = img.size
    if min(w, h) < MIN_IMAGE_DIM:
        return True, f"Image is small ({w}x{h}); results may be poor."
    return False, ""


def dump_json(obj: Any, path: Path) -> None:
    """Serialize an object (dict or dataclass) to JSON."""
    if hasattr(obj, "to_dict"):
        obj = obj.to_dict()
    path.write_text(json.dumps(obj, indent=2, default=str))


def save_step_log(run_dir: Path, step: int, name: str, payload: Any) -> Path:
    """Save intermediate agent output for orchestration traceability."""
    log_path = run_dir / f"step_{step}_{name}.json"
    dump_json(payload, log_path)
    return log_path
