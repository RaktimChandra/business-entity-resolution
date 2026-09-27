"""Lightweight run tracking: structured logs, stage timings, peak memory.

Every stage of the pipeline is wrapped in ``tracker.stage(name)``; timings,
peak RSS and arbitrary metrics are appended to ``run_log.json`` so a run can
be audited after the fact.
"""
from __future__ import annotations

import json
import logging
import os
import sys
import time
from contextlib import contextmanager

try:
    import resource  # POSIX only
except ImportError:  # pragma: no cover - Windows
    resource = None


def _peak_rss_mb() -> float:
    if resource is None:
        return float("nan")
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # Linux reports KiB, macOS reports bytes.
    return peak / (1024 * 1024) if sys.platform == "darwin" else peak / 1024


def get_logger(log_path: str | None = None) -> logging.Logger:
    logger = logging.getLogger("ber")
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", "%H:%M:%S")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    if log_path:
        fh = logging.FileHandler(log_path, encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger


class Tracker:
    def __init__(self, work_dir: str):
        os.makedirs(work_dir, exist_ok=True)
        self.path = os.path.join(work_dir, "run_log.json")
        self.log = get_logger(os.path.join(work_dir, "run.log"))
        self.stages: list[dict] = []
        self.metrics: dict = {}
        self._t0 = time.time()

    @contextmanager
    def stage(self, name: str):
        self.log.info("▶ %s", name)
        t = time.time()
        try:
            yield
        finally:
            dt = time.time() - t
            rec = {"stage": name, "seconds": round(dt, 2), "peak_rss_mb": round(_peak_rss_mb(), 1)}
            self.stages.append(rec)
            self.log.info("✔ %s  (%.1fs, peak RSS %.0f MB)", name, dt, rec["peak_rss_mb"])
            self.flush()

    def record(self, key: str, value) -> None:
        self.metrics[key] = value
        self.flush()

    def flush(self) -> None:
        payload = {
            "total_seconds": round(time.time() - self._t0, 2),
            "stages": self.stages,
            "metrics": self.metrics,
        }
        with open(self.path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, default=_json_default)


def _json_default(o):
    try:
        import numpy as np
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
    except ImportError:  # pragma: no cover
        pass
    return str(o)
