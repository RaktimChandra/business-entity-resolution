"""Build the final submission archive in the exact structure required.

<team_name>_submission.zip
├── output/{matching_results.tsv, candidate_pairs.tsv}
├── code/business_entity_resolution/{src/, README.md, requirements.txt}
└── Documentation_template.md
"""
from __future__ import annotations

import os
import zipfile

_SKIP_DIRS = {"__pycache__", ".pytest_cache", ".ipynb_checkpoints", "work", "output", "dataset"}
_SKIP_EXT = {".pyc", ".pyo", ".parquet", ".tsv", ".zip"}


def build_zip(zip_path: str, out_dir: str, code_dir: str, doc_path: str, data_dir: str | None = None) -> str:
    required = [os.path.join(out_dir, "matching_results.tsv"), os.path.join(out_dir, "candidate_pairs.tsv"),
                os.path.join(code_dir, "README.md"), os.path.join(code_dir, "requirements.txt"), doc_path]
    missing = [p for p in required if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(f"cannot package, missing: {missing}")
    data_abs = os.path.abspath(data_dir) if data_dir else None
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.write(required[0], "output/matching_results.tsv")
        z.write(required[1], "output/candidate_pairs.tsv")
        for root, dirs, files in os.walk(code_dir):
            dirs[:] = [d for d in dirs if d not in _SKIP_DIRS
                       and not (data_abs and os.path.abspath(os.path.join(root, d)) == data_abs)]
            for f in files:
                if os.path.splitext(f)[1] in _SKIP_EXT:
                    continue
                full = os.path.join(root, f)
                rel = os.path.relpath(full, code_dir)
                z.write(full, os.path.join("code", "business_entity_resolution", rel))
        z.write(doc_path, "Documentation_template.md")
    return zip_path


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--code-dir", required=True)
    ap.add_argument("--doc", required=True)
    a = ap.parse_args()
    print(build_zip(a.zip, a.out_dir, a.code_dir, a.doc))
