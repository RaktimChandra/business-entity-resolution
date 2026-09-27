"""Independent submission checker mirroring every rule in the problem statement.

Checked for both ``matching_results.tsv`` and ``candidate_pairs.tsv``:
* exact header, tab-separated, two columns per line;
* exactly one row per test Source 1 entity, no extra or duplicate rows;
* IDs are S2-/S3- only, exist in the test set, no duplicates within a list;
* every matched ID also appears among the candidates of the same entity.
"""
from __future__ import annotations

import os
import subprocess
import sys


def _read_ids(path: str) -> set[str]:
    ids = set()
    with open(path, encoding="utf-8") as fh:
        header = fh.readline().rstrip("\n").split("\t")
        col = header.index("entity_id")
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) > col:
                ids.add(parts[col].strip())
    return ids


def _check_file(path, header_col, s1_ids, target_ids, errors, label):
    rows = {}
    if not os.path.exists(path):
        errors.append(f"{label}: file not found: {path}")
        return rows
    with open(path, encoding="utf-8") as fh:
        header = fh.readline().rstrip("\n").rstrip("\r")
        if header != f"source1_entity_id\t{header_col}":
            errors.append(f"{label}: bad header {header!r}")
        for ln, line in enumerate(fh, start=2):
            line = line.rstrip("\n").rstrip("\r")
            parts = line.split("\t")
            if len(parts) != 2:
                errors.append(f"{label}: line {ln} has {len(parts)} columns")
                continue
            s1, lst = parts
            if s1 in rows:
                errors.append(f"{label}: duplicate row for {s1}")
            if s1 not in s1_ids:
                errors.append(f"{label}: unknown source1 id {s1} (line {ln})")
            ids = [x for x in lst.split(",")] if lst else []
            if any(x == "" or x != x.strip() for x in ids):
                errors.append(f"{label}: empty/whitespace id in list for {s1}")
            if len(set(ids)) != len(ids):
                errors.append(f"{label}: duplicate ids in list for {s1}")
            for x in ids:
                if not (x.startswith("S2-") or x.startswith("S3-")):
                    errors.append(f"{label}: non S2/S3 id {x} for {s1}")
                elif x not in target_ids:
                    errors.append(f"{label}: id {x} not in test set ({s1})")
            rows[s1] = set(ids)
            if len(errors) > 50:
                errors.append("... (truncated)")
                return rows
    missing = s1_ids - set(rows)
    if missing:
        errors.append(f"{label}: {len(missing)} source1 entities missing, e.g. {sorted(missing)[:3]}")
    return rows


def validate(matching: str, candidate: str, test_dir: str) -> list[str]:
    s1 = _read_ids(os.path.join(test_dir, "test_source1.tsv"))
    tgt = _read_ids(os.path.join(test_dir, "test_source2.tsv")) | _read_ids(os.path.join(test_dir, "test_source3.tsv"))
    errors: list[str] = []
    m = _check_file(matching, "matched_entity_ids", s1, tgt, errors, "matching")
    c = _check_file(candidate, "candidate_entity_ids", s1, tgt, errors, "candidate")
    not_subset = [k for k, v in m.items() if not v <= c.get(k, set())]
    if not_subset:
        errors.append(f"{len(not_subset)} entities have matches that are not candidates, e.g. {not_subset[:3]}")
    return errors


def run_official(validator_path: str, matching: str, candidate: str, test_dir: str):
    """Run the organisers' validator if it is available; returns (ran, passed, output)."""
    if not validator_path or not os.path.exists(validator_path):
        return False, None, "official validator not found"
    proc = subprocess.run([sys.executable, validator_path, "--matching", matching,
                           "--candidate", candidate, "--test-dir", test_dir],
                          capture_output=True, text=True)
    return True, proc.returncode == 0, (proc.stdout + proc.stderr).strip()


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--matching", required=True)
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--test-dir", required=True)
    a = ap.parse_args()
    errs = validate(a.matching, a.candidate, a.test_dir)
    print("PASS" if not errs else "\n".join(f"{i + 1}. {e}" for i, e in enumerate(errs)))
    sys.exit(0 if not errs else 1)
