"""Reading the challenge TSV files and writing submission files."""
from __future__ import annotations

import csv
import os

import numpy as np
import pandas as pd

SOURCE_COLUMNS = ["entity_id", "business_name", "business_address", "country"]


def read_tsv(path: str) -> pd.DataFrame:
    """Read a challenge TSV exactly as-is.

    * tab separator (addresses and ID lists contain commas)
    * every column as string; empty cells stay empty strings (no NaN)
    * quoting disabled so stray quote characters inside names survive
    """
    return pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        na_values=[],
        quoting=csv.QUOTE_NONE,
        engine="c",
        encoding="utf-8",
    )


def read_source(data_dir: str, split: str, source: int) -> pd.DataFrame:
    path = os.path.join(data_dir, split, f"{split}_source{source}.tsv")
    df = read_tsv(path)
    missing = [c for c in SOURCE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing columns {missing}; found {list(df.columns)}")
    df = df[SOURCE_COLUMNS].copy()
    df["entity_id"] = df["entity_id"].str.strip()
    df["country"] = df["country"].str.strip()
    df["source"] = np.int8(source)
    return df


def read_ground_truth(data_dir: str) -> dict[str, list[str]]:
    path = os.path.join(data_dir, "train", "train_ground_truth.tsv")
    gt = read_tsv(path)
    out: dict[str, list[str]] = {}
    for s1, ids in zip(gt["source1_entity_id"].str.strip(), gt["matched_entity_ids"]):
        out[s1] = [x.strip() for x in ids.split(",") if x.strip()]
    return out


def write_id_lists(path: str, s1_ids, lists, value_column: str) -> None:
    """Write one row per S1 entity: ``source1_entity_id<TAB>id,id,id``."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as fh:
        fh.write(f"source1_entity_id\t{value_column}\n")
        for s1, ids in zip(s1_ids, lists):
            fh.write(f"{s1}\t{','.join(ids)}\n")
