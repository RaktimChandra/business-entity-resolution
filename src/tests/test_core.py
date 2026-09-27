import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.dirname(HERE)
sys.path.insert(0, SRC)

from ber.decision import decode_expected_f, decode_threshold, enforce_exclusivity  # noqa: E402
from ber.metrics import GroundTruth, macro_f05  # noqa: E402
from ber.normalize import normalize_record, phonetic_token  # noqa: E402


def test_metric_matches_official_example():
    # S1-00001 predicts [S2-00047, S2-00193, S3-00812]; truth [S2-00047, S3-00812] -> 0.714
    gt = GroundTruth(["S1-00001"], ["S2-00047", "S2-00193", "S3-00812"],
                     {"S1-00001": ["S2-00047", "S3-00812"]})
    r = macro_f05(gt, np.array([0]), np.array([0, 0, 0]), np.array([0, 1, 2]))
    assert abs(r["macro_f05"] - 0.7142857) < 1e-6


def test_metric_singletons():
    gt = GroundTruth(["a", "b"], ["x"], {"a": [], "b": ["x"]})
    # empty prediction for singleton = 1, correct match = 1
    assert macro_f05(gt, np.array([0, 1]), np.array([1]), np.array([0]))["macro_f05"] == 1.0
    # false merge on singleton = 0 -> mean 0.5
    assert macro_f05(gt, np.array([0, 1]), np.array([0, 1]), np.array([0, 0]))["macro_f05"] == 0.5
    # missing a matched entity = 0
    assert macro_f05(gt, np.array([0, 1]), np.array([], int), np.array([], int))["macro_f05"] == 0.5


def test_normalization_views():
    n = dict(zip(["name_norm", "name_core", "name_phon", "addr_norm", "nums", "postal", "codes",
                  "translit", "addr_missing"],
                 normalize_record("Sharma Traders Pvt. Ltd.", "12, M.G. Rd, Bangalore 560001")))
    assert n["name_core"] == "sharma traders"
    assert "road" in n["addr_norm"] and "bengaluru" in n["addr_norm"]
    assert n["postal"] == "560001" and "12" in n["nums"].split()
    d = dict(zip(["name_norm", "name_core"], normalize_record("एसएस फूड प्राइवेट लिमिटेड", "None")))
    assert d["name_core"].isascii() and "limited" not in d["name_core"]
    f = normalize_record("Café Crème SARL", "12 av. des Champs-Élysées, 75008 Paris")
    assert f[1] == "cafe creme" and "avenue" in f[3]
    assert normalize_record("None", "null")[-1] == 1


def test_phonetic_transliteration_variants():
    assert phonetic_token("lakshmi") == phonetic_token("laxmi")
    assert phonetic_token("private") == phonetic_token("praivet")


def test_decoders():
    s1 = np.array([0, 0, 0, 1, 1, 2])
    p = np.array([0.95, 0.9, 0.1, 0.2, 0.15, 0.6])
    sel = decode_threshold(s1, p, 0.5, 0.8)
    assert sel.tolist() == [True, True, False, False, False, True]
    sel = decode_expected_f(s1, p)
    assert sel[0] and sel[1] and not sel[2] and not sel[3] and not sel[4]
    # exclusivity keeps the target on the entity with the highest probability
    t = np.array([7, 8, 9, 7, 5, 6])
    ex = enforce_exclusivity(s1, t, np.array([0.6, 0.9, 0.1, 0.95, 0.1, 0.6]), np.array([1, 1, 0, 1, 0, 1], bool))
    assert ex.tolist() == [False, True, False, True, False, True]


def test_end_to_end(tmp_path):
    data = tmp_path / "dataset"
    subprocess.run([sys.executable, os.path.join(SRC, "tools", "make_synthetic.py"), "--out", str(data),
                    "--n-train", "800", "--n-test", "600"], check=True)
    r = subprocess.run([sys.executable, os.path.join(SRC, "main.py"), "run", "--data-dir", str(data),
                        "--out-dir", str(tmp_path / "output"), "--work-dir", str(tmp_path / "work"),
                        "--n-jobs", "1", "--n-folds", "2", "--lgb-rounds", "80",
                        "--zip", str(tmp_path / "sub.zip")], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    from ber.validate import validate
    assert validate(str(tmp_path / "output" / "matching_results.tsv"),
                    str(tmp_path / "output" / "candidate_pairs.tsv"), str(data / "test")) == []
    from tools.score import score
    s = score(str(tmp_path / "output" / "matching_results.tsv"), str(data / "test_ground_truth_hidden.tsv"))
    assert s["macro_f05"] > 0.85
    doc = (tmp_path / "work" / "Documentation_template.md").read_text()
    assert "{{" not in doc
    import zipfile
    names = set(zipfile.ZipFile(tmp_path / "sub.zip").namelist())
    for req in ("output/matching_results.tsv", "output/candidate_pairs.tsv", "Documentation_template.md",
                "code/business_entity_resolution/README.md", "code/business_entity_resolution/requirements.txt",
                "code/business_entity_resolution/src/main.py"):
        assert req in names
