"""Multilingual, country-agnostic normalization of business names and addresses.

Design goals
------------
* Script-agnostic: Indic scripts (Devanagari, Tamil, Bengali, ...) and accented
  Latin (French) are transliterated to ASCII so that string similarity works
  across scripts.
* Country-agnostic: no rule depends on the country label, so unseen countries
  in the test set (France) are handled by exactly the same code path.
* Loss-aware: several views are produced per field (full, core, phonetic,
  numeric) so that downstream features can decide which view to trust.
"""
from __future__ import annotations

import re
import unicodedata
from multiprocessing import Pool

import numpy as np
import pandas as pd

try:
    from anyascii import anyascii as _to_ascii
except ImportError:  # pragma: no cover - dependency is pinned in requirements
    def _to_ascii(s: str) -> str:
        return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()

# ---------------------------------------------------------------------------
# Vocabularies. These are generic linguistic abbreviations, not data lookups.
# ---------------------------------------------------------------------------
MISSING_VALUES = {"", "none", "null", "nan", "na", "n/a", "n a", "-", "--", "nil", "not available", "unknown"}

NAME_ABBREV = {
    "pvt": "private", "pvtltd": "private limited", "prv": "private", "priv": "private",
    "ltd": "limited", "ltda": "limited", "lmt": "limited", "ld": "limited",
    "corp": "corporation", "corpn": "corporation", "co": "company", "cos": "companies",
    "inc": "incorporated", "incorp": "incorporated", "intl": "international", "int'l": "international",
    "natl": "national", "mfg": "manufacturing", "mfrs": "manufacturers", "mfr": "manufacturer",
    "svc": "services", "svcs": "services", "serv": "services", "srvs": "services",
    "assoc": "associates", "assocs": "associates", "bros": "brothers", "bro": "brother",
    "mgmt": "management", "mgt": "management", "dev": "development", "devt": "development",
    "eng": "engineering", "engg": "engineering", "engr": "engineering",
    "ent": "enterprises", "entp": "enterprises", "enterprise": "enterprises",
    "ind": "industries", "inds": "industries", "industry": "industries",
    "tech": "technologies", "technology": "technologies", "tecnologies": "technologies",
    "sys": "systems", "sol": "solutions", "soln": "solutions", "solns": "solutions",
    "grp": "group", "hldgs": "holdings", "hldg": "holdings", "invest": "investments",
    "dept": "department", "univ": "university", "inst": "institute", "hosp": "hospital",
    "mkt": "market", "mkts": "markets", "rest": "restaurant", "resto": "restaurant",
    "ctr": "center", "centre": "center", "cntr": "center", "phar": "pharmacy",
    "st": "saint", "ste": "sainte", "mt": "mount", "et": "and", "und": "and",
    "cie": "compagnie", "ets": "etablissements", "sct": "societe",
}

ADDR_ABBREV = {
    "rd": "road", "st": "street", "str": "street", "ave": "avenue", "av": "avenue", "avn": "avenue",
    "blvd": "boulevard", "bd": "boulevard", "boul": "boulevard", "bvd": "boulevard",
    "ln": "lane", "dr": "drive", "drv": "drive", "ct": "court", "crt": "court", "pl": "place",
    "hwy": "highway", "hway": "highway", "pkwy": "parkway", "pky": "parkway", "sq": "square",
    "ste": "suite", "apt": "apartment", "apts": "apartments", "fl": "floor", "flr": "floor",
    "bldg": "building", "bld": "building", "cir": "circle", "ter": "terrace", "terr": "terrace",
    "trl": "trail", "expy": "expressway", "fwy": "freeway", "tpke": "turnpike", "cres": "crescent",
    "n": "north", "s": "south", "e": "east", "w": "west",
    "ne": "northeast", "nw": "northwest", "se": "southeast", "sw": "southwest",
    "mt": "mount", "ft": "fort", "pt": "point", "hts": "heights", "jct": "junction",
    "opp": "opposite", "nr": "near", "nxt": "next", "bhd": "behind",
    "mkt": "market", "stn": "station", "rly": "railway", "sec": "sector", "sect": "sector",
    "ph": "", "tel": "", "phone": "", "mob": "", "mobile": "", "contact": "", "extn": "extension", "ext": "extension", "chk": "chowk", "clny": "colony",
    "col": "colony", "ngr": "nagar", "bazar": "bazaar", "sal": "salai", "mg": "mahatma gandhi",
    "r": "rue", "ch": "chemin", "imp": "impasse", "all": "allee", "rte": "route", "fbg": "faubourg",
    "qu": "quai", "crs": "cours", "za": "zone activite", "zi": "zone industrielle",
    "cedex": "", "bp": "",
    # city renamings that commonly co-exist in the same source
    "bangalore": "bengaluru", "bombay": "mumbai", "madras": "chennai", "calcutta": "kolkata",
    "gurgaon": "gurugram", "poona": "pune", "mysore": "mysuru", "trivandrum": "thiruvananthapuram",
    "baroda": "vadodara", "cochin": "kochi", "pondicherry": "puducherry", "vizag": "visakhapatnam",
    "benares": "varanasi", "banaras": "varanasi", "allahabad": "prayagraj", "belgaum": "belagavi",
}

LEGAL_TOKENS = {
    "private", "limited", "llp", "llc", "lc", "incorporated", "corporation", "company", "companies",
    "plc", "lp", "pllc", "pc", "pa", "gmbh", "ag", "bv", "nv", "opc",
    "sarl", "sas", "sasu", "sa", "eurl", "sci", "snc", "scp", "selarl", "eirl", "ei", "sca", "scs",
    "compagnie", "etablissements", "societe", "the", "and", "of", "de", "du", "des", "la", "le",
    "les", "l", "d", "et", "fils", "ltd",
}

ADDR_FILLER = {
    "near", "opposite", "behind", "beside", "next", "to", "the", "of", "at", "and", "in", "on",
    "no", "num", "number", "plot", "door", "flat", "hno", "h", "house", "shop", "unit", "suite",
    "floor", "apartment", "building", "ground", "first", "second", "third", "de", "du", "des",
    "la", "le", "les", "l", "d", "et", "a", "an",
}

_RE_LETTER_SEP_DIGIT = re.compile(r"(?<=[a-z])[\-/.](?=\d)|(?<=\d)[\-/.](?=[a-z])")
_RE_NON_ALNUM = re.compile(r"[^a-z0-9]+")
_RE_DIGITS = re.compile(r"\d+")
_RE_SPACES = re.compile(r"\s+")

# ordered phonetic folds for transliteration-robust keys
_PHON_RULES = [
    ("ksh", "x"), ("ck", "k"), ("ph", "f"), ("sh", "s"), ("ch", "c"), ("th", "t"),
    ("dh", "d"), ("bh", "b"), ("kh", "k"), ("gh", "g"), ("jh", "j"), ("q", "k"),
    ("w", "v"), ("z", "j"), ("y", "i"), ("ee", "i"), ("oo", "u"), ("ou", "u"),
]
_VOWELS = set("aeiou")


def _base_clean(text: str) -> tuple[str, bool]:
    """Lowercase, transliterate to ASCII, unify separators. Returns (text, transliterated)."""
    if text is None:
        return "", False
    s = str(text)
    translit = False
    if not s.isascii():
        # Indic / other scripts -> Latin; accents stripped.
        translit = any(ord(ch) > 0x2FF for ch in s)
        s = _to_ascii(s)
    s = s.lower().strip()
    if s in MISSING_VALUES:
        return "", translit
    s = s.replace("&", " and ").replace("@", " at ").replace("+", " and ")
    s = s.replace("'", "")  # o'brien -> obrien, int'l -> intl
    s = _RE_LETTER_SEP_DIGIT.sub("", s)  # af-0684 -> af0684 ; 12-b -> 12b
    s = _RE_NON_ALNUM.sub(" ", s)
    s = _RE_SPACES.sub(" ", s).strip()
    if s in MISSING_VALUES:
        return "", translit
    return s, translit


def _join_initials(tokens: list[str]) -> list[str]:
    """Merge runs of single letters produced by dotted initials: 'm g road' -> 'mg road'."""
    out: list[str] = []
    run: list[str] = []
    for t in tokens:
        if len(t) == 1 and t.isalpha():
            run.append(t)
            continue
        if run:
            out.append("".join(run) if len(run) > 1 else run[0])
            run = []
        out.append(t)
    if run:
        out.append("".join(run) if len(run) > 1 else run[0])
    return out


def _expand(tokens: list[str], table: dict[str, str]) -> list[str]:
    out: list[str] = []
    for t in tokens:
        r = table.get(t)
        if r is None:
            out.append(t)
        elif r:
            out.extend(r.split())
    return out


def phonetic_token(tok: str) -> str:
    """Transliteration-robust skeleton: fold digraphs, drop inner vowels, collapse repeats."""
    if not tok or tok.isdigit():
        return tok
    s = tok
    for a, b in _PHON_RULES:
        s = s.replace(a, b)
    head, tail = s[0], s[1:]
    tail = "".join(c for c in tail if c not in _VOWELS and c != "h")
    s = head + tail
    out = [s[0]]
    for c in s[1:]:
        if c != out[-1]:
            out.append(c)
    return "".join(out)


# phonetic skeletons of long legal words: catches transliterated forms such as
# "praivet" (private) or "limitad" (limited) coming out of non-Latin scripts.
# Only applied to trailing tokens (legal suffixes sit at the end of a name).
_LEGAL_PHON = {phonetic_token(t) for t in ("private", "limited", "corporation", "incorporated")}


def _numbers(tokens: list[str]) -> list[str]:
    nums = []
    for t in tokens:
        for d in _RE_DIGITS.findall(t):
            d2 = d.lstrip("0") or "0"
            nums.append(d2)
    return nums


def normalize_record(name: str, addr: str) -> tuple:
    """Return all normalized views for a single record."""
    n_clean, n_tr = _base_clean(name)
    a_clean, a_tr = _base_clean(addr)

    n_tok = _expand(_join_initials(n_clean.split()), NAME_ABBREV)
    name_norm = " ".join(n_tok)
    last = len(n_tok) - 1
    core_tok = [t for i, t in enumerate(n_tok)
                if t not in LEGAL_TOKENS
                and not (i > 0 and i >= last - 1 and len(t) >= 5 and phonetic_token(t) in _LEGAL_PHON)]
    if not core_tok:
        core_tok = n_tok
    name_core = " ".join(core_tok)
    name_phon = " ".join(phonetic_token(t) for t in core_tok)

    a_tok = _expand(_join_initials(a_clean.split()), ADDR_ABBREV)
    a_keep = [t for t in a_tok if t not in ADDR_FILLER]
    addr_norm = " ".join(a_keep)

    nums_all = _numbers(a_tok) + _numbers(n_tok)
    nums = " ".join(sorted(set(nums_all)))
    postal = " ".join(sorted({d for d in nums_all if 5 <= len(d) <= 6}))
    codes = " ".join(sorted({t for t in a_tok + n_tok if not t.isdigit() and any(c.isdigit() for c in t)}))

    return (name_norm, name_core, name_phon, addr_norm, nums, postal, codes,
            np.int8(n_tr or a_tr), np.int8(addr_norm == ""))


NORM_COLUMNS = ["name_norm", "name_core", "name_phon", "addr_norm", "nums", "postal", "codes",
                "translit", "addr_missing"]


def _normalize_chunk(args):
    names, addrs = args
    return [normalize_record(n, a) for n, a in zip(names, addrs)]


def normalize_frame(df: pd.DataFrame, n_jobs: int = 1, chunk: int = 200_000) -> pd.DataFrame:
    """Add normalized columns to a source frame (parallel over chunks)."""
    names = df["business_name"].tolist()
    addrs = df["business_address"].tolist()
    parts = [(names[i:i + chunk], addrs[i:i + chunk]) for i in range(0, len(names), chunk)]
    if n_jobs > 1 and len(parts) > 1:
        with Pool(min(n_jobs, len(parts))) as pool:
            results = pool.map(_normalize_chunk, parts)
    else:
        results = [_normalize_chunk(p) for p in parts]
    rows = [r for part in results for r in part]
    out = df.copy()
    if rows:
        cols = list(zip(*rows))
    else:
        cols = [[] for _ in NORM_COLUMNS]
    for c, vals in zip(NORM_COLUMNS, cols):
        if c in ("translit", "addr_missing"):
            out[c] = np.asarray(vals, dtype=np.int8)
        else:
            out[c] = pd.Series(vals, index=out.index, dtype=object)
    return out
