"""Synthetic dataset in the exact challenge format, used only for automated
end-to-end tests of the pipeline (never for training a submission model).

It reproduces the documented noise patterns: abbreviations, legal-suffix
drift, typos, word-order swaps, transliteration into Devanagari, missing
address parts, landmark references, chains with the same name at different
addresses, singletons, and a test-only country.
"""
from __future__ import annotations

import argparse
import os
import random

WORDS_IN = ["Sharma", "Gupta", "Agarwal", "Lakshmi", "Ganesh", "Balaji", "Sri", "Sai", "Om", "Krishna",
            "Traders", "Stores", "Enterprises", "Medical", "Hardware", "Electronics", "Sweets", "Food",
            "Textiles", "Motors", "Jewellers", "Pharma", "Agencies", "Industries"]
WORDS_US = ["Blue", "Ridge", "Summit", "Pioneer", "Golden", "Eagle", "Harbor", "Maple", "Liberty", "Joe's",
            "Pizza", "Grill", "Plumbing", "Dental", "Auto", "Repair", "Bakery", "Consulting", "Logistics",
            "Realty", "Fitness", "Coffee", "Hardware", "Cleaning"]
WORDS_FR = ["Boulangerie", "Dupont", "Martin", "Bernard", "Petit", "Durand", "Café", "Pharmacie", "Garage",
            "Coiffure", "Élégance", "Atelier", "Fromagerie", "Librairie", "Boucherie", "Crème", "Lefèvre"]
LEGAL = {"India": ["Pvt Ltd", "Private Limited", "LLP", ""], "US": ["Inc", "LLC", "Corp", "Corporation", ""],
         "France": ["SARL", "SAS", "EURL", ""]}
STREETS = {"India": ["MG Road", "Anna Salai", "Station Road", "Gandhi Nagar", "Nehru Street", "Park Road"],
           "US": ["Main Street", "Oak Avenue", "Maple Drive", "Belden Avenue", "Sunset Boulevard", "Elm Street"],
           "France": ["rue de la Paix", "avenue Victor Hugo", "boulevard Haussmann", "rue Lafayette"]}
CITIES = {"India": [("Bengaluru", "5600"), ("Chennai", "6000"), ("Mumbai", "4000"), ("Kolkata", "7000")],
          "US": [("Chicago", "606"), ("Austin", "787"), ("Denver", "802"), ("Seattle", "981")],
          "France": [("Paris", "750"), ("Lyon", "690"), ("Marseille", "130")]}
DEVANAGARI = {"Sharma": "शर्मा", "Gupta": "गुप्ता", "Traders": "ट्रेडर्स", "Stores": "स्टोर्स", "Food": "फूड",
              "Medical": "मेडिकल", "Sweets": "स्वीट्स", "Private": "प्राइवेट", "Limited": "लिमिटेड",
              "Agarwal": "अग्रवाल", "Hardware": "हार्डवेयर", "Krishna": "कृष्णा", "Sai": "साई"}
ABBR = {"Private Limited": "Pvt Ltd", "Pvt Ltd": "Private Limited", "Corporation": "Corp", "Corp": "Corporation",
        "Road": "Rd", "Street": "St", "Avenue": "Ave", "Boulevard": "Blvd", "Drive": "Dr", "avenue": "av.",
        "boulevard": "bd", "Incorporated": "Inc", "Inc": "Incorporated", "&": "and"}


def typo(s, rng):
    if len(s) < 4:
        return s
    i = rng.randrange(1, len(s) - 1)
    op = rng.random()
    if op < 0.33:
        return s[:i] + s[i + 1:]
    if op < 0.66:
        return s[:i] + s[i + 1] + s[i] + s[i + 2:]
    return s[:i] + rng.choice("aeioustnr") + s[i + 1:]


def make_entity(country, rng, used):
    words = {"India": WORDS_IN, "US": WORDS_US, "France": WORDS_FR}[country]
    while True:
        name = " ".join(rng.sample(words, rng.choice([2, 2, 3])))
        if name not in used:
            used.add(name)
            break
    legal = rng.choice(LEGAL[country])
    city, pre = rng.choice(CITIES[country])
    num = str(rng.randint(1, 2999))
    street = rng.choice(STREETS[country])
    if country == "India":
        pin = pre + str(rng.randint(10, 99))
        addr = f"{num}, {street}, {city} {pin}"
    elif country == "US":
        zipc = pre + str(rng.randint(10, 99))
        addr = f"{num} {street}, {city}, {zipc}"
    else:
        cp = pre + str(rng.randint(1, 20)).zfill(2)
        addr = f"{num} {street}, {cp} {city}"
    return {"name": (name + " " + legal).strip(), "addr": addr, "country": country, "city": city,
            "street": street, "num": num}


def noisy(ent, rng):
    name, addr = ent["name"], ent["addr"]
    for k, v in ABBR.items():
        if k in name and rng.random() < 0.5:
            name = name.replace(k, v)
        if k in addr and rng.random() < 0.5:
            addr = addr.replace(k, v)
    if rng.random() < 0.25:
        name = typo(name, rng)
    if rng.random() < 0.15:
        toks = name.split()
        if len(toks) > 2:
            toks[0], toks[1] = toks[1], toks[0]
            name = " ".join(toks)
    if rng.random() < 0.2:
        name = name.upper()
    if ent["country"] == "India" and rng.random() < 0.15:
        name = " ".join(DEVANAGARI.get(w, w) for w in name.replace("Pvt Ltd", "Private Limited").split())
    r = rng.random()
    if r < 0.12:
        addr = addr.rsplit(" ", 1)[0]  # drop postal code / city
    elif r < 0.2:
        addr = f"Near SBI ATM, {ent['street']}, {ent['city']}" if ent["country"] == "India" else addr
    elif r < 0.25:
        addr = "None"
    if rng.random() < 0.15:
        addr = typo(addr, rng)
    return name, addr


def generate(out, n_s1, countries, split, rng, start_ids):
    used = set()
    s1, s2, s3, gt = [], [], [], []
    c1, c2, c3 = start_ids
    for _ in range(n_s1):
        country = rng.choice(countries)
        ent = make_entity(country, rng, used)
        c1 += 1
        sid = f"S1-{c1:06d}"
        n1, a1 = noisy(ent, rng) if rng.random() < 0.3 else (ent["name"], ent["addr"])
        s1.append((sid, n1, a1, country))
        matches = []
        if rng.random() > 0.06:
            k = min(11, max(1, int(rng.expovariate(1 / 3.0)) + 1))
            for _ in range(k):
                n, a = noisy(ent, rng)
                if rng.random() < 0.5:
                    c2 += 1
                    tid = f"S2-{c2:07d}"
                    s2.append((tid, n, a, country))
                else:
                    c3 += 1
                    tid = f"S3-{c3:07d}"
                    s3.append((tid, n, a, country))
                matches.append(tid)
        gt.append((sid, ",".join(matches)))
        # hard negative: same brand, different branch
        if rng.random() < 0.15:
            other = dict(ent)
            other["num"] = str(int(ent["num"]) + rng.randint(1, 40))
            other["addr"] = ent["addr"].replace(ent["num"], other["num"], 1)
            n, a = noisy(other, rng)
            c2 += 1
            s2.append((f"S2-{c2:07d}", n, a, country))
    # unrelated noise records in the target sources
    for _ in range(n_s1 // 3):
        country = rng.choice(countries)
        ent = make_entity(country, rng, used)
        n, a = noisy(ent, rng)
        c3 += 1
        s3.append((f"S3-{c3:07d}", n, a, country))
    rng.shuffle(s2)
    rng.shuffle(s3)
    d = os.path.join(out, split)
    os.makedirs(d, exist_ok=True)
    header = "entity_id\tbusiness_name\tbusiness_address\tcountry\n"
    for idx, rows in ((1, s1), (2, s2), (3, s3)):
        with open(os.path.join(d, f"{split}_source{idx}.tsv"), "w", encoding="utf-8") as fh:
            fh.write(header)
            for r in rows:
                fh.write("\t".join(x.replace("\t", " ") for x in r) + "\n")
    return gt, (c1, c2, c3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-train", type=int, default=4000)
    ap.add_argument("--n-test", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    gt, ids = generate(a.out, a.n_train, ["US", "India"], "train", rng, (0, 0, 0))
    with open(os.path.join(a.out, "train", "train_ground_truth.tsv"), "w", encoding="utf-8") as fh:
        fh.write("source1_entity_id\tmatched_entity_ids\n")
        for s, m in gt:
            fh.write(f"{s}\t{m}\n")
    gt_test, _ = generate(a.out, a.n_test, ["US", "India", "France"], "test", rng, ids)
    with open(os.path.join(a.out, "test_ground_truth_hidden.tsv"), "w", encoding="utf-8") as fh:
        fh.write("source1_entity_id\tmatched_entity_ids\n")
        for s, m in gt_test:
            fh.write(f"{s}\t{m}\n")


if __name__ == "__main__":
    main()
