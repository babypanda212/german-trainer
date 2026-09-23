"""Fill target_vocab. A1-B1 from DWDS (official Goethe lists), B2-C2 from docs/vocabulary provenance JSON.
Usage: python -m scripts.load_vocab [--db data/trainer.db]
"""
import argparse, csv, json, sys, urllib.request
from pathlib import Path
from trainer.store import Store

ROOT = Path(__file__).resolve().parents[1]
DWDS_URL = "https://www.dwds.de/api/lemma/goethe/{level}.csv"
PROV_JSON = ROOT / "docs/vocabulary/cleaned/entries-with-provenance.json"
SOURCE_BY_LEVEL = {"B2": "aspekte_neu", "C1": "aspekte_neu", "C2": "radicalrampage"}

def load_dwds_csv(store: Store, path: Path, level: str) -> int:
    n = 0
    with path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            lemma = row["Lemma"].strip()
            if not lemma:
                continue
            art = row.get("Artikel", "").strip()
            gloss = f"{art} {lemma}" if art else lemma
            store.add_target(lemma, level, "goethe_dwds", gloss)
            n += 1
    store.commit()
    return n

def gloss_from_variants(variants: list[dict]) -> str | None:
    if not variants:
        return None
    for v in variants:
        if v.get("translation"):
            return v["translation"]
    return variants[0].get("text")

def load_provenance_json(store: Store, path: Path) -> int:
    entries = json.loads(path.read_text(encoding="utf-8"))
    n = 0
    for e in entries:
        level = e["level"]
        store.add_target(e["headword"], level, SOURCE_BY_LEVEL[level], gloss_from_variants(e["variants"]))
        n += 1
    store.commit()
    return n

def fetch_dwds(level: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        urllib.request.urlretrieve(DWDS_URL.format(level=level), dest)
    return dest

def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=str(ROOT / "data/trainer.db"))
    args = ap.parse_args(argv)
    store = Store(Path(args.db))
    for level in ("A1", "A2", "B1"):
        p = fetch_dwds(level, ROOT / "data/vocab" / f"{level}.csv")
        print(level, load_dwds_csv(store, p, level))
    print("B2-C2", load_provenance_json(store, PROV_JSON))
    print(store.target_counts())

if __name__ == "__main__":
    main()
