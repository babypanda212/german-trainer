import json
from pathlib import Path
from scripts.load_vocab import load_dwds_csv, load_provenance_json, gloss_from_variants

DWDS_SAMPLE = 'Lemma,URL,Wortart,Genus,Artikel,nur_im_Plural\nHaus,https://x,Substantiv,neutr.,das,nein\nAuto,https://y,Substantiv,neutr.,das,nein\n'

def test_load_dwds_csv(store, tmp_path):
    p = tmp_path / "A1.csv"; p.write_text(DWDS_SAMPLE, encoding="utf-8")
    n = load_dwds_csv(store, p, "A1")
    assert n == 2
    assert store.target_counts() == {"A1": 2}
    assert store.next_targets("A1", 5)[0]["gloss"] in ("das Haus", "das Auto")

def test_load_provenance_json(store, tmp_path):
    data = [
      {"headword": "Abbau", "level": "B2", "variants": [{"level": "B2", "text": "Abbau, der (Sg.)", "source": "x"}]},
      {"headword": "aberrant", "level": "C2", "variants": [{"level": "C2", "text": "aberrant", "translation": "aberrant", "source": "y"}]},
    ]
    p = tmp_path / "prov.json"; p.write_text(json.dumps(data), encoding="utf-8")
    n = load_provenance_json(store, p)
    assert n == 2
    assert store.target_counts() == {"B2": 1, "C2": 1}
    c2 = store.next_targets("C2", 1)[0]
    assert c2["source"] == "radicalrampage" and c2["gloss"] == "aberrant"
    b2 = store.next_targets("B2", 1)[0]
    assert b2["source"] == "aspekte_neu" and b2["gloss"] == "Abbau, der (Sg.)"

def test_gloss_prefers_translation():
    assert gloss_from_variants([{"text": "x", "translation": "y"}]) == "y"
    assert gloss_from_variants([{"text": "x"}]) == "x"
    assert gloss_from_variants([]) is None
