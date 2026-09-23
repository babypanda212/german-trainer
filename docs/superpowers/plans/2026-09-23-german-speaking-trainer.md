# German Speaking Trainer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Local web app for spoken German practice: mic in, German tutor voice out, corrections logged, CEFR trend, target vocabulary from Goethe/Aspekte lists, spaced vocab review.

**Architecture:** FastAPI backend. Audio → faster-whisper (local) → Claude via Agent SDK persistent session (subscription login) → Piper TTS (local) → SQLite. One static HTML page for practice, one for progress. Tutor returns structured JSON via `output_format`.

**Tech Stack:** Python 3.12 (conda env `gt`), FastAPI + uvicorn, faster-whisper, piper-tts, claude-agent-sdk, sqlite3 (stdlib), pytest, httpx (for route tests). Frontend: plain HTML/JS, Chart.js from cdnjs.

Spec: `docs/superpowers/specs/2026-09-23-german-speaking-trainer-design.md`

---

## File structure

```
german-trainer/
├── trainer/
│   ├── __init__.py
│   ├── store.py        # SQLite: sessions, turns, mistakes, vocab, target_vocab; recap; SM-2
│   ├── cefr.py         # derive_level, rubric text
│   ├── tutor.py        # Tutor: Agent SDK session wrapper, prompts, JSON schemas
│   ├── stt.py          # transcribe(webm_bytes) -> str
│   ├── tts.py          # speak(text) -> Path
│   └── server.py       # FastAPI app
├── scripts/
│   └── load_vocab.py   # fills target_vocab from DWDS CSVs + provenance JSON
├── static/
│   ├── index.html
│   └── progress.html
├── data/               # trainer.db, audio/, voices/  (gitignored)
├── tests/
│   ├── conftest.py
│   ├── test_store.py
│   ├── test_cefr.py
│   ├── test_load_vocab.py
│   ├── test_tutor.py
│   ├── test_server.py
│   ├── test_stt.py
│   └── test_tts.py
├── pyproject.toml
└── README.md
```

Mistake types (fixed list, used everywhere): `case, gender, verb_position, conjugation, word_order, preposition, article, vocab, other`.

---

### Task 0: Environment and skeleton

**Files:**
- Create: `pyproject.toml`, `.gitignore`, `trainer/__init__.py`, `tests/conftest.py`, `data/.gitkeep`

- [ ] **Step 1: Create conda env (system Python is 3.9, Agent SDK needs 3.10+)**

```bash
conda create -y -n gt python=3.12
conda activate gt
pip install fastapi uvicorn[standard] python-multipart faster-whisper piper-tts claude-agent-sdk pytest httpx
```

Expected: all install without error. `python -c "import claude_agent_sdk, faster_whisper, piper, fastapi"` prints nothing.

- [ ] **Step 2: Verify Claude Code login works headlessly**

```bash
claude -p "Antworte nur mit: ok" --output-format json | head -c 300
```

Expected: JSON with `"result":"ok"`. If it asks to log in, run `claude` once interactively and log in with the subscription.

- [ ] **Step 3: Write pyproject.toml**

```toml
[project]
name = "german-trainer"
version = "0.1.0"
requires-python = ">=3.12"

[tool.pytest.ini_options]
testpaths = ["tests"]
asyncio_mode = "auto"
```

- [ ] **Step 4: Write .gitignore**

```
data/
!data/.gitkeep
__pycache__/
*.pyc
.pytest_cache/
```

- [ ] **Step 5: Write empty `trainer/__init__.py`, `data/.gitkeep`, and `tests/conftest.py`**

```python
# tests/conftest.py
import pytest
from pathlib import Path
from trainer.store import Store

@pytest.fixture
def store(tmp_path: Path) -> Store:
    return Store(tmp_path / "t.db")
```

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml .gitignore trainer/__init__.py data/.gitkeep tests/conftest.py
git commit -m "chore: project skeleton"
```

---

### Task 1: store.py — schema, sessions, turns, mistakes

**Files:**
- Create: `trainer/store.py`
- Test: `tests/test_store.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_store.py
from datetime import datetime, timedelta
from trainer.store import Store, MISTAKE_TYPES

def test_schema_creates_tables(store):
    names = {r[0] for r in store.conn.execute("select name from sqlite_master where type='table'")}
    assert {"sessions", "turns", "mistakes", "vocab", "target_vocab"} <= names

def test_session_lifecycle(store):
    sid = store.start_session()
    assert store.open_session_id() == sid
    tid = store.add_turn(sid, idx=0, user_de="Ich habe gegangen.", tutor_de="Fast! Ich bin gegangen.", audio_path="a.wav", words=3)
    store.add_mistake(tid, "conjugation", "habe gegangen", "bin gegangen", "sein with movement verbs")
    store.end_session(sid, level="B1", scores={"range": 3, "accuracy": 2, "fluency": 3, "coherence": 3}, summary="ok")
    assert store.open_session_id() is None
    s = store.get_session(sid)
    assert s["level_est"] == "B1" and s["accuracy"] == 2 and s["ended_at"] is not None

def test_add_mistake_rejects_unknown_type(store):
    sid = store.start_session()
    tid = store.add_turn(sid, 0, "x", "y", None, 1)
    import pytest
    with pytest.raises(ValueError):
        store.add_mistake(tid, "spelling", "a", "b", "c")

def test_mistake_counts_last_n_sessions(store):
    for i in range(3):
        sid = store.start_session()
        tid = store.add_turn(sid, 0, "x", "y", None, 1)
        store.add_mistake(tid, "case", "a", "b", "c")
        if i == 0:
            store.add_mistake(tid, "gender", "a", "b", "c")
        store.end_session(sid, "A2", {"range":2,"accuracy":2,"fluency":2,"coherence":2}, "")
    counts = store.mistake_counts(last_n_sessions=2)
    assert counts == {"case": 2}

def test_error_rate_per_100_words(store):
    sid = store.start_session()
    tid = store.add_turn(sid, 0, "a b c d", "y", None, 50)
    store.add_mistake(tid, "case", "a", "b", "c")
    assert store.error_rate(sid) == 2.0
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_store.py -v`
Expected: FAIL, `ModuleNotFoundError: trainer.store`

- [ ] **Step 3: Implement**

```python
# trainer/store.py
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path

MISTAKE_TYPES = ["case", "gender", "verb_position", "conjugation", "word_order",
                 "preposition", "article", "vocab", "other"]

SCHEMA = """
create table if not exists sessions (
  id integer primary key, started_at text not null, ended_at text,
  level_est text, range integer, accuracy integer, fluency integer, coherence integer, summary text);
create table if not exists turns (
  id integer primary key, session_id integer not null references sessions(id),
  idx integer not null, user_de text not null, tutor_de text not null, audio_path text, words integer not null);
create table if not exists mistakes (
  id integer primary key, turn_id integer not null references turns(id),
  type text not null, original text, corrected text, explanation text);
create table if not exists vocab (
  id integer primary key, word text unique not null, gloss text, source_turn_id integer,
  due_at text not null, interval_days real not null default 1, ease real not null default 2.5);
create table if not exists target_vocab (
  id integer primary key, word text not null, level text not null, source text not null,
  gloss text, times_used integer not null default 0, introduced_at text,
  unique(word, level));
"""

def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")

class Store:
    def __init__(self, path: Path):
        self.conn = sqlite3.connect(str(path))
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)

    # sessions
    def start_session(self) -> int:
        cur = self.conn.execute("insert into sessions(started_at) values (?)", (_now(),))
        self.conn.commit()
        return cur.lastrowid

    def open_session_id(self) -> int | None:
        r = self.conn.execute("select id from sessions where ended_at is null order by id desc limit 1").fetchone()
        return r["id"] if r else None

    def end_session(self, sid: int, level: str, scores: dict, summary: str) -> None:
        self.conn.execute(
            "update sessions set ended_at=?, level_est=?, range=?, accuracy=?, fluency=?, coherence=?, summary=? where id=?",
            (_now(), level, scores["range"], scores["accuracy"], scores["fluency"], scores["coherence"], summary, sid))
        self.conn.commit()

    def get_session(self, sid: int) -> dict:
        return dict(self.conn.execute("select * from sessions where id=?", (sid,)).fetchone())

    def list_sessions(self) -> list[dict]:
        return [dict(r) for r in self.conn.execute("select * from sessions order by id")]

    # turns and mistakes
    def add_turn(self, sid: int, idx: int, user_de: str, tutor_de: str, audio_path: str | None, words: int) -> int:
        cur = self.conn.execute(
            "insert into turns(session_id, idx, user_de, tutor_de, audio_path, words) values (?,?,?,?,?,?)",
            (sid, idx, user_de, tutor_de, audio_path, words))
        self.conn.commit()
        return cur.lastrowid

    def turns(self, sid: int) -> list[dict]:
        return [dict(r) for r in self.conn.execute("select * from turns where session_id=? order by idx", (sid,))]

    def add_mistake(self, turn_id: int, type_: str, original: str, corrected: str, explanation: str) -> int:
        if type_ not in MISTAKE_TYPES:
            raise ValueError(f"unknown mistake type {type_!r}")
        cur = self.conn.execute(
            "insert into mistakes(turn_id, type, original, corrected, explanation) values (?,?,?,?,?)",
            (turn_id, type_, original, corrected, explanation))
        self.conn.commit()
        return cur.lastrowid

    def mistakes(self, sid: int) -> list[dict]:
        return [dict(r) for r in self.conn.execute(
            "select m.* from mistakes m join turns t on t.id=m.turn_id where t.session_id=? order by m.id", (sid,))]

    def mistake_counts(self, last_n_sessions: int = 10) -> dict[str, int]:
        rows = self.conn.execute("""
            select m.type, count(*) n from mistakes m
            join turns t on t.id=m.turn_id
            where t.session_id in (select id from sessions where ended_at is not null order by id desc limit ?)
            group by m.type order by n desc""", (last_n_sessions,)).fetchall()
        return {r["type"]: r["n"] for r in rows}

    def error_rate(self, sid: int) -> float:
        words = self.conn.execute("select coalesce(sum(words),0) w from turns where session_id=?", (sid,)).fetchone()["w"]
        errs = len(self.mistakes(sid))
        return round(100.0 * errs / words, 2) if words else 0.0
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_store.py -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add trainer/store.py tests/test_store.py
git commit -m "feat(store): schema, sessions, turns, mistakes"
```

---

### Task 2: store.py — vocab (SM-2), target_vocab, recap

**Files:**
- Modify: `trainer/store.py`
- Test: `tests/test_store.py`

- [ ] **Step 1: Write failing tests (append)**

```python
# tests/test_store.py (append)
from datetime import datetime, timedelta

def test_vocab_sm2_schedule(store):
    store.add_vocab("Abbau", "dismantling", source_turn_id=None)
    v = store.get_vocab("Abbau")
    assert v["interval_days"] == 1
    store.review_vocab("Abbau", quality=4)
    v = store.get_vocab("Abbau")
    assert v["interval_days"] == 6           # second review interval
    assert v["ease"] == 2.5
    store.review_vocab("Abbau", quality=2)   # fail → reset
    v = store.get_vocab("Abbau")
    assert v["interval_days"] == 1
    assert v["ease"] < 2.5

def test_vocab_due(store):
    store.add_vocab("jetzt", "now", None)
    store.add_vocab("später", "later", None)
    store.conn.execute("update vocab set due_at=? where word='später'",
                       ((datetime.now() + timedelta(days=3)).isoformat(),))
    store.conn.commit()
    assert [v["word"] for v in store.due_vocab()] == ["jetzt"]

def test_target_vocab_next_and_mark(store):
    store.add_target("Haus", "A1", "goethe_dwds", "house")
    store.add_target("Auto", "A1", "goethe_dwds", "car")
    store.add_target("Abbau", "B2", "aspekte_neu", None)
    store.mark_target_used("Haus", "A1")
    nxt = store.next_targets("A1", n=1)
    assert nxt[0]["word"] == "Auto"           # least used first
    assert store.next_targets("C2") == []

def test_recap(store):
    sid = store.start_session()
    tid = store.add_turn(sid, 0, "x", "y", None, 10)
    store.add_mistake(tid, "gender", "a", "b", "c")
    store.end_session(sid, "B1", {"range":3,"accuracy":3,"fluency":3,"coherence":3}, "")
    store.add_vocab("Abbau", "dismantling", tid)
    r = store.recap()
    assert r["level"] == "B1"
    assert r["top_mistakes"] == ["gender"]
    assert r["due_vocab"][0]["word"] == "Abbau"

def test_recap_default_level_when_no_sessions(store):
    assert store.recap()["level"] == "A2"
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_store.py -v`
Expected: 5 new tests FAIL with AttributeError

- [ ] **Step 3: Implement (append to Store class)**

```python
    # vocab (SM-2)
    def add_vocab(self, word: str, gloss: str | None, source_turn_id: int | None) -> None:
        self.conn.execute(
            "insert or ignore into vocab(word, gloss, source_turn_id, due_at) values (?,?,?,?)",
            (word, gloss, source_turn_id, _now()))
        self.conn.commit()

    def get_vocab(self, word: str) -> dict | None:
        r = self.conn.execute("select * from vocab where word=?", (word,)).fetchone()
        return dict(r) if r else None

    def review_vocab(self, word: str, quality: int) -> None:
        """SM-2. quality 0-5; <3 resets interval."""
        v = self.get_vocab(word)
        if v is None:
            return
        ease = max(1.3, v["ease"] + 0.1 - (5 - quality) * (0.08 + (5 - quality) * 0.02))
        if quality < 3:
            interval = 1.0
        elif v["interval_days"] <= 1:
            interval = 6.0
        else:
            interval = round(v["interval_days"] * ease, 1)
        due = (datetime.now() + timedelta(days=interval)).isoformat(timespec="seconds")
        self.conn.execute("update vocab set interval_days=?, ease=?, due_at=? where word=?",
                          (interval, ease, due, word))
        self.conn.commit()

    def due_vocab(self, limit: int = 5) -> list[dict]:
        return [dict(r) for r in self.conn.execute(
            "select * from vocab where due_at <= ? order by due_at limit ?", (_now(), limit))]

    def all_vocab(self) -> list[dict]:
        return [dict(r) for r in self.conn.execute("select * from vocab order by word")]

    # target vocab
    def add_target(self, word: str, level: str, source: str, gloss: str | None) -> None:
        self.conn.execute(
            "insert or ignore into target_vocab(word, level, source, gloss) values (?,?,?,?)",
            (word, level, source, gloss))

    def commit(self) -> None:
        self.conn.commit()

    def next_targets(self, level: str, n: int = 10) -> list[dict]:
        return [dict(r) for r in self.conn.execute(
            "select * from target_vocab where level=? order by times_used, random() limit ?", (level, n))]

    def mark_target_used(self, word: str, level: str) -> None:
        self.conn.execute(
            "update target_vocab set times_used=times_used+1, introduced_at=coalesce(introduced_at, ?) where word=? and level=?",
            (_now(), word, level))
        self.conn.commit()

    def target_counts(self) -> dict[str, int]:
        return {r["level"]: r["n"] for r in self.conn.execute(
            "select level, count(*) n from target_vocab group by level")}

    # recap for session start
    def recap(self) -> dict:
        last = self.conn.execute(
            "select level_est from sessions where ended_at is not null order by id desc limit 1").fetchone()
        level = last["level_est"] if last and last["level_est"] else "A2"
        top = list(self.mistake_counts(10).keys())[:3]
        return {"level": level, "top_mistakes": top, "due_vocab": self.due_vocab(5)}
```

Note `add_target` does not commit per row (bulk load); `commit()` is called once by the loader. `next_targets` and `mark_target_used` still commit.

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_store.py -v`
Expected: 10 passed

- [ ] **Step 5: Commit**

```bash
git add trainer/store.py tests/test_store.py
git commit -m "feat(store): SM-2 vocab, target vocab, recap"
```

---

### Task 3: cefr.py

**Files:**
- Create: `trainer/cefr.py`
- Test: `tests/test_cefr.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_cefr.py
import pytest
from trainer.cefr import derive_level, RUBRIC, LEVELS

def test_levels_order():
    assert LEVELS == ["A1", "A2", "B1", "B2", "C1", "C2"]

@pytest.mark.parametrize("scores,expected", [
    ((1, 1, 1, 1), "A1"),
    ((4, 4, 4, 4), "B2"),
    ((6, 6, 6, 6), "C2"),
    ((4, 4, 4, 3), "B2"),   # mean 3.75 → 4
    ((5, 5, 5, 2), "B1"),   # mean 4.25 → 4, but capped at min+1 = 3
])
def test_derive_level(scores, expected):
    assert derive_level(*scores) == expected

def test_rejects_out_of_range():
    with pytest.raises(ValueError):
        derive_level(0, 3, 3, 3)
    with pytest.raises(ValueError):
        derive_level(7, 3, 3, 3)

def test_rubric_mentions_goethe_traits():
    assert "Durchhaltevermögen" in RUBRIC and "Konnotation" in RUBRIC
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_cefr.py -v`
Expected: FAIL, ModuleNotFoundError

- [ ] **Step 3: Implement**

```python
# trainer/cefr.py
LEVELS = ["A1", "A2", "B1", "B2", "C1", "C2"]

RUBRIC = """Score the learner's German speaking on four criteria, each 1-6 (1=A1 ... 6=C2).
Score only the learner's own turns, never the tutor's. Use these anchors, taken from the
Goethe-Institut exam descriptions:

range (Wortschatzspektrum):
  3 (B1): minimal means used maximally, everyday topics only.
  4 (B2): large vocabulary in own field and most general topics; gaps cause hesitation and paraphrase.
  5 (C1): broad repertoire; gaps bridged skilfully by paraphrase; idioms and colloquial phrases controlled.
  6 (C2): very rich including idioms; aware of Konnotationen; consistently apt word choice.
accuracy (Korrektheit):
  3 (B1): frequent errors but meaning clear.
  4 (B2): good control; avoids gross errors (krasse Formulierungsfehler); occasional slips, often self-corrected.
  5 (C1): consistently high accuracy; errors rare and hardly noticed.
  6 (C2): correct even in complex language.
fluency (Flüssigkeit):
  3 (B1): short, structurally reduced sentences; needs repetition.
  4 (B2): spontaneous, structured, no longer only short sentences; Durchhaltevermögen im Diskurs (holds a line of argument over several turns).
  5 (C1): beinahe mühelos; longer utterances in less time; adapts to social context; can make allusions or jokes.
  6 (C2): very fluent and precise; complex content presented coherently.
coherence (Kohärenz/Register):
  3 (B1): simple linking (und, aber, weil).
  4 (B2): effective argumentation; pros and cons systematically; Sprachbewusstsein.
  5 (C1): control of text patterns and connectors; differentiated use of formal vs informal register.
  6 (C2): stylistic differentiation; removes ambiguity; emphasises and nuances deliberately.
"""

def derive_level(range_: int, accuracy: int, fluency: int, coherence: int) -> str:
    scores = [range_, accuracy, fluency, coherence]
    if any(not (1 <= s <= 6) for s in scores):
        raise ValueError(f"scores must be 1-6, got {scores}")
    mean = round(sum(scores) / 4)
    capped = min(mean, min(scores) + 1)
    return LEVELS[capped - 1]
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_cefr.py -v`
Expected: 8 passed

- [ ] **Step 5: Commit**

```bash
git add trainer/cefr.py tests/test_cefr.py
git commit -m "feat(cefr): level derivation and Goethe-based rubric"
```

---

### Task 4: scripts/load_vocab.py

**Files:**
- Create: `scripts/load_vocab.py`
- Test: `tests/test_load_vocab.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_load_vocab.py
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
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_load_vocab.py -v`
Expected: FAIL, ModuleNotFoundError

- [ ] **Step 3: Implement**

```python
# scripts/load_vocab.py
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
```

Also create empty `scripts/__init__.py` so `scripts.load_vocab` imports in tests.

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_load_vocab.py -v`
Expected: 3 passed

- [ ] **Step 5: Run the real load**

```bash
mkdir -p data && python -m scripts.load_vocab
```

Expected: prints counts, roughly `{'A1': ~650, 'A2': ~1300, 'B1': ~2400, 'B2': 2387, 'C1': 2599, 'C2': 402}`. If DWDS download fails (403), download the three CSVs in a browser from `https://www.dwds.de/api/lemma/goethe/A1.csv` etc. into `data/vocab/` and rerun.

- [ ] **Step 6: Commit**

```bash
git add scripts/__init__.py scripts/load_vocab.py tests/test_load_vocab.py
git commit -m "feat: load target vocabulary from DWDS and Aspekte/RadicalRampage lists"
```

---

### Task 5: tutor.py

**Files:**
- Create: `trainer/tutor.py`
- Test: `tests/test_tutor.py`

Design: `Tutor` takes a `session_factory` so tests inject a fake. The real factory builds a `ClaudeSDKClient`. Two schemas: `TURN_SCHEMA` (per turn) and `SUMMARY_SCHEMA` (session end). `output_format` is fixed per client, so the tutor uses one client for turns and a second, single-shot `query()` for the summary.

- [ ] **Step 1: Write failing tests**

```python
# tests/test_tutor.py
import pytest
from trainer.tutor import Tutor, build_system_prompt, TURN_SCHEMA, SUMMARY_SCHEMA

class FakeSession:
    def __init__(self, replies):
        self.replies = list(replies); self.prompts = []; self.closed = False
    async def turn(self, prompt: str) -> dict | None:
        self.prompts.append(prompt)
        return self.replies.pop(0)
    async def close(self):
        self.closed = True

RECAP = {"level": "B1", "top_mistakes": ["gender", "case"], "due_vocab": [{"word": "Abbau", "gloss": "dismantling"}]}
TARGETS = [{"word": "Haus", "gloss": "das Haus"}, {"word": "Auto", "gloss": "das Auto"}]

def test_system_prompt_contains_recap_targets_and_types():
    p = build_system_prompt(RECAP, TARGETS)
    assert "B1" in p and "gender" in p and "Abbau" in p and "Haus" in p
    assert "verb_position" in p          # fixed mistake list
    assert "Durchhaltevermögen" in p     # rubric included

def test_system_prompt_c1_adds_word_formation():
    p = build_system_prompt({**RECAP, "level": "C1"}, [])
    assert "Wortbildung" in p

@pytest.mark.asyncio
async def test_turn_parses_reply():
    reply = {"reply_de": "Gut! Wo wohnst du?", "spoken_correction": None,
             "corrections": [{"type": "gender", "original": "der Haus", "corrected": "das Haus", "explanation": "Haus is neuter"}],
             "targets_used": ["Haus"]}
    fake = FakeSession([reply])
    t = Tutor(session_factory=lambda sp, schema: fake)
    await t.start(RECAP, TARGETS)
    r = await t.turn("Ich wohne in der Haus.")
    assert r.reply_de == "Gut! Wo wohnst du?"
    assert r.corrections[0]["type"] == "gender"
    assert r.targets_used == ["Haus"]
    assert fake.prompts[-1] == "Ich wohne in der Haus."

@pytest.mark.asyncio
async def test_turn_retries_once_then_degrades():
    good = {"reply_de": "Ok.", "spoken_correction": None, "corrections": [], "targets_used": []}
    fake = FakeSession([None, good])
    t = Tutor(session_factory=lambda sp, schema: fake)
    await t.start(RECAP, TARGETS)
    r = await t.turn("Hallo")
    assert r.reply_de == "Ok." and len(fake.prompts) == 2

    fake2 = FakeSession([None, None])
    t2 = Tutor(session_factory=lambda sp, schema: fake2)
    await t2.start(RECAP, TARGETS)
    r2 = await t2.turn("Hallo")
    assert r2.degraded is True and r2.corrections == []

@pytest.mark.asyncio
async def test_close_returns_summary_and_closes_session():
    summary = {"range": 3, "accuracy": 3, "fluency": 4, "coherence": 3, "summary": "fine",
               "vocab": [{"word": "Abbau", "gloss": "dismantling"}]}
    fake = FakeSession([])
    summ_fake = FakeSession([summary])
    made = []
    def factory(sp, schema):
        made.append(schema); return fake if schema is TURN_SCHEMA else summ_fake
    t = Tutor(session_factory=factory)
    await t.start(RECAP, TARGETS)
    s = await t.close(user_turns=["Ich bin gegangen."], error_rate=2.0)
    assert s["level"] == "B1" and s["fluency"] == 4
    assert fake.closed and summ_fake.closed
    assert SUMMARY_SCHEMA in made
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_tutor.py -v`
Expected: FAIL, ModuleNotFoundError

- [ ] **Step 3: Implement**

```python
# trainer/tutor.py
from dataclasses import dataclass, field
from typing import Callable, Awaitable, Protocol
from trainer.cefr import RUBRIC, derive_level
from trainer.store import MISTAKE_TYPES

TURN_SCHEMA = {
    "type": "object",
    "properties": {
        "reply_de": {"type": "string"},
        "spoken_correction": {"type": ["string", "null"]},
        "corrections": {"type": "array", "items": {
            "type": "object",
            "properties": {
                "type": {"type": "string", "enum": MISTAKE_TYPES},
                "original": {"type": "string"},
                "corrected": {"type": "string"},
                "explanation": {"type": "string"}},
            "required": ["type", "original", "corrected", "explanation"]}},
        "targets_used": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["reply_de", "spoken_correction", "corrections", "targets_used"],
}

SUMMARY_SCHEMA = {
    "type": "object",
    "properties": {
        "range": {"type": "integer", "minimum": 1, "maximum": 6},
        "accuracy": {"type": "integer", "minimum": 1, "maximum": 6},
        "fluency": {"type": "integer", "minimum": 1, "maximum": 6},
        "coherence": {"type": "integer", "minimum": 1, "maximum": 6},
        "summary": {"type": "string"},
        "vocab": {"type": "array", "items": {
            "type": "object",
            "properties": {"word": {"type": "string"}, "gloss": {"type": "string"}},
            "required": ["word", "gloss"]}},
    },
    "required": ["range", "accuracy", "fluency", "coherence", "summary", "vocab"],
}

def build_system_prompt(recap: dict, targets: list[dict]) -> str:
    level = recap["level"]
    targets_txt = ", ".join(f"{t['word']} ({t['gloss']})" if t.get("gloss") else t["word"] for t in targets) or "none"
    due_txt = ", ".join(f"{v['word']} ({v.get('gloss') or ''})" for v in recap["due_vocab"]) or "none"
    top = ", ".join(recap["top_mistakes"]) or "none yet"
    extra = ""
    if level in ("C1", "C2"):
        extra = ("\nAt this level, also point out Wortbildung: when a useful compound, prefix verb or derivation "
                 "comes up, say briefly how it is built from known parts.\n")
    return f"""You are a German conversation tutor. The learner's current estimated level is {level}.
Speak German only in reply_de, pitched at {level}. Keep replies to 1-3 sentences and end with a question
or prompt that keeps the conversation going. Never switch to English in reply_de.

Learner's recurring mistake types (target these gently): {top}.
Vocabulary due for review, work these into your questions naturally: {due_txt}.
Target vocabulary for this session, work each into the conversation at least once and list the ones you used
in targets_used (exact word as given): {targets_txt}.
{extra}
Corrections: log every error in the learner's turn in `corrections` with type from
{MISTAKE_TYPES}, the original fragment, the corrected fragment, and a one-sentence English explanation.
In `spoken_correction` put at most ONE short German correction to say aloud (the most important one),
or null if the turn was fine. Do not put corrections inside reply_de.

{RUBRIC}
"""

@dataclass
class TurnResult:
    reply_de: str
    spoken_correction: str | None
    corrections: list[dict]
    targets_used: list[str]
    degraded: bool = False

class Session(Protocol):
    async def turn(self, prompt: str) -> dict | None: ...
    async def close(self) -> None: ...

SessionFactory = Callable[[str, dict], Session]

class Tutor:
    def __init__(self, session_factory: SessionFactory):
        self._factory = session_factory
        self._session: Session | None = None
        self._recap: dict = {}
        self._targets: list[dict] = []

    async def start(self, recap: dict, targets: list[dict]) -> None:
        self._recap, self._targets = recap, targets
        self._session = self._factory(build_system_prompt(recap, targets), TURN_SCHEMA)

    async def turn(self, user_de: str) -> TurnResult:
        assert self._session is not None, "call start() first"
        data = await self._session.turn(user_de)
        if data is None:
            data = await self._session.turn(user_de)
        if data is None:
            return TurnResult("Entschuldigung, kannst du das noch einmal sagen?", None, [], [], degraded=True)
        return TurnResult(data["reply_de"], data.get("spoken_correction"), data.get("corrections", []),
                          data.get("targets_used", []))

    async def close(self, user_turns: list[str], error_rate: float) -> dict:
        if self._session is not None:
            await self._session.close()
            self._session = None
        prompt = (f"{RUBRIC}\nThe learner's turns this session (tutor turns omitted):\n"
                  + "\n".join(f"- {t}" for t in user_turns)
                  + f"\n\nLogged error rate: {error_rate} errors per 100 words.\n"
                  "Score the four criteria 1-6, write a 3-sentence English summary of what to work on, "
                  "and list up to 8 useful German words the learner used or was corrected on, with a short English gloss.")
        summ = self._factory("You are a strict CEFR rater for German speaking.", SUMMARY_SCHEMA)
        data = await summ.turn(prompt)
        await summ.close()
        if data is None:
            data = {"range": 3, "accuracy": 3, "fluency": 3, "coherence": 3,
                    "summary": "Rating failed; scores are placeholders.", "vocab": []}
        data["level"] = derive_level(data["range"], data["accuracy"], data["fluency"], data["coherence"])
        return data


# --- real session backed by the Agent SDK -------------------------------------------------

class ClaudeSession:
    """One persistent Claude Code session (uses the local `claude` login, i.e. the subscription)."""
    def __init__(self, system_prompt: str, schema: dict, model: str | None = None):
        from claude_agent_sdk import ClaudeSDKClient, ClaudeAgentOptions
        self._client = ClaudeSDKClient(options=ClaudeAgentOptions(
            system_prompt=system_prompt,
            allowed_tools=[],
            model=model,
            permission_mode="bypassPermissions",
            output_format={"type": "json_schema", "schema": schema},
        ))
        self._connected = False

    async def turn(self, prompt: str) -> dict | None:
        from claude_agent_sdk import ResultMessage
        if not self._connected:
            await self._client.connect(); self._connected = True
        await self._client.query(prompt)
        out = None
        async for m in self._client.receive_response():
            if isinstance(m, ResultMessage) and m.subtype == "success" and m.structured_output:
                out = m.structured_output
        return out

    async def close(self) -> None:
        if self._connected:
            await self._client.disconnect(); self._connected = False

def claude_session_factory(system_prompt: str, schema: dict) -> Session:
    return ClaudeSession(system_prompt, schema)
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_tutor.py -v`
Expected: 5 passed

- [ ] **Step 5: Manual smoke against the real session (not a test)**

```bash
python - <<'EOF'
import asyncio
from trainer.tutor import Tutor, claude_session_factory
async def main():
    t = Tutor(claude_session_factory)
    await t.start({"level":"B1","top_mistakes":[],"due_vocab":[]}, [{"word":"Bahnhof","gloss":"der Bahnhof"}])
    print(await t.turn("Hallo! Ich habe gestern in die Stadt gegangen."))
    print(await t.close(["Ich habe gestern in die Stadt gegangen."], 10.0))
asyncio.run(main())
EOF
```

Expected: a TurnResult with German reply and a `conjugation` or `case` correction, then a summary dict with level. If it errors on `permission_mode`, drop that argument.

- [ ] **Step 6: Commit**

```bash
git add trainer/tutor.py tests/test_tutor.py
git commit -m "feat(tutor): Agent SDK session wrapper with structured turns and CEFR summary"
```

---

### Task 6: stt.py

**Files:**
- Create: `trainer/stt.py`
- Test: `tests/test_stt.py`

- [ ] **Step 1: Write test (smoke, skipped without model)**

```python
# tests/test_stt.py
import pytest, subprocess, shutil
from trainer.stt import transcribe, MIN_WORDS

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg missing")

def test_silence_returns_empty(tmp_path):
    wav = tmp_path / "s.wav"
    subprocess.run(["ffmpeg", "-y", "-f", "lavfi", "-i", "anullsrc=r=16000:cl=mono", "-t", "1", str(wav)],
                   check=True, capture_output=True)
    assert transcribe(wav.read_bytes()) == ""

def test_min_words_constant():
    assert MIN_WORDS == 2
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_stt.py -v`
Expected: FAIL, ModuleNotFoundError

- [ ] **Step 3: Implement**

```python
# trainer/stt.py
import subprocess, tempfile
from pathlib import Path
from functools import lru_cache

MIN_WORDS = 2
MODEL_SIZE = "small"   # bump to "medium" if German accuracy is poor

@lru_cache(maxsize=1)
def _model():
    from faster_whisper import WhisperModel
    return WhisperModel(MODEL_SIZE, device="cpu", compute_type="int8")

def _to_wav16k(audio_bytes: bytes) -> Path:
    """Browser sends WebM/Opus (or anything ffmpeg reads). Normalise to 16 kHz mono WAV."""
    src = tempfile.NamedTemporaryFile(suffix=".in", delete=False); src.write(audio_bytes); src.close()
    dst = Path(src.name).with_suffix(".wav")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", src.name, "-ar", "16000", "-ac", "1", str(dst)],
                   check=True)
    Path(src.name).unlink(missing_ok=True)
    return dst

def transcribe(audio_bytes: bytes) -> str:
    wav = _to_wav16k(audio_bytes)
    try:
        segments, _ = _model().transcribe(str(wav), language="de", vad_filter=True, beam_size=5)
        text = " ".join(s.text.strip() for s in segments).strip()
    finally:
        wav.unlink(missing_ok=True)
    return text if len(text.split()) >= MIN_WORDS else ""
```

- [ ] **Step 4: Run tests** (first run downloads the `small` model, ~500 MB)

Run: `pytest tests/test_stt.py -v`
Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
git add trainer/stt.py tests/test_stt.py
git commit -m "feat(stt): faster-whisper German transcription with ffmpeg normalisation"
```

---

### Task 7: tts.py

**Files:**
- Create: `trainer/tts.py`
- Test: `tests/test_tts.py`

- [ ] **Step 1: Download the voice**

```bash
mkdir -p data/voices && cd data/voices && python -m piper.download_voices de_DE-thorsten-medium && cd ../..
ls data/voices
```

Expected: `de_DE-thorsten-medium.onnx` and `de_DE-thorsten-medium.onnx.json`.

- [ ] **Step 2: Write test**

```python
# tests/test_tts.py
import pytest, wave
from pathlib import Path
from trainer.tts import speak, VOICE_PATH

pytestmark = pytest.mark.skipif(not VOICE_PATH.exists(), reason="piper voice not downloaded")

def test_speak_writes_wav(tmp_path):
    out = speak("Guten Tag, wie geht es dir?", out_dir=tmp_path)
    assert out.exists() and out.suffix == ".wav"
    with wave.open(str(out)) as w:
        assert w.getnframes() > 1000
```

- [ ] **Step 3: Run to verify failure**

Run: `pytest tests/test_tts.py -v`
Expected: FAIL, ModuleNotFoundError

- [ ] **Step 4: Implement**

```python
# trainer/tts.py
import uuid, wave
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VOICE_PATH = ROOT / "data/voices/de_DE-thorsten-medium.onnx"
AUDIO_DIR = ROOT / "data/audio"

@lru_cache(maxsize=1)
def _voice():
    from piper import PiperVoice
    return PiperVoice.load(str(VOICE_PATH))

def speak(text_de: str, out_dir: Path = AUDIO_DIR) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{uuid.uuid4().hex}.wav"
    with wave.open(str(out), "wb") as wf:
        _voice().synthesize_wav(text_de, wf)
    return out
```

- [ ] **Step 5: Run tests**

Run: `pytest tests/test_tts.py -v`
Expected: 1 passed. Play it: `afplay <path>` to confirm it sounds like German.

- [ ] **Step 6: Commit**

```bash
git add trainer/tts.py tests/test_tts.py
git commit -m "feat(tts): Piper German voice"
```

---

### Task 8: server.py

**Files:**
- Create: `trainer/server.py`
- Test: `tests/test_server.py`

Pipeline per turn: transcribe → tutor.turn → tts → store. Store writes: turn row, mistakes, `mark_target_used` for `targets_used`, `review_vocab` for due words (quality 4 if used without a correction on it, 2 if corrected). Dependencies (stt, tts, session factory) are module-level names so tests can monkeypatch them.

- [ ] **Step 1: Write failing tests**

```python
# tests/test_server.py
import pytest
from pathlib import Path
from httpx import AsyncClient, ASGITransport
from trainer import server
from trainer.store import Store

class FakeSession:
    def __init__(self, sp, schema): self.sp = sp; self.schema = schema
    async def turn(self, prompt):
        if "CEFR rater" in self.sp:
            return {"range": 3, "accuracy": 3, "fluency": 3, "coherence": 3, "summary": "s",
                    "vocab": [{"word": "Bahnhof", "gloss": "station"}]}
        return {"reply_de": "Schön! Und dann?", "spoken_correction": "Ich bin gegangen.",
                "corrections": [{"type": "conjugation", "original": "habe gegangen", "corrected": "bin gegangen", "explanation": "e"}],
                "targets_used": ["Bahnhof"]}
    async def close(self): pass

@pytest.fixture
def app(tmp_path, monkeypatch):
    store = Store(tmp_path / "t.db")
    store.add_target("Bahnhof", "A2", "goethe_dwds", "der Bahnhof"); store.commit()
    monkeypatch.setattr(server, "store", store)
    monkeypatch.setattr(server, "session_factory", FakeSession)
    monkeypatch.setattr(server, "transcribe", lambda b: "Ich habe zum Bahnhof gegangen.")
    def fake_speak(text, out_dir=None):
        p = tmp_path / "x.wav"; p.write_bytes(b"RIFF"); return p
    monkeypatch.setattr(server, "speak", fake_speak)
    server.tutor = None
    return server.app

@pytest.fixture
async def client(app):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        yield c

async def test_full_session_flow(client):
    r = await client.post("/session/start"); assert r.status_code == 200
    sid = r.json()["session_id"]
    r = await client.post("/turn", files={"audio": ("a.webm", b"xx", "audio/webm")})
    body = r.json()
    assert body["transcript"] == "Ich habe zum Bahnhof gegangen."
    assert body["reply_de"] == "Schön! Und dann?"
    assert body["corrections"][0]["type"] == "conjugation"
    assert body["audio_url"].startswith("/audio/")
    r = await client.post("/session/end"); s = r.json()
    assert s["level"] == "B1" and s["session_id"] == sid
    st = server.store
    assert len(st.turns(sid)) == 1 and len(st.mistakes(sid)) == 1
    assert st.next_targets("A2", 1)[0]["times_used"] == 1
    assert st.get_vocab("Bahnhof") is not None

async def test_turn_text_bypasses_stt(client, monkeypatch):
    monkeypatch.setattr(server, "transcribe", lambda b: (_ for _ in ()).throw(AssertionError("stt called")))
    await client.post("/session/start")
    r = await client.post("/turn/text", json={"text": "Hallo, wie geht es?"})
    assert r.status_code == 200 and r.json()["transcript"] == "Hallo, wie geht es?"

async def test_turn_without_session_is_400(client):
    r = await client.post("/turn/text", json={"text": "x"})
    assert r.status_code == 400

async def test_empty_transcript_returns_needs_retry(client, monkeypatch):
    monkeypatch.setattr(server, "transcribe", lambda b: "")
    await client.post("/session/start")
    r = await client.post("/turn", files={"audio": ("a.webm", b"xx", "audio/webm")})
    assert r.status_code == 200 and r.json() == {"transcript": "", "needs_retry": True}

async def test_start_closes_stale_open_session(client):
    await client.post("/session/start")
    r = await client.post("/session/start")
    assert r.json()["closed_previous"] is True

async def test_progress(client):
    await client.post("/session/start")
    await client.post("/turn/text", json={"text": "Ich habe gegangen."})
    await client.post("/session/end")
    r = await client.get("/progress"); p = r.json()
    assert p["sessions"][0]["level_est"] == "B1"
    assert p["mistakes_by_session"][0]["conjugation"] == 1
    assert "vocab_due" in p and "target_counts" in p
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_server.py -v`
Expected: FAIL, ModuleNotFoundError. (If `asyncio_mode` errors, `pip install pytest-asyncio`.)

- [ ] **Step 3: Implement**

```python
# trainer/server.py
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from trainer.store import Store
from trainer.tutor import Tutor, claude_session_factory
from trainer.stt import transcribe
from trainer.tts import speak, AUDIO_DIR

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "data/trainer.db"
TARGETS_PER_SESSION = 10

app = FastAPI(title="german-trainer")
store: Store = Store(DB_PATH)
session_factory = claude_session_factory
tutor: Tutor | None = None
_sid: int | None = None
_idx = 0
_level = "A2"
_due_words: set[str] = set()

class TextTurn(BaseModel):
    text: str

@app.post("/session/start")
async def session_start():
    global tutor, _sid, _idx, _level, _due_words
    closed_previous = False
    stale = store.open_session_id()
    if stale is not None:
        await _end(stale); closed_previous = True
    recap = store.recap()
    targets = store.next_targets(recap["level"], TARGETS_PER_SESSION)
    tutor = Tutor(session_factory)
    await tutor.start(recap, targets)
    _sid = store.start_session(); _idx = 0; _level = recap["level"]
    _due_words = {v["word"] for v in recap["due_vocab"]}
    return {"session_id": _sid, "level": recap["level"], "targets": [t["word"] for t in targets],
            "closed_previous": closed_previous}

async def _process(user_de: str, audio_path: str | None):
    global _idx
    r = await tutor.turn(user_de)
    tid = store.add_turn(_sid, _idx, user_de, r.reply_de, audio_path, len(user_de.split()))
    _idx += 1
    corrected_words = set()
    for c in r.corrections:
        store.add_mistake(tid, c["type"], c["original"], c["corrected"], c["explanation"])
        corrected_words.update(c["original"].split())
    for w in r.targets_used:
        store.mark_target_used(w, _level)
        store.add_vocab(w, None, tid)
    for w in _due_words:
        if w in user_de:
            store.review_vocab(w, 2 if w in corrected_words else 4)
    audio_url = None
    try:
        p = speak(r.reply_de); audio_url = f"/audio/{p.name}"
    except Exception:
        pass
    return {"transcript": user_de, "reply_de": r.reply_de, "spoken_correction": r.spoken_correction,
            "corrections": r.corrections, "audio_url": audio_url, "degraded": r.degraded}

@app.post("/turn")
async def turn(audio: UploadFile = File(...)):
    if tutor is None or _sid is None:
        raise HTTPException(400, "no open session")
    text = transcribe(await audio.read())
    if not text:
        return {"transcript": "", "needs_retry": True}
    return await _process(text, None)

@app.post("/turn/text")
async def turn_text(body: TextTurn):
    if tutor is None or _sid is None:
        raise HTTPException(400, "no open session")
    return await _process(body.text.strip(), None)

async def _end(sid: int) -> dict:
    global tutor, _sid
    turns = store.turns(sid)
    if tutor is None:
        tutor = Tutor(session_factory)
    summ = await tutor.close([t["user_de"] for t in turns], store.error_rate(sid))
    store.end_session(sid, summ["level"], summ, summ["summary"])
    for v in summ.get("vocab", []):
        store.add_vocab(v["word"], v["gloss"], None)
    tutor = None; _sid = None
    return {**summ, "session_id": sid}

@app.post("/session/end")
async def session_end():
    if _sid is None:
        raise HTTPException(400, "no open session")
    return await _end(_sid)

@app.get("/progress")
def progress():
    sessions = store.list_sessions()
    by_session = []
    for s in sessions:
        counts: dict[str, int] = {}
        for m in store.mistakes(s["id"]):
            counts[m["type"]] = counts.get(m["type"], 0) + 1
        by_session.append({"session_id": s["id"], **counts})
    return {"sessions": sessions, "mistakes_by_session": by_session,
            "vocab_due": len(store.due_vocab(1000)), "vocab_total": len(store.all_vocab()),
            "target_counts": store.target_counts()}

@app.get("/session/{sid}")
def session_detail(sid: int):
    return {"session": store.get_session(sid), "turns": store.turns(sid), "mistakes": store.mistakes(sid)}

@app.get("/audio/{name}")
def audio(name: str):
    p = AUDIO_DIR / name
    if not p.exists():
        raise HTTPException(404)
    return FileResponse(p, media_type="audio/wav")

app.mount("/", StaticFiles(directory=str(ROOT / "static"), html=True), name="static")
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_server.py -v`
Expected: 6 passed. `static/` must exist for the mount: `mkdir -p static && touch static/.gitkeep` if Task 9 is not done yet.

- [ ] **Step 5: Commit**

```bash
git add trainer/server.py tests/test_server.py static/.gitkeep
git commit -m "feat(server): session, turn, end, progress routes"
```

---

### Task 9: static/index.html (practice page)

**Files:**
- Create: `static/index.html`

- [ ] **Step 1: Write the page**

```html
<!doctype html>
<html lang="de">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>German Trainer</title>
<style>
  :root { --bg:#fff; --fg:#111; --muted:#666; --accent:#1a5fb4; --ok:#2e7d32; --warn:#c62828; --card:#f5f5f5; }
  @media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) { --bg:#121212; --fg:#eee; --muted:#aaa; --card:#1e1e1e; } }
  body { margin:0; padding:16px; background:var(--bg); color:var(--fg); font:16px/1.5 system-ui, sans-serif; max-width:720px; margin-inline:auto; }
  header { display:flex; justify-content:space-between; align-items:center; gap:8px; flex-wrap:wrap; }
  button { font:inherit; padding:10px 16px; border-radius:8px; border:1px solid var(--muted); background:var(--card); color:var(--fg); cursor:pointer; }
  #rec { width:100%; padding:24px; font-size:20px; margin:16px 0; }
  #rec.active { background:var(--warn); color:#fff; }
  .turn { padding:12px; border-radius:8px; background:var(--card); margin:8px 0; }
  .you { border-left:4px solid var(--muted); } .tutor { border-left:4px solid var(--accent); }
  .corr { color:var(--warn); font-size:14px; } .spoken { color:var(--ok); }
  .muted { color:var(--muted); font-size:14px; }
  #confirm { display:none; margin:8px 0; } #confirm textarea { width:100%; font:inherit; min-height:60px; }
  a { color:var(--accent); }
</style>
</head>
<body>
<header>
  <div><strong>German Trainer</strong> <span id="level" class="muted"></span></div>
  <div><a href="/progress.html">Fortschritt</a> <button id="end" disabled>Sitzung beenden</button></div>
</header>
<button id="start">Sitzung starten</button>
<div id="targets" class="muted"></div>
<button id="rec" disabled>Halten und sprechen</button>
<div id="confirm">
  <div class="muted">Verstanden als (bearbeiten, dann senden):</div>
  <textarea id="transcript"></textarea>
  <button id="send">Senden</button> <button id="retry">Nochmal aufnehmen</button>
</div>
<div id="log"></div>
<div id="summary"></div>
<script>
const $ = id => document.getElementById(id);
let rec, chunks = [], lastAudio = null;
const post = (u, o) => fetch(u, o).then(r => r.json());

$('start').onclick = async () => {
  const s = await post('/session/start', {method:'POST'});
  $('level').textContent = `Niveau ${s.level}`;
  $('targets').textContent = 'Zielwörter: ' + s.targets.join(', ');
  $('rec').disabled = false; $('end').disabled = false; $('start').disabled = true;
  $('log').innerHTML = ''; $('summary').innerHTML = '';
};

$('end').onclick = async () => {
  const s = await post('/session/end', {method:'POST'});
  $('summary').innerHTML = `<div class="turn"><strong>${s.level}</strong> (range ${s.range}, accuracy ${s.accuracy}, fluency ${s.fluency}, coherence ${s.coherence})<br>${s.summary}</div>`;
  $('rec').disabled = true; $('end').disabled = true; $('start').disabled = false;
};

async function startRec() {
  const stream = await navigator.mediaDevices.getUserMedia({audio:true});
  rec = new MediaRecorder(stream, {mimeType:'audio/webm'}); chunks = [];
  rec.ondataavailable = e => chunks.push(e.data);
  rec.onstop = async () => {
    stream.getTracks().forEach(t => t.stop());
    const blob = new Blob(chunks, {type:'audio/webm'});
    const fd = new FormData(); fd.append('audio', blob, 'a.webm');
    const r = await post('/turn', {method:'POST', body: fd});
    if (r.needs_retry) { $('transcript').value = ''; $('confirm').style.display = 'block'; return; }
    render(r);
  };
  rec.start(); $('rec').classList.add('active');
}
function stopRec() { if (rec && rec.state === 'recording') rec.stop(); $('rec').classList.remove('active'); }
$('rec').onpointerdown = startRec; $('rec').onpointerup = stopRec; $('rec').onpointerleave = stopRec;

$('send').onclick = async () => {
  const text = $('transcript').value.trim(); if (!text) return;
  $('confirm').style.display = 'none';
  render(await post('/turn/text', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({text})}));
};
$('retry').onclick = () => { $('confirm').style.display = 'none'; };

function render(r) {
  const log = $('log');
  const corr = r.corrections.map(c => `<div class="corr">${c.type}: „${c.original}“ → „${c.corrected}“ <span class="muted">${c.explanation}</span></div>`).join('');
  log.insertAdjacentHTML('beforeend', `<div class="turn you">${r.transcript}${corr}</div>`);
  const spoken = r.spoken_correction ? `<div class="spoken">${r.spoken_correction}</div>` : '';
  log.insertAdjacentHTML('beforeend', `<div class="turn tutor">${spoken}${r.reply_de}${r.degraded ? ' <span class="muted">(Antwort ohne Korrekturen)</span>' : ''}</div>`);
  if (r.audio_url) { lastAudio = new Audio(r.audio_url); lastAudio.play(); }
  window.scrollTo(0, document.body.scrollHeight);
}
</script>
</body>
</html>
```

- [ ] **Step 2: Run the server and try one turn**

```bash
uvicorn trainer.server:app --port 8765
```

Open `http://localhost:8765`, start session, hold the button, say a German sentence, release. Expected: transcript + corrections appear, tutor reply plays. End session shows level and summary.

- [ ] **Step 3: Commit**

```bash
git add static/index.html
git commit -m "feat(ui): practice page with hold-to-talk"
```

---

### Task 10: static/progress.html

**Files:**
- Create: `static/progress.html`

- [ ] **Step 1: Write the page**

```html
<!doctype html>
<html lang="de">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Fortschritt</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
<style>
  :root { --bg:#fff; --fg:#111; --muted:#666; --card:#f5f5f5; --accent:#1a5fb4; }
  @media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) { --bg:#121212; --fg:#eee; --muted:#aaa; --card:#1e1e1e; } }
  body { margin:0; padding:16px; background:var(--bg); color:var(--fg); font:16px/1.5 system-ui, sans-serif; max-width:900px; margin-inline:auto; }
  .card { background:var(--card); border-radius:8px; padding:12px; margin:12px 0; }
  canvas { max-height:280px; } .muted { color:var(--muted); font-size:14px; } a { color:var(--accent); }
  details { margin:6px 0; } pre { white-space:pre-wrap; font:inherit; }
</style>
</head>
<body>
<a href="/">← Üben</a>
<h2>Fortschritt</h2>
<div id="stats" class="muted"></div>
<div class="card"><canvas id="level"></canvas></div>
<div class="card"><canvas id="mistakes"></canvas></div>
<div class="card"><h3>Sitzungen</h3><div id="sessions"></div></div>
<script>
const LEVELS = ["A1","A2","B1","B2","C1","C2"];
const TYPES = ["case","gender","verb_position","conjugation","word_order","preposition","article","vocab","other"];
const COLORS = ["#1a5fb4","#c62828","#2e7d32","#ef6c00","#6a1b9a","#00838f","#9e9d24","#5d4037","#616161"];
fetch('/progress').then(r => r.json()).then(p => {
  const done = p.sessions.filter(s => s.ended_at && s.level_est);
  $('stats').textContent = `${done.length} Sitzungen · Vokabeln: ${p.vocab_total} (${p.vocab_due} fällig) · Zielwörter: ${Object.entries(p.target_counts).map(([k,v])=>k+' '+v).join(', ')}`;
  const labels = done.map(s => s.started_at.slice(0,10));
  const lvl = done.map(s => LEVELS.indexOf(s.level_est) + 1);
  const ma = lvl.map((_, i) => { const w = lvl.slice(Math.max(0,i-4), i+1); return w.reduce((a,b)=>a+b,0)/w.length; });
  new Chart($('level'), { type:'line', data:{ labels, datasets:[
      {label:'Niveau', data:lvl, borderColor:'#1a5fb4', pointRadius:4, tension:0},
      {label:'Ø 5 Sitzungen', data:ma, borderColor:'#9e9d24', borderDash:[6,4], pointRadius:0}]},
    options:{ scales:{ y:{ min:1, max:6, ticks:{ callback:v => LEVELS[v-1] || '' } } } } });
  const bySess = Object.fromEntries(p.mistakes_by_session.map(m => [m.session_id, m]));
  new Chart($('mistakes'), { type:'bar', data:{ labels, datasets: TYPES.map((t,i) => ({
      label:t, backgroundColor:COLORS[i], data: done.map(s => (bySess[s.id]||{})[t] || 0) })) },
    options:{ scales:{ x:{stacked:true}, y:{stacked:true, title:{display:true, text:'Fehler'}} } } });
  $('sessions').innerHTML = done.slice().reverse().map(s =>
    `<details><summary>${s.started_at.slice(0,16)} · ${s.level_est} · ${s.summary || ''}</summary><div id="s${s.id}" class="muted">…</div></details>`).join('');
  document.querySelectorAll('details').forEach(d => d.addEventListener('toggle', async () => {
    if (!d.open) return; const id = d.querySelector('div').id.slice(1);
    const x = await fetch('/session/'+id).then(r => r.json());
    d.querySelector('div').innerHTML = x.turns.map(t => `<pre><b>Du:</b> ${t.user_de}\n<b>Tutor:</b> ${t.tutor_de}</pre>`).join('');
  }, {once:true}));
});
function $(id){ return document.getElementById(id); }
</script>
</body>
</html>
```

- [ ] **Step 2: Check in browser**

Open `http://localhost:8765/progress.html` after at least one finished session. Expected: level line, stacked mistake bars, expandable session transcripts.

- [ ] **Step 3: Commit**

```bash
git add static/progress.html
git commit -m "feat(ui): progress page"
```

---

### Task 11: README and full test run

**Files:**
- Create: `README.md`

- [ ] **Step 1: Write README**

```markdown
# german-trainer

Spoken German practice with a Claude tutor. Local speech-to-text and text-to-speech, Claude via your Claude Code subscription login, progress in SQLite.

## Setup (once)

    conda create -y -n gt python=3.12 && conda activate gt
    pip install fastapi uvicorn[standard] python-multipart faster-whisper piper-tts claude-agent-sdk pytest pytest-asyncio httpx
    brew install ffmpeg            # if missing
    claude                         # log in once with the subscription, then exit
    mkdir -p data/voices && (cd data/voices && python -m piper.download_voices de_DE-thorsten-medium)
    python -m scripts.load_vocab   # fills target vocabulary (DWDS A1-B1, Aspekte B2/C1, RadicalRampage C2)

## Run

    uvicorn trainer.server:app --port 8765

Open http://localhost:8765. Hold the button, speak, release. `/progress.html` for charts.

## Tests

    pytest -q

STT/TTS tests skip if the models are not downloaded.

## Design

`docs/superpowers/specs/2026-09-23-german-speaking-trainer-design.md`
```

- [ ] **Step 2: Run everything**

Run: `pytest -q`
Expected: all passed (STT/TTS may show `s` for skipped if models absent).

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs: README"
```

---

## Self-review

- Spec coverage: store (T1, T2), cefr + Goethe rubric (T3), target vocab + loader (T4), tutor with recap/targets/Wortbildung/JSON/retry/summary (T5), stt empty-guard (T6), tts (T7), routes incl. text retype, stale session close, progress (T8), practice UI with confirm/retype (T9), charts + transcripts (T10). Error table: STT empty → `needs_retry` (T8/T9); malformed JSON → retry then degraded (T5); Piper failure → `audio_url: null` (T8); stale open session → closed on next start (T8). Session-dies restart banner from spec is not implemented: a dead SDK session surfaces as `degraded: true`; user ends and restarts. Acceptable for v1, noted here.
- Types: `Store.recap()` returns `level, top_mistakes, due_vocab` used identically in tutor and server. `TurnResult` fields match server usage. `Tutor.close()` returns dict with `level` + four scores + `summary` + `vocab`, consumed by `_end`.
- `_level` is fixed at session start and used for `mark_target_used`, matching the level the targets were drawn at.
