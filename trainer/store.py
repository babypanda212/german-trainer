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

    def target_vocab_glosses(self) -> dict[str, str | None]:
        """word -> gloss across every level. Used to fill in a gloss when logging a vocab
        exposure detected in the learner's own turn (see vocab_signals.py) - the word itself
        may be matched several times across levels; the first one wins, arbitrarily."""
        out: dict[str, str | None] = {}
        for r in self.conn.execute("select word, gloss from target_vocab"):
            out.setdefault(r["word"], r["gloss"])
        return out

    # recap for session start
    def recap(self) -> dict:
        last = self.conn.execute(
            "select level_est from sessions where ended_at is not null order by id desc limit 1").fetchone()
        level = last["level_est"] if last and last["level_est"] else "A2"
        top = list(self.mistake_counts(10).keys())[:3]
        return {"level": level, "top_mistakes": top, "due_vocab": self.due_vocab(5)}
