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
