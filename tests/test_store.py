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
