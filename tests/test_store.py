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
