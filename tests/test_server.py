import asyncio
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
        return {"reply_de": "Schön! Und dann?",
                "corrections": [{"type": "conjugation", "original": "habe gegangen", "corrected": "bin gegangen",
                                  "explanation": "e", "recast": "Ah, du meinst: „ich bin gegangen“."}],
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
    body = r.json()
    sid = body["session_id"]
    assert body["greeting_de"] == "Schön! Und dann?"   # opening line, from the same fake reply
    assert body["audio_url"].startswith("/audio/")
    r = await client.post("/turn", files={"audio": ("a.webm", b"xx", "audio/webm")})
    body = r.json()
    assert body["transcript"] == "Ich habe zum Bahnhof gegangen."
    assert body["reply_de"] == "Schön! Und dann?"
    assert body["spoken_correction"] == "Ah, du meinst: „ich bin gegangen“."   # the recast, spoken separately
    assert body["correction_audio_url"].startswith("/audio/")
    assert body["corrections"][0]["type"] == "conjugation"   # full list still logged
    assert body["audio_url"].startswith("/audio/")
    assert "degraded" not in body   # no fabricated-fallback field
    r = await client.post("/session/end"); s = r.json()
    assert s["level"] == "B1" and s["session_id"] == sid
    st = server.store
    assert len(st.turns(sid)) == 1 and len(st.mistakes(sid)) == 1
    assert st.next_targets("A2", 1)[0]["times_used"] == 2   # once from the opening, once from the real turn
    assert st.get_vocab("Bahnhof") is not None

async def test_start_reports_no_greeting_when_tutor_fails(client, monkeypatch):
    class AlwaysFailingSession:
        def __init__(self, sp, schema): pass
        async def turn(self, prompt): return None
        async def close(self): pass
    monkeypatch.setattr(server, "session_factory", AlwaysFailingSession)
    r = await client.post("/session/start")
    body = r.json()
    assert r.status_code == 200
    assert body["greeting_de"] is None and body["audio_url"] is None   # honest omission, no fake greeting

async def test_turn_returns_502_when_tutor_fails(client, monkeypatch):
    class AlwaysFailingSession:
        def __init__(self, sp, schema): pass
        async def turn(self, prompt): return None
        async def close(self): pass
    monkeypatch.setattr(server, "session_factory", AlwaysFailingSession)
    await client.post("/session/start")   # opening fails silently too (greeting_de: None), session still starts
    r = await client.post("/turn/text", json={"text": "Hallo"})
    assert r.status_code == 502

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

async def test_turn_cannot_race_a_slow_session_end(client, monkeypatch):
    # Regression test: a VAD-triggered /turn landing while /session/end is mid-flight (e.g.
    # waiting on the closing rating call) used to crash with AssertionError because the two
    # requests shared unlocked module state. The lock must serialize them: /turn either
    # completes cleanly before /end starts, or is cleanly rejected (400) once queued behind it -
    # never a 500.
    class SlowFakeSession(FakeSession):
        async def turn(self, prompt):
            if "CEFR rater" in self.sp:
                await asyncio.sleep(0.2)
            return await super().turn(prompt)
    monkeypatch.setattr(server, "session_factory", SlowFakeSession)
    await client.post("/session/start")
    end_task = asyncio.create_task(client.post("/session/end"))
    await asyncio.sleep(0.05)   # let /session/end acquire the lock and enter its slow await
    turn_resp = await client.post("/turn/text", json={"text": "Hallo"})
    end_resp = await end_task
    assert end_resp.status_code == 200
    assert turn_resp.status_code == 400   # queued behind end, session already gone by its turn
    assert turn_resp.json()["detail"] == "no open session"
