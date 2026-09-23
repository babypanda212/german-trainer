import asyncio
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from trainer.store import Store
from trainer.tutor import Tutor, TutorError, claude_session_factory
from trainer.stt import transcribe
from trainer.tts import speak, AUDIO_DIR
from trainer.vocab_signals import compute_vocab_signals

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
# Session state (tutor/_sid/...) is mutable module globals shared across requests. Without
# serializing the routes that touch it, a slow /session/end (awaiting the closing rating call)
# can race a VAD-triggered /turn and hand it a half-torn-down session — crashing with an
# AssertionError instead of a clean error. This lock makes those routes queue instead of race.
_lock = asyncio.Lock()

class TextTurn(BaseModel):
    text: str

@app.post("/session/start")
async def session_start():
    async with _lock:
        global tutor, _sid, _idx, _level
        closed_previous = False
        stale = store.open_session_id()
        if stale is not None:
            await _end(stale); closed_previous = True
        recap = store.recap()
        targets = store.next_targets(recap["level"], TARGETS_PER_SESSION)
        tutor = Tutor(session_factory)
        await tutor.start(recap, targets)
        _sid = store.start_session(); _idx = 0; _level = recap["level"]
        greeting_de = None
        audio_url = None
        try:
            g = await tutor.opening()
            greeting_de = g.reply_de
            for w in g.targets_used:
                store.mark_target_used(w, _level)
                store.add_vocab(w, None, None)
            try:
                p = speak(greeting_de); audio_url = f"/audio/{p.name}"
            except Exception:
                pass
        except TutorError:
            pass  # no greeting available; the learner can still speak first — nothing is fabricated
        return {"session_id": _sid, "level": recap["level"], "targets": [t["word"] for t in targets],
                "closed_previous": closed_previous, "greeting_de": greeting_de, "audio_url": audio_url}

async def _process(user_de: str, audio_path: str | None):
    global _idx
    try:
        r = await tutor.turn(user_de)
    except TutorError as e:
        raise HTTPException(502, str(e))
    tid = store.add_turn(_sid, _idx, user_de, r.reply_de, audio_path, len(user_de.split()))
    _idx += 1
    for c in r.corrections:
        store.add_mistake(tid, c["type"], c["original"], c["corrected"], c["explanation"])
    for w in r.targets_used:
        store.mark_target_used(w, _level)
        store.add_vocab(w, None, tid)
    # Vocabulary knowledge signal, deterministic given r.asked_about (the one thing the model
    # judges): asking about a word means "don't know it" and excludes every other target word
    # in the same sentence from counting as known — she never asks about more than one at a
    # time, so their silence isn't evidence either way. See vocab_signals.py.
    target_glosses = store.target_vocab_glosses()
    signals = compute_vocab_signals(user_de, set(target_glosses.keys()), r.asked_about)
    for word, quality in signals.items():
        gloss = r.asked_about_gloss if word == r.asked_about else target_glosses.get(word)
        store.add_vocab(word, gloss, tid)
        store.review_vocab(word, quality)
    # Correction (if any) is woven into reply_de as a single elicitation turn (see tutor.py) —
    # one utterance, one audio clip, not a separate paced correction beat.
    audio_url = None
    try:
        p = speak(r.reply_de); audio_url = f"/audio/{p.name}"
    except Exception:
        pass
    return {"transcript": user_de, "reply_de": r.reply_de,
            "corrections": r.corrections, "audio_url": audio_url}

@app.post("/turn")
async def turn(audio: UploadFile = File(...)):
    async with _lock:
        if tutor is None or _sid is None:
            raise HTTPException(400, "no open session")
        text = transcribe(await audio.read())
        if not text:
            return {"transcript": "", "needs_retry": True}
        return await _process(text, None)

@app.post("/turn/text")
async def turn_text(body: TextTurn):
    async with _lock:
        if tutor is None or _sid is None:
            raise HTTPException(400, "no open session")
        return await _process(body.text.strip(), None)

async def _end(sid: int) -> dict:
    global tutor, _sid
    turns = store.turns(sid)
    if tutor is None:
        tutor = Tutor(session_factory)
    try:
        summ = await tutor.close([t["user_de"] for t in turns], store.error_rate(sid))
    except TutorError as e:
        raise HTTPException(502, str(e))
    store.end_session(sid, summ["level"], summ, summ["summary"])
    for v in summ.get("vocab", []):
        store.add_vocab(v["word"], v["gloss"], None)
    tutor = None; _sid = None
    return {**summ, "session_id": sid}

@app.post("/session/end")
async def session_end():
    async with _lock:
        if _sid is None:
            raise HTTPException(400, "no open session")
        return await _end(_sid)

@app.get("/progress")
async def progress():
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
async def session_detail(sid: int):
    return {"session": store.get_session(sid), "turns": store.turns(sid), "mistakes": store.mistakes(sid)}

@app.get("/audio/{name}")
def audio(name: str):
    p = AUDIO_DIR / name
    if not p.exists():
        raise HTTPException(404)
    return FileResponse(p, media_type="audio/wav")

app.mount("/", StaticFiles(directory=str(ROOT / "static"), html=True), name="static")
