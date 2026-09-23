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
