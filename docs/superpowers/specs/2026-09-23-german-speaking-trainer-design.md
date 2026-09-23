# German speaking trainer: design

Date: 2026-09-23
Status: approved in chat, awaiting spec review

## Goal

Single-user, local web app for spoken German practice with persistent progress. Fills the gap in existing tools: nothing open-source combines AI voice conversation in German with memory across sessions.

## Decisions

| Decision | Choice | Rejected |
|---|---|---|
| Users | One, on this Mac | Shared / hosted |
| Model placement | Hybrid: local STT and TTS, cloud LLM | Fully local (weak German), fully cloud (audio leaves machine, cost) |
| LLM access | Claude via Agent SDK, one persistent session, subscription login | Messages API (needs key, subscription unusable); `claude -p` per turn (2 to 4 s startup per turn); Claude Code skill (no voice, no charts) |
| Interface | One local web page | Terminal push-to-talk |
| Progress | Mistake memory, CEFR trend, vocab with spaced review, session log | None dropped |
| Target vocabulary | A1 to B1: official Goethe lists via DWDS. B2, C1: Klett Aspekte neu textbook word lists (already in `docs/vocabulary/`, deduplicated). C2: RadicalRampage list (402 words, unverified, flagged) | Profile deutsch CD-ROM (out of print, 2005 Windows DB); authentic-text vocab extraction (dropped for v1 now that real B2/C1 lists exist) |

## Vocabulary sources (verified 2026-09-23)

- Goethe publishes official Wortlisten for A1, A2, B1 only. DWDS mirrors them with CSV/JSON export (`dwds.de/lemma/wortschatz-goethe-zertifikat`). Goethe copyright, personal use.
- B2: Goethe's Prüfungsziele §4.4 points to Profile deutsch (Glaboniat 2005) rather than publishing a list.
- C1/C2: Prüfungsziele §4.4 states no inventory exists because authentic texts are used; candidates are expected to derive unknown words (*erschließen*) from known parts.
- All "Goethe B2/C1 Wortliste" PDFs online are third-party compilations.

## Architecture

```
browser (static/index.html)
  mic → WebM/Opus blob ──POST /turn──▶ FastAPI (server.py)
                                        │ stt.py: faster-whisper, language=de → transcript
                                        │ tutor.py: Agent SDK session → JSON
                                        │           {reply_de, corrections[], level_signal}
                                        │ tts.py: Piper de_DE-thorsten → wav
                                        │ store.py: write turn, mistakes, vocab
                                        ◀── {transcript, reply_de, corrections, audio_url}
  plays audio, shows reply and corrections
```

Files: `server.py`, `stt.py`, `tutor.py`, `tts.py`, `store.py`, `cefr.py`, `static/index.html`, `static/progress.html`, `data/trainer.db`, `data/audio/`.

## Components

### stt.py
- `transcribe(webm_bytes) -> str`
- faster-whisper, model `small` (upgrade to `medium` if accuracy is poor), `language="de"`.
- Returns empty string if fewer than 2 words or no speech detected.

### tutor.py
- `Tutor.start(recap: Recap) -> None`: opens one Agent SDK session for the sitting. System prompt includes the rubric, the fixed mistake-type list, the JSON schema, and the recap (top 3 recurring mistake types, up to 5 vocab items due).
- `Tutor.turn(user_de: str) -> TurnResult`: streams the user text in, parses JSON reply.
- `Tutor.close() -> SessionSummary`: one final call. Input: the user's turns only, plus error rate per 100 words from logged mistakes. Output: four rubric sub-scores, derived level, summary text, vocab extracted.
- Tutor rules in prompt: reply in German at the estimated level, at most one correction spoken aloud per turn, all corrections in the JSON, work due vocab into the conversation naturally.

### cefr.py
- `derive_level(range_, accuracy, fluency, coherence) -> str`
- Each sub-score 1 to 6 maps to A1 to C2. Level = rounded mean, capped at min sub-score + 1. Pure function, unit-tested.
- Rubric text given to the tutor uses Goethe's own level descriptors (Prüfungsziele B2/C1/C2 §4.5/4.6): B2 = effective argumentation, staying power in discourse, avoids gross errors, gaps cause hesitation; C1 = near-effortless, longer utterances in less time, paraphrases gaps skilfully, adapts register, allusions and jokes; C2 = precise, idiomatic, aware of connotations, removes ambiguity. Scoring is about what is done with the vocabulary, not its size.

### target_vocab (in store.py)
```
target_vocab  id, word, level, source, times_used, introduced_at
```
- `source` ∈ {goethe_dwds, aspekte_neu, radicalrampage}. Loaded once by `scripts/load_vocab.py` from DWDS CSV (A1 to B1, `https://www.dwds.de/api/lemma/goethe/{A1,A2,B1}.csv`, columns Lemma, URL, Wortart, Genus, Artikel, nur_im_Plural) and from `docs/vocabulary/cleaned/entries-with-provenance.json` (B2, C1, C2; fields headword, level, variants[].text/translation/source).
- Each session start: `next_targets(level, n=10)` returns the least-used words at the user's current level. Tutor must work them into the conversation. Words the user then produces correctly are promoted into `vocab` for spaced review.
- At C1/C2 the tutor is additionally instructed to point out word-formation (compounds, prefixes, derivations), since Goethe's C1/C2 specs test deriving unknown words rather than a fixed list.
- Stored as a trend. Session-to-session noise is expected; the progress page shows a 5-session moving average alongside raw points.

### tts.py
- `speak(text_de: str) -> Path`: Piper `de_DE-thorsten-medium`, writes wav to `data/audio/`, returns path.

### store.py
SQLite, schema:

```
sessions   id, started_at, ended_at, level_est, range, accuracy, fluency, coherence, summary
turns      id, session_id, idx, user_de, tutor_de, audio_path, words
mistakes   id, turn_id, type, original, corrected, explanation
vocab      id, word, gloss, source_turn_id, due_at, interval_days, ease
```

- `type` is one of: case, gender, verb_position, conjugation, word_order, preposition, article, vocab, other. Fixed list shared by tutor prompt, store, and charts.
- Vocab scheduling: SM-2. `due()` returns items with `due_at <= now`. Reviewed implicitly: if the word appears in a user turn without a correction, mark reviewed with quality 4; if corrected, quality 2.
- `recap() -> Recap`: top 3 mistake types over last 10 sessions, up to 5 due vocab.

### server.py
Routes:
- `POST /session/start` → starts Tutor with recap, returns session id
- `POST /turn` (audio blob) → full pipeline, returns transcript, reply, corrections, audio_url
- `POST /turn/text` (typed text) → same without STT, used for retype after bad transcription
- `POST /session/end` → Tutor.close(), stores summary, returns it
- `GET /progress` → JSON for charts
- static files

### static/index.html
- Hold-to-record button, transcript shown before sending (confirm or retype), reply text, spoken-correction highlight, end-session button.
- Plain HTML + JS, no build step.

### static/progress.html
- Three charts: mistakes by type over time (stacked), level trend (raw + moving average), vocab due count. Session list with transcripts and summaries. Chart library loaded from cdnjs.

## Error handling

| Failure | Behaviour |
|---|---|
| STT empty or garbage | Show transcript field editable, do not send to tutor until confirmed |
| Agent SDK session dies | Restart with recap, banner "session restarted" |
| Tutor JSON malformed | Retry once with "return only JSON"; then show raw text, log zero corrections, warn |
| Piper fails | Show text reply, no audio, warn |
| Session ended without close (browser closed) | Next start finds open session, runs close() on it first |

## Testing

- `store.py`: full unit tests (schema, recap, SM-2 scheduling, mistake aggregation).
- `cefr.py`: full unit tests, including cap rule.
- `tutor.py`: tested against a fake session object returning canned JSON, including malformed JSON path.
- `server.py`: route tests with stt/tts/tutor stubbed.
- `stt.py`, `tts.py`: one smoke test each, skipped if models not installed.

## Out of scope (v1)

- Multiple users, auth, hosting.
- Pronunciation scoring.
- Flashcard UI (vocab review is conversational).
- Scenario/roleplay modes. Free conversation only; scenarios can be added as prompt presets later.
