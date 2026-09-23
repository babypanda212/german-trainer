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
`docs/superpowers/plans/2026-09-23-german-speaking-trainer.md`
