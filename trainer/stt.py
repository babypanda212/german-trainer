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
