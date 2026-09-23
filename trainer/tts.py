import uuid, wave
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VOICE_PATH = ROOT / "data/voices/de_DE-kerstin-low.onnx"
AUDIO_DIR = ROOT / "data/audio"

@lru_cache(maxsize=1)
def _voice():
    from piper import PiperVoice
    return PiperVoice.load(str(VOICE_PATH))

def speak(text_de: str, out_dir: Path = AUDIO_DIR) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{uuid.uuid4().hex}.wav"
    with wave.open(str(out), "wb") as wf:
        _voice().synthesize_wav(text_de, wf)
    return out
