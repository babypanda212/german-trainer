import pytest, wave
from pathlib import Path
from trainer.tts import speak, VOICE_PATH

pytestmark = pytest.mark.skipif(not VOICE_PATH.exists(), reason="piper voice not downloaded")

def test_speak_writes_wav(tmp_path):
    out = speak("Guten Tag, wie geht es dir?", out_dir=tmp_path)
    assert out.exists() and out.suffix == ".wav"
    with wave.open(str(out)) as w:
        assert w.getnframes() > 1000
