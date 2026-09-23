import pytest, subprocess, shutil
from trainer.stt import transcribe, MIN_WORDS

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg missing")

def test_silence_returns_empty(tmp_path):
    wav = tmp_path / "s.wav"
    subprocess.run(["ffmpeg", "-y", "-f", "lavfi", "-i", "anullsrc=r=16000:cl=mono", "-t", "1", str(wav)],
                   check=True, capture_output=True)
    assert transcribe(wav.read_bytes()) == ""

def test_min_words_constant():
    assert MIN_WORDS == 2
