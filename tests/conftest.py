import pytest
from pathlib import Path
from trainer.store import Store

@pytest.fixture
def store(tmp_path: Path) -> Store:
    return Store(tmp_path / "t.db")
