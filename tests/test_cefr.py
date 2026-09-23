import pytest
from trainer.cefr import derive_level, RUBRIC, LEVELS

def test_levels_order():
    assert LEVELS == ["A1", "A2", "B1", "B2", "C1", "C2"]

@pytest.mark.parametrize("scores,expected", [
    ((1, 1, 1, 1), "A1"),
    ((4, 4, 4, 4), "B2"),
    ((6, 6, 6, 6), "C2"),
    ((4, 4, 4, 3), "B2"),   # mean 3.75 → 4
    ((5, 5, 5, 2), "B1"),   # mean 4.25 → 4, but capped at min+1 = 3
])
def test_derive_level(scores, expected):
    assert derive_level(*scores) == expected

def test_rejects_out_of_range():
    with pytest.raises(ValueError):
        derive_level(0, 3, 3, 3)
    with pytest.raises(ValueError):
        derive_level(7, 3, 3, 3)

def test_rubric_mentions_goethe_traits():
    assert "Durchhaltevermögen" in RUBRIC and "Konnotation" in RUBRIC
