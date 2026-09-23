import pytest
from pathlib import Path
from trainer.tutor import Tutor, build_system_prompt, TURN_SCHEMA, SUMMARY_SCHEMA

class FakeSession:
    def __init__(self, replies):
        self.replies = list(replies); self.prompts = []; self.closed = False
    async def turn(self, prompt: str) -> dict | None:
        self.prompts.append(prompt)
        return self.replies.pop(0)
    async def close(self):
        self.closed = True

RECAP = {"level": "B1", "top_mistakes": ["gender", "case"], "due_vocab": [{"word": "Abbau", "gloss": "dismantling"}]}
TARGETS = [{"word": "Haus", "gloss": "das Haus"}, {"word": "Auto", "gloss": "das Auto"}]

def test_system_prompt_contains_recap_targets_and_types():
    p = build_system_prompt(RECAP, TARGETS)
    assert "B1" in p and "gender" in p and "Abbau" in p and "Haus" in p
    assert "verb_position" in p          # fixed mistake list
    assert "Durchhaltevermögen" in p     # rubric included

def test_system_prompt_c1_adds_word_formation():
    p = build_system_prompt({**RECAP, "level": "C1"}, [])
    assert "Wortbildung" in p

@pytest.mark.asyncio
async def test_turn_parses_reply():
    reply = {"reply_de": "Gut! Wo wohnst du?", "spoken_correction": None,
             "corrections": [{"type": "gender", "original": "der Haus", "corrected": "das Haus", "explanation": "Haus is neuter"}],
             "targets_used": ["Haus"]}
    fake = FakeSession([reply])
    t = Tutor(session_factory=lambda sp, schema: fake)
    await t.start(RECAP, TARGETS)
    r = await t.turn("Ich wohne in der Haus.")
    assert r.reply_de == "Gut! Wo wohnst du?"
    assert r.corrections[0]["type"] == "gender"
    assert r.targets_used == ["Haus"]
    assert fake.prompts[-1] == "Ich wohne in der Haus."

@pytest.mark.asyncio
async def test_turn_retries_once_then_degrades():
    good = {"reply_de": "Ok.", "spoken_correction": None, "corrections": [], "targets_used": []}
    fake = FakeSession([None, good])
    t = Tutor(session_factory=lambda sp, schema: fake)
    await t.start(RECAP, TARGETS)
    r = await t.turn("Hallo")
    assert r.reply_de == "Ok." and len(fake.prompts) == 2

    fake2 = FakeSession([None, None])
    t2 = Tutor(session_factory=lambda sp, schema: fake2)
    await t2.start(RECAP, TARGETS)
    r2 = await t2.turn("Hallo")
    assert r2.degraded is True and r2.corrections == []

@pytest.mark.asyncio
async def test_close_returns_summary_and_closes_session():
    summary = {"range": 3, "accuracy": 3, "fluency": 4, "coherence": 3, "summary": "fine",
               "vocab": [{"word": "Abbau", "gloss": "dismantling"}]}
    fake = FakeSession([])
    summ_fake = FakeSession([summary])
    made = []
    def factory(sp, schema):
        made.append(schema); return fake if schema is TURN_SCHEMA else summ_fake
    t = Tutor(session_factory=factory)
    await t.start(RECAP, TARGETS)
    s = await t.close(user_turns=["Ich bin gegangen."], error_rate=2.0)
    assert s["level"] == "B1" and s["fluency"] == 4
    assert fake.closed and summ_fake.closed
    assert SUMMARY_SCHEMA in made
