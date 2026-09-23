from dataclasses import dataclass, field
from typing import Callable, Awaitable, Protocol
from trainer.cefr import RUBRIC, derive_level
from trainer.store import MISTAKE_TYPES

TURN_SCHEMA = {
    "type": "object",
    "properties": {
        "reply_de": {"type": "string"},
        "spoken_correction": {"type": ["string", "null"]},
        "corrections": {"type": "array", "items": {
            "type": "object",
            "properties": {
                "type": {"type": "string", "enum": MISTAKE_TYPES},
                "original": {"type": "string"},
                "corrected": {"type": "string"},
                "explanation": {"type": "string"}},
            "required": ["type", "original", "corrected", "explanation"]}},
        "targets_used": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["reply_de", "spoken_correction", "corrections", "targets_used"],
}

SUMMARY_SCHEMA = {
    "type": "object",
    "properties": {
        "range": {"type": "integer", "minimum": 1, "maximum": 6},
        "accuracy": {"type": "integer", "minimum": 1, "maximum": 6},
        "fluency": {"type": "integer", "minimum": 1, "maximum": 6},
        "coherence": {"type": "integer", "minimum": 1, "maximum": 6},
        "summary": {"type": "string"},
        "vocab": {"type": "array", "items": {
            "type": "object",
            "properties": {"word": {"type": "string"}, "gloss": {"type": "string"}},
            "required": ["word", "gloss"]}},
    },
    "required": ["range", "accuracy", "fluency", "coherence", "summary", "vocab"],
}

def build_system_prompt(recap: dict, targets: list[dict]) -> str:
    level = recap["level"]
    targets_txt = ", ".join(f"{t['word']} ({t['gloss']})" if t.get("gloss") else t["word"] for t in targets) or "none"
    due_txt = ", ".join(f"{v['word']} ({v.get('gloss') or ''})" for v in recap["due_vocab"]) or "none"
    top = ", ".join(recap["top_mistakes"]) or "none yet"
    extra = ""
    if level in ("C1", "C2"):
        extra = ("\nAt this level, also point out Wortbildung: when a useful compound, prefix verb or derivation "
                 "comes up, say briefly how it is built from known parts.\n")
    return f"""You are a German conversation tutor. The learner's current estimated level is {level}.
Speak German only in reply_de, pitched at {level}. Keep replies to 1-3 sentences and end with a question
or prompt that keeps the conversation going. Never switch to English in reply_de.

Learner's recurring mistake types (target these gently): {top}.
Vocabulary due for review, work these into your questions naturally: {due_txt}.
Target vocabulary for this session, work each into the conversation at least once and list the ones you used
in targets_used (exact word as given): {targets_txt}.
{extra}
Corrections: log every error in the learner's turn in `corrections` with type from
{MISTAKE_TYPES}, the original fragment, the corrected fragment, and a one-sentence English explanation.
In `spoken_correction` put at most ONE short German correction to say aloud (the most important one),
or null if the turn was fine. Do not put corrections inside reply_de.

{RUBRIC}
"""

@dataclass
class TurnResult:
    reply_de: str
    spoken_correction: str | None
    corrections: list[dict]
    targets_used: list[str]
    degraded: bool = False

class Session(Protocol):
    async def turn(self, prompt: str) -> dict | None: ...
    async def close(self) -> None: ...

SessionFactory = Callable[[str, dict], Session]

class Tutor:
    def __init__(self, session_factory: SessionFactory):
        self._factory = session_factory
        self._session: Session | None = None
        self._recap: dict = {}
        self._targets: list[dict] = []

    async def start(self, recap: dict, targets: list[dict]) -> None:
        self._recap, self._targets = recap, targets
        self._session = self._factory(build_system_prompt(recap, targets), TURN_SCHEMA)

    async def turn(self, user_de: str) -> TurnResult:
        assert self._session is not None, "call start() first"
        data = await self._session.turn(user_de)
        if data is None:
            data = await self._session.turn(user_de)
        if data is None:
            return TurnResult("Entschuldigung, kannst du das noch einmal sagen?", None, [], [], degraded=True)
        return TurnResult(data["reply_de"], data.get("spoken_correction"), data.get("corrections", []),
                          data.get("targets_used", []))

    async def close(self, user_turns: list[str], error_rate: float) -> dict:
        if self._session is not None:
            await self._session.close()
            self._session = None
        prompt = (f"{RUBRIC}\nThe learner's turns this session (tutor turns omitted):\n"
                  + "\n".join(f"- {t}" for t in user_turns)
                  + f"\n\nLogged error rate: {error_rate} errors per 100 words.\n"
                  "Score the four criteria 1-6, write a 3-sentence English summary of what to work on, "
                  "and list up to 8 useful German words the learner used or was corrected on, with a short English gloss.")
        summ = self._factory("You are a strict CEFR rater for German speaking.", SUMMARY_SCHEMA)
        data = await summ.turn(prompt)
        await summ.close()
        if data is None:
            data = {"range": 3, "accuracy": 3, "fluency": 3, "coherence": 3,
                    "summary": "Rating failed; scores are placeholders.", "vocab": []}
        data["level"] = derive_level(data["range"], data["accuracy"], data["fluency"], data["coherence"])
        return data


# --- real session backed by the Agent SDK -------------------------------------------------

class ClaudeSession:
    """One persistent Claude Code session (uses the local `claude` login, i.e. the subscription)."""
    def __init__(self, system_prompt: str, schema: dict, model: str | None = None):
        from claude_agent_sdk import ClaudeSDKClient, ClaudeAgentOptions
        self._client = ClaudeSDKClient(options=ClaudeAgentOptions(
            system_prompt=system_prompt,
            allowed_tools=[],
            model=model,
            permission_mode="bypassPermissions",
            output_format={"type": "json_schema", "schema": schema},
        ))
        self._connected = False

    async def turn(self, prompt: str) -> dict | None:
        from claude_agent_sdk import ResultMessage
        if not self._connected:
            await self._client.connect(); self._connected = True
        await self._client.query(prompt)
        out = None
        async for m in self._client.receive_response():
            if isinstance(m, ResultMessage) and m.subtype == "success" and m.structured_output:
                out = m.structured_output
        return out

    async def close(self) -> None:
        if self._connected:
            await self._client.disconnect(); self._connected = False

def claude_session_factory(system_prompt: str, schema: dict) -> Session:
    return ClaudeSession(system_prompt, schema)
