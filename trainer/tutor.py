from dataclasses import dataclass, field
from typing import Callable, Awaitable, Protocol
from trainer.cefr import RUBRIC, derive_level
from trainer.store import MISTAKE_TYPES

TURN_SCHEMA = {
    "type": "object",
    "properties": {
        "reply_de": {"type": "string"},
        "corrections": {"type": "array", "items": {
            "type": "object",
            "properties": {
                "type": {"type": "string", "enum": MISTAKE_TYPES},
                "original": {"type": "string"},
                "corrected": {"type": "string"},
                "explanation": {"type": "string"}},
            "required": ["type", "original", "corrected", "explanation"]}},
        "targets_used": {"type": "array", "items": {"type": "string"}},
        "asked_about": {"type": ["string", "null"]},
        "asked_about_gloss": {"type": ["string", "null"]},
    },
    "required": ["reply_de", "corrections", "targets_used", "asked_about", "asked_about_gloss"],
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
This log is for progress tracking only and is never shown to the learner directly — it is separate
from what you say in reply_de.

If there is at least one correction, choose ONE to build your reply around — prefer one matching the
learner's recurring mistake types ({top}) if such an error occurred this turn, otherwise the first
correction found. Never feature more than one, even if several errors occurred. Weave reply_de around
that one correction as a single natural continuation of the conversation (1-3 sentences, still ending
in a question):
  1. Model the correct form briefly in passing — a natural recast, not a grammar explanation, not
     sounding like a teacher.
  2. Then ask a follow-up question that can only be answered by using that same corrected structure
     again. This is the actual point: the learner must produce it themselves, not just hear it once.
If there were no corrections this turn, just continue the conversation normally with a question.

Vocabulary knowledge: if the learner explicitly asks what a German word means, or asks for its
translation, set `asked_about` to that exact word (as they wrote or said it) and `asked_about_gloss`
to a short English gloss. Otherwise both are null. At most one word per turn — the learner will
never ask about more than one word at once, so if several could apply, name only the clearest one.

{RUBRIC}
"""

class TutorError(RuntimeError):
    """Raised when the tutor backend fails to produce a usable structured reply.
    Never papered over with a canned/placeholder result — callers must handle it explicitly."""

OPENING_INSTRUCTION = (
    "(No learner turn: this is the start of the session. Greet the learner in German and ask one short, "
    "level-appropriate question to open the conversation. corrections must be an empty list — there is "
    "nothing to correct yet.)"
)

@dataclass
class TurnResult:
    reply_de: str
    corrections: list[dict]
    targets_used: list[str]
    asked_about: str | None = None
    asked_about_gloss: str | None = None
    model_source: str = "claude"   # "claude" or "local" — which backend actually produced this turn

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
            raise TutorError("Tutor did not return a valid structured reply after one retry.")
        source = getattr(self._session, "last_source", "claude")
        return TurnResult(data["reply_de"], data.get("corrections", []), data.get("targets_used", []),
                          data.get("asked_about"), data.get("asked_about_gloss"), source)

    async def opening(self) -> TurnResult:
        """First line of the session: no learner turn to reply to, so this bypasses `turn`'s
        normal (transcript, reply) pairing. Caller must not log this as a learner turn."""
        return await self.turn(OPENING_INSTRUCTION)

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
        try:
            data = await summ.turn(prompt)
        finally:
            await summ.close()
        if data is None:
            raise TutorError("Tutor did not return a valid session summary after the rating call.")
        data["level"] = derive_level(data["range"], data["accuracy"], data["fluency"], data["coherence"])
        data["model_source"] = getattr(summ, "last_source", "claude")   # local CEFR ratings are much less reliable
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


# --- local fallback, used when the Claude backend is unavailable (e.g. subscription usage
# exhausted) --------------------------------------------------------------------------------

OLLAMA_URL = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "qwen2.5:7b"

class OllamaSession:
    """Local session via Ollama. Its /api/chat endpoint is stateless per call, unlike the Agent
    SDK's persistent session, so this class keeps its own running message history and resends
    it every turn. Quality is noticeably below Claude - this exists to stay available, not to
    match it."""
    def __init__(self, system_prompt: str, schema: dict, model: str = OLLAMA_MODEL):
        self._model = model
        self._schema = schema
        self._messages = [{"role": "system", "content": system_prompt}]

    async def turn(self, prompt: str) -> dict | None:
        import httpx, json
        self._messages.append({"role": "user", "content": prompt})
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                resp = await client.post(OLLAMA_URL, json={
                    "model": self._model, "messages": self._messages, "stream": False,
                    "format": self._schema,
                })
                resp.raise_for_status()
                content = resp.json()["message"]["content"]
                data = json.loads(content)
        except Exception:
            return None
        self._messages.append({"role": "assistant", "content": content})
        return data

    async def close(self) -> None:
        pass

def ollama_session_factory(system_prompt: str, schema: dict) -> Session:
    return OllamaSession(system_prompt, schema)


class FallbackSession:
    """Tries the primary session first; on failure, switches to the fallback for the rest of
    this session's lifetime - a fresh local session has no memory of turns before the switch,
    so staying switched (rather than retrying the primary every turn) avoids repeatedly losing
    context. Never fabricates: if the fallback also fails, turn() returns None like any other
    session, so Tutor's existing retry-then-raise logic still applies unchanged."""
    def __init__(self, system_prompt: str, schema: dict, primary_factory: SessionFactory, fallback_factory: SessionFactory):
        self._primary: Session | None = primary_factory(system_prompt, schema)
        self._fallback_factory = fallback_factory
        self._system_prompt = system_prompt
        self._schema = schema
        self._fallback: Session | None = None
        self.last_source = "claude"

    async def turn(self, prompt: str) -> dict | None:
        if self._primary is not None:
            try:
                data = await self._primary.turn(prompt)
            except Exception:
                data = None
            if data is not None:
                self.last_source = "claude"
                return data
            try:
                await self._primary.close()
            except Exception:
                pass
            self._primary = None
            self._fallback = self._fallback_factory(self._system_prompt, self._schema)
        try:
            data = await self._fallback.turn(prompt)
        except Exception:
            data = None
        if data is not None:
            self.last_source = "local"
        return data

    async def close(self) -> None:
        if self._primary is not None:
            try:
                await self._primary.close()
            except Exception:
                pass
        if self._fallback is not None:
            try:
                await self._fallback.close()
            except Exception:
                pass

def fallback_session_factory(primary: SessionFactory, fallback: SessionFactory) -> SessionFactory:
    def factory(system_prompt: str, schema: dict) -> Session:
        return FallbackSession(system_prompt, schema, primary, fallback)
    return factory
