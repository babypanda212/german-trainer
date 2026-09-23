LEVELS = ["A1", "A2", "B1", "B2", "C1", "C2"]

RUBRIC = """Score the learner's German speaking on four criteria, each 1-6 (1=A1 ... 6=C2).
Score only the learner's own turns, never the tutor's. Use these anchors, taken from the
Goethe-Institut exam descriptions:

range (Wortschatzspektrum):
  3 (B1): minimal means used maximally, everyday topics only.
  4 (B2): large vocabulary in own field and most general topics; gaps cause hesitation and paraphrase.
  5 (C1): broad repertoire; gaps bridged skilfully by paraphrase; idioms and colloquial phrases controlled.
  6 (C2): very rich including idioms; aware of Konnotationen; consistently apt word choice.
accuracy (Korrektheit):
  3 (B1): frequent errors but meaning clear.
  4 (B2): good control; avoids gross errors (krasse Formulierungsfehler); occasional slips, often self-corrected.
  5 (C1): consistently high accuracy; errors rare and hardly noticed.
  6 (C2): correct even in complex language.
fluency (Flüssigkeit):
  3 (B1): short, structurally reduced sentences; needs repetition.
  4 (B2): spontaneous, structured, no longer only short sentences; Durchhaltevermögen im Diskurs (holds a line of argument over several turns).
  5 (C1): beinahe mühelos; longer utterances in less time; adapts to social context; can make allusions or jokes.
  6 (C2): very fluent and precise; complex content presented coherently.
coherence (Kohärenz/Register):
  3 (B1): simple linking (und, aber, weil).
  4 (B2): effective argumentation; pros and cons systematically; Sprachbewusstsein.
  5 (C1): control of text patterns and connectors; differentiated use of formal vs informal register.
  6 (C2): stylistic differentiation; removes ambiguity; emphasises and nuances deliberately.
"""

def derive_level(range_: int, accuracy: int, fluency: int, coherence: int) -> str:
    scores = [range_, accuracy, fluency, coherence]
    if any(not (1 <= s <= 6) for s in scores):
        raise ValueError(f"scores must be 1-6, got {scores}")
    mean = round(sum(scores) / 4)
    capped = min(mean, min(scores) + 1)
    return LEVELS[capped - 1]
