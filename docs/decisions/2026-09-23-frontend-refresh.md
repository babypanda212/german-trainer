# Frontend refresh

User approved the warm off-white, muted teal, conversation-first design on 2026-09-23.

- Kept the plain HTML and JavaScript approach so the app still needs no build step.
- Added a shared stylesheet for the speaking and progress pages. The speaking screen has a conversation panel, a vocabulary sidebar and a stacked narrow-screen layout.
- Connected the existing opening greeting response. Added session/loading/error states, audio replay, keyboard recording and an always-available typed reply during active sessions.
- Used textContent for tutor messages and corrections; escaped progress summaries and transcripts before insertion.
- Preserved existing progress calculations and chart colors. No study records or vocabulary data changed.
- Added recording guards for overlapping operations and releasing the control before microphone permission resolves. Uses a browser-supported recording format rather than requiring WebM.
- Kept original source modules and backend behavior outside this change. An unrelated change to trainer/tts.py was already present and was not edited.

Validation: existing suite passed, 39 tests. JavaScript syntax checks passed. Browser checks with a temporary isolated sample-response server verified starting, greeting, typing, corrections, reply rendering and ending with a summary. Desktop and narrow viewport layouts were inspected, including horizontal overflow. The live progress page loaded its existing history and charts. No live microphone recording or model call was used for this frontend verification.

## Processing feedback

Added on user request: a prominent processing panel above the microphone, animated dots, elapsed seconds, and context-specific text for session start, turn processing and session summary. After 15 seconds, explicitly states the request is still pending without inventing progress percentages or backend stage information. The panel disappears on success or failure; playback receives separate speaking/correction labels. Reduced-motion settings disable dot animation. Verified pending, success, network failure and HTTP error behavior with the actual request/indicator functions, plus JavaScript syntax and whitespace checks.
