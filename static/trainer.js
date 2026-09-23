'use strict';
const $ = id => document.getElementById(id);
let active = false, busy = false, requestingMic = false;
let recorder = null, stream = null, lastAudio = null, turns = 0;
// Continuous listening (voice-activity detection): the mic opens once per session and
// stays open. `listening` gates whether the VAD loop is allowed to auto-start a recording -
// it's false while busy, while audio is playing, or while the learner has paused it manually.
let listening = false, pausedByUser = false, vadTimer = null;
let audioCtx = null, analyser = null, vadData = null, speechStartedAt = null, silenceStartedAt = null;
let fallbackNoticeShown = false;   // show the "local model" notice once per session, not every turn
const SPEECH_RMS = 0.02, SILENCE_MS = 900, MIN_SPEECH_MS = 400, RESUME_PAUSE_MS = 300;
function status(text) { $('status').textContent = text; }
function controls() {
  $('start').disabled = busy || active;
  $('restart').disabled = busy || active;
  $('rec').disabled = !active || busy || requestingMic;
  $('type').disabled = !active || busy || requestingMic;
  $('end').disabled = !active || busy || requestingMic;
  $('send').disabled = !active || busy;
  $('retry').disabled = busy;
}
function error(text) { $('error').textContent = text; $('error').hidden = false; }
function clearError() { $('error').hidden = true; }
let processingTimer = null;
function processing(url) {
  const messages = {
    '/session/start': ['Dein Tutor macht sich bereit …', 'Das Gespräch wird vorbereitet.'],
    '/session/end': ['Dein Rückblick entsteht …', 'Dein Tutor wertet die Sitzung aus.'],
    '/turn': ['Dein Tutor denkt nach …', 'Deine Aufnahme wird verarbeitet und die Antwort vorbereitet.'],
    '/turn/text': ['Dein Tutor denkt nach …', 'Deine Antwort wird verarbeitet.']
  };
  const [title, detail] = messages[url] || ['Bitte warten …', 'Die Anfrage wird verarbeitet.'];
  clearInterval(processingTimer);
  const began = Date.now();
  $('processing-title').textContent = title;
  $('processing-detail').textContent = detail;
  $('processing-time').textContent = '0 s';
  $('processing').hidden = false;
  $('status').classList.add('is-processing');
  status(title);
  $('record-title').textContent = 'Einen Moment bitte.';
  $('record-hint').textContent = 'Dein Tutor antwortet gleich. Du musst nichts drücken.';
  processingTimer = setInterval(() => {
    const seconds = Math.floor((Date.now() - began) / 1000);
    $('processing-time').textContent = `${seconds} s`;
    if (seconds >= 15) $('processing-detail').textContent = 'Es dauert etwas länger. Die Anfrage läuft noch.';
  }, 1000);
}
function processingDone() {
  clearInterval(processingTimer); processingTimer = null;
  $('processing').hidden = true;
  $('status').classList.remove('is-processing');
}
async function request(url, options) {
  processing(url);
  try {
    const response = await fetch(url, options);
    if (!response.ok) throw new Error('Die Anfrage konnte nicht abgeschlossen werden. Bitte versuche es erneut.');
    return await response.json();
  } finally { processingDone(); }
}

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}
function scrollLog() { $('log').scrollTop = $('log').scrollHeight; }
function play(url) {
  if (lastAudio) lastAudio.pause();
  lastAudio = new Audio(url);
  lastAudio.play().catch(() => error('Die Audioantwort konnte nicht abgespielt werden. Du kannst sie erneut anhören oder den Text lesen.'));
}
// Awaited version used for the tutor's reply audio, so the mic stays paused until the tutor
// is actually done talking. Never rejects: a playback glitch shouldn't block the conversation,
// it just resolves early.
function playAwait(url) {
  return new Promise(resolve => {
    if (!url) { resolve(); return; }
    if (lastAudio) lastAudio.pause();
    lastAudio = new Audio(url);
    lastAudio.onended = resolve; lastAudio.onerror = resolve;
    lastAudio.play().catch(resolve);
  });
}
function replayableTurn(className, label, text, audio) {
  const node = el('article', `turn ${className}`);
  node.append(el('div', 'turn-label', label), el('div', 'turn-body', text));
  if (audio) {
    const replay = el('button', 'replay', '↻ Noch einmal anhören');
    replay.onclick = () => play(audio); node.append(replay);
  }
  $('log').append(node); scrollLog();
  return node;
}
// One turn, one bubble, one audio clip. Any correction is already woven into the text itself
// as a brief recast + a follow-up question that requires reusing the corrected form (see
// tutor.py) - not a separate paced beat before the reply.
async function tutorTurn(text, audio, modelSource) {
  if (modelSource === 'local' && !fallbackNoticeShown) {
    fallbackNoticeShown = true;
    $('log').append(el('p', 'muted turn', 'Claude ist gerade nicht verfügbar - für den Rest dieser Sitzung antwortet ein lokales Modell. Antworten sind einfacher und weniger zuverlässig.'));
    scrollLog();
  }
  status(audio ? 'Dein Tutor spricht …' : 'Antwort erhalten');
  $('record-title').textContent = audio ? 'Dein Tutor spricht.' : 'Antwort erhalten.';
  $('record-hint').textContent = audio ? 'Hör kurz zu. Danach bist du wieder dran.' : 'Du kannst auf die Antwort reagieren.';
  replayableTurn('tutor', 'Dein Tutor', text, audio);
  await playAwait(audio);
}
async function startSession() {
  busy = true; clearError(); controls(); status('Dein Tutor macht sich bereit …');
  try {
    const s = await request('/session/start', {method:'POST'});
    active = true; turns = 0; $('record-title').textContent = 'Deine Stimme ist dran.';
    // Keep the start button mounted so it can also recover from a failed request.
    $('empty').hidden = true;
    $('log').querySelectorAll('.turn').forEach(node => node.remove());
    $('summary').hidden = true; $('restart').hidden = true; $('end').hidden = false;
    $('confirm').hidden = true; $('transcript').value = '';
    $('level').textContent = s.level;
    $('targets').replaceChildren(...s.targets.map(word => el('li', '', word)));
    if (!s.targets.length) $('targets').append(el('li', '', 'Heute frei sprechen'));
    $('turn-count').textContent = 'Dein Gespräch kann beginnen.';
    $('record-hint').textContent = 'Sprich einfach - ich reagiere automatisch, sobald du pausierst.';
    pausedByUser = false; $('rec').setAttribute('aria-pressed', 'false');
    fallbackNoticeShown = false;
    if (s.greeting_de) await tutorTurn(s.greeting_de, s.audio_url, s.model_source);
    else $('log').append(el('p','muted turn','Dein Tutor ist bereit. Beginne mit einem Thema deiner Wahl.'));
    await ensureMic();
    if (stream) { startVadLoop(); resumeListening(); }
  } catch (e) { error(e.message); status('Start nicht möglich'); }
  finally { busy = false; controls(); }
}
$('start').onclick = startSession;
$('restart').onclick = startSession;
$('end').onclick = async () => {
  busy = true; controls(); clearError(); status('Dein Rückblick entsteht …');
  listening = false;   // stop new auto-recordings starting while the request is in flight
  if (lastAudio) lastAudio.pause();
  try {
    const s = await request('/session/end', {method:'POST'});
    active = false; releaseMic(); $('confirm').hidden = true;
    const summary = $('summary'); summary.replaceChildren();
    summary.append(el('p', 'eyebrow', 'Dein Rückblick'), el('h2', '', `Geschafft. Dein geschätztes Niveau: ${s.level}`), el('p', '', s.summary));
    const scores = el('div','scores');
    for (const [key,label] of [['range','Ausdruck'],['accuracy','Genauigkeit'],['fluency','Flüssigkeit'],['coherence','Zusammenhang']]) scores.append(el('span','',`${label}: ${s[key]}/6`));
    summary.append(scores,el('p','muted','Eine Einschätzung dieser Sitzung, kein Prüfungsergebnis.')); summary.hidden = false;
    $('restart').hidden = false; $('end').hidden = true;
    $('record-title').textContent = 'Gut gemacht.'; $('record-hint').textContent = 'Dein nächstes Gespräch wartet auf dich.';
    status('Sitzung abgeschlossen');
  } catch (e) { error(e.message); status('Beenden fehlgeschlagen'); resumeListening(); }
  finally { busy = false; controls(); }
};
function showText(label = 'Deine Antwort auf Deutsch') {
  $('transcript-label').textContent = label; $('confirm').hidden = false; $('transcript').focus();
}
$('type').onclick = () => showText();
$('retry').onclick = () => { $('confirm').hidden = true; $('rec').focus(); };
// Every detected mistake is still logged to your progress data (see /progress). Only the one
// the tutor chose to build its reply around is ever surfaced live, woven into reply_de itself.
async function render(r) {
  const node = el('article', 'turn you');
  node.append(el('div', 'turn-label', 'Du'), el('div', 'turn-body', r.transcript));
  $('log').append(node);
  await tutorTurn(r.reply_de, r.audio_url, r.model_source);
  turns += 1; $('turn-count').textContent = `${turns} ${turns === 1 ? 'Antwort' : 'Antworten'} von dir`;
}
async function submitTurn(url, options) {
  busy = true; clearError(); controls(); status('Dein Tutor hört zu …');
  try {
    const r = await request(url, options);
    if (r.needs_retry) { showText('Nicht sicher verstanden. Schreibe deine Antwort oder nimm sie erneut auf.'); status('Bitte noch einmal'); return; }
    await render(r); $('confirm').hidden = true; $('transcript').value = '';
  } catch (e) { error(e.message); status('Bitte erneut versuchen'); }
  finally { busy = false; controls(); resumeListening(); }
}
$('confirm').onsubmit = async event => {
  event.preventDefault(); const text = $('transcript').value.trim();
  if (!text || busy || !active) return;
  await submitTurn('/turn/text', {method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({text})});
};

// --- continuous listening: mic opens once per session, a small volume-threshold loop
// detects when the learner starts and stops talking, and records only that span. ------------
async function ensureMic() {
  if (stream) return;
  requestingMic = true; controls(); status('Mikrofon wird vorbereitet …');
  try {
    if (!navigator.mediaDevices?.getUserMedia || !window.MediaRecorder) throw new Error('Die Aufnahme ist in diesem Browser nicht verfügbar. Schreibe deine Antwort stattdessen.');
    stream = await navigator.mediaDevices.getUserMedia({audio:true});
    audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    const source = audioCtx.createMediaStreamSource(stream);
    analyser = audioCtx.createAnalyser(); analyser.fftSize = 2048;
    vadData = new Uint8Array(analyser.fftSize);
    source.connect(analyser);
  } catch (e) {
    error(e.name === 'NotAllowedError' ? 'Bitte erlaube den Mikrofonzugriff oder schreibe deine Antwort.' : e.message);
    status('Mikrofon nicht verfügbar'); stream = null;
  } finally { requestingMic = false; controls(); }
}
function releaseMic() {
  stopVadLoop();
  if (audioCtx) { audioCtx.close(); audioCtx = null; }
  stream?.getTracks().forEach(t => t.stop()); stream = null; analyser = null; vadData = null;
}
function rms() {
  analyser.getByteTimeDomainData(vadData);
  let sum = 0;
  for (let i = 0; i < vadData.length; i++) { const v = (vadData[i] - 128) / 128; sum += v * v; }
  return Math.sqrt(sum / vadData.length);
}
function startVadLoop() {
  if (vadTimer || !analyser) return;
  vadTimer = setInterval(() => {
    if (!listening) return;
    // Browsers auto-suspend an AudioContext after inactivity or when the tab loses focus;
    // once suspended, the analyser keeps returning stale/silent data forever with no error -
    // this is the main cause of listening silently going dead. Cheap no-op when already running.
    if (audioCtx && audioCtx.state === 'suspended') { audioCtx.resume(); return; }
    const level = rms(), now = performance.now();
    if (level > SPEECH_RMS) {
      silenceStartedAt = null;
      if (!speechStartedAt) { speechStartedAt = now; beginAutoRecording(); }
    } else if (speechStartedAt) {
      if (!silenceStartedAt) silenceStartedAt = now;
      if (now - silenceStartedAt > SILENCE_MS && now - speechStartedAt > MIN_SPEECH_MS) finishAutoRecording();
    }
  }, 100);
}
function stopVadLoop() { if (vadTimer) { clearInterval(vadTimer); vadTimer = null; } speechStartedAt = null; silenceStartedAt = null; }
function resumeListening() {
  if (!active || pausedByUser || !stream) return;
  status('Ich höre zu …'); $('record-title').textContent = 'Ich höre zu.';
  $('record-hint').textContent = 'Sprich einfach - ich reagiere automatisch, sobald du pausierst.';
  setTimeout(() => { listening = true; }, RESUME_PAUSE_MS);
}
function beginAutoRecording() {
  const mime = ['audio/webm;codecs=opus','audio/webm','audio/mp4'].find(type => MediaRecorder.isTypeSupported(type));
  recorder = new MediaRecorder(stream, mime ? {mimeType:mime} : undefined);
  const chunks = [];
  recorder.ondataavailable = event => { if (event.data.size) chunks.push(event.data); };
  recorder.onstop = async () => {
    if (!active || !chunks.length) { busy = false; controls(); resumeListening(); return; }
    try {
      const type = recorder.mimeType;
      const fd = new FormData(); fd.append('audio', new Blob(chunks,{type}), type.includes('mp4') ? 'recording.mp4' : 'recording.webm');
      await submitTurn('/turn', {method:'POST', body: fd});
    } catch (e) {
      // Never let a packaging failure leave busy/listening stuck forever - that's the other
      // way "it stops listening" can happen silently.
      error(e.message); busy = false; controls(); resumeListening();
    }
  };
  recorder.start(); $('rec').classList.add('active');
  $('record-title').textContent = 'Ich höre dir zu.'; $('record-hint').textContent = 'Sprich weiter … ich sende automatisch, sobald du pausierst.';
  status('Aufnahme läuft');
}
function finishAutoRecording() {
  speechStartedAt = null; silenceStartedAt = null; listening = false;   // don't record over our own reply
  if (recorder?.state === 'recording') { busy = true; controls(); recorder.stop(); }
  $('rec').classList.remove('active');
}
// The mic button is a manual mute/resume toggle, not a hold-to-talk button - listening is
// automatic by default; this just lets the learner pause it (background noise, a pause to think).
$('rec').setAttribute('aria-label', 'Zuhören pausieren oder fortsetzen');
$('rec').onclick = () => {
  pausedByUser = !pausedByUser;
  $('rec').setAttribute('aria-pressed', String(pausedByUser));
  if (pausedByUser) {
    listening = false;
    $('record-title').textContent = 'Zuhören pausiert.'; $('record-hint').textContent = 'Klicke erneut, um weiter zuzuhören.';
    status('Pausiert');
  } else {
    resumeListening();
  }
};
