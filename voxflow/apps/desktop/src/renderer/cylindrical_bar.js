(() => {
  'use strict';

  // ─── DOM References ────────────────────────────────────────────────────────
  const barWrapper      = document.getElementById('bar-wrapper');
  const cylindricalBar  = document.getElementById('cylindrical-bar');
  const badgeState      = document.getElementById('badge-state');
  const badgeLang       = document.getElementById('badge-lang');
  const btnEmojiToggle  = document.getElementById('btn-emoji-toggle');
  const transcriptLive  = document.getElementById('transcript-live');
  const quickInput      = document.getElementById('quick-input');
  const btnToggle       = document.getElementById('btn-toggle');
  const btnKeyboard     = document.getElementById('btn-keyboard');
  const statusIndicator = document.getElementById('status-indicator');
  const waveBars        = document.querySelectorAll('.wave-bars .bar');

  // ─── State Management ─────────────────────────────────────────────────────
  // 'idle' | 'listening' | 'processing' | 'pasted'
  let state = 'idle';
  let liveBuffer = '';
  let pendingFinalUtterance = '';
  let history = [];
  let emojiInsertionEnabled = false;
  let lastProcessedUtteranceHash = '';
  let silenceTimer = null;
  let recognition = null;
  let isRecognitionRunning = false;
  let ws = null;
  let audioCtx = null;
  let analyser = null;
  let dataArray = null;
  let micStream = null;
  let animWaveId = null;
  let wavePhase = 0;
  const processedUtteranceIds = new Set();

  const WS_URL = (window.voxflow && window.voxflow.wsUrl) || 'ws://127.0.0.1:8000/ws';

  // ─── Part D: Reactive State Accessors & Display Text Renderer ───────────
  function renderDisplayText() {
    const historyPrefix = history.length > 0 ? history.join(' ') + ' ' : '';
    const activePortion = pendingFinalUtterance
      ? pendingFinalUtterance + (liveBuffer ? ' ' + liveBuffer : '')
      : liveBuffer;
    const displayText = (historyPrefix + activePortion).trim();
    if (transcriptLive && displayText) {
      transcriptLive.textContent = displayText;
    }
    return displayText;
  }

  function setLiveBuffer(interimText) {
    liveBuffer = (interimText || '').trim();
    return renderDisplayText();
  }

  function clearLiveBuffer() {
    liveBuffer = '';
    pendingFinalUtterance = '';
    renderDisplayText();
  }

  function commitToHistory(finalizedText) {
    if (!finalizedText || !finalizedText.trim()) return;
    history.push(finalizedText.trim());
    liveBuffer = '';
    pendingFinalUtterance = '';
    renderDisplayText();
  }

  // ─── Local Fallback Normalization Rules (if ML engine offline) ────────────
  const NORMS = [
    [/\bgonna\b/gi,   'going to'],
    [/\bwanna\b/gi,   'want to'],
    [/\bgotta\b/gi,   'got to'],
    [/\bkinda\b/gi,   'kind of'],
    [/\bsorta\b/gi,   'sort of'],
    [/\bdunno\b/gi,   "don't know"],
    [/\bcoulda\b/gi,  'could have'],
    [/\bwoulda\b/gi,  'would have'],
    [/\bshoulda\b/gi, 'should have'],
    [/\blemme\b/gi,   'let me'],
    [/\bgimme\b/gi,   'give me'],
    [/\byeah\b/gi,    'yes'],
    [/\bnope\b/gi,    'no'],
  ];
  const FILLERS = /\b(um|uh|er|ah|like|you\s+know|okay|ok|alright|right|so|well|basically|literally|honestly|actually|obviously|clearly|anyway|just|really|very|quite|rather)\b/gi;

  function localRefine(text) {
    let t = (text || '').trim();
    for (const [p, r] of NORMS) t = t.replace(p, r);
    t = t.replace(FILLERS, '').replace(/\s+/g, ' ').trim();
    if (t) t = t[0].toUpperCase() + t.slice(1);
    if (t && !/[.!?]$/.test(t) && t.split(' ').length >= 4) t += '.';
    return t || text;
  }

  // ─── WebSocket Connection ─────────────────────────────────────────────────
  function connectWebSocket() {
    try {
      ws = new WebSocket(WS_URL);
      ws.onopen = () => {
        console.log('[VoxFlow Desktop] Connected to ML Engine:', WS_URL);
      };
      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          if (data.type === 'transcript_interim') {
            if (state === 'listening' && data.text) {
              transcriptLive.textContent = data.text;
            }
          }
        } catch (_) {}
      };
      ws.onclose = () => {
        setTimeout(connectWebSocket, 3000);
      };
      ws.onerror = () => {
        // Handled silently
      };
    } catch (_) {
      setTimeout(connectWebSocket, 3000);
    }
  }

  connectWebSocket();

  // ─── State Switcher ───────────────────────────────────────────────────────
  function setState(newState, message) {
    state = newState;
    barWrapper.className = `bar-wrapper state-${newState}`;

    switch (newState) {
      case 'idle':
        badgeState.textContent = 'READY';
        transcriptLive.textContent = message || 'Say “VoxFlow” or click to talk';
        btnToggle.title = 'Start voice capture';
        break;

      case 'listening':
        badgeState.textContent = 'LISTENING';
        transcriptLive.textContent = message || 'Listening…';
        btnToggle.title = 'Finish recording & refine';
        break;

      case 'processing':
        badgeState.textContent = 'ADAPTING';
        transcriptLive.textContent = message || 'Refining with Adaptive Memory…';
        btnToggle.title = 'Processing…';
        break;

      case 'pasted':
        badgeState.textContent = 'PASTED';
        transcriptLive.textContent = message || '✓ Auto-pasted into active app';
        btnToggle.title = 'Success';
        break;
    }

    if (newState === 'listening') {
      initMicAnalyzer();
      startWaveAnimation();
    } else {
      stopMicAnalyzer();
      stopWaveAnimation();
    }
  }

  // ─── Audio Waveform Visualizer & Animation ────────────────────────────────
  async function initMicAnalyzer() {
    if (audioCtx) return;
    try {
      if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
        micStream = await navigator.mediaDevices.getUserMedia({ audio: true });
        audioCtx = new (window.AudioContext || window.webkitAudioContext)();
        const source = audioCtx.createMediaStreamSource(micStream);
        analyser = audioCtx.createAnalyser();
        analyser.fftSize = 64;
        source.connect(analyser);
        dataArray = new Uint8Array(analyser.frequencyBinCount);
      }
    } catch (err) {
      console.warn('[VoxFlow] Real mic analyzer unavailable; using synthetic wave', err);
    }
  }

  function stopMicAnalyzer() {
    if (micStream) {
      try {
        micStream.getTracks().forEach(t => t.stop());
      } catch (_) {}
    }
    if (audioCtx && audioCtx.state !== 'closed') {
      try {
        audioCtx.close();
      } catch (_) {}
    }
    micStream = null;
    audioCtx = null;
    analyser = null;
    dataArray = null;
  }

  function startWaveAnimation() {
    cancelAnimationFrame(animWaveId);

    function step() {
      if (state !== 'listening') return;

      if (analyser && dataArray) {
        analyser.getByteFrequencyData(dataArray);
        for (let i = 0; i < waveBars.length; i++) {
          const val = dataArray[i * 2 + 1] || 0;
          const height = Math.max(5, Math.min(22, 5 + (val / 255) * 20));
          waveBars[i].style.height = `${height}px`;
        }
      } else {
        // High-aesthetic organic sine wave fallback
        wavePhase += 0.16;
        for (let i = 0; i < waveBars.length; i++) {
          const h = 5 + Math.abs(Math.sin(wavePhase + i * 0.5)) * 14;
          waveBars[i].style.height = `${h}px`;
        }
      }

      animWaveId = requestAnimationFrame(step);
    }

    animWaveId = requestAnimationFrame(step);
  }

  function stopWaveAnimation() {
    cancelAnimationFrame(animWaveId);
    waveBars.forEach(b => {
      b.style.height = '6px';
    });
  }

  // ─── Wake-Word Pattern Matcher ────────────────────────────────────────────
  function extractWakeCommand(phrase) {
    const p = (phrase || '').toLowerCase();
    const regex = /\b(?:(?:hi|hey|ok|okay|hello)\s+)?(?:vox\s*flow|box\s*flow|vox\s*flo|fox\s*flow|voxflow|boss\s*flow)\b(.*)/i;
    const match = p.match(regex);
    if (match) {
      return match[1].trim();
    }
    return null;
  }

  // ─── Speech Recognition Engine ────────────────────────────────────────────
  const SpeechRec = window.SpeechRecognition || window.webkitSpeechRecognition;

  function initSpeechEngine() {
    if (!SpeechRec) {
      console.warn('[VoxFlow] Web Speech API not available in this environment.');
      transcriptLive.textContent = 'Click to speak or type thoughts';
      return;
    }
    startWakeListener();
  }

  function startWakeListener() {
    if (state !== 'idle' || !SpeechRec) return;

    try {
      if (recognition) recognition.abort();
    } catch (_) {}

    try {
      recognition = new SpeechRec();
      recognition.continuous = true;
      recognition.interimResults = true;
      recognition.lang = 'en-US';

      recognition.onresult = (e) => {
        if (state !== 'idle') return;
        for (let i = e.resultIndex; i < e.results.length; i++) {
          const transcript = e.results[i][0].transcript;
          const command = extractWakeCommand(transcript);
          if (command !== null) {
            console.log('[VoxFlow] Wake-word activated:', transcript);
            triggerWake(command);
            return;
          }
        }
      };

      recognition.onerror = () => {
        isRecognitionRunning = false;
        if (state === 'idle') setTimeout(startWakeListener, 1200);
      };

      recognition.onend = () => {
        isRecognitionRunning = false;
        if (state === 'idle') setTimeout(startWakeListener, 500);
      };

      recognition.start();
      isRecognitionRunning = true;
    } catch (_) {
      isRecognitionRunning = false;
    }
  }

  // ─── Trigger Active Listening ─────────────────────────────────────────────
  function triggerWake(initialText = '') {
    try {
      if (recognition) recognition.abort();
    } catch (_) {}

    pendingFinalUtterance = initialText ? initialText.trim() : '';
    liveBuffer = '';
    setState('listening');

    if (!SpeechRec) {
      // If Web Speech is missing in Electron, toggle quick text input or prompt
      showQuickInput();
      return;
    }

    try {
      recognition = new SpeechRec();
      recognition.continuous = true;
      recognition.interimResults = true;
      recognition.lang = 'en-US';

      recognition.onresult = (e) => {
        if (state !== 'listening') return;
        clearTimeout(silenceTimer);

        let interim = '';
        let newFinal = '';
        for (let i = e.resultIndex; i < e.results.length; i++) {
          const text = e.results[i][0].transcript;
          if (e.results[i].isFinal) {
            newFinal += text + ' ';
          } else {
            interim += text;
          }
        }

        if (newFinal.trim()) {
          pendingFinalUtterance = (pendingFinalUtterance ? pendingFinalUtterance + ' ' : '') + newFinal.trim();
        }
        const currentActive = setLiveBuffer(interim);

        // Tier 1 live preview via WebSocket
        if (ws && ws.readyState === WebSocket.OPEN && currentActive) {
          ws.send(JSON.stringify({ type: 'transcript_interim', text: currentActive }));
        }

        // Auto-finalize after 1.8 seconds of silence
        silenceTimer = setTimeout(finishListening, 1800);
      };

      if (pendingFinalUtterance) {
        transcriptLive.textContent = pendingFinalUtterance;
        silenceTimer = setTimeout(finishListening, 1800);
      }

      recognition.onerror = () => finishListening();
      recognition.onend = () => {
        if (state === 'listening') finishListening();
      };

      recognition.start();
      isRecognitionRunning = true;

      // 30s safety timeout
      setTimeout(() => {
        if (state === 'listening') finishListening();
      }, 30000);
    } catch (err) {
      console.warn('[VoxFlow] Failed to start active speech capture', err);
      showQuickInput();
    }
  }

  // ─── Finalize & Send to Adaptive Memory ────────────────────────────────────
  function finishListening() {
    clearTimeout(silenceTimer);
    try {
      if (recognition) recognition.abort();
    } catch (_) {}

    const utterance = (pendingFinalUtterance + (liveBuffer ? ' ' + liveBuffer : '')).trim();
    liveBuffer = '';
    pendingFinalUtterance = '';

    if (!utterance) {
      setState('idle');
      startWakeListener();
      return;
    }

    // Deduplication guard
    if (utterance === lastProcessedUtteranceHash) {
      return;
    }
    lastProcessedUtteranceHash = utterance;

    processUtterance(utterance);
  }

  function processUtterance(rawUtterance) {
    setState('processing');

    if (ws && ws.readyState === WebSocket.OPEN) {
      let resolved = false;

      const responseHandler = (event) => {
        try {
          const data = JSON.parse(event.data);
          if (data.type === 'transcript') {
            resolved = true;
            ws.removeEventListener('message', responseHandler);

            // Part G: Smart Listener Discard Signal Check
            if (data.intent_action === 'discard' || (!data.text && data.discarded_text)) {
              console.log('[VoxFlow SmartListener] Speech discarded by user:', data.discarded_text);
              clearLiveBuffer();
              if (transcriptLive) {
                transcriptLive.textContent = 'Cancelled';
              }
              setTimeout(() => {
                setState('idle');
                startWakeListener();
              }, 600);
              return;
            }

            const refined = data.text || localRefine(rawUtterance);
            if (!refined.trim()) {
              clearLiveBuffer();
              setState('idle');
              startWakeListener();
              return;
            }

            // Update language badge
            if (badgeLang && data.detected_language) {
              const langLabel = data.is_code_switched ? 'MIXED' : data.detected_language.toUpperCase().replace('_', '-');
              badgeLang.textContent = langLabel;
            }

            // Commit to history and auto-paste into focused window
            commitToHistory(refined);
            executeAutoPaste(refined);
          }

        } catch (_) {}
      };

      ws.addEventListener('message', responseHandler);
      ws.send(JSON.stringify({
        type: 'transcript',
        text: rawUtterance,
        enable_emoji: emojiInsertionEnabled,
      }));

      // Fallback timeout: 5 seconds
      setTimeout(() => {
        if (!resolved) {
          ws.removeEventListener('message', responseHandler);
          const fallback = localRefine(rawUtterance);
          commitToHistory(fallback);
          executeAutoPaste(fallback);
        }
      }, 5000);
    } else {
      // Local fallback
      setTimeout(() => {
        const fallback = localRefine(rawUtterance);
        commitToHistory(fallback);
        executeAutoPaste(fallback);
      }, 400);
    }
  }

  // ─── Auto-Paste Execution & UI Feedback ───────────────────────────────────
  async function executeAutoPaste(refinedText) {
    setState('pasted', refinedText);

    // Let the user see the refined result for 900ms
    await new Promise(r => setTimeout(r, 900));

    if (window.voxflow && window.voxflow.pasteText) {
      await window.voxflow.pasteText(refinedText);
    } else {
      try {
        await navigator.clipboard.writeText(refinedText);
      } catch (_) {}
    }

    // Return to idle state after paste
    setTimeout(() => {
      setState('idle');
      hideQuickInput();
      startWakeListener();
    }, 1200);
  }

  // ─── Quick Text Input Mode ────────────────────────────────────────────────
  function showQuickInput() {
    transcriptLive.style.display = 'none';
    quickInput.classList.add('active');
    quickInput.value = (pendingFinalUtterance || liveBuffer || '').trim();
    quickInput.focus();
  }

  function hideQuickInput() {
    quickInput.classList.remove('active');
    quickInput.value = '';
    transcriptLive.style.display = 'block';
  }

  quickInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') {
      const val = quickInput.value.trim();
      hideQuickInput();
      if (val) {
        processUtterance(val);
      } else {
        setState('idle');
        startWakeListener();
      }
    } else if (e.key === 'Escape') {
      hideQuickInput();
      setState('idle');
      startWakeListener();
    }
  });

  // ─── User Interactions ────────────────────────────────────────────────────
  cylindricalBar.addEventListener('click', (e) => {
    // Avoid triggering when buttons, input, or drag grip are clicked
    if (e.target.closest('#btn-toggle') ||
        e.target.closest('#btn-keyboard') ||
        e.target.closest('#btn-emoji-toggle') ||
        e.target.closest('#quick-input') ||
        e.target.closest('#drag-grip')) {
      return;
    }

    if (state === 'idle') {
      triggerWake();
    } else if (state === 'listening') {
      finishListening();
    }
  });

  if (btnEmojiToggle) {
    btnEmojiToggle.addEventListener('click', (e) => {
      e.stopPropagation();
      emojiInsertionEnabled = !emojiInsertionEnabled;
      btnEmojiToggle.classList.toggle('active', emojiInsertionEnabled);
      btnEmojiToggle.title = `Contextual Emojis: ${emojiInsertionEnabled ? 'ON' : 'OFF'}`;
      console.log('[VoxFlow] Contextual emoji mode:', emojiInsertionEnabled ? 'ON' : 'OFF');
    });
  }

  statusIndicator.addEventListener('click', (e) => {
    e.stopPropagation();
    if (state === 'idle') {
      triggerWake();
    } else if (state === 'listening') {
      finishListening();
    }
  });

  btnToggle.addEventListener('click', (e) => {
    e.stopPropagation();
    if (state === 'idle') {
      triggerWake();
    } else if (state === 'listening') {
      finishListening();
    }
  });

  btnKeyboard.addEventListener('click', (e) => {
    e.stopPropagation();
    if (quickInput.classList.contains('active')) {
      hideQuickInput();
    } else {
      showQuickInput();
    }
  });

  // Hotkey Trigger from Main Process (Ctrl+Shift+Space / Cmd+Shift+Space)
  if (window.voxflow && window.voxflow.onWakeCommand) {
    window.voxflow.onWakeCommand(() => {
      if (state === 'idle') {
        triggerWake();
      } else if (state === 'listening') {
        finishListening();
      }
    });
  }

  // ─── Boot ─────────────────────────────────────────────────────────────────
  setState('idle');
  initSpeechEngine();
})();
