// True Tone SIH - Cyber Defense Dashboard Controller

let currentSampleType = 'spoof';
let currentLiveMode = 'spoof';
let selectedFile = null;
let liveSocket = null;
let callTimerInterval = null;
let callSeconds = 0;

document.addEventListener('DOMContentLoaded', () => {
  refreshAllData();
  populateAudioDevices();
  // Poll telemetry and blocklist every 3.5 seconds
  setInterval(refreshAllData, 3500);

  setupDropzone();
});

// Enumerate local audio devices (Microphones, Headsets, Stereo Mix loopback)
async function populateAudioDevices() {
  const select = document.getElementById('audioDeviceSelect');
  if (!select || !navigator.mediaDevices?.enumerateDevices) return;

  try {
    const devices = await navigator.mediaDevices.enumerateDevices();
    const audioInputs = devices.filter(d => d.kind === 'audioinput');
    if (audioInputs.length > 0) {
      select.innerHTML = '';
      const defaultOpt = document.createElement('option');
      defaultOpt.value = 'default';
      defaultOpt.innerText = '🎙️ Default System Microphone / Line-In';
      select.appendChild(defaultOpt);

      audioInputs.forEach((d, idx) => {
        if (!d.deviceId || d.deviceId === 'default' || d.deviceId === 'communications') return;
        const label = d.label || `Audio Device ${idx + 1}`;
        const isStereoMix = label.toLowerCase().includes('stereo mix') || label.toLowerCase().includes('what u hear');
        const opt = document.createElement('option');
        opt.value = d.deviceId;
        opt.innerText = (isStereoMix ? '🎛️ ' : '🎙️ ') + label;
        select.appendChild(opt);
      });
    }
  } catch (e) {
    console.log('Audio device enumeration skipped:', e);
  }
}

// Tab Navigation
function switchTab(mode) {
  const tabLive = document.getElementById('tabLiveCall');
  const tabSim = document.getElementById('tabSimulator');
  const tabUp = document.getElementById('tabUpload');
  
  const viewLive = document.getElementById('viewLiveCall');
  const viewSim = document.getElementById('viewSimulator');
  const viewUp = document.getElementById('viewUpload');

  [tabLive, tabSim, tabUp].forEach(t => t && t.classList.remove('active'));
  [viewLive, viewSim, viewUp].forEach(v => v && (v.style.display = 'none'));

  if (mode === 'livecall') {
    tabLive.classList.add('active');
    viewLive.style.display = 'block';
  } else if (mode === 'simulator') {
    tabSim.classList.add('active');
    viewSim.style.display = 'block';
  } else {
    tabUp.classList.add('active');
    viewUp.style.display = 'block';
  }
}

let selectedAudioSource = 'mic';
let simVoiceType = 'spoof';
let audioContext = null;
let mediaStream = null;
let analyserNode = null;
let scriptProcessor = null;
let pcmBuffer = [];
let animFrameId = null;

// Audio Source Selection
function selectAudioSource(source) {
  selectedAudioSource = source;
  document.getElementById('sourceMic').classList.toggle('selected', source === 'mic');
  document.getElementById('sourceTab').classList.toggle('selected', source === 'tab');
  document.getElementById('sourceSim').classList.toggle('selected', source === 'sim');

  const simPicker = document.getElementById('simulatedSubPicker');
  if (simPicker) {
    simPicker.style.display = source === 'sim' ? 'block' : 'none';
  }

  const tabGuide = document.getElementById('tabAudioGuide');
  if (tabGuide) {
    tabGuide.style.display = source === 'tab' ? 'block' : 'none';
  }

  const micPicker = document.getElementById('micDevicePicker');
  if (micPicker) {
    micPicker.style.display = source === 'mic' ? 'block' : 'none';
  }
}

function setSimVoiceType(type) {
  simVoiceType = type;
  document.getElementById('simSpoofBtn').classList.toggle('active', type === 'spoof');
  document.getElementById('simHumanBtn').classList.toggle('active', type === 'bonafide');
}

// Downsample audio buffer from source sample rate to target sample rate
// Uses linear interpolation for clean downsampling (e.g., 48kHz → 16kHz)
function downsampleBuffer(buffer, fromRate, toRate) {
  if (fromRate === toRate) return buffer;
  if (fromRate < toRate) {
    console.warn('[TrueTone] Cannot upsample, returning original buffer');
    return buffer;
  }
  const ratio = fromRate / toRate;
  const newLength = Math.round(buffer.length / ratio);
  const result = new Array(newLength);
  for (let i = 0; i < newLength; i++) {
    const srcIdx = i * ratio;
    const srcIdxFloor = Math.floor(srcIdx);
    const srcIdxCeil = Math.min(srcIdxFloor + 1, buffer.length - 1);
    const frac = srcIdx - srcIdxFloor;
    result[i] = buffer[srcIdxFloor] * (1 - frac) + buffer[srcIdxCeil] * frac;
  }
  return result;
}

// 16-bit PCM WAV Encoder in Pure JS (16kHz Mono)
function encodeWAV(samples, sampleRate = 16000) {
  const buffer = new ArrayBuffer(44 + samples.length * 2);
  const view = new DataView(buffer);

  function writeString(v, offset, str) {
    for (let i = 0; i < str.length; i++) {
      v.setUint8(offset + i, str.charCodeAt(i));
    }
  }

  writeString(view, 0, 'RIFF');
  view.setUint32(4, 36 + samples.length * 2, true);
  writeString(view, 8, 'WAVE');
  writeString(view, 12, 'fmt ');
  view.setUint32(16, 16, true);          // SubChunk1Size (16 for PCM)
  view.setUint16(20, 1, true);           // AudioFormat (1 = PCM)
  view.setUint16(22, 1, true);           // NumChannels (1 = Mono)
  view.setUint32(24, sampleRate, true);  // SampleRate
  view.setUint32(28, sampleRate * 2, true); // ByteRate (SampleRate * NumChannels * BitsPerSample/8)
  view.setUint16(32, 2, true);           // BlockAlign
  view.setUint16(34, 16, true);          // BitsPerSample (16 bits)
  writeString(view, 36, 'data');
  view.setUint32(40, samples.length * 2, true); // SubChunk2Size

  let offset = 44;
  for (let i = 0; i < samples.length; i++, offset += 2) {
    const s = Math.max(-1, Math.min(1, samples[i]));
    view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
  }
  return new Uint8Array(buffer);
}

// Real-Time Audio Signal VU Meter
function updateAudioSignalMeter(rms) {
  const meter = document.getElementById('audioSignalMeter');
  const label = document.getElementById('audioSignalText');
  if (!meter || !label) return;

  // Map RMS to percentage (0.15 is loud speech)
  const pct = Math.min(100, Math.round(rms * 500));
  meter.style.width = `${pct}%`;

  if (rms < 0.004) {
    label.innerText = 'SILENCE / WAITING FOR VOICE';
    label.style.color = 'var(--text-muted)';
  } else if (rms < 0.020) {
    label.innerText = 'AMBIENT BACKGROUND SOUND';
    label.style.color = '#94a3b8';
  } else {
    const db = Math.round(20 * Math.log10(rms + 1e-6));
    label.innerText = `VOICE ACTIVE (${db} dB)`;
    label.style.color = '#00f2fe';
  }
}

// Canvas Waveform Oscilloscope Visualizer
function startCanvasVisualizer() {
  const canvas = document.getElementById('liveAudioCanvas');
  if (!canvas || !analyserNode) return;
  canvas.style.display = 'block';

  // Fix internal coordinate resolution to match DOM element
  canvas.width = canvas.offsetWidth || 600;
  canvas.height = 60;

  const ctx = canvas.getContext('2d');
  const bufferLength = analyserNode.frequencyBinCount;
  const dataArray = new Uint8Array(bufferLength);

  function render() {
    animFrameId = requestAnimationFrame(render);
    analyserNode.getByteTimeDomainData(dataArray);

    ctx.fillStyle = 'rgba(10, 15, 25, 0.4)';
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    ctx.lineWidth = 2;
    ctx.strokeStyle = '#00f2fe';
    ctx.beginPath();

    const sliceWidth = canvas.width * 1.0 / bufferLength;
    let x = 0;

    for (let i = 0; i < bufferLength; i++) {
      const v = dataArray[i] / 128.0;
      const y = v * canvas.height / 2;

      if (i === 0) {
        ctx.moveTo(x, y);
      } else {
        ctx.lineTo(x, y);
      }
      x += sliceWidth;
    }

    ctx.lineTo(canvas.width, canvas.height / 2);
    ctx.stroke();
  }

  render();
}

// Start Real-Time Live In-Call Audio Interception
async function startLiveIpCall() {
  const callerIp = document.getElementById('liveCallerIp').value.trim() || '198.51.100.110';
  const statusText = document.getElementById('callStatusText');
  const pulseDot = document.getElementById('callPulseDot');
  const timer = document.getElementById('callTimer');
  const banner = document.getElementById('liveDetectionBanner');
  const bannerIcon = document.getElementById('liveBannerIcon');
  const bannerMsg = document.getElementById('liveBannerMsg');
  const btnStart = document.getElementById('btnStartCall');
  const btnEnd = document.getElementById('btnEndCall');

  // STEP 1: Prompt for and acquire Media Stream IMMEDIATELY while user gesture is active
  if (selectedAudioSource === 'mic' || selectedAudioSource === 'tab') {
    statusText.innerText = 'INITIALIZING AUDIO CAPTURE DEVICE...';
    pulseDot.className = 'call-pulse-dot active';

    try {
      if (selectedAudioSource === 'mic') {
        const deviceId = document.getElementById('audioDeviceSelect')?.value;
        const audioConstraints = {
          channelCount: 1,
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true
        };
        if (deviceId && deviceId !== 'default') {
          audioConstraints.deviceId = { exact: deviceId };
        }
        mediaStream = await navigator.mediaDevices.getUserMedia({ audio: audioConstraints });
      } else {
        // Tab / System audio: capture the remote caller's voice from the call tab
        // CRITICAL: video track MUST stay alive — stopping it kills audio on Chrome
        try {
          mediaStream = await navigator.mediaDevices.getDisplayMedia({
            video: {
              width: { ideal: 1 },
              height: { ideal: 1 },
              frameRate: { ideal: 1 }
            },
            audio: {
              echoCancellation: false,
              noiseSuppression: false,
              autoGainControl: false,
              suppressLocalAudioPlayback: false
            },
            systemAudio: 'include',
            selfBrowserSurface: 'exclude',
            surfaceSwitching: 'include',
            preferCurrentTab: false
          });
        } catch (displayErr) {
          console.warn('Tab audio sharing cancelled or rejected:', displayErr);
          statusText.innerText = 'CALL GUARDIAN: STANDBY';
          pulseDot.className = 'call-pulse-dot';
          showToast('Call audio sharing was cancelled or denied.', 'warning');
          return;
        }

        // Validate that an audio track was actually selected
        const audioTracks = mediaStream.getAudioTracks();
        console.log('[TrueTone] getDisplayMedia tracks:', {
          totalTracks: mediaStream.getTracks().length,
          audioTracks: audioTracks.length,
          videoTracks: mediaStream.getVideoTracks().length,
          audioTrackSettings: audioTracks.length > 0 ? audioTracks[0].getSettings() : null,
          audioTrackLabel: audioTracks.length > 0 ? audioTracks[0].label : 'NONE'
        });

        if (!audioTracks || audioTracks.length === 0) {
          mediaStream.getTracks().forEach(track => track.stop());
          statusText.innerText = 'CALL GUARDIAN: STANDBY';
          pulseDot.className = 'call-pulse-dot';
          openTabAudioHelpModal();
          showToast('No audio track selected! Please check "Also share tab audio" or "Share system audio".', 'danger');
          return;
        }

        // DO NOT stop video tracks! On Chrome, getDisplayMedia ties audio+video
        // together — stopping video will TERMINATE the audio track too, causing
        // all-zero sample buffers. The video is already minimal (1x1 @ 1fps).

        audioTracks[0].onended = () => {
          console.log('In-call audio track ended by user.');
          endLiveIpCall();
          showToast('Call audio stream stopped.', 'info');
        };
      }

      // Initialize Web Audio API at the NATIVE system sample rate
      // CRITICAL: Do NOT force sampleRate to 16000! Chrome's getDisplayMedia
      // audio runs at the system rate (48000Hz). Forcing 16kHz causes the
      // MediaStreamSource to produce all-zero buffers because Chrome cannot
      // resample the incoming capture stream to a non-native context rate.
      const AudioCtxClass = window.AudioContext || window.webkitAudioContext;
      audioContext = new AudioCtxClass();
      if (audioContext.state === 'suspended') {
        await audioContext.resume();
      }
      window._activeAudioContext = audioContext;
      const nativeSr = audioContext.sampleRate; // Typically 44100 or 48000
      console.log('[TrueTone] AudioContext sampleRate:', nativeSr, 'state:', audioContext.state);

      // Connect source node from media stream (use ORIGINAL stream, not a clone)
      const sourceNode = audioContext.createMediaStreamSource(mediaStream);
      window._activeSourceNode = sourceNode;

      // Analyser Node for Oscilloscope
      analyserNode = audioContext.createAnalyser();
      analyserNode.fftSize = 512;
      sourceNode.connect(analyserNode);
      startCanvasVisualizer();

      // Detect actual channel count from the audio stream
      const trackSettings = mediaStream.getAudioTracks()[0]?.getSettings() || {};
      const streamChannels = trackSettings.channelCount || 1;
      console.log('[TrueTone] Audio track channel count:', streamChannels);

      // ScriptProcessorNode — match channel count to stream (1 for tab audio, 1-2 for mic)
      const procChannels = Math.min(streamChannels, 2);
      // Accumulate 2 seconds of audio at the NATIVE rate, then downsample to 16kHz
      const targetSeconds = 2.0;
      const targetChunkSamples = Math.round(nativeSr * targetSeconds);
      const overlapSamples = Math.round(nativeSr * 1.0);
      scriptProcessor = audioContext.createScriptProcessor(4096, procChannels, 1);
      window._activeScriptProcessor = scriptProcessor;
      pcmBuffer = [];

      // Audio flow diagnostics
      let _totalSamplesReceived = 0;
      let _nonZeroSamples = 0;
      let _diagInterval = null;
      _diagInterval = setInterval(() => {
        const pct = _totalSamplesReceived > 0 ? ((_nonZeroSamples / _totalSamplesReceived) * 100).toFixed(1) : '0.0';
        console.log(`[TrueTone AudioDiag] sr=${nativeSr}, Total samples: ${_totalSamplesReceived}, Non-zero: ${_nonZeroSamples} (${pct}%), Buffer: ${pcmBuffer.length}`);
      }, 5000);
      window._audioDiagInterval = _diagInterval;

      scriptProcessor.onaudioprocess = (e) => {
        const inBuf = e.inputBuffer;
        const numCh = inBuf.numberOfChannels;
        const ch0 = inBuf.getChannelData(0);

        // Mute output to prevent local feedback/echo
        for (let c = 0; c < e.outputBuffer.numberOfChannels; c++) {
          e.outputBuffer.getChannelData(c).fill(0);
        }

        // Mix down channels (for mic stereo) or use mono directly (for tab audio)
        let sumSquares = 0;
        if (numCh > 1) {
          const ch1 = inBuf.getChannelData(1);
          for (let i = 0; i < ch0.length; i++) {
            const sample = (ch0[i] + ch1[i]) * 0.5;
            pcmBuffer.push(sample);
            sumSquares += sample * sample;
            _totalSamplesReceived++;
            if (Math.abs(sample) > 1e-6) _nonZeroSamples++;
          }
        } else {
          for (let i = 0; i < ch0.length; i++) {
            pcmBuffer.push(ch0[i]);
            sumSquares += ch0[i] * ch0[i];
            _totalSamplesReceived++;
            if (Math.abs(ch0[i]) > 1e-6) _nonZeroSamples++;
          }
        }

        const rms = Math.sqrt(sumSquares / ch0.length);
        updateAudioSignalMeter(rms);

        if (pcmBuffer.length >= targetChunkSamples) {
          const chunk = pcmBuffer.slice(0, targetChunkSamples);
          pcmBuffer = pcmBuffer.slice(overlapSamples);

          // Downsample from native rate to 16kHz before sending to backend
          const downsampled = downsampleBuffer(chunk, nativeSr, 16000);

          // Calculate chunk peak amplitude and RMS energy
          let peakAmp = 0;
          let sumSquaresChunk = 0;
          for (let i = 0; i < downsampled.length; i++) {
            const abs = Math.abs(downsampled[i]);
            if (abs > peakAmp) peakAmp = abs;
            sumSquaresChunk += downsampled[i] * downsampled[i];
          }
          const chunkRms = Math.sqrt(sumSquaresChunk / downsampled.length);

          // Speech-gated normalization:
          // When active speech is present, normalize to standard 0.85 reference amplitude
          // without amplifying near-silent background pause
          if (peakAmp >= 0.03 && chunkRms >= 0.005) {
            const normGain = Math.min(10.0, 0.85 / peakAmp);
            for (let i = 0; i < downsampled.length; i++) {
              downsampled[i] *= normGain;
            }
          }

          const wavBytes = encodeWAV(downsampled, 16000);
          if (liveSocket && liveSocket.readyState === WebSocket.OPEN) {
            liveSocket.send(wavBytes);
            console.log(`[TrueTone] Sent WAV chunk: ${wavBytes.length} bytes, ${downsampled.length} samples @ 16kHz, peak=${peakAmp.toFixed(4)}, chunkRMS=${chunkRms.toFixed(4)}`);
          }
        }
      };

      // Dynamics control and gain node for clean capture without clipping:
      // Tab/system audio gets gentle 2.5x gain (+8dB) through a DynamicsCompressorNode
      // which automatically prevents hard clipping on loud speech while keeping quiet speech clear.
      const boostGain = audioContext.createGain();
      const compressor = audioContext.createDynamicsCompressor();
      compressor.threshold.value = -12; // dB
      compressor.knee.value = 8;
      compressor.ratio.value = 4;
      compressor.attack.value = 0.003;
      compressor.release.value = 0.25;

      if (selectedAudioSource === 'tab') {
        boostGain.gain.value = 2.5; // Gentle +8dB boost (safe, non-clipping)
        console.log('[TrueTone] Applied 2.5x gentle gain boost with dynamics compressor for tab audio');
      } else {
        boostGain.gain.value = 1.0;
      }

      // Connect: source -> boostGain -> compressor -> scriptProcessor -> silentOutput -> destination
      // Also: source -> analyser (already connected above)
      const silentGain = audioContext.createGain();
      silentGain.gain.value = 0.0;
      sourceNode.connect(boostGain);
      boostGain.connect(compressor);
      compressor.connect(scriptProcessor);
      scriptProcessor.connect(silentGain);
      silentGain.connect(audioContext.destination);

    } catch (err) {
      console.warn('Audio capture setup failed:', err);
      statusText.innerText = 'CALL GUARDIAN: STANDBY';
      pulseDot.className = 'call-pulse-dot';
      showToast('Audio capture failed: ' + (err.message || err), 'danger');
      return;
    }
  }

  // STEP 2: Transition UI to active state
  btnStart.style.display = 'none';
  btnEnd.style.display = 'inline-flex';
  statusText.innerText = 'CONNECTING TO DEFENSE GATEWAY...';
  pulseDot.className = 'call-pulse-dot active';
  banner.className = 'live-detection-banner';
  bannerIcon.innerText = '⏳';
  bannerMsg.innerText = 'Connecting to True Tone Security Gateway...';

  callSeconds = 0;
  clearInterval(callTimerInterval);
  callTimerInterval = setInterval(() => {
    callSeconds++;
    const m = String(Math.floor(callSeconds / 60)).padStart(2, '0');
    const s = String(callSeconds % 60).padStart(2, '0');
    timer.innerText = `${m}:${s}`;
  }, 1000);

  // STEP 3: Connect WebSocket to Security Gateway
  const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  const wsUrl = `${wsProtocol}//${window.location.host}/ws/live-call?caller_ip=${encodeURIComponent(callerIp)}`;
  liveSocket = new WebSocket(wsUrl);

  liveSocket.onopen = () => {
    statusText.innerText = 'BACKGROUND GUARDIAN ACTIVE: ' + callerIp;
    bannerIcon.innerText = '🎙️';
    bannerMsg.innerText = 'Sniffing in-call voice stream in background... Continuous acoustic inspection running.';

    if (selectedAudioSource === 'sim') {
      sendSimulatedFeed();
    }
  };

  liveSocket.onmessage = (event) => {
    const data = JSON.parse(event.data);
    handleLiveCallEvent(data);
  };

  liveSocket.onclose = (e) => {
    endLiveIpCall();
    if (e.code === 4403) {
      statusText.innerText = '🚨 CALL DROPPED: CALLER IP BANNED';
      pulseDot.className = 'call-pulse-dot danger';
    } else {
      statusText.innerText = 'CALL GUARDIAN: STANDBY';
      pulseDot.className = 'call-pulse-dot';
    }
    refreshAllData();
  };

  liveSocket.onerror = (err) => {
    console.error('WebSocket Error:', err);
    statusText.innerText = 'GATEWAY CONNECTION ERROR';
  };
}

function sendSimulatedFeed() {
  const wave = document.getElementById('waveVisualizer');
  if (wave) wave.style.display = 'flex';
  wave.className = 'wave-visualizer active';

  setTimeout(() => {
    if (liveSocket && liveSocket.readyState === WebSocket.OPEN) {
      liveSocket.send(JSON.stringify({
        command: 'simulate_chunk',
        sample_type: simVoiceType
      }));
    }
  }, 1200);
}

function handleLiveCallEvent(data) {
  const statusText = document.getElementById('callStatusText');
  const pulseDot = document.getElementById('callPulseDot');
  const banner = document.getElementById('liveDetectionBanner');
  const bannerIcon = document.getElementById('liveBannerIcon');
  const bannerMsg = document.getElementById('liveBannerMsg');

  if (data.event === 'AMBIENT_LISTENING') {
    statusText.innerText = '🎙️ LISTENING · WAITING FOR VOICE';
    pulseDot.className = 'call-pulse-dot active';
    banner.className = 'live-detection-banner';
    bannerIcon.innerText = '🎙️';
    bannerMsg.innerHTML = `<strong>BACKGROUND GUARDIAN LISTENING:</strong> Inter-speech pause / ambient sound detected. Continuous monitoring active on <code>${data.caller_ip}</code>...`;
  } else if (data.event === 'THREAT_SUSPECTED') {
    statusText.innerText = '⚠️ ELEVATED RISK · VERIFYING BUFFER...';
    pulseDot.className = 'call-pulse-dot danger';
    banner.className = 'live-detection-banner warning';
    bannerIcon.innerText = '⚠️';
    bannerMsg.innerHTML = `<strong>ANALYZING CALLER PHONEMES:</strong> Elevated spoof signature detected (${data.spoof_percentage}%). Cross-verifying next frame before threat enforcement...`;
  } else if (data.event === 'CALL_TERMINATED_IP_BANNED') {
    statusText.innerText = '🚨 DEEPFAKE DETECTED! CALL DROPPED & IP BANNED';
    pulseDot.className = 'call-pulse-dot danger';
    banner.className = 'live-detection-banner danger';
    bannerIcon.innerText = '🛑';
    bannerMsg.innerHTML = `<strong>EMERGENCY THREAT NEUTRALIZED!</strong> AI Voice Deepfake detected (${data.spoof_percentage}% risk). Call session killed and origin IP <code>${data.caller_ip}</code> quarantined at Firewall! <button class="btn-unblock" style="margin-left: 0.75rem; padding: 0.25rem 0.6rem; vertical-align: middle; background: rgba(16, 185, 129, 0.25); color: #10b981; border: 1px solid #10b981;" onclick="unblockIp('${data.caller_ip}')">🔓 Unblock Now</button>`;

    // Render result card
    renderResult({
      status: 'SUCCESS',
      caller_ip: data.caller_ip,
      ai_detection: data.detection
    });

    playThreatAlertSound();
    endLiveIpCall();
  } else if (data.event === 'VOICE_VERIFIED_AUTHENTIC') {
    statusText.innerText = '✅ REAL HUMAN VERIFIED &middot; IN-CALL GUARDIAN ACTIVE';
    pulseDot.className = 'call-pulse-dot active';
    banner.className = 'live-detection-banner safe';
    bannerIcon.innerText = '✅';
    bannerMsg.innerHTML = `<strong>AUTHENTIC HUMAN VOICE:</strong> Natural vocal tract verified (${data.spoof_percentage}% spoof risk). In-call monitoring continuing silently in background.`;

    renderResult({
      status: 'SUCCESS',
      caller_ip: data.caller_ip,
      ai_detection: data.detection
    });
  } else if (data.event === 'CALL_REJECTED') {
    statusText.innerText = '🚫 CALL REJECTED: IP IS BLACKLISTED';
    pulseDot.className = 'call-pulse-dot danger';
    banner.className = 'live-detection-banner danger';
    bannerIcon.innerText = '🛑';
    bannerMsg.innerText = data.message;
  }
}

function playThreatAlertSound() {
  try {
    const ctx = new (window.AudioContext || window.webkitAudioContext)();
    const osc = ctx.createOscillator();
    const gain = ctx.createGain();
    osc.type = 'sawtooth';
    osc.frequency.setValueAtTime(880, ctx.currentTime);
    osc.frequency.exponentialRampToValueAtTime(220, ctx.currentTime + 0.4);
    gain.gain.setValueAtTime(0.3, ctx.currentTime);
    gain.gain.linearRampToValueAtTime(0.01, ctx.currentTime + 0.4);
    osc.connect(gain);
    gain.connect(ctx.destination);
    osc.start();
    osc.stop(ctx.currentTime + 0.4);
  } catch (e) {}
}

function endLiveIpCall() {
  if (liveSocket && liveSocket.readyState === WebSocket.OPEN) {
    try {
      liveSocket.send(JSON.stringify({ command: 'end_call' }));
      liveSocket.close();
    } catch (e) {}
  }
  clearInterval(callTimerInterval);
  cancelAnimationFrame(animFrameId);

  // Clear audio diagnostics interval
  if (window._audioDiagInterval) {
    clearInterval(window._audioDiagInterval);
    window._audioDiagInterval = null;
  }

  if (scriptProcessor) {
    try { scriptProcessor.disconnect(); } catch (e) {}
    scriptProcessor = null;
    window._activeScriptProcessor = null;
  }
  if (mediaStream) {
    try { mediaStream.getTracks().forEach(track => track.stop()); } catch (e) {}
    mediaStream = null;
  }
  if (audioContext && audioContext.state !== 'closed') {
    try { audioContext.close(); } catch (e) {}
    audioContext = null;
    window._activeAudioContext = null;
  }

  const canvas = document.getElementById('liveAudioCanvas');
  if (canvas) canvas.style.display = 'none';

  const meter = document.getElementById('audioSignalMeter');
  if (meter) meter.style.width = '0%';
  const label = document.getElementById('audioSignalText');
  if (label) {
    label.innerText = 'STANDBY / NO SIGNAL';
    label.style.color = 'var(--text-muted)';
  }

  document.getElementById('btnStartCall').style.display = 'inline-flex';
  document.getElementById('btnEndCall').style.display = 'none';
  document.getElementById('callStatusText').innerText = 'CALL GUARDIAN: STANDBY';
  document.getElementById('callPulseDot').className = 'call-pulse-dot';
}

// Sample Selection
function selectSample(type) {
  currentSampleType = type;
  const cardSpoof = document.getElementById('sampleCardSpoof');
  const cardBonafide = document.getElementById('sampleCardBonafide');

  if (type === 'spoof') {
    cardSpoof.classList.add('selected');
    cardBonafide.classList.remove('selected');
  } else {
    cardBonafide.classList.add('selected');
    cardSpoof.classList.remove('selected');
  }
}

// File Drag & Drop
function setupDropzone() {
  const dropzone = document.getElementById('audioDropzone');
  ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
    dropzone.addEventListener(eventName, preventDefaults, false);
  });

  function preventDefaults(e) {
    e.preventDefault();
    e.stopPropagation();
  }

  dropzone.addEventListener('dragover', () => dropzone.classList.add('dragover'));
  dropzone.addEventListener('dragleave', () => dropzone.classList.remove('dragover'));
  dropzone.addEventListener('drop', (e) => {
    dropzone.classList.remove('dragover');
    const dt = e.dataTransfer;
    if (dt.files && dt.files.length > 0) {
      selectedFile = dt.files[0];
      document.getElementById('selectedFileName').innerText = `Selected: ${selectedFile.name} (${(selectedFile.size / 1024).toFixed(1)} KB)`;
    }
  });
}

function handleFileSelected(event) {
  if (event.target.files && event.target.files.length > 0) {
    selectedFile = event.target.files[0];
    document.getElementById('selectedFileName').innerText = `Selected: ${selectedFile.name} (${(selectedFile.size / 1024).toFixed(1)} KB)`;
  }
}

// Execute Simulation
async function runSimulation() {
  const callerIp = document.getElementById('simCallerIp').value.trim() || '198.51.100.42';
  const btn = document.getElementById('btnRunSim');
  btn.disabled = true;
  btn.innerHTML = '<span>⏳</span><span>Analyzing Voice Stream with ML Model...</span>';

  try {
    const res = await fetch('/api/simulate-call', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        caller_ip: callerIp,
        sample_type: currentSampleType
      })
    });

    const data = await res.json();
    renderResult(data);
    await refreshAllData();
  } catch (err) {
    alert('Simulation request failed: ' + err);
  } finally {
    btn.disabled = false;
    btn.innerHTML = '<span>⚡</span><span>Transmit Voice Stream & Test Defense</span>';
  }
}

// Upload & Inspect Custom Audio
async function uploadAndInspectAudio() {
  if (!selectedFile) {
    alert('Please select or drop an audio file (.flac, .wav, .mp3) first.');
    return;
  }

  const callerIp = document.getElementById('uploadCallerIp').value.trim() || '203.0.113.88';
  const btn = document.getElementById('btnUploadInspect');
  btn.disabled = true;
  btn.innerHTML = '<span>⏳</span><span>Running Neural Inference & LFCC Extractor...</span>';

  const formData = new FormData();
  formData.append('file', selectedFile);
  formData.append('caller_ip', callerIp);

  try {
    const res = await fetch('/api/detect', {
      method: 'POST',
      body: formData
    });

    const data = await res.json();
    renderResult(data);
    await refreshAllData();
  } catch (err) {
    alert('Audio inspection failed: ' + err);
  } finally {
    btn.disabled = false;
    btn.innerHTML = '<span>🔍</span><span>Analyze Audio with ML Model</span>';
  }
}

// Render Result Box
function renderResult(data) {
  const resultBox = document.getElementById('resultBox');
  const verdictText = document.getElementById('verdictText');
  const actionPill = document.getElementById('actionPill');
  const spoofPercentLabel = document.getElementById('spoofPercentLabel');
  const meterFill = document.getElementById('meterFill');
  const riskLevelVal = document.getElementById('riskLevelVal');
  const latencyVal = document.getElementById('latencyVal');
  const ipActionVal = document.getElementById('ipActionVal');

  resultBox.classList.add('active');

  // Handle case where caller was already blocked
  if (data.status === 'FORBIDDEN') {
    resultBox.className = 'result-box active danger';
    verdictText.innerText = 'CALL REJECTED: IP BANNED';
    actionPill.className = 'threat-action-pill';
    actionPill.innerText = 'CONNECTION REFUSED (HTTP 403)';
    spoofPercentLabel.innerText = '100.0%';
    meterFill.style.width = '100%';
    riskLevelVal.innerText = 'BANNED';
    riskLevelVal.style.color = 'var(--accent-crimson)';
    latencyVal.innerText = '< 1 ms (Gateway Drop)';
    ipActionVal.innerText = 'BLOCKED IN FIREWALL';
    ipActionVal.style.color = 'var(--accent-crimson)';
    return;
  }

  const det = data.ai_detection;
  const isSpoof = det.is_spoof;
  const isSilent = det.is_silent || !det.is_speech;
  const probPercent = det.spoof_percentage;

  if (isSilent) {
    resultBox.className = 'result-box active idle';
    verdictText.innerText = det.verdict || 'AMBIENT NOISE / WAITING FOR VOICE';
    actionPill.className = 'threat-action-pill';
    actionPill.style.background = 'rgba(148, 163, 184, 0.15)';
    actionPill.style.color = '#94a3b8';
    actionPill.style.borderColor = 'rgba(148, 163, 184, 0.3)';
    actionPill.innerText = 'NO VOICE DETECTED (AMBIENT)';
    riskLevelVal.innerText = 'IDLE';
    riskLevelVal.style.color = 'var(--text-muted)';
    ipActionVal.innerText = 'MONITORING';
    ipActionVal.style.color = 'var(--text-muted)';
  } else if (isSpoof) {
    resultBox.className = 'result-box active danger';
    verdictText.innerText = 'SYNTHETIC AI VOICE / SPOOF';
    actionPill.className = 'threat-action-pill';
    actionPill.innerText = `AUTO-BLOCKED IP: ${data.caller_ip}`;
    riskLevelVal.innerText = det.risk_level;
    riskLevelVal.style.color = 'var(--accent-crimson)';
    ipActionVal.innerText = 'BANNED AT GATEWAY';
    ipActionVal.style.color = 'var(--accent-crimson)';
  } else {
    resultBox.className = 'result-box active success';
    verdictText.innerText = 'AUTHENTIC HUMAN / BONAFIDE';
    actionPill.className = 'threat-action-pill safe';
    actionPill.innerText = 'CALL PERMITTED';
    riskLevelVal.innerText = det.risk_level;
    riskLevelVal.style.color = 'var(--accent-emerald)';
    ipActionVal.innerText = 'AUTHORIZED';
    ipActionVal.style.color = 'var(--accent-emerald)';
  }

  spoofPercentLabel.innerText = `${probPercent}%`;
  meterFill.style.width = `${Math.min(100, Math.max(0, probPercent))}%`;
  latencyVal.innerText = `${det.total_latency_ms} ms (${det.forward_latency_ms}ms NN)`;
}

// Refresh Telemetry & Tables
async function refreshAllData() {
  try {
    // 1. Telemetry
    const statsRes = await fetch('/api/stats');
    if (statsRes.ok) {
      const stats = await statsRes.json();
      document.getElementById('telemetryActiveBans').innerText = stats.active_bans_count;
      document.getElementById('telemetrySpoofsStopped').innerText = stats.synthetic_spoofs_stopped;
      document.getElementById('telemetryBonafideCalls').innerText = stats.bonafide_calls_permitted;
      document.getElementById('telemetryTotalCalls').innerText = stats.total_calls_inspected;
    }

    // 2. Blocklist Table
    const blockRes = await fetch('/api/blocked-ips');
    if (blockRes.ok) {
      const blockData = await blockRes.json();
      renderBlocklist(blockData.blocked_ips || []);
    }

    // 3. Incident Logs Feed
    const incRes = await fetch('/api/incidents?limit=25');
    if (incRes.ok) {
      const incData = await incRes.json();
      renderIncidentFeed(incData.incidents || []);
    }
  } catch (e) {
    console.warn('Telemetry update error:', e);
  }
}

function renderBlocklist(list) {
  const tbody = document.getElementById('blocklistTbody');
  if (list.length === 0) {
    tbody.innerHTML = `
      <tr>
        <td colspan="4" style="text-align: center; color: var(--text-muted); padding: 2rem;">
          No IPs currently banned. Firewall is actively listening.
        </td>
      </tr>
    `;
    return;
  }

  tbody.innerHTML = list.map(item => {
    const ttlText = item.remaining_ttl_seconds != null
      ? `${Math.floor(item.remaining_ttl_seconds / 60)}m ${item.remaining_ttl_seconds % 60}s`
      : 'Permanent';

    return `
      <tr>
        <td><span class="ip-tag">${item.ip}</span></td>
        <td style="max-width: 200px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;" title="${item.reason}">
          ${item.reason}
        </td>
        <td style="font-family: var(--font-mono); color: var(--accent-warning);">${ttlText}</td>
        <td>
          <button class="btn-unblock" onclick="unblockIp('${item.ip}')">Unblock</button>
        </td>
      </tr>
    `;
  }).join('');
}

function renderIncidentFeed(logs) {
  const feed = document.getElementById('incidentFeed');
  if (logs.length === 0) {
    feed.innerHTML = `
      <div style="text-align: center; color: var(--text-muted); padding: 1.5rem;">
        No incidents logged yet.
      </div>
    `;
    return;
  }

  feed.innerHTML = logs.map(log => {
    const isThreat = log.event_type.includes('SPOOF') || log.event_type.includes('BLOCK');
    const isSafe = log.event_type.includes('BONAFIDE');
    const cls = isThreat ? 'danger' : (isSafe ? 'success' : '');
    const icon = isThreat ? '🛑' : (isSafe ? '✅' : 'ℹ️');

    const dt = new Date(log.timestamp);
    const timeStr = dt.toLocaleTimeString();

    return `
      <div class="incident-item ${cls}">
        <div class="incident-icon">${icon}</div>
        <div class="incident-content">
          <div class="incident-title">
            <span>${log.event_type} &middot; <code style="color:var(--accent-cyan)">${log.ip}</code></span>
            <span class="incident-time">${timeStr}</span>
          </div>
          <div class="incident-desc">${log.action_taken}</div>
        </div>
      </div>
    `;
  }).join('');
}

// Toast Notification System
function showToast(message, type = 'success') {
  const container = document.getElementById('toastContainer');
  if (!container) return;

  const toast = document.createElement('div');
  const bg = type === 'success' ? 'rgba(16, 185, 129, 0.95)' : (type === 'danger' ? 'rgba(239, 68, 68, 0.95)' : 'rgba(14, 165, 233, 0.95)');
  toast.style.cssText = `
    background: ${bg};
    color: #fff;
    padding: 0.75rem 1.25rem;
    border-radius: 8px;
    font-size: 0.85rem;
    font-weight: 600;
    box-shadow: 0 10px 25px rgba(0,0,0,0.5);
    display: flex;
    align-items: center;
    gap: 0.5rem;
    pointer-events: auto;
    animation: fadeInSlide 0.3s ease forwards;
    transition: opacity 0.3s ease;
  `;
  toast.innerHTML = `<span>${type === 'success' ? '✅' : (type === 'danger' ? '🛑' : 'ℹ️')}</span> <span>${message}</span>`;
  container.appendChild(toast);

  setTimeout(() => {
    toast.style.opacity = '0';
    setTimeout(() => toast.remove(), 300);
  }, 4000);
}

// Unblock IP Handler
async function unblockIp(ip, reason = 'Command Center Action') {
  if (!ip) {
    showToast('Please specify a valid IP address.', 'danger');
    return;
  }
  const cleanIp = ip.trim();
  try {
    const res = await fetch('/api/unblock', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ ip: cleanIp, reason })
    });
    const data = await res.json();
    showToast(`IP ${cleanIp} successfully unblocked!`, 'success');

    // Reset banner if this was the caller IP
    const currentCaller = document.getElementById('liveCallerIp').value.trim();
    if (currentCaller === cleanIp) {
      const banner = document.getElementById('liveDetectionBanner');
      const bannerIcon = document.getElementById('liveBannerIcon');
      const bannerMsg = document.getElementById('liveBannerMsg');
      banner.className = 'live-detection-banner safe';
      bannerIcon.innerText = '🔓';
      bannerMsg.innerHTML = `<strong>IP UNBLOCKED:</strong> Origin IP <code>${cleanIp}</code> is restored and cleared from Firewall. Ready to call.`;
      document.getElementById('callStatusText').innerText = 'CALL GUARDIAN: READY';
      document.getElementById('callPulseDot').className = 'call-pulse-dot';
    }

    await refreshAllData();
  } catch (e) {
    showToast('Unblock failed: ' + e, 'danger');
  }
}

// Unblock Current Caller IP from Input
async function unblockCurrentCallerIp() {
  const ip = document.getElementById('liveCallerIp').value.trim();
  if (!ip) {
    showToast('Enter an IP address to unblock.', 'danger');
    return;
  }
  await unblockIp(ip, 'Manual Unblock from Live Call Console');
}

// Unblock All Active Quarantined IPs
async function unblockAllIps() {
  try {
    const res = await fetch('/api/unblock-all', { method: 'POST' });
    const data = await res.json();
    showToast(`All active bans cleared! (${data.unblocked_count} restored)`, 'success');
    await refreshAllData();
  } catch (e) {
    showToast('Clear all failed: ' + e, 'danger');
  }
}

// Quick Unblock Modal Handlers
function openQuickUnblockModal() {
  const currentIp = document.getElementById('liveCallerIp').value.trim() || '198.51.100.110';
  document.getElementById('quickUnblockIpInput').value = currentIp;
  document.getElementById('quickUnblockModal').classList.add('active');
}

function closeQuickUnblockModal() {
  document.getElementById('quickUnblockModal').classList.remove('active');
}

async function submitQuickUnblock() {
  const ip = document.getElementById('quickUnblockIpInput').value.trim();
  if (!ip) {
    showToast('Please enter an IP address.', 'danger');
    return;
  }
  closeQuickUnblockModal();
  await unblockIp(ip, 'Quick Unblock Modal Action');
}

// Manual Quarantine Modal
function openManualBlockModal() {
  document.getElementById('manualBlockModal').classList.add('active');
}

function closeManualBlockModal() {
  document.getElementById('manualBlockModal').classList.remove('active');
}

async function submitManualBlock() {
  const ip = document.getElementById('manualIpInput').value.trim();
  const reason = document.getElementById('manualReasonInput').value.trim() || 'Manual Quarantine';
  const duration = parseInt(document.getElementById('manualTtlInput').value, 10) || 3600;

  if (!ip) {
    showToast('Please enter a target IP address.', 'danger');
    return;
  }

  try {
    await fetch('/api/block-manual', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        ip,
        reason,
        duration_seconds: duration
      })
    });
    closeManualBlockModal();
    document.getElementById('manualIpInput').value = '';
    showToast(`IP ${ip} quarantined at Firewall.`, 'danger');
    await refreshAllData();
  } catch (e) {
    showToast('Failed to ban IP: ' + e, 'danger');
  }
}

// Tab Audio Help Modal Handlers
function openTabAudioHelpModal() {
  const modal = document.getElementById('tabAudioHelpModal');
  if (modal) modal.classList.add('active');
}

function closeTabAudioHelpModal() {
  const modal = document.getElementById('tabAudioHelpModal');
  if (modal) modal.classList.remove('active');
}

function retryTabAudioCapture() {
  closeTabAudioHelpModal();
  startLiveIpCall();
}


