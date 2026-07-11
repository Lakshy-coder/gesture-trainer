// Media pipe set up
const video = document.getElementById("video");
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");

const holistic = new Holistic({
  locateFile: f => `https://cdn.jsdelivr.net/npm/@mediapipe/holistic/${f}`
});
holistic.setOptions({ modelComplexity: 1, smoothLandmarks: true });

const cam = new Camera(video, {
  width: 640,
  height: 480,
  onFrame: async () => await holistic.send({ image: video })
});
cam.start();

holistic.onResults(res => {
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  // Mirror the feed
  ctx.save();
  ctx.scale(-1, 1);
  ctx.drawImage(video, -canvas.width, 0, canvas.width, canvas.height);
  ctx.restore();

  // Draw hand skeletons
  if (res.rightHandLandmarks) drawHand(res.rightHandLandmarks);
  if (res.leftHandLandmarks) drawHand(res.leftHandLandmarks);

  // Capture data if recording
  captureFrame(res.rightHandLandmarks, res.leftHandLandmarks);

  // Live inference
  if (model && res.rightHandLandmarks && res.leftHandLandmarks) {
    const input = tf.tensor2d([extract(res.rightHandLandmarks, res.leftHandLandmarks)]);
    const prob = model.predict(input).dataSync()[0];
    input.dispose();
    updateConf(prob);
  }
});

// Hand skeleton drawing
function drawHand(lm) {
  const segs = [
    [0, 1, 2, 3, 4],
    [0, 5, 6, 7, 8],
    [0, 9, 10, 11, 12],
    [0, 13, 14, 15, 16],
    [0, 17, 18, 19, 20]
  ];
  ctx.strokeStyle = "#22c55e";
  ctx.lineWidth = 2;

  for (const seg of segs) {
    ctx.beginPath();
    seg.forEach((i, idx) => {
      const x = canvas.width - lm[i].x * canvas.width;
      const y = lm[i].y * canvas.height;
      idx === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    });
    ctx.stroke();
  }

  lm.forEach(p => {
    ctx.beginPath();
    ctx.arc(canvas.width - p.x * canvas.width, p.y * canvas.height, 3, 0, Math.PI * 2);
    ctx.fillStyle = "#ef4444";
    ctx.fill();
  });
}

// landmark processing — wrist-relative, scale-normalized (21 landmarks * 3 coords = 63 per hand)
function normalize(lm) {
  const w = lm[0], mcp = lm[9];
  const scale = Math.sqrt(
    (mcp.x - w.x) ** 2 + (mcp.y - w.y) ** 2 + (mcp.z - w.z) ** 2
  ) || 1;
  const out = [];
  for (let i = 0; i < 21; i++) {
    out.push((lm[i].x - w.x) / scale);
    out.push((lm[i].y - w.y) / scale);
    out.push((lm[i].z - w.z) / scale);
  }
  return out;
}

function extract(right, left) {
  return [...normalize(right), ...normalize(left)]; // 126-dim feature vector
}
// Config
const COUNTDOWN = 3;              // countdown before recording starts (s)
const RECORD_TIME = 4;            // recording time per session (s)
const VAL_RATIO = 0.25;           // fraction of SESSIONS (not frames) held out per class
const MIN_SESSIONS_PER_CLASS = 3; // minimum sessions/class before validation is computed

// Data collection state

// Each sample is { x: [126 floats], session: "<label>_<timestamp>" }.
// The session id ties every frame captured in one countdown+record cycle
// together, so a train/validation split can keep a whole session on one
// side only (see splitSessionsForClass below).
let samples = { clone_sign: [], not_sign: [] };
let recording = null;
let currentSessionId = null;
let model = null; // the "live" model used for on-page testing + export

const statusEl = document.getElementById("train-status");

function updateCounts() {
  document.getElementById("count-clone").textContent = samples.clone_sign.length;
  document.getElementById("count-other").textContent = samples.not_sign.length;
  document.getElementById("sessions-clone").textContent = groupBySession(samples.clone_sign).size;
  document.getElementById("sessions-other").textContent = groupBySession(samples.not_sign).size;
}

function captureFrame(right, left) {
  if (!recording || !right || !left) return;
  samples[recording].push({ x: extract(right, left), session: currentSessionId });
  updateCounts();
}

let countdownTimer = null;
let recordTimer = null;

function startCountdown(label) {
  // Cancel any running session
  cancelRecording();

  const badge = document.getElementById("rec-badge");
  let remaining = COUNTDOWN;

  badge.classList.add("active");
  badge.textContent = `GET READY… ${remaining}`;
  statusEl.textContent = `Recording "${label === "clone_sign" ? "clone sign" : "other"}" in ${remaining}s — get into position!`;

  countdownTimer = setInterval(() => {
    remaining--;
    if (remaining > 0) {
      badge.textContent = `GET READY… ${remaining}`;
      statusEl.textContent = `Recording in ${remaining}s — get into position!`;
    } else {
      clearInterval(countdownTimer);
      countdownTimer = null;
      startRec(label);
    }
  }, 1000);
}

function startRec(label) {
  recording = label;
  currentSessionId = `${label}_${Date.now()}`; // one id per countdown+record cycle

  const badge = document.getElementById("rec-badge");
  badge.classList.add("active");

  let remaining = RECORD_TIME;
  badge.textContent = `● REC ${remaining}s`;
  statusEl.textContent = `Recording "${label === "clone_sign" ? "clone sign" : "other"}" — hold your pose!`;

  recordTimer = setInterval(() => {
    remaining--;
    if (remaining > 0) {
      badge.textContent = `● REC ${remaining}s`;
    } else {
      stopRec();
      statusEl.textContent = `Done! Captured samples. Record more (3+ sessions/class unlocks validation) or train.`;
    }
  }, 1000);
}

function stopRec() {
  recording = null;
  currentSessionId = null;
  clearInterval(recordTimer);
  recordTimer = null;
  document.getElementById("rec-badge").classList.remove("active");
}

function cancelRecording() {
  clearInterval(countdownTimer);
  clearInterval(recordTimer);
  countdownTimer = null;
  recordTimer = null;
  recording = null;
  currentSessionId = null;
  document.getElementById("rec-badge").classList.remove("active");
}

// Click-to-toggle buttons
["btn-rec-clone", "btn-rec-other"].forEach(id => {
  const label = id.includes("clone") ? "clone_sign" : "not_sign";
  document.getElementById(id).addEventListener("click", () => startCountdown(label));
});

// Keyboard: tap 1 / 2
const keyMap = { "1": "clone_sign", "2": "not_sign" };
document.addEventListener("keydown", e => {
  if (!e.repeat && keyMap[e.key]) startCountdown(keyMap[e.key]);
});

// ---------------------------------------------------------------------------
// Shuffling helpers
// ---------------------------------------------------------------------------
function shuffle(arr) {
  const a = arr.slice();
  for (let i = a.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [a[i], a[j]] = [a[j], a[i]];
  }
  return a;
}

function shuffleParallel(xs, ys) {
  const idx = shuffle(xs.map((_, i) => i));
  return { xs: idx.map(i => xs[i]), ys: idx.map(i => ys[i]) };
}

// Session-level stratified split

// WHY: a single 4s recording captures dozens of frames of (basically) the
// same pose, same lighting, same background — they're near-duplicates of
// each other. If those frames get randomly shuffled into both train and
// validation, validation "sees" near-copies of what it trained on and
// reports inflated accuracy that says nothing about real generalization
// (e.g. to a new session, angle, or lighting). Splitting by whole SESSION
// instead of by frame keeps every frame from one recording on the same
// side of the split, so validation is only ever tested against poses the
// model has never (even approximately) seen during training.

function groupBySession(list) {
  const map = new Map();
  list.forEach(s => {
    const key = s.session || "unknown";
    if (!map.has(key)) map.set(key, []);
    map.get(key).push(s.x);
  });
  return map;
}

function splitSessionsForClass(list) {
  const sessions = groupBySession(list);
  const ids = shuffle([...sessions.keys()]);
  const total = ids.length;

  let valCount = Math.max(1, Math.round(total * VAL_RATIO));
  if (total - valCount < 1) valCount = Math.max(0, total - 1); // always keep >=1 train session

  const valIds = ids.slice(0, valCount);
  const trainIds = ids.slice(valCount);

  const trainX = trainIds.flatMap(id => sessions.get(id));
  const valX = valIds.flatMap(id => sessions.get(id));

  return {
    trainX, valX, trainIds, valIds,
    numSessions: total,
    numTrainSessions: trainIds.length,
    numValSessions: valIds.length
  };
}

function buildSplit() {
  const pos = splitSessionsForClass(samples.clone_sign); // label 1
  const neg = splitSessionsForClass(samples.not_sign);   // label 0

  return {
    trainX: [...pos.trainX, ...neg.trainX],
    trainY: [...pos.trainX.map(() => 1), ...neg.trainX.map(() => 0)],
    valX: [...pos.valX, ...neg.valX],
    valY: [...pos.valX.map(() => 1), ...neg.valX.map(() => 0)],
    pos, neg
  };
}


// Model

function buildModel() {
  const m = tf.sequential();
  // Mess around with the NN model topology to try and get better performance.
  // Keep in mind bias-variance tradeoffs and over/under fitting
  m.add(tf.layers.dense({ inputShape: [126], units: 64, activation: "relu" }));
  m.add(tf.layers.dropout({ rate: 0.3 }));
  m.add(tf.layers.dense({ units: 32, activation: "relu" }));
  m.add(tf.layers.dense({ units: 1, activation: "sigmoid" }));
  m.compile({ optimizer: "adam", loss: "binaryCrossentropy", metrics: ["accuracy"] });
  return m;
}

function computeMetrics(m, xs, ys) {
  const xT = tf.tensor2d(xs);
  const preds = m.predict(xT).dataSync();
  xT.dispose();

  let tp = 0, fp = 0, tn = 0, fn = 0;
  preds.forEach((p, i) => {
    const pred = p >= 0.5 ? 1 : 0;
    const actual = ys[i];
    if (pred === 1 && actual === 1) tp++;
    else if (pred === 1 && actual === 0) fp++;
    else if (pred === 0 && actual === 0) tn++;
    else fn++;
  });

  const n = tp + fp + tn + fn;
  const accuracy = n ? (tp + tn) / n : 0;
  const precision = (tp + fp) ? tp / (tp + fp) : 0;
  const recall = (tp + fn) ? tp / (tp + fn) : 0;
  const f1 = (precision + recall) ? (2 * precision * recall) / (precision + recall) : 0;

  return { tp, fp, tn, fn, n, accuracy, precision, recall, f1 };
}


// Metrics panel UI

const metricsSection = document.getElementById("metrics-section");

function renderMetrics(metrics, split) {
  document.getElementById("m-acc").textContent = `${(metrics.accuracy * 100).toFixed(1)}%`;
  document.getElementById("m-prec").textContent = `${(metrics.precision * 100).toFixed(1)}%`;
  document.getElementById("m-rec").textContent = `${(metrics.recall * 100).toFixed(1)}%`;
  document.getElementById("m-f1").textContent = `${(metrics.f1 * 100).toFixed(1)}%`;

  document.getElementById("cm-tp").textContent = metrics.tp;
  document.getElementById("cm-fn").textContent = metrics.fn;
  document.getElementById("cm-fp").textContent = metrics.fp;
  document.getElementById("cm-tn").textContent = metrics.tn;

  document.getElementById("metrics-meta").textContent =
    `Trained on ${split.pos.numTrainSessions + split.neg.numTrainSessions} sessions ` +
    `(${split.pos.numTrainSessions} clone / ${split.neg.numTrainSessions} other) — ` +
    `validated on ${split.pos.numValSessions + split.neg.numValSessions} held-out sessions ` +
    `(${split.pos.numValSessions} clone / ${split.neg.numValSessions} other), ` +
    `${metrics.n} validation frames total. No session appears on both sides.`;

  metricsSection.classList.remove("hidden");
}

function hideMetrics() {
  metricsSection.classList.add("hidden");
}

// Training the model

document.getElementById("btn-train").addEventListener("click", async () => {
  const nP = samples.clone_sign.length;
  const nN = samples.not_sign.length;

  if (nP < 5 || nN < 5) {
    statusEl.textContent = "Need at least 5 samples each.";
    return;
  }

  const trainBtn = document.getElementById("btn-train");
  trainBtn.disabled = true;

  const sessionsP = groupBySession(samples.clone_sign).size;
  const sessionsN = groupBySession(samples.not_sign).size;
  const canValidate = sessionsP >= MIN_SESSIONS_PER_CLASS && sessionsN >= MIN_SESSIONS_PER_CLASS;

  let metrics = null;
  let split = null;

  if (canValidate) {
    // --- Phase 1: held-out validation ---
    split = buildSplit();
    const { xs: trainXs, ys: trainYs } = shuffleParallel(split.trainX, split.trainY);

    const valModel = buildModel();
    const xT = tf.tensor2d(trainXs);
    const yT = tf.tensor1d(trainYs);

    statusEl.textContent = "Validation pass: training on train-sessions only...";
    await valModel.fit(xT, yT, {
      epochs: 50,
      batchSize: 16,
      shuffle: true,
      callbacks: {
        onEpochEnd: ep => { statusEl.textContent = `Validation pass — epoch ${ep + 1}/50`; }
      }
    });
    xT.dispose();
    yT.dispose();

    metrics = computeMetrics(valModel, split.valX, split.valY);
    valModel.dispose();
    renderMetrics(metrics, split);
  } else {
    hideMetrics();
  }

  // --- Phase 2: final "live" model, trained on ALL collected data ---
  // (Held-out split above is only for an honest accuracy estimate; once
  // we have that estimate, using every sample for the deployed model is
  // standard practice and gives the on-page demo the best shot it can get.)
  const allXs = [...samples.clone_sign.map(s => s.x), ...samples.not_sign.map(s => s.x)];
  const allYs = [...samples.clone_sign.map(() => 1), ...samples.not_sign.map(() => 0)];
  const { xs, ys } = shuffleParallel(allXs, allYs);

  if (model) model.dispose();
  model = buildModel();

  const xT2 = tf.tensor2d(xs);
  const yT2 = tf.tensor1d(ys);

  statusEl.textContent = canValidate ? "Validated. Training final model on all data..." : "Training...";
  await model.fit(xT2, yT2, {
    epochs: 50,
    batchSize: 16,
    shuffle: true,
    callbacks: {
      onEpochEnd: (ep, logs) => {
        statusEl.textContent = `Final model — epoch ${ep + 1}/50 — train acc: ${((logs.accuracy || 0) * 100).toFixed(1)}%`;
      }
    }
  });
  xT2.dispose();
  yT2.dispose();

  trainBtn.disabled = false;

  if (canValidate) {
    statusEl.textContent = `Done! Held-out validation accuracy: ${(metrics.accuracy * 100).toFixed(1)}% ` +
      `(${metrics.n} val frames, ${split.pos.numValSessions + split.neg.numValSessions} sessions). ` +
      `Live model retrained on all ${nP + nN} samples — test your sign above.`;
  } else {
    statusEl.textContent = `Done! ${nP + nN} samples. Model is live — test your sign above. ` +
      `(Record ${MIN_SESSIONS_PER_CLASS}+ separate sessions per gesture to unlock validation accuracy — currently ${sessionsP}/${sessionsN}.)`;
  }
});

// confidence bar
function updateConf(prob) {
  document.getElementById("conf-fill").style.width = `${(prob * 100).toFixed(0)}%`;
  document.getElementById("conf-label").textContent = `${(prob * 100).toFixed(0)}%`;
}

// Export / import

document.getElementById("btn-export-data").addEventListener("click", () => {
  const payload = {
    formatVersion: 2,
    exportedAt: new Date().toISOString(),
    samples
  };
  const blob = new Blob([JSON.stringify(payload)], { type: "application/json" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = "gesture-data.json";
  a.click();
});

document.getElementById("btn-import-data").addEventListener("click", () => {
  document.getElementById("import-data-input").click();
});

// Accepts both the new session-aware export (formatVersion 2) and old flat
// exports (plain 126-number arrays, no session info). Legacy frames are
// grouped into one conservative "legacy" session per class per import, so
// they can never be split across train/val — safe, if not ideal.
function normalizeImportPayload(raw) {
  const isV2 = raw && raw.formatVersion === 2 && raw.samples;
  const src = isV2 ? raw.samples : raw;
  const importTag = `legacy-import_${Date.now()}`;

  const out = { clone_sign: [], not_sign: [] };
  ["clone_sign", "not_sign"].forEach(key => {
    (src[key] || []).forEach(entry => {
      if (Array.isArray(entry)) {
        out[key].push({ x: entry, session: `${importTag}_${key}` });
      } else if (entry && entry.x) {
        out[key].push(entry);
      }
    });
  });
  return out;
}

document.getElementById("import-data-input").addEventListener("change", e => {
  const reader = new FileReader();
  reader.onload = ev => {
    const raw = JSON.parse(ev.target.result);
    const normalized = normalizeImportPayload(raw);
    samples.clone_sign.push(...normalized.clone_sign);
    samples.not_sign.push(...normalized.not_sign);
    updateCounts();
    statusEl.textContent = "Data imported.";
  };
  reader.readAsText(e.target.files[0]);
});

// save / clear model
document.getElementById("btn-save-model").addEventListener("click", async () => {
  if (!model) {
    statusEl.textContent = "Train a model first.";
    return;
  }
  await model.save("downloads://gesture-model");
  statusEl.textContent = "Model saved — you'll get gesture-model.json + gesture-model.weights.bin";
});

document.getElementById("btn-clear-data").addEventListener("click", () => {
  samples = { clone_sign: [], not_sign: [] };
  updateCounts();
  hideMetrics();
  statusEl.textContent = "Data cleared.";
});

updateCounts();