const API = 'http://localhost:8000';

const CLASSES = [
  'airplane', 'automobile', 'bird', 'cat', 'deer',
  'dog', 'frog', 'horse', 'ship', 'truck'
];

const SAMPLES = CLASSES.flatMap(cls => [
  { cls, label: cls, src: `samples/${cls}_1.jpg` },
  { cls, label: cls, src: `samples/${cls}_2.jpg` },
]);

// ── STATE ──────────────────────────────────────────────────────
let activeSource = null; // { type: 'sample'|'upload', blob, objectUrl }

// ── ELEMENT REFS ──────────────────────────────────────────────
const filmstrip    = document.getElementById('filmstrip');
const uploadZone   = document.getElementById('uploadZone');
const uploadIdle   = document.getElementById('uploadIdle');
const uploadPreview= document.getElementById('uploadPreview');
const uploadThumb  = document.getElementById('uploadThumb');
const uploadName   = document.getElementById('uploadName');
const uploadClear  = document.getElementById('uploadClear');
const fileInput    = document.getElementById('fileInput');
const statusDot    = document.getElementById('statusDot');
const statusLabel  = document.getElementById('statusLabel');
const offlineBanner= document.getElementById('offlineBanner');
const toast        = document.getElementById('toast');

// ── TOAST ──────────────────────────────────────────────────────
let toastTimer;
function showToast(msg) {
  toast.textContent = msg;
  toast.classList.add('show');
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => toast.classList.remove('show'), 2800);
}

// ── BACKEND STATUS CHECK ───────────────────────────────────────
async function checkBackend() {
  try {
    const res = await fetch(`${API}/`, { signal: AbortSignal.timeout(3000) });
    if (res.ok) {
      statusDot.className = 'status-dot online';
      statusLabel.textContent = 'backend online';
      offlineBanner.classList.add('hidden');
      return true;
    }
  } catch (_) {}
  statusDot.className = 'status-dot offline';
  statusLabel.textContent = 'backend offline';
  offlineBanner.classList.remove('hidden');
  return false;
}

// ── FILMSTRIP BUILD ────────────────────────────────────────────
function buildFilmstrip() {
  let lastCls = null;
  SAMPLES.forEach((item, i) => {
    if (item.cls !== lastCls && lastCls !== null) {
      const div = document.createElement('div');
      div.className = 'film-divider';
      filmstrip.appendChild(div);
    }
    lastCls = item.cls;

    const el = document.createElement('div');
    el.className = 'film-item';
    el.dataset.index = i;
    el.innerHTML = `
      <div class="film-thumb-wrap">
        <img src="${item.src}" alt="${item.label}" loading="lazy" />
      </div>
      <span class="film-label">${item.label}</span>
    `;
    el.addEventListener('click', () => onFilmstripClick(item, el));
    filmstrip.appendChild(el);
  });
}

function clearFilmstripActive() {
  filmstrip.querySelectorAll('.film-item').forEach(el => el.classList.remove('active'));
}

// ── FILMSTRIP CLICK ────────────────────────────────────────────
async function onFilmstripClick(item, el) {
  if (activeSource?.type === 'upload') {
    showToast('remove your uploaded image first');
    return;
  }

  clearFilmstripActive();
  el.classList.add('active');

  try {
    const res  = await fetch(item.src);
    const blob = await res.blob();
    const url  = URL.createObjectURL(blob);

    if (activeSource?.objectUrl) URL.revokeObjectURL(activeSource.objectUrl);
    activeSource = { type: 'sample', blob, objectUrl: url };

    runBothModels(blob, url);
  } catch (e) {
    showToast('could not load sample image');
  }
}

// ── UPLOAD FLOW ────────────────────────────────────────────────
uploadZone.addEventListener('click', (e) => {
  if (e.target === uploadClear || uploadClear.contains(e.target)) return;
  fileInput.click();
});

uploadZone.addEventListener('dragover', (e) => {
  e.preventDefault();
  uploadZone.classList.add('drag-over');
});

uploadZone.addEventListener('dragleave', () => {
  uploadZone.classList.remove('drag-over');
});

uploadZone.addEventListener('drop', (e) => {
  e.preventDefault();
  uploadZone.classList.remove('drag-over');
  const file = e.dataTransfer.files[0];
  if (file) handleUpload(file);
});

fileInput.addEventListener('change', () => {
  const file = fileInput.files[0];
  if (file) handleUpload(file);
  fileInput.value = '';
});

uploadClear.addEventListener('click', (e) => {
  e.stopPropagation();
  clearUpload();
});

function handleUpload(file) {
  if (!file.type.match(/image\/(jpeg|png|webp)/)) {
    showToast('unsupported file type — use JPG, PNG or WEBP');
    return;
  }

  clearFilmstripActive();
  if (activeSource?.objectUrl) URL.revokeObjectURL(activeSource.objectUrl);

  const url = URL.createObjectURL(file);
  activeSource = { type: 'upload', blob: file, objectUrl: url };

  uploadThumb.src = url;
  uploadName.textContent = file.name;
  uploadIdle.classList.add('hidden');
  uploadPreview.classList.remove('hidden');

  runBothModels(file, url);
}

function clearUpload() {
  if (activeSource?.objectUrl) URL.revokeObjectURL(activeSource.objectUrl);
  activeSource = null;

  uploadIdle.classList.remove('hidden');
  uploadPreview.classList.add('hidden');
  uploadThumb.src = '';
  uploadName.textContent = '';

  resetCards();
}

// ── CARD STATE MANAGEMENT ──────────────────────────────────────
function setCardState(suffix, state) {
  ['stats', 'loading', 'result', 'error'].forEach(s => {
    const el = document.getElementById(`state${suffix}-${s}`);
    if (el) el.classList.toggle('hidden', s !== state);
  });
}

function resetCards() {
  setCardState('70', 'stats');
  setCardState('50', 'stats');
}

// ── INFERENCE ─────────────────────────────────────────────────
async function predict(blob, modelId) {
  const form = new FormData();
  form.append('file', blob, 'image.jpg');

  const start = performance.now();
  const res = await fetch(`${API}/predict?model_id=${modelId}`, {
    method: 'POST',
    body: form,
  });
  const latency = performance.now() - start;

  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  const data = await res.json();
  return { ...data, latency_ms: latency };
}

function renderResult(suffix, data, imageUrl) {
  const thumb   = document.getElementById(`thumb${suffix}`);
  const cls     = document.getElementById(`class${suffix}`);
  const conf    = document.getElementById(`conf${suffix}`);
  const lat     = document.getElementById(`lat${suffix}`);
  const probsEl = document.getElementById(`probs${suffix}`);

  thumb.src = imageUrl;
  cls.textContent  = data.predicted_class;
  conf.textContent = `${data.confidence.toFixed(1)}%`;
  lat.textContent  = `latency · ${data.latency_ms.toFixed(0)} ms`;

  const sorted = Object.entries(data.all_probs).sort((a, b) => b[1] - a[1]);
  const topCls = sorted[0][0];

  probsEl.innerHTML = sorted.map(([name, prob]) => {
    const isTop = name === topCls;
    return `
      <div class="prob-row">
        <span class="prob-label ${isTop ? 'top' : ''}">${name}</span>
        <div class="prob-bar-track">
          <div class="prob-bar-fill ${isTop ? 'top' : ''}" style="width:${prob}%"></div>
        </div>
        <span class="prob-val ${isTop ? 'top' : ''}">${prob.toFixed(1)}%</span>
      </div>
    `;
  }).join('');

  setCardState(suffix, 'result');
}

async function runBothModels(blob, imageUrl) {
  setCardState('70', 'loading');
  setCardState('50', 'loading');

  const [r70, r50] = await Promise.allSettled([
    predict(blob, 'pruned_70_fp32'),
    predict(blob, 'pruned_50_fp32'),
  ]);

  if (r70.status === 'fulfilled') {
    renderResult('70', r70.value, imageUrl);
  } else {
    console.error('70% model error:', r70.reason);
    setCardState('70', 'error');
  }

  if (r50.status === 'fulfilled') {
    renderResult('50', r50.value, imageUrl);
  } else {
    console.error('50% model error:', r50.reason);
    setCardState('50', 'error');
  }
}

// ── INIT ───────────────────────────────────────────────────────
buildFilmstrip();
checkBackend();
setInterval(checkBackend, 30000);
