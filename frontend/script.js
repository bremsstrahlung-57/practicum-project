const API = "http://localhost:8000";

const CLASSES = [
  "airplane",
  "automobile",
  "bird",
  "cat",
  "deer",
  "dog",
  "frog",
  "horse",
  "ship",
  "truck",
];

const SAMPLES = CLASSES.flatMap((cls) => [
  { cls, label: cls, src: `samples/${cls}_1.jpg` },
  { cls, label: cls, src: `samples/${cls}_2.jpg` },
]);

let activeSource = null;
let models = [];

const cards = new Map();
const filmstrip = document.getElementById("filmstrip");
const modelsSection = document.getElementById("modelsSection");
const uploadZone = document.getElementById("uploadZone");
const uploadIdle = document.getElementById("uploadIdle");
const uploadPreview = document.getElementById("uploadPreview");
const uploadThumb = document.getElementById("uploadThumb");
const uploadName = document.getElementById("uploadName");
const uploadClear = document.getElementById("uploadClear");
const fileInput = document.getElementById("fileInput");
const statusDot = document.getElementById("statusDot");
const statusLabel = document.getElementById("statusLabel");
const offlineBanner = document.getElementById("offlineBanner");
const toast = document.getElementById("toast");

let toastTimer;

function showToast(msg) {
  toast.textContent = msg;
  toast.classList.add("show");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => toast.classList.remove("show"), 2800);
}

async function checkBackend() {
  try {
    const res = await fetch(`${API}/`, { signal: AbortSignal.timeout(3000) });
    if (res.ok) {
      statusDot.className = "status-dot online";
      statusLabel.textContent = "backend online";
      offlineBanner.classList.add("hidden");
      return true;
    }
  } catch (_) {}

  statusDot.className = "status-dot offline";
  statusLabel.textContent = "backend offline";
  offlineBanner.classList.remove("hidden");
  return false;
}

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

function buildFilmstrip() {
  let lastCls = null;
  SAMPLES.forEach((item, i) => {
    if (item.cls !== lastCls && lastCls !== null) {
      filmstrip.appendChild(el("div", "film-divider"));
    }
    lastCls = item.cls;

    const itemEl = el("button", "film-item");
    itemEl.type = "button";
    itemEl.dataset.index = i;

    const thumbWrap = el("div", "film-thumb-wrap");
    const image = el("img");
    image.src = item.src;
    image.alt = item.label;
    image.loading = "lazy";

    thumbWrap.appendChild(image);
    itemEl.append(thumbWrap, el("span", "film-label", item.label));
    itemEl.addEventListener("click", () => onFilmstripClick(item, itemEl));
    filmstrip.appendChild(itemEl);
  });
}

function clearFilmstripActive() {
  filmstrip
    .querySelectorAll(".film-item")
    .forEach((itemEl) => itemEl.classList.remove("active"));
}

async function onFilmstripClick(item, itemEl) {
  if (activeSource?.type === "upload") {
    showToast("remove your uploaded image first");
    return;
  }

  clearFilmstripActive();
  itemEl.classList.add("active");

  try {
    const res = await fetch(item.src);
    const blob = await res.blob();
    const url = URL.createObjectURL(blob);

    if (activeSource?.objectUrl) URL.revokeObjectURL(activeSource.objectUrl);
    activeSource = { type: "sample", blob, objectUrl: url };

    runModels(blob, url);
  } catch (_) {
    showToast("could not load sample image");
  }
}

uploadZone.addEventListener("click", (event) => {
  if (event.target === uploadClear || uploadClear.contains(event.target)) return;
  fileInput.click();
});

uploadZone.addEventListener("dragover", (event) => {
  event.preventDefault();
  uploadZone.classList.add("drag-over");
});

uploadZone.addEventListener("dragleave", () => {
  uploadZone.classList.remove("drag-over");
});

uploadZone.addEventListener("drop", (event) => {
  event.preventDefault();
  uploadZone.classList.remove("drag-over");
  const file = event.dataTransfer.files[0];
  if (file) handleUpload(file);
});

fileInput.addEventListener("change", () => {
  const file = fileInput.files[0];
  if (file) handleUpload(file);
  fileInput.value = "";
});

uploadClear.addEventListener("click", (event) => {
  event.stopPropagation();
  clearUpload();
});

function handleUpload(file) {
  if (!file.type.match(/image\/(jpeg|png|webp)/)) {
    showToast("unsupported file type - use JPG, PNG or WEBP");
    return;
  }

  clearFilmstripActive();
  if (activeSource?.objectUrl) URL.revokeObjectURL(activeSource.objectUrl);

  const url = URL.createObjectURL(file);
  activeSource = { type: "upload", blob: file, objectUrl: url };

  uploadThumb.src = url;
  uploadName.textContent = file.name;
  uploadIdle.classList.add("hidden");
  uploadPreview.classList.remove("hidden");

  runModels(file, url);
}

function clearUpload() {
  if (activeSource?.objectUrl) URL.revokeObjectURL(activeSource.objectUrl);
  activeSource = null;

  uploadIdle.classList.remove("hidden");
  uploadPreview.classList.add("hidden");
  uploadThumb.src = "";
  uploadName.textContent = "";

  resetCards();
}

function setCardState(modelId, state) {
  const card = cards.get(modelId);
  if (!card) return;

  Object.entries(card.states).forEach(([stateName, node]) => {
    node.classList.toggle("hidden", stateName !== state);
  });
}

function resetCards() {
  models.forEach((model) => {
    setCardState(model.id, model.available ? "stats" : "error");
  });
}

async function loadModels() {
  try {
    const res = await fetch(`${API}/models`, { signal: AbortSignal.timeout(5000) });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    models = await res.json();
  } catch (error) {
    console.error("model metadata error:", error);
    models = [];
    modelsSection.replaceChildren(
      el("div", "models-empty", "model metadata unavailable - start the backend"),
    );
    return;
  }

  buildModelCards();
}

function buildModelCards() {
  cards.clear();
  modelsSection.replaceChildren(...models.map(createModelCard));
}

function createModelCard(model) {
  const card = el("article", `model-card ${model.available ? "" : "unavailable"}`);
  card.id = `card-${model.id}`;

  const header = el("div", "card-header");
  const title = el("div", "card-title");
  const ratio = el("span", "card-ratio", model.short_name);
  const label = el("span", "card-label", model.name);
  const tags = el("div", "card-tags");

  model.tags.forEach((tag) => {
    tags.appendChild(el("span", `model-tag tag-${tagSlug(tag)}`, tag));
  });

  title.append(ratio, label, tags);
  header.append(title, el("div", "card-badge", model.id));

  const body = el("div", "card-body");
  const stats = createStatsState(model);
  const loading = createLoadingState();
  const result = createResultState();
  const error = createErrorState(model);

  if (!model.available) {
    stats.classList.add("hidden");
    error.classList.remove("hidden");
  }

  body.append(stats, loading, result, error);
  card.append(header, body);

  cards.set(model.id, {
    root: card,
    states: { stats, loading, result, error },
    result: {
      thumb: result.querySelector(".result-thumb"),
      className: result.querySelector(".result-class"),
      confidence: result.querySelector(".result-conf"),
      latency: result.querySelector(".result-latency"),
      probs: result.querySelector(".probs-list"),
    },
  });

  return card;
}

function tagSlug(tag) {
  return tag.toLowerCase().replace(/[^a-z0-9]+/g, "-");
}

function createStatsState(model) {
  const state = el("div", "card-state state-stats");
  const table = el("table", "stats-table");
  const tbody = el("tbody");

  model.stats.forEach((stat) => {
    const row = el("tr");
    row.append(
      el("td", "stat-key", stat.label),
      el("td", "stat-val", stat.value),
    );
    tbody.appendChild(row);
  });

  table.appendChild(tbody);
  state.append(
    table,
    el("div", "waiting-hint", "select an image to predict"),
  );
  return state;
}

function createLoadingState() {
  const state = el("div", "card-state state-loading hidden");
  state.append(el("div", "spinner"), el("div", "loading-text", "running inference..."));
  return state;
}

function createResultState() {
  const state = el("div", "card-state state-result hidden");
  const resultTop = el("div", "result-top");
  const thumb = el("img", "result-thumb");
  const info = el("div", "result-info");

  thumb.alt = "";
  info.append(
    el("div", "result-class", "-"),
    el("div", "result-conf", "-"),
    el("div", "result-latency", "-"),
  );
  resultTop.append(thumb, info);
  state.append(resultTop, el("div", "probs-list"));
  return state;
}

function createErrorState(model) {
  const state = el("div", "card-state state-error hidden");
  const message = model.available
    ? "prediction failed - check backend"
    : model.error || "model unavailable";
  state.appendChild(el("div", "error-text", message));
  return state;
}

async function predict(blob, modelId) {
  const form = new FormData();
  form.append("file", blob, "image.jpg");

  const start = performance.now();
  const res = await fetch(`${API}/predict?model_id=${encodeURIComponent(modelId)}`, {
    method: "POST",
    body: form,
  });
  const latency = performance.now() - start;

  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  const data = await res.json();
  return { ...data, latency_ms: latency };
}

function renderResult(modelId, data, imageUrl) {
  const card = cards.get(modelId);
  if (!card) return;

  card.result.thumb.src = imageUrl;
  card.result.className.textContent = data.predicted_class;
  card.result.confidence.textContent = `${data.confidence.toFixed(1)}%`;
  card.result.latency.textContent = `latency · ${data.latency_ms.toFixed(0)} ms`;
  card.result.probs.replaceChildren(...data.probabilities.map(createProbabilityRow));

  setCardState(modelId, "result");
}

function createProbabilityRow(item) {
  const row = el("div", "prob-row");
  const label = el("span", `prob-label ${item.is_top ? "top" : ""}`, item.class_name);
  const track = el("div", "prob-bar-track");
  const fill = el("div", `prob-bar-fill ${item.is_top ? "top" : ""}`);
  const value = el(
    "span",
    `prob-val ${item.is_top ? "top" : ""}`,
    `${item.probability.toFixed(1)}%`,
  );

  fill.style.width = `${Math.max(0, Math.min(100, item.probability))}%`;
  track.appendChild(fill);
  row.append(label, track, value);
  return row;
}

async function runModels(blob, imageUrl) {
  const runnableModels = models.filter((model) => model.available);
  if (!runnableModels.length) {
    showToast("no models loaded");
    return;
  }

  runnableModels.forEach((model) => setCardState(model.id, "loading"));

  const results = await Promise.allSettled(
    runnableModels.map(async (model) => {
      try {
        return {
          model,
          result: await predict(blob, model.id),
        };
      } catch (error) {
        error.model = model;
        throw error;
      }
    }),
  );

  results.forEach((entry) => {
    if (entry.status === "fulfilled") {
      renderResult(entry.value.model.id, entry.value.result, imageUrl);
      return;
    }

    console.error("model error:", entry.reason);
    const failedId = entry.reason?.model?.id;
    if (failedId) setCardState(failedId, "error");
  });
}

buildFilmstrip();
checkBackend();
loadModels();
setInterval(checkBackend, 30000);
