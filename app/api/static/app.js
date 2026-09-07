// Vector Database Demo Dashboard Controller

const state = {
  totalVectors: 50000,
  currentVector: null,
  exactResults: null,
  ivfResults: null,
  exactLatency: null,
  ivfLatency: null,
};

// DOM Elements
const apiStatusBadge = document.getElementById("api-status-badge");
const apiStatusText = document.getElementById("api-status-text");

const statVectors = document.getElementById("stat-vectors");
const statDimension = document.getElementById("stat-dimension");
const statClusters = document.getElementById("stat-clusters");
const statNprobe = document.getElementById("stat-nprobe");

const btnSample = document.getElementById("btn-sample");
const sampleIdTag = document.getElementById("sample-id-tag");
const vectorPreview = document.getElementById("vector-preview");
const inputK = document.getElementById("input-k");
const sliderNprobe = document.getElementById("slider-nprobe");
const valNprobe = document.getElementById("val-nprobe");

const btnExact = document.getElementById("btn-exact");
const btnIvf = document.getElementById("btn-ivf");
const btnCompare = document.getElementById("btn-compare");
const searchError = document.getElementById("search-error");

const exactLatencyEl = document.getElementById("exact-latency");
const exactCountEl = document.getElementById("exact-count");
const exactQpsEl = document.getElementById("exact-qps");
const exactTableBody = document.getElementById("exact-results-body");

const ivfLatencyEl = document.getElementById("ivf-latency");
const ivfCandidatesEl = document.getElementById("ivf-candidates");
const ivfReductionEl = document.getElementById("ivf-reduction");
const ivfTableBody = document.getElementById("ivf-results-body");

const compRecallEl = document.getElementById("comp-recall");
const compSpeedupEl = document.getElementById("comp-speedup");

// Initialize application
document.addEventListener("DOMContentLoaded", async () => {
  setupEventListeners();
  await checkHealth();
  await loadStats();
  await loadSampleVector();
});

function setupEventListeners() {
  btnSample.addEventListener("click", loadSampleVector);

  sliderNprobe.addEventListener("input", (e) => {
    valNprobe.textContent = e.target.value;
  });

  btnExact.addEventListener("click", () => executeExactSearch());
  btnIvf.addEventListener("click", () => executeIvfSearch());
  btnCompare.addEventListener("click", () => executeCompareBoth());
}

// 1. Health check
async function checkHealth() {
  try {
    const res = await fetch("/health");
    if (res.ok) {
      const data = await res.json();
      if (data.status === "ok") {
        apiStatusBadge.className = "status-badge online";
        apiStatusText.textContent = "API Online";
        return;
      }
    }
    throw new Error("Invalid status");
  } catch (err) {
    apiStatusBadge.className = "status-badge offline";
    apiStatusText.textContent = "API Offline";
  }
}

// 2. Load Stats
async function loadStats() {
  try {
    const res = await fetch("/stats");
    if (!res.ok) throw new Error("Failed to fetch stats");
    const data = await res.json();

    state.totalVectors = data.exact_count || 50000;
    statVectors.textContent = Number(data.exact_count).toLocaleString();
    statDimension.textContent = data.dimension;
    statClusters.textContent = data.ivf_clusters;
    statNprobe.textContent = data.default_nprobe;

    sliderNprobe.value = data.default_nprobe || 5;
    valNprobe.textContent = sliderNprobe.value;
  } catch (err) {
    console.error("Error loading stats:", err);
  }
}

// 3. Load Sample Vector
async function loadSampleVector() {
  btnSample.disabled = true;
  btnSample.textContent = "Loading...";
  clearError();

  try {
    const res = await fetch("/sample");
    if (!res.ok) throw new Error("Failed to fetch sample vector");
    const data = await res.json();

    state.currentVector = data.vector;
    sampleIdTag.textContent = `Sample Vector #${data.id}`;
    vectorPreview.value = JSON.stringify(data.vector);
  } catch (err) {
    showError("Unable to load sample vector from API.");
  } finally {
    btnSample.disabled = false;
    btnSample.textContent = "⚡ Sample Random Vector";
  }
}

// 4. Exact Search
async function executeExactSearch() {
  if (!state.currentVector) {
    showError("Please load a sample vector first.");
    return null;
  }

  const k = parseInt(inputK.value, 10) || 10;
  clearError();
  setSearchingState(true, "exact");

  const startTime = performance.now();
  try {
    const res = await fetch("/exact/search", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ vector: state.currentVector, k }),
    });

    const endTime = performance.now();
    const clientLatency = endTime - startTime;

    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || "Exact search failed");
    }

    const data = await res.json();
    state.exactResults = data.results;
    state.exactLatency = clientLatency;

    // Update Exact Card
    exactLatencyEl.textContent = `${clientLatency.toFixed(2)} ms`;
    exactCountEl.textContent = `${data.count} items`;
    const qps = (1000.0 / clientLatency).toFixed(1);
    exactQpsEl.textContent = `${qps} QPS`;

    renderResultsTable(exactTableBody, data.results);
    updateComparison();
    return data;
  } catch (err) {
    showError(`Search failed: ${err.message}`);
    return null;
  } finally {
    setSearchingState(false, "exact");
  }
}

// 5. IVF Search
async function executeIvfSearch() {
  if (!state.currentVector) {
    showError("Please load a sample vector first.");
    return null;
  }

  const k = parseInt(inputK.value, 10) || 10;
  const nprobe = parseInt(sliderNprobe.value, 10) || 5;
  clearError();
  setSearchingState(true, "ivf");

  const startTime = performance.now();
  try {
    const res = await fetch("/ivf/search", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ vector: state.currentVector, k, nprobe }),
    });

    const endTime = performance.now();
    const clientLatency = endTime - startTime;

    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || "IVF search failed");
    }

    const data = await res.json();
    state.ivfResults = data.results;
    state.ivfLatency = clientLatency;

    // Update IVF Card
    ivfLatencyEl.textContent = `${clientLatency.toFixed(2)} ms`;
    const candidates = data.candidates || 0;
    ivfCandidatesEl.textContent = `${Number(candidates).toLocaleString()} / ${Number(state.totalVectors).toLocaleString()}`;

    const reduction = ((1.0 - (candidates / state.totalVectors)) * 100.0).toFixed(2);
    ivfReductionEl.textContent = `${reduction}%`;

    renderResultsTable(ivfTableBody, data.results);
    updateComparison();
    return data;
  } catch (err) {
    showError(`Search failed: ${err.message}`);
    return null;
  } finally {
    setSearchingState(false, "ivf");
  }
}

// 6. Compare Both
async function executeCompareBoth() {
  if (!state.currentVector) {
    showError("Please load a sample vector first.");
    return;
  }

  clearError();
  setSearchingState(true, "all");

  try {
    await executeExactSearch();
    await executeIvfSearch();
  } finally {
    setSearchingState(false, "all");
  }
}

// 7. Update Single-Query Comparison Card
function updateComparison() {
  if (!state.exactResults || !state.ivfResults) {
    return;
  }

  const exactIds = state.exactResults.map((r) => r.id);
  const ivfIds = state.ivfResults.map((r) => r.id);

  // Compute Recall@k
  const exactIdSet = new Set(exactIds);
  const matchedCount = ivfIds.filter((id) => exactIdSet.has(id)).length;
  const recallPct = exactIds.length > 0 ? (matchedCount / exactIds.length) * 100.0 : 0.0;
  const kVal = exactIds.length;

  compRecallEl.textContent = `Recall@${kVal}: ${recallPct.toFixed(2)}%`;

  // Compute Speedup factor
  if (state.exactLatency && state.ivfLatency && state.ivfLatency > 0) {
    const speedup = state.exactLatency / state.ivfLatency;
    compSpeedupEl.textContent = `Speedup: ${speedup.toFixed(2)}×`;
  }

  // Highlight matching rows across tables
  highlightMatchingRows(exactIdSet);
}

// 8. Render Results Table
function renderResultsTable(tbodyEl, results) {
  tbodyEl.innerHTML = "";
  if (!results || results.length === 0) {
    tbodyEl.innerHTML = `<tr><td colspan="3" class="table-empty">No results returned</td></tr>`;
    return;
  }

  results.forEach((item, index) => {
    const row = document.createElement("tr");
    row.dataset.vectorId = item.id;
    row.innerHTML = `
      <td>#${index + 1}</td>
      <td>${item.id}</td>
      <td>${Number(item.score).toFixed(6)}</td>
    `;
    tbodyEl.appendChild(row);
  });
}

// 9. Highlight Matching Rows
function highlightMatchingRows(exactIdSet) {
  const ivfRows = ivfTableBody.querySelectorAll("tr");
  ivfRows.forEach((row) => {
    const id = parseInt(row.dataset.vectorId, 10);
    if (exactIdSet.has(id)) {
      row.classList.add("match-row");
    } else {
      row.classList.remove("match-row");
    }
  });

  const exactRows = exactTableBody.querySelectorAll("tr");
  const ivfIdSet = new Set(state.ivfResults.map((r) => r.id));
  exactRows.forEach((row) => {
    const id = parseInt(row.dataset.vectorId, 10);
    if (ivfIdSet.has(id)) {
      row.classList.add("match-row");
    } else {
      row.classList.remove("match-row");
    }
  });
}

// UI Helpers
function setSearchingState(isLoading, target) {
  if (isLoading) {
    btnExact.disabled = true;
    btnIvf.disabled = true;
    btnCompare.disabled = true;
    if (target === "exact") btnExact.textContent = "Searching Exact...";
    if (target === "ivf") btnIvf.textContent = "Searching IVF...";
    if (target === "all") btnCompare.textContent = "Comparing Both...";
  } else {
    btnExact.disabled = false;
    btnIvf.disabled = false;
    btnCompare.disabled = false;
    btnExact.textContent = "Run Exact Search";
    btnIvf.textContent = "Run IVF Search";
    btnCompare.textContent = "⚡ Compare Both";
  }
}

function showError(msg) {
  searchError.textContent = msg;
  searchError.style.display = "block";
}

function clearError() {
  searchError.textContent = "";
  searchError.style.display = "none";
}
