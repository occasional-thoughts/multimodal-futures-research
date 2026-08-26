const API = "http://localhost:8000";
const VARIANT_LABELS = { concat: "Concatenation", weighted: "Weighted", coattention: "Co-Attention" };

const tickerSelect = document.getElementById("ticker-select");
const dateSelect = document.getElementById("date-select");
const regimeBadge = document.getElementById("regime-badge");
const predictionsEl = document.getElementById("predictions");
const attributionEl = document.getElementById("attribution");
const cfTabsEl = document.getElementById("cf-tabs");
const counterfactualEl = document.getElementById("counterfactual");
const consistencyEl = document.getElementById("consistency");

let activeCfVariant = "concat";

async function fetchJSON(path) {
  const res = await fetch(`${API}${path}`);
  if (!res.ok) throw new Error(`${path} -> ${res.status}`);
  return res.json();
}

async function init() {
  try {
    const stocks = await fetchJSON("/stocks");
    if (!stocks.length) {
      predictionsEl.innerHTML = `<div class="error">No trained artifacts found. Run backend/scripts/run_pipeline.py first.</div>`;
      return;
    }
    tickerSelect.innerHTML = stocks
      .map((s) => `<option value="${s.ticker}">${s.ticker} — ${s.company_name}</option>`)
      .join("");
    tickerSelect.addEventListener("change", onTickerChange);
    dateSelect.addEventListener("change", refreshDate);
    await onTickerChange();
  } catch (e) {
    predictionsEl.innerHTML = `<div class="error">Could not reach API at ${API}. Is the backend running? (${e.message})</div>`;
  }
}

async function onTickerChange() {
  const ticker = tickerSelect.value;
  const dates = await fetchJSON(`/dates/${ticker}`);
  dateSelect.innerHTML = dates.map((d) => `<option value="${d.date}">${d.date} (${d.regime})</option>`).join("");
  // Run independently -- a consistency-report failure (e.g. a regime with too
  // few test instances for this ticker) must not block predictions from loading.
  loadConsistency(ticker).catch((e) => {
    consistencyEl.innerHTML = `<div class="error">Could not load consistency report: ${e.message}</div>`;
  });
  await refreshDate();
}

async function refreshDate() {
  const ticker = tickerSelect.value;
  const date = dateSelect.value;
  if (!date) return;

  predictionsEl.innerHTML = `<div class="loading">Loading predictions…</div>`;
  attributionEl.innerHTML = `<div class="loading">Loading attribution…</div>`;

  const pred = await fetchJSON(`/predict/${ticker}/${date}`);
  renderRegime(pred.regime);
  renderPredictions(pred.predictions);

  const variants = Object.keys(pred.predictions);
  cfTabsEl.innerHTML = variants
    .map((v) => `<button data-variant="${v}" class="${v === activeCfVariant ? "active" : ""}">${VARIANT_LABELS[v]}</button>`)
    .join("");
  cfTabsEl.querySelectorAll("button").forEach((btn) =>
    btn.addEventListener("click", () => {
      activeCfVariant = btn.dataset.variant;
      cfTabsEl.querySelectorAll("button").forEach((b) => b.classList.toggle("active", b === btn));
      loadCounterfactual(ticker, date, activeCfVariant);
    })
  );

  const attrEntries = await Promise.all(
    variants.map((v) => fetchJSON(`/explain/${ticker}/${date}/${v}`).then((r) => [v, r]))
  );
  renderAttribution(Object.fromEntries(attrEntries));

  loadCounterfactual(ticker, date, activeCfVariant);
}

function renderRegime(regime) {
  regimeBadge.textContent = regime === "embargo" ? "embargo (excluded)" : regime;
  regimeBadge.className = `badge ${regime}`;
}

function renderPredictions(predictions) {
  predictionsEl.innerHTML = Object.entries(predictions)
    .map(([variant, p]) => {
      const dir = p.prediction;
      const pct = Math.round(p.prob_up * 100);
      return `
      <div class="card">
        <h3>${VARIANT_LABELS[variant]}</h3>
        <div class="pred-value ${dir}">${dir === "up" ? "▲ UP" : "▼ DOWN"}</div>
        <div class="pred-meta">P(up) = ${pct}% · test accuracy ${(p.accuracy * 100).toFixed(1)}%</div>
      </div>`;
    })
    .join("");
}

function renderAttribution(byVariant) {
  attributionEl.innerHTML = Object.entries(byVariant)
    .map(([variant, a]) => {
      const techPct = Math.round(a.tech_share * 100);
      const newsPct = 100 - techPct;
      const features = a.top_features
        .map(
          (f) =>
            `<li><span>${f.feature}</span><span>${f.shap >= 0 ? "+" : ""}${f.shap.toFixed(3)}</span></li>`
        )
        .join("");
      return `
      <div class="card">
        <h3>${VARIANT_LABELS[variant]}</h3>
        <div class="bar-row">
          <div class="bar-label"><span>Technical ${techPct}%</span><span>News ${newsPct}%</span></div>
          <div class="bar-track">
            <div class="bar-tech" style="width:${techPct}%"></div>
            <div class="bar-news" style="width:${newsPct}%"></div>
          </div>
        </div>
        <ul class="feature-list">${features}</ul>
      </div>`;
    })
    .join("");
}

async function loadCounterfactual(ticker, date, variant) {
  counterfactualEl.innerHTML = `<div class="loading">Generating counterfactual (genetic search)…</div>`;
  try {
    const cf = await fetchJSON(`/counterfactual/${ticker}/${date}/${variant}`);
    if (!cf.changes.length) {
      counterfactualEl.innerHTML = `<div class="cf-empty">No counterfactual within the permitted feature ranges was found for this instance.</div>`;
      return;
    }
    const changes = cf.changes
      .map(
        (c) =>
          `<li><span>${c.feature}</span><span>${c.from.toFixed(3)} → ${c.to.toFixed(3)}</span></li>`
      )
      .join("");
    counterfactualEl.innerHTML = `
      <div class="cf-flip">Current prediction: <strong>${cf.current_prediction.toUpperCase()}</strong> → to flip to <strong>${cf.counterfactual_prediction.toUpperCase()}</strong>, the minimal change found was:</div>
      <ul class="cf-changes">${changes}</ul>`;
  } catch (e) {
    counterfactualEl.innerHTML = `<div class="error">Counterfactual generation failed: ${e.message}</div>`;
  }
}

async function loadConsistency(ticker) {
  consistencyEl.innerHTML = `<div class="loading">Loading consistency report…</div>`;
  const report = await fetchJSON(`/consistency-report/${ticker}`);
  const rows = [];
  for (const [regime, pairs] of Object.entries(report.by_regime)) {
    for (const [pair, stat] of Object.entries(pairs)) {
      const rho = stat.rho == null || isNaN(stat.rho) ? "—" : stat.rho.toFixed(2);
      const p = stat.p == null || isNaN(stat.p) ? "—" : stat.p.toFixed(3);
      rows.push(`<tr><td>${regime}</td><td>${pair.replace("_vs_", " vs. ")}</td><td>${rho}</td><td>${p}</td></tr>`);
    }
  }
  const verdictClass = report.verdict === "architecture-robust" ? "robust" : "artifact";
  consistencyEl.innerHTML = `
    <table class="consistency-table">
      <thead><tr><th>Regime</th><th>Fusion pair</th><th>Spearman ρ</th><th>p-value</th></tr></thead>
      <tbody>${rows.join("")}</tbody>
    </table>
    <div class="verdict ${verdictClass}">
      Verdict: ${report.verdict} (threshold τ = ${report.threshold}, mean ρ = ${report.overall_consistency.mean_rho?.toFixed(2) ?? "—"}
      [${report.overall_consistency.ci_low?.toFixed(2) ?? "—"}, ${report.overall_consistency.ci_high?.toFixed(2) ?? "—"}])
    </div>`;
}

init();
