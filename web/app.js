let pollingTimer = null;
let currentSlug = null;

const PIPELINE_STEPS = [
  { id: "scout", name: "Scout Web", detail: "Searching sources" },
  { id: "fetch", name: "Fetch Papers", detail: "Ingesting candidate catalogs" },
  { id: "extract", name: "Extract Claims", detail: "Analyzing scientific protocol" },
  { id: "cross", name: "Cross-Check", detail: "Validating experiment code" },
  { id: "clean", name: "Clean Trace", detail: "Running ML simulations" },
  { id: "feed", name: "Feed Model", detail: "Calibrating final results" }
];

function renderModel(models = {}) {
  const model = models.world_model || models.agent_model || "not configured";
  document.querySelector("#model-label").textContent = model;
}

function text(value) {
  return String(value ?? "").replace(/[&<>"']/g, (char) => ({
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    '"': "&quot;",
    "'": "&#39;",
  }[char]));
}

// Map run step number to pipeline index
function getPipelineStatus(step, runStatus, stepIndex) {
  // If the run has finished completely
  if (runStatus === "completed") {
    return "done";
  }

  let activeIndex = -1;
  if (step === 1 || step === 2) {
    activeIndex = 0; // Scout Web
  } else if (step === 3 || step === 4) {
    activeIndex = 1; // Fetch Papers
  } else if (step === 5) {
    activeIndex = 2; // Extract Claims
  } else if (step === 6) {
    activeIndex = 3; // Cross-Check
  } else if (step === 7) {
    activeIndex = 4; // Clean Trace
  } else if (step >= 8) {
    activeIndex = 5; // Feed Model
  }

  if (stepIndex < activeIndex) {
    return "done";
  } else if (stepIndex === activeIndex) {
    return "running";
  } else {
    return "queued";
  }
}

// Function to download paper metadata as TXT file
function downloadPaper(title, authors, year, url, abstract) {
  const content = `TITLE: ${title}\nAUTHORS: ${authors.join(", ")}\nYEAR: ${year}\nURL: ${url}\n\nABSTRACT:\n${abstract}`;
  const blob = new Blob([content], { type: "text/plain;charset=utf-8" });
  const downloadUrl = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = downloadUrl;
  a.download = `${title.replace(/[^a-z0-9]/gi, '_').toLowerCase()}.txt`;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(downloadUrl);
}

// Update the entire dashboard UI based on the fetched status data
function updateUI(data) {
  // Shell transition is no longer needed, UI is static.

  // Update Topbar
  const statusLabel = document.querySelector("#status-label");
  const modeLabel = document.querySelector("#mode-label");
  
  if (data.running) {
    statusLabel.textContent = "AWAKE";
    statusLabel.parentElement.parentElement.classList.add("pulse");
    modeLabel.textContent = "PREDICT";
  } else {
    statusLabel.textContent = "ARCHIVED";
    statusLabel.parentElement.parentElement.classList.remove("pulse");
    modeLabel.textContent = "PREVIOUS RUN";
  }
  
  renderModel(data.models || {});
  const reportButton = document.querySelector("#download-report");
  if (reportButton) {
    reportButton.disabled = !data.slug;
    reportButton.onclick = () => {
      if (data.slug) window.location.href = `/api/report?slug=${encodeURIComponent(data.slug)}`;
    };
  }

  // Determine current step of execution
  const currentStep = data.notebook.length > 0 
    ? Math.max(...data.notebook.map(n => n.step)) + 1
    : (data.events.length > 0 ? 1 : 0);

  // Update Right Sidebar Pipeline
  const pipelineEl = document.querySelector("#pipeline");
  pipelineEl.innerHTML = PIPELINE_STEPS.map((stepInfo, idx) => {
    const status = getPipelineStatus(currentStep, data.result ? "completed" : "running", idx);
    let iconClass = "scout";
    if (idx === 1) iconClass = "fetch";
    else if (idx === 2) iconClass = "extract";
    else if (idx === 3) iconClass = "cross";
    else if (idx === 4) iconClass = "clean";
    else if (idx === 5) iconClass = "feed";
    
    return `
      <li>
        <div class="gicon ${iconClass}"></div>
        <div class="step-num">${idx + 1}</div>
        <div class="step-text">
          <strong>${stepInfo.name.toUpperCase()}</strong>
          <p>${stepInfo.detail}</p>
        </div>
        <div class="badge ${status}">${status.toUpperCase()}</div>
      </li>
    `;
  }).join("");

  // Update Live Logs
  const logEl = document.querySelector("#live-log");
  if (data.error) {
    logEl.textContent = `RUN FAILED\n${data.error}`;
  } else if (data.events.length > 0) {
    logEl.textContent = data.events.map(event => {
      const timeStr = event.ts ? new Date(event.ts).toLocaleTimeString("en-GB") : "";
      return `${timeStr}  [${event.event.toUpperCase()}]  ${event.message}`;
    }).join("\n");
    logEl.scrollTop = logEl.scrollHeight;
  } else {
    logEl.textContent = "Initializing command sequence...";
  }

  // Update Current State Box
  const stateCopyEl = document.querySelector("#current-state-copy");
  const latestEvent = data.events[data.events.length - 1];
  const queryId = data.result ? `Q_${data.slug.replace("open-", "").substring(0, 8).toUpperCase()}` : `Q_PENDING`;
  
  document.querySelector("#status-query-id").textContent = queryId;
  
  stateCopyEl.innerHTML = `
    <div><dt>State ID</dt><dd class="monospace">${data.slug.substring(0, 10).toUpperCase()}</dd></div>
    <div><dt>Timestamp (UTC)</dt><dd class="monospace">${latestEvent ? new Date(latestEvent.ts).toUTCString() : "Pending"}</dd></div>
    <div><dt>Context</dt><dd>${latestEvent ? latestEvent.message : "Ingesting research protocol."}</dd></div>
  `;

  // Update Horizon and Temperature metrics
  document.querySelector("#horizon").textContent = data.result ? data.result.lab_question.budget : "8";
  document.querySelector("#temperature").textContent = "0.20";

  // Update Prediction Cards (Dynamic)
  const branchesContainer = document.querySelector("#predicted-branches");
  if (data.predictions && data.predictions.length > 0) {
    const recentPreds = data.predictions.slice(-3).reverse();
    branchesContainer.innerHTML = recentPreds.map((pred, i) => {
      const prob = (pred.prediction && pred.prediction.claim_support_probability) !== undefined ? pred.prediction.claim_support_probability : 0.0;
      const probPercent = Math.round(prob * 100);
      const actionData = pred.action || {};
      const stateId = (actionData.action_id || "STATE_UNKNOWN").substring(0, 16).toUpperCase();
      const descText = actionData.kind ? actionData.kind.replace(/_/g, ' ') : "Unknown Action";
      let badgeHtml = '<span class="badge anomaly">ANOMALY</span>';
      
      if (prob > 0.6) badgeHtml = '<span class="badge done">LIKELY</span>';
      else if (prob < 0.4) badgeHtml = '<span class="badge low-confidence">LOW CONF</span>';

      return `
        <div class="branch-card" id="pred-card-${i+1}" style="top: ${30 + i * 100}px;">
          <div class="branch-number">${i+1}</div>
          <div class="thumb thumb-${i+1}"></div>
          <div class="branch-info">
            <div class="branch-header">
              <div class="b-col"><span class="lbl">STATE ID</span><span class="val monospace">${stateId}</span></div>
              <div class="b-col"><span class="lbl">PROB.</span><span class="val probability monospace">${prob.toFixed(2)}</span></div>
            </div>
            <div class="pred-progress"><div class="pred-progress-bar bg-red" style="width: ${probPercent}%;"></div></div>
            <div class="branch-footer">
              ${badgeHtml}
              <span class="desc"><span class="gicon skull-micro"></span> ${descText}</span>
            </div>
          </div>
        </div>
      `;
    }).join("");
  }

  // Draw active curves glow on SVG paths
  const linkB1 = document.querySelector(".link-b1");
  const linkB2 = document.querySelector(".link-b2");
  const linkB3 = document.querySelector(".link-b3");
  
  [linkB1, linkB2, linkB3].forEach(l => l.classList.remove("active"));
  if (data.running) {
    if (currentStep === 6) linkB3.classList.add("active");
    else if (currentStep === 7) linkB1.classList.add("active");
  }

  // Update Verdict Banner
  const verdictTitle = document.querySelector("#verdict-title");
  const verdictDesc = document.querySelector("#verdict-panel p");
  
  if (data.result) {
    const claims = Array.isArray(data.result.claims) ? data.result.claims : (Array.isArray(data.claims) ? data.claims : []);
    const isBlocked = claims.some(c => c.verdict === "blocked");
    if (isBlocked) {
      verdictTitle.textContent = "CLAIM BLOCKED.";
      verdictTitle.style.color = "var(--red)";
      verdictDesc.textContent = "The verifier rejected the robust superiority claim. Baseline logistic regression stands.";
    } else {
      verdictTitle.textContent = "CLAIM SUPPORTED.";
      verdictTitle.style.color = "var(--green)";
      verdictDesc.textContent = "The empirical evidence supports the hypothesis. Nonlinear gains are validated.";
    }
  } else {
    verdictTitle.textContent = "SUFFERING IS INEVITABLE.";
    verdictTitle.style.color = "var(--red)";
    verdictDesc.textContent = "The path ahead bleeds. Resistance is a prelude to ruin.";
  }

  // Update Papers List
  const papersEl = document.querySelector("#papers-list");
  if (data.papers && data.papers.length > 0) {
    papersEl.innerHTML = data.papers.slice(0, 3).map((paper, idx) => {
      // Bind click function for download
      window[`downloadPaper_${idx}`] = () => {
        downloadPaper(paper.title, paper.authors, paper.year, paper.url, paper.abstract);
      };
      
      const authorShort = paper.authors.length > 0 ? `${paper.authors[0]} et al.` : "Unknown";
      const safeUrl = text(paper.url || "#");
      
      return `
        <li>
          <div class="paper-info">
            <span class="paper-title" onclick="window.open('${safeUrl}', '_blank')">${text(paper.title)}</span>
            <span class="paper-meta">
              ${text(authorShort)} | ${text(paper.year)}
              <span class="paper-meta-badge">${text(paper.arxiv_id || paper.citation_id || 'Ref')}</span>
            </span>
          </div>
          <button class="btn-download-paper" onclick="window.downloadPaper_${idx}()" title="Download metadata">
            TXT
          </button>
        </li>
      `;
    }).join("");
  } else {
    papersEl.innerHTML = `<li class="empty-row">No papers downloaded yet.</li>`;
  }

  // Update Clean Traces Table
  const traceEl = document.querySelector("#trace-rows");
  if (data.events.length > 0) {
    const lastEvents = data.events.slice(-5).reverse();
    traceEl.innerHTML = lastEvents.map(event => {
      const timeStr = event.ts ? new Date(event.ts).toLocaleTimeString("en-GB") : "";
      
      // Map event names to readable source tags
      let sourceTag = "SCOUT";
      if (event.event.includes("literature")) sourceTag = "SCOUT";
      else if (event.event.includes("dataset")) sourceTag = "FETCH";
      else if (event.event.includes("protocol")) sourceTag = "EXTRACT";
      else if (event.event.includes("code") || event.event.includes("dry_run")) sourceTag = "CROSS";
      else if (event.event.includes("action") || event.event.includes("run_")) sourceTag = "CLEAN";
      else if (event.event.includes("completed")) sourceTag = "FEED";
      
      const statusClass = event.event.includes("failed") || event.event.includes("timeout") ? "bad" : "clean";
      const statusText = event.event.includes("failed") || event.event.includes("timeout") ? "FAILED" : "CLEAN";

      return `
        <tr>
          <td class="monospace">${timeStr}</td>
          <td>${sourceTag}</td>
          <td>${event.message}</td>
          <td class="${statusClass}">${statusText}</td>
        </tr>
      `;
    }).join("");
  } else {
    traceEl.innerHTML = `<tr><td colspan="4" style="text-align:center;">No traces logged yet.</td></tr>`;
  }

  // Update State Diffs Table from real world-model predictions.
  const diffEl = document.querySelector("#diff-rows");
  const predictionRows = (data.predictions || []).slice(-5).reverse();
  if (predictionRows.length > 0) {
    diffEl.innerHTML = predictionRows.map((item) => {
      const action = item.action || {};
      const pred = item.prediction || {};
      const actionId = text(action.action_id || "action");
      const kind = text((action.kind || "unknown").replace(/_/g, " "));
      const rec = text(pred.recommendation || "unknown");
      const claimDelta = text(pred.expected_claim_delta || "none");
      const waste = Number(pred.compute_waste_risk ?? 0);
      const voi = Number(pred.value_of_information ?? 0);
      const conf = Number(pred.claim_support_probability ?? 0);
      const tone = conf >= 0.6 ? "clean" : conf <= 0.35 ? "bad" : "warn";
      return `
        <tr>
          <td><div class="diff-col monospace"><span>${actionId}</span><span class="diff-sub">${kind}</span></div></td>
          <td><div class="diff-col"><span>Waste: ${waste.toFixed(2)}</span><span class="diff-sub">VOI: ${voi.toFixed(2)}</span></div></td>
          <td><div class="diff-col"><span>${rec}</span><span class="diff-sub">${claimDelta}</span></div></td>
          <td><div class="diff-col ${tone}"><span>${text(pred.expected_best_model || "model unknown")}</span><span class="diff-sub">${text(pred.runtime_risk || "runtime unknown")}</span></div></td>
          <td class="${tone} monospace">${conf.toFixed(2)}</td>
        </tr>`;
    }).join("");
  } else {
    diffEl.innerHTML = `<tr><td colspan="5" class="empty-state">Awaiting world model predictions.</td></tr>`;
  }

}

// Poll status data for the active slug
function pollStatus() {
  if (!currentSlug) return;
  
  fetch(`/api/status?slug=${currentSlug}`)
    .then(res => res.json())
    .then(data => {
      updateUI(data);
      
      // Stop polling when finished
      if (!data.running) {
        clearInterval(pollingTimer);
        pollingTimer = null;
        loadPreviousRuns();
      }
    })
    .catch(err => {
      console.error("Error polling status:", err);
    });
}

// Load a specific run's archived data
function loadRun(slug) {
  clearInterval(pollingTimer);
  pollingTimer = null;
  currentSlug = slug;
  
  fetch(`/api/status?slug=${slug}`)
    .then(res => res.json())
    .then(data => {
      updateUI(data);
    })
    .catch(err => {
      console.error("Error loading run:", err);
    });
}

// Previous runs are hidden in the new UI design, keeping function as a no-op
function loadPreviousRuns() {
  // No-op
}

document.querySelectorAll(".nav-item[data-target]").forEach((button) => {
  button.addEventListener("click", () => {
    document.querySelectorAll(".nav-item").forEach((item) => item.classList.remove("active"));
    button.classList.add("active");
    const target = document.querySelector(button.dataset.target);
    if (target) {
      target.scrollIntoView({ behavior: "smooth", block: "center" });
    }
  });
});

// Handle form submit to start a run
document.querySelector("#ask-form").addEventListener("submit", (event) => {
  event.preventDefault();
  
  const questionInput = document.querySelector("#question");
  const question = questionInput.value.trim();
  const budget = Math.max(1, Math.min(32, Number(document.querySelector("#turn-count").value) || 8));
  
  if (!question) return;
  
  // Submit command run to backend
  fetch("/api/run", {
    method: "POST",
    headers: {
      "Content-Type": "application/json"
    },
    body: JSON.stringify({ question, budget })
  })
    .then(res => res.json())
    .then(data => {
    if (data.status === "started") {
        currentSlug = data.slug;
        
        // Immediate UI transition
        document.querySelector("#live-log").textContent = "Ingesting command...";
        
        // Start polling
        clearInterval(pollingTimer);
        pollStatus();
        pollingTimer = setInterval(pollStatus, 1500);
      }
    })
    .catch(err => {
      console.error("Error starting run:", err);
    });
});

// Clear live logs
document.querySelector("#clear-log").addEventListener("click", () => {
  document.querySelector("#live-log").textContent = "";
});

// Initialize on page load
window.addEventListener("load", () => {
  fetch("/api/config")
    .then(res => res.json())
    .then(config => renderModel(config.models || {}))
    .catch(() => {});

  // Load the most recent run automatically
  fetch("/api/previous-runs")
    .then(res => res.json())
    .then(runs => {
      if (runs && runs.length > 0) {
        loadRun(runs[0].slug);
      }
    })
    .catch(err => console.error(err));
  
  // Make helper functions globally accessible for HTML onclick bindings
  window.loadRun = loadRun;
});
