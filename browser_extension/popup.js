/**
 * PassGuard Browser Extension — popup.js
 * Communicates with the locally running Flask server at http://127.0.0.1:5000
 *
 * Routes used:
 *   POST /transform  — phrase → password
 *   POST /generate   — generate password
 *   POST /store      — save to vault
 *   POST /retrieve   — retrieve from vault
 *   GET  /           — server health check
 */

const BASE = "http://127.0.0.1:5000";

// ── state ────────────────────────────────────────────────────────────────────
let serverOnline  = false;
let currentMode   = "passphrase";
let lastPassword  = "";   // last generated password (for fill-on-page)

// ── init ─────────────────────────────────────────────────────────────────────
document.addEventListener("DOMContentLoaded", async () => {
  await checkServer();
  setupTabs();
  setupTransformPanel();
  setupGeneratePanel();
  setupVaultPanel();
});

// ── server health check ───────────────────────────────────────────────────────
async function checkServer() {
  try {
    const res = await fetch(BASE + "/", { method: "GET", signal: AbortSignal.timeout(2000) });
    if (res.ok || res.status === 302) {
      setOnline(true);
    } else {
      setOnline(false);
    }
  } catch {
    setOnline(false);
  }
}

function setOnline(online) {
  serverOnline = online;
  const dot    = document.getElementById("status-dot");
  const text   = document.getElementById("status-text");
  const notice = document.getElementById("offline-notice");
  const tabs   = document.getElementById("main-tabs");

  if (online) {
    dot.className    = "dot online";
    text.textContent = "connected";
    text.style.color = "#2ecc71";
    notice.classList.add("hidden");
    tabs.style.display = "flex";
  } else {
    dot.className    = "dot offline";
    text.textContent = "offline";
    text.style.color = "#e74c3c";
    notice.classList.remove("hidden");
    tabs.style.display = "none";
    document.querySelectorAll(".panel").forEach(p => p.classList.remove("active"));
    document.querySelectorAll(".panel").forEach(p => p.classList.add("hidden"));
  }
}

// ── tabs ──────────────────────────────────────────────────────────────────────
function setupTabs() {
  document.querySelectorAll(".tab").forEach(btn => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".tab").forEach(b => b.classList.remove("active"));
      document.querySelectorAll(".panel").forEach(p => {
        p.classList.remove("active");
        p.classList.add("hidden");
      });
      btn.classList.add("active");
      const panel = document.getElementById("tab-" + btn.dataset.tab);
      if (panel) {
        panel.classList.remove("hidden");
        panel.classList.add("active");
      }
    });
  });
}

// ── API helper ────────────────────────────────────────────────────────────────
async function postJSON(path, data) {
  try {
    const res = await fetch(BASE + path, {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify(data),
      signal:  AbortSignal.timeout(10000),
    });
    return await res.json();
  } catch (e) {
    return { success: false, error: e.message || "Request failed" };
  }
}

// ── fill password into active tab's password field ────────────────────────────
async function fillPasswordOnPage(password) {
  if (!password) return;
  try {
    const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
    await chrome.scripting.executeScript({
      target: { tabId: tab.id },
      func: (pwd) => {
        // Try to find password input fields on the page
        const inputs = [
          ...document.querySelectorAll('input[type="password"]'),
          ...document.querySelectorAll('input[name*="pass"]'),
          ...document.querySelectorAll('input[id*="pass"]'),
          ...document.querySelectorAll('input[placeholder*="assword"]'),
        ];

        // Deduplicate
        const unique = [...new Set(inputs)];

        if (unique.length === 0) {
          alert("PassGuard: No password field found on this page.");
          return false;
        }

        // Fill the first visible password field
        for (const input of unique) {
          if (input.offsetParent !== null) {  // visible check
            input.value = pwd;
            input.dispatchEvent(new Event("input",  { bubbles: true }));
            input.dispatchEvent(new Event("change", { bubbles: true }));
            input.style.outline = "2px solid #7c6af7";
            setTimeout(() => input.style.outline = "", 2000);
            return true;
          }
        }

        // Fallback: fill first one even if not visible
        unique[0].value = pwd;
        unique[0].dispatchEvent(new Event("input",  { bubbles: true }));
        unique[0].dispatchEvent(new Event("change", { bubbles: true }));
        return true;
      },
      args: [password],
    });
    showToast("Password filled on page!", "success");
  } catch (e) {
    // Fallback: just copy to clipboard
    await navigator.clipboard.writeText(password);
    showToast("Copied! Paste manually into password field.", "info");
  }
}

// ── copy to clipboard ─────────────────────────────────────────────────────────
async function copyText(text, btn) {
  await navigator.clipboard.writeText(text);
  const orig = btn.textContent;
  btn.textContent = "Copied!";
  setTimeout(() => btn.textContent = orig, 1500);
}

// ── meters ────────────────────────────────────────────────────────────────────
function setMeter(fillId, labelId, value, text) {
  const fill  = document.getElementById(fillId);
  const label = document.getElementById(labelId);
  const pct   = Math.round((value || 0) * 100);
  fill.style.width      = pct + "%";
  fill.style.background = pct >= 70 ? "#2ecc71" : pct >= 40 ? "#f39c12" : "#e74c3c";
  label.textContent     = text || pct + "%";
}

// ── toast ─────────────────────────────────────────────────────────────────────
function showToast(msg, type = "info") {
  const t = document.getElementById("toast");
  t.textContent = msg;
  t.className   = `toast ${type} show`;
  clearTimeout(window._toastTimer);
  window._toastTimer = setTimeout(() => t.classList.remove("show"), 2500);
}

// ══ TRANSFORM PANEL ══════════════════════════════════════════════════════════

function setupTransformPanel() {
  const btn = document.getElementById("btn-transform");

  btn.addEventListener("click", async () => {
    const phrase = document.getElementById("phrase-input").value.trim();
    const year   = document.getElementById("year-input").value.trim();

    if (!phrase) { showToast("Please enter a phrase.", "error"); return; }

    btn.textContent = "Transforming...";
    btn.disabled    = true;

    const payload = { phrase };
    if (year) payload.year = parseInt(year);

    const res = await postJSON("/transform", payload);

    btn.textContent = "Transform";
    btn.disabled    = false;

    if (!res.success) {
      showToast(res.error || "Transform failed.", "error");
      return;
    }

    lastPassword = res.password;

    // Show password
    document.getElementById("transform-password").textContent = res.password;
    document.getElementById("transform-result").classList.remove("hidden");

    // ML scores
    if (res.ml_scores) {
      const s = res.ml_scores;
      setMeter("tm-str", "tm-str-lbl", s.strength_score,     s.strength     || "");
      setMeter("tm-mem", "tm-mem-lbl", s.memorability_score, s.memorability || "");
      document.getElementById("transform-scores").classList.remove("hidden");
    }

    // Pipeline breakdown
    const p = res.pipeline;
    if (p && typeof p === "object" && p.original_phrase) {
      const pipe = document.getElementById("transform-pipeline");
      pipe.innerHTML =
        `Phrase   : "${p.original_phrase}"<br>` +
        `Kept     : [${(p.filtered_words || []).join(", ")}]<br>` +
        `Anchors  : [${(p.anchor_words   || []).join(", ")}]<br>` +
        `Applied  : ${p.substitution || "—"}<br>` +
        `Core     : ${p.core || "—"}<br>` +
        `Output   : <strong>${p.final || res.password}</strong>`;
      pipe.classList.remove("hidden");
    }

    // Alternative candidates
    if (res.candidates && res.candidates.length) {
      const box = document.getElementById("transform-candidates");
      box.innerHTML = '<div style="font-size:10px;color:#666;margin-bottom:4px;">Alternatives</div>' +
        res.candidates.map((c, i) =>
          `<div class="cand-item">
            <span class="cand-pwd">${c}</span>
            <button class="cand-use" data-pwd="${c}">Use</button>
          </div>`
        ).join("");
      box.querySelectorAll(".cand-use").forEach(b => {
        b.addEventListener("click", () => {
          lastPassword = b.dataset.pwd;
          document.getElementById("transform-password").textContent = b.dataset.pwd;
          showToast("Switched to this variant.", "info");
        });
      });
      box.classList.remove("hidden");
    }
  });

  // Copy
  document.getElementById("btn-copy-transform").addEventListener("click", async () => {
    const pwd = document.getElementById("transform-password").textContent;
    await copyText(pwd, document.getElementById("btn-copy-transform"));
  });

  // Fill on page
  document.getElementById("btn-fill-transform").addEventListener("click", async () => {
    const pwd = document.getElementById("transform-password").textContent;
    await fillPasswordOnPage(pwd);
  });

  // Save to vault
  document.getElementById("btn-save-transform").addEventListener("click", () => {
    const pwd = document.getElementById("transform-password").textContent;
    // Switch to vault tab and pre-fill password
    document.querySelector('[data-tab="vault"]').click();
    document.getElementById("vault-password").value = pwd;
    document.getElementById("vault-label").focus();
    showToast("Switch to Vault tab — add a label and click Store.", "info");
  });
}

// ══ GENERATE PANEL ════════════════════════════════════════════════════════════

function setupGeneratePanel() {
  // Mode buttons
  document.querySelectorAll("[data-mode]").forEach(btn => {
    btn.addEventListener("click", () => {
      currentMode = btn.dataset.mode;
      document.querySelectorAll("[data-mode]").forEach(b => {
        b.className = b.classList.contains("btn-sm")
          ? "btn btn-outline btn-sm"
          : "btn btn-outline";
      });
      btn.className = btn.classList.contains("btn-sm")
        ? "btn btn-primary btn-sm"
        : "btn btn-primary";

      ["passphrase","pattern","random"].forEach(m => {
        document.getElementById("opts-" + m).classList.toggle("hidden", m !== currentMode);
      });
    });
  });

  // Generate
  document.getElementById("btn-generate").addEventListener("click", async () => {
    const btn = document.getElementById("btn-generate");
    btn.textContent = "Generating...";
    btn.disabled    = true;

    const payload = { mode: currentMode, n_candidates: 5 };
    if (currentMode === "passphrase") {
      payload.n_words   = parseInt(document.getElementById("n-words").value) || 4;
      payload.separator = "-";
    } else if (currentMode === "pattern") {
      payload.pattern = document.getElementById("pattern-input").value;
      if (!payload.pattern) {
        showToast("Please enter a pattern.", "error");
        btn.textContent = "Generate"; btn.disabled = false; return;
      }
    } else {
      payload.length = parseInt(document.getElementById("rand-length").value) || 16;
    }

    const res = await postJSON("/generate", payload);
    btn.textContent = "Generate";
    btn.disabled    = false;

    if (!res.success) { showToast(res.error || "Generate failed.", "error"); return; }

    lastPassword = res.password;
    document.getElementById("gen-password").textContent = res.password;
    setMeter("gen-str", "gen-str-lbl", res.strength_proba,    res.strength_label    || "");
    setMeter("gen-mem", "gen-mem-lbl", res.memorability_proba, res.memorability_label || "");
    document.getElementById("gen-result").classList.remove("hidden");

    if (res.all_candidates && res.all_candidates.length > 1) {
      const box = document.getElementById("gen-candidates");
      box.innerHTML = '<div style="font-size:10px;color:#666;margin-bottom:4px;">All candidates</div>' +
        res.all_candidates.slice(1).map(c =>
          `<div class="cand-item">
            <span class="cand-pwd">${c.password}</span>
            <button class="cand-use" data-pwd="${c.password}">Use</button>
          </div>`
        ).join("");
      box.querySelectorAll(".cand-use").forEach(b => {
        b.addEventListener("click", () => {
          lastPassword = b.dataset.pwd;
          document.getElementById("gen-password").textContent = b.dataset.pwd;
          showToast("Switched to this variant.", "info");
        });
      });
      box.classList.remove("hidden");
    }
  });

  // Copy
  document.getElementById("btn-copy-gen").addEventListener("click", async () => {
    await copyText(document.getElementById("gen-password").textContent,
                   document.getElementById("btn-copy-gen"));
  });

  // Fill
  document.getElementById("btn-fill-gen").addEventListener("click", async () => {
    await fillPasswordOnPage(document.getElementById("gen-password").textContent);
  });

  // Save
  document.getElementById("btn-save-gen").addEventListener("click", () => {
    const pwd = document.getElementById("gen-password").textContent;
    document.querySelector('[data-tab="vault"]').click();
    document.getElementById("vault-password").value = pwd;
    document.getElementById("vault-label").focus();
    showToast("Switch to Vault tab — add a label and click Store.", "info");
  });
}

// ══ VAULT PANEL ═══════════════════════════════════════════════════════════════

function setupVaultPanel() {
  // Store
  document.getElementById("btn-store").addEventListener("click", async () => {
    const label    = document.getElementById("vault-label").value.trim();
    const password = document.getElementById("vault-password").value.trim();
    if (!label || !password) {
      showToast("Label and password are required.", "error"); return;
    }
    const res = await postJSON("/store", { label, password });
    if (!res.success) { showToast(res.error || "Store failed.", "error"); return; }
    showToast(`'${label}' stored securely!`, "success");
    document.getElementById("vault-label").value    = "";
    document.getElementById("vault-password").value = "";
  });

  // Retrieve
  document.getElementById("btn-retrieve").addEventListener("click", async () => {
    const label = document.getElementById("retrieve-label").value.trim();
    if (!label) { showToast("Enter a label to retrieve.", "error"); return; }
    const res = await postJSON("/retrieve", { label });
    if (!res.success) { showToast(res.error || "Not found.", "error"); return; }

    document.getElementById("retrieved-pwd").textContent = res.password;
    document.getElementById("retrieve-result").classList.remove("hidden");
    lastPassword = res.password;
  });

  // Copy retrieved
  document.getElementById("btn-copy-retrieved").addEventListener("click", async () => {
    await copyText(document.getElementById("retrieved-pwd").textContent,
                   document.getElementById("btn-copy-retrieved"));
  });

  // Fill retrieved
  document.getElementById("btn-fill-retrieved").addEventListener("click", async () => {
    await fillPasswordOnPage(document.getElementById("retrieved-pwd").textContent);
  });
}
