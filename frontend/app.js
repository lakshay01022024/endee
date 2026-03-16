/**
 * Endee AI Knowledge Base — Frontend Logic
 *
 * Handles search, RAG queries, document ingestion,
 * and dynamic UI rendering.
 */

// ── Configuration ────────────────────────────────

const API_BASE = "/api";
let currentMode = "ask"; // "ask" or "search"

// ── DOM Elements ─────────────────────────────────

const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => document.querySelectorAll(sel);

const searchForm = $("#search-form");
const searchInput = $("#search-input");
const searchBtn = $("#btn-search");
const searchLoader = $("#search-loader");
const btnText = searchBtn.querySelector(".btn-text");

const resultsSection = $("#results-section");
const aiAnswer = $("#ai-answer");
const answerText = $("#answer-text");
const modelUsed = $("#model-used");
const sourcesTitle = $("#sources-title");
const resultsCount = $("#results-count");
const resultsGrid = $("#results-grid");
const emptyState = $("#empty-state");

const ingestPanel = $("#ingest-panel");
const ingestForm = $("#ingest-form");
const ingestStatus = $("#ingest-status");

const healthDot = $("#health-dot");
const healthText = $("#health-text");

// ── Background Particles ─────────────────────────

function createParticles() {
    const container = $("#particles");
    const count = 20;
    for (let i = 0; i < count; i++) {
        const particle = document.createElement("div");
        particle.className = "particle";
        const size = Math.random() * 4 + 2;
        particle.style.width = `${size}px`;
        particle.style.height = `${size}px`;
        particle.style.left = `${Math.random() * 100}%`;
        particle.style.animationDuration = `${Math.random() * 20 + 15}s`;
        particle.style.animationDelay = `${Math.random() * 20}s`;
        container.appendChild(particle);
    }
}

// ── Health Check ─────────────────────────────────

async function checkHealth() {
    try {
        const res = await fetch(`${API_BASE}/health`);
        const data = await res.json();
        const status = data.status || "error";

        healthDot.className = `status-dot ${status}`;
        healthText.textContent =
            status === "healthy" ? "Connected" : status === "degraded" ? "Degraded" : "Error";
    } catch {
        healthDot.className = "status-dot error";
        healthText.textContent = "Offline";
    }
}

// ── Mode Toggle ──────────────────────────────────

$$(".mode-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
        $$(".mode-btn").forEach((b) => b.classList.remove("active"));
        btn.classList.add("active");
        currentMode = btn.dataset.mode;

        searchInput.placeholder =
            currentMode === "ask"
                ? "e.g., How do vector databases work?"
                : "e.g., vector database cosine similarity";
    });
});

// ── Search / Ask ─────────────────────────────────

searchForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    const query = searchInput.value.trim();
    if (!query) return;

    setLoading(true);

    try {
        if (currentMode === "ask") {
            await performAsk(query);
        } else {
            await performSearch(query);
        }
    } catch (err) {
        console.error(err);
        showError("Request failed. Is the backend running?");
    } finally {
        setLoading(false);
    }
});

async function performSearch(query) {
    const res = await fetch(`${API_BASE}/search`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query, top_k: 5 }),
    });

    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();

    aiAnswer.classList.add("hidden");
    sourcesTitle.textContent = "Search Results";
    renderResults(data.results);
}

async function performAsk(question) {
    const res = await fetch(`${API_BASE}/ask`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question, top_k: 5 }),
    });

    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();

    // Show AI answer
    answerText.textContent = data.answer;
    modelUsed.textContent = `Model: ${data.model_used}`;
    aiAnswer.classList.remove("hidden");

    sourcesTitle.textContent = "Sources";
    renderResults(data.sources);
}

// ── Render Results ───────────────────────────────

function renderResults(results) {
    emptyState.classList.add("hidden");
    resultsSection.classList.remove("hidden");
    resultsCount.textContent = `${results.length} result${results.length !== 1 ? "s" : ""}`;

    resultsGrid.innerHTML = results
        .map((r, i) => {
            const score = (r.score * 100).toFixed(1);
            const scoreClass =
                r.score >= 0.7 ? "high" : r.score >= 0.4 ? "medium" : "low";

            return `
                <div class="result-card" style="animation-delay: ${i * 60}ms">
                    <div class="result-card-header">
                        <div class="result-card-title">${escapeHtml(r.title || "Untitled")}</div>
                        <div class="score-badge ${scoreClass}">
                            ${score}% match
                        </div>
                    </div>
                    <div class="result-card-content">${escapeHtml(r.content || "")}</div>
                    <div class="result-card-meta">
                        ${r.category ? `<span class="meta-tag">${escapeHtml(r.category)}</span>` : ""}
                        ${r.source ? `<span class="meta-tag">src: ${escapeHtml(r.source)}</span>` : ""}
                        ${r.chunk_index !== undefined ? `<span class="meta-tag">chunk ${r.chunk_index}</span>` : ""}
                    </div>
                </div>
            `;
        })
        .join("");
}

// ── Hint Chips ───────────────────────────────────

$$(".hint-chip").forEach((chip) => {
    chip.addEventListener("click", () => {
        searchInput.value = chip.dataset.query;
        searchForm.dispatchEvent(new Event("submit"));
    });
});

// ── Ingest Panel ─────────────────────────────────

$("#btn-toggle-ingest").addEventListener("click", () => {
    ingestPanel.classList.toggle("open");
});

ingestForm.addEventListener("submit", async (e) => {
    e.preventDefault();

    const title = $("#doc-title").value.trim();
    const content = $("#doc-content").value.trim();
    const category = $("#doc-category").value;
    const source = $("#doc-source").value.trim() || "manual";

    if (!title || !content) return;

    const btn = $("#btn-ingest");
    btn.disabled = true;
    btn.textContent = "Ingesting...";

    try {
        const res = await fetch(`${API_BASE}/ingest`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ title, content, category, source }),
        });

        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const data = await res.json();

        showIngestStatus(
            `✓ ${data.message} (${data.chunks_created} chunks created)`,
            "success"
        );
        ingestForm.reset();
    } catch (err) {
        showIngestStatus(`✗ Ingestion failed: ${err.message}`, "error");
    } finally {
        btn.disabled = false;
        btn.textContent = "Ingest Document";
    }
});

function showIngestStatus(message, type) {
    ingestStatus.textContent = message;
    ingestStatus.className = `status-message ${type}`;
    ingestStatus.classList.remove("hidden");
    setTimeout(() => ingestStatus.classList.add("hidden"), 5000);
}

// ── Utilities ────────────────────────────────────

function setLoading(loading) {
    searchBtn.disabled = loading;
    if (loading) {
        btnText.classList.add("hidden");
        searchLoader.classList.remove("hidden");
    } else {
        btnText.classList.remove("hidden");
        searchLoader.classList.add("hidden");
    }
}

function showError(message) {
    emptyState.classList.add("hidden");
    resultsSection.classList.remove("hidden");
    aiAnswer.classList.add("hidden");
    resultsGrid.innerHTML = `
        <div class="glass-card" style="text-align: center; color: var(--error);">
            <p>⚠️ ${escapeHtml(message)}</p>
        </div>
    `;
    resultsCount.textContent = "";
}

function escapeHtml(text) {
    const div = document.createElement("div");
    div.textContent = text;
    return div.innerHTML;
}

// ── Initialize ───────────────────────────────────

document.addEventListener("DOMContentLoaded", () => {
    createParticles();
    checkHealth();
    setInterval(checkHealth, 30000);
});
