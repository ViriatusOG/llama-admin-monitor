// ─── App shell: navigation + responsive drawer ────────────────────────────

const SECTIONS = ['monitor', 'logs', 'presets', 'bench', 'chat', 'models', 'install'];
let activeSection = 'monitor';

function switchTab(name) {
    if (!SECTIONS.includes(name)) return;
    activeSection = name;
    SECTIONS.forEach(s => {
        const panel = document.getElementById('section-' + s);
        if (panel) panel.hidden = s !== name;
    });
    document.querySelectorAll('.nav-item[data-section]').forEach(item => {
        const selected = item.dataset.section === name;
        item.classList.toggle('active', selected);
        if (selected) item.setAttribute('aria-current', 'page');
        else item.removeAttribute('aria-current');
    });
    if (name === 'models') loadModelsTab();
    if (name === 'bench') populateBenchModels();
    if (name === 'presets') renderPresetsPage();
    if (name === 'install') loadInstallPage(false);
    if (name === 'chat') setTimeout(() => document.getElementById('chat-input').focus(), 50);
    if (name === 'logs') { logsUnread = 0; renderLogsNav(); const el = document.getElementById('app-log'); el.scrollTop = el.scrollHeight; }

    const wasOpen = document.getElementById('sidebar').classList.contains('open');
    setNavigationOpen(false);
    if (wasOpen) {
        const heading = document.querySelector('#section-' + name + ' .page-title');
        if (heading) {
            heading.setAttribute('tabindex', '-1');
            heading.focus({ preventScroll: true });
        }
    }
}

const mobileQuery = window.matchMedia('(max-width: 900px)');

function setNavigationOpen(open, returnFocus = false) {
    const sidebar = document.getElementById('sidebar');
    open = Boolean(open && mobileQuery.matches);
    const toggle = document.getElementById('mobile-toggle');
    sidebar.classList.toggle('open', open);
    sidebar.inert = mobileQuery.matches && !open;
    document.querySelector('.main-content').inert = open;
    document.getElementById('sidebar-backdrop').hidden = !open;
    toggle.setAttribute('aria-expanded', String(open));
    if (open) document.getElementById('sidebar-close').focus();
    else if (returnFocus) toggle.focus();
}

document.querySelectorAll('.nav-item[data-section]').forEach(item => {
    item.addEventListener('click', () => switchTab(item.dataset.section));
});
document.querySelectorAll('[data-goto]').forEach(btn => {
    btn.addEventListener('click', () => switchTab(btn.dataset.goto));
});
document.getElementById('mobile-toggle').addEventListener('click', () => setNavigationOpen(true));
document.getElementById('sidebar-close').addEventListener('click', () => setNavigationOpen(false, true));
document.getElementById('sidebar-backdrop').addEventListener('click', () => setNavigationOpen(false, true));
mobileQuery.addEventListener('change', () => setNavigationOpen(false));
document.getElementById('sidebar').addEventListener('keydown', e => {
    if (!mobileQuery.matches || !document.getElementById('sidebar').classList.contains('open')) return;
    if (e.key === 'Escape' && !e.defaultPrevented) {
        e.preventDefault();
        setNavigationOpen(false, true);
    }
});
setNavigationOpen(false);

if (window.LlamaAdmin && window.LlamaAdmin.theme) window.LlamaAdmin.theme.init();

// ─── Shared state ─────────────────────────────────────────────────────────

let presets = [];
let serverRunning = false;
let prevLogLen = 0;
let totalVramMb = 0;
let usedVramMb = 0;
let allModelsCache = [];
let lastGpuCount = -1;

// --- Settings Persistence (backend) ---

let settingsSaveTimer = null;

function collectSettings() {
    return {
        preset_id: document.getElementById('preset-select').value,
        port: parseInt(document.getElementById('port').value) || 8080,
        llama_server_path: document.getElementById('set-server-path').value,
        llama_server_cwd: document.getElementById('set-server-cwd').value,
        models_dir: document.getElementById('set-models-dir').value,
    };
}

function saveSettings() {
    // Debounce: wait 400ms of inactivity before saving
    clearTimeout(settingsSaveTimer);
    settingsSaveTimer = setTimeout(() => {
        fetch('/api/settings', {
            method: 'PUT',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(collectSettings()),
        }).catch(() => {});
    }, 400);
}

function applySettings(s) {
    if (!s) return;
    if (s.port) document.getElementById('port').value = s.port;
    if (s.llama_server_path !== undefined) document.getElementById('set-server-path').value = s.llama_server_path;
    if (s.llama_server_cwd !== undefined) document.getElementById('set-server-cwd').value = s.llama_server_cwd;
    if (s.models_dir !== undefined) document.getElementById('set-models-dir').value = s.models_dir;
}

// Auto-save on any launch-form change
document.getElementById('controls').addEventListener('input', saveSettings);
document.getElementById('controls').addEventListener('change', saveSettings);

// Load presets and populate dropdown
async function loadPresets(selectId) {
    const [presetsResp, settingsResp] = await Promise.all([
        fetch('/api/presets'),
        selectId === undefined ? fetch('/api/settings') : Promise.resolve(null),
    ]);
    presets = await presetsResp.json();
    const saved = settingsResp ? await settingsResp.json() : null;

    const sel = document.getElementById('preset-select');
    sel.innerHTML = '';
    presets.forEach(p => {
        const opt = document.createElement('option');
        opt.value = p.id;
        opt.textContent = p.name;
        sel.appendChild(opt);
    });

    const targetId = selectId ?? (saved?.preset_id || null);
    if (targetId && presets.find(p => p.id === targetId)) {
        sel.value = targetId;
    } else if (presets.length > 0) {
        sel.value = presets[0].id;
    }

    if (selectId === undefined && saved) applySettings(saved);
    saveSettings();
    document.getElementById('nav-count-presets').textContent = presets.length || '';
    renderPresetsPage();
    renderRuntime();
}

// Initial load
loadPresets();
loadGpuEnv();

// --- GPU Environment ---

async function loadGpuEnv() {
    try {
        const resp = await fetch('/api/gpu-env');
        const data = await resp.json();
        const env = data.env;
        const archs = data.architectures;
        const detected = data.detected;

        const sel = document.getElementById('gpu-env-arch');
        sel.innerHTML = '';
        archs.forEach(a => {
            const opt = document.createElement('option');
            opt.value = a.id;
            let label = a.name;
            if (detected && detected.arch === a.id) label += ' (detected)';
            opt.textContent = label;
            sel.appendChild(opt);
        });
        sel.value = env.arch;

        document.getElementById('gpu-env-devices').value = env.devices;
        document.getElementById('gpu-env-rocm-path').value = env.rocm_path || '/opt/rocm';

        const infoEl = document.getElementById('gpu-detected-info');
        const summaryInfo = document.getElementById('gpu-env-info');
        if (detected) {
            infoEl.textContent = 'Detected: ' + detected.count + ' GPU(s): ' + detected.names.join(', ');
            summaryInfo.textContent = '\u2014 ' + detected.count + ' GPU(s) detected';
        } else {
            infoEl.textContent = 'No GPU detected via rocminfo/nvidia-smi';
            summaryInfo.textContent = '';
        }
    } catch (err) {
        console.error('Failed to load GPU env:', err);
    }
}

// --- Modals: shared open/close helpers ---

function openModal(id) { document.getElementById(id).classList.add('open'); }
function closeModal(id) { document.getElementById(id).classList.remove('open'); }
function modalIsOpen(id) { return document.getElementById(id).classList.contains('open'); }

// --- Config Modal ---

function openConfigModal(section) {
    openModal('config-modal');
    checkAppUpdates();
    if (section === 'updates') {
        const details = document.getElementById('config-updates-section');
        details.open = true;
        setTimeout(() => details.scrollIntoView({ block: 'start', behavior: 'smooth' }), 60);
    }
}
function closeConfigModal() { closeModal('config-modal'); }

document.getElementById('config-modal').addEventListener('click', e => {
    if (e.target === e.currentTarget) closeConfigModal();
});

function saveConfig() {
    // Save server paths via settings
    clearTimeout(settingsSaveTimer);
    fetch('/api/settings', {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(collectSettings()),
    }).catch(() => {});

    // Save GPU env
    const env = {
        arch: document.getElementById('gpu-env-arch').value,
        devices: document.getElementById('gpu-env-devices').value.trim(),
        rocm_path: document.getElementById('gpu-env-rocm-path').value.trim() || '/opt/rocm',
        extra_env: [],
    };
    fetch('/api/gpu-env', {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(env),
    }).catch(() => {});

    closeConfigModal();
    showToast('Configuration saved', 'success');
}

// --- File Browser ---

let fbTargetId = '';
let fbFilter = '';
let fbCurrentPath = '';

function openFileBrowser(targetId, filter) {
    fbTargetId = targetId;
    fbFilter = filter === 'dir' ? '' : (filter || '');
    const modal = document.getElementById('file-browser-modal');
    // If target already has a path, start there; otherwise home
    const current = document.getElementById(targetId).value;
    let startPath = '';
    if (current) {
        // Use parent directory of current value
        const parts = current.split('/');
        parts.pop();
        startPath = parts.join('/') || '/';
    }
    // Show/hide "Select this folder" for dir-mode
    const selectBtn = modal.querySelector('.modal-buttons .btn-primary');
    selectBtn.classList.toggle('hidden', filter !== 'dir');
    modal.classList.add('open');
    fileBrowserGo(startPath);
}

function closeFileBrowser() { closeModal('file-browser-modal'); }

document.getElementById('file-browser-modal').addEventListener('click', e => {
    if (e.target === e.currentTarget) closeFileBrowser();
});

async function fileBrowserGo(path) {
    const entriesEl = document.getElementById('fb-entries');
    entriesEl.innerHTML = '<div class="fb-empty">Loading...</div>';
    const params = new URLSearchParams();
    if (path) params.set('path', path);
    if (fbFilter) params.set('filter', fbFilter);
    try {
        const resp = await fetch('/api/browse?' + params);
        const data = await resp.json();
        if (data.error) {
            entriesEl.innerHTML = '<div class="fb-empty">' + escapeHtml(data.error) + '</div>';
            return;
        }
        fbCurrentPath = data.path;
        document.getElementById('fb-path-input').value = data.path;
        if (data.entries.length === 0) {
            entriesEl.innerHTML = '<div class="fb-empty">Empty directory</div>';
            return;
        }
        fbEntries = data.entries;
        renderFileBrowser();
    } catch (err) {
        entriesEl.innerHTML = '<div class="fb-empty">Error: ' + escapeHtml(err.message) + '</div>';
    }
}

let fbEntries = [];
let fbSortKey = 'name';
let fbSortDir = 'asc';

function sortFileBrowser(key) {
    if (fbSortKey === key) {
        fbSortDir = fbSortDir === 'asc' ? 'desc' : 'asc';
    } else {
        fbSortKey = key;
        fbSortDir = key === 'size' ? 'desc' : 'asc';
    }
    renderFileBrowser();
}

function renderFileBrowser() {
    const entriesEl = document.getElementById('fb-entries');
    const headerEl = document.getElementById('fb-header');
    if (headerEl) {
        headerEl.innerHTML =
            '<span class="sortable" onclick="sortFileBrowser(\'name\')">Name' +
            sortIndicator(fbSortKey, 'name', fbSortDir) + '</span>' +
            '<span class="sortable" onclick="sortFileBrowser(\'size\')">Size' +
            sortIndicator(fbSortKey, 'size', fbSortDir) + '</span>';
    }
    if (!fbEntries || fbEntries.length === 0) {
        entriesEl.innerHTML = '<div class="fb-empty">Empty directory</div>';
        return;
    }
    // Directories always stay above files -- sorting a mixed list by size
    // would scatter folders through it, which reads as broken.
    const sorted = fbEntries.slice().sort((x, y) => {
        if (x.is_dir !== y.is_dir) return x.is_dir ? -1 : 1;
        return compareValues(x[fbSortKey], y[fbSortKey], fbSortDir);
    });
    entriesEl.innerHTML = sorted.map(e => {
        if (e.is_dir) {
            return '<div class="fb-entry fb-entry-dir" onclick="fileBrowserGo(\'' + jsStr(e.path) + '\')">' +
                '<span class="fb-entry-icon">\u{1F4C1}</span>' +
                '<span class="fb-entry-name">' + escapeHtml(e.name) + '</span></div>';
        } else {
            return '<div class="fb-entry fb-entry-file fb-match" onclick="fileBrowserSelect(\'' + jsStr(e.path) + '\')">' +
                '<span class="fb-entry-icon">\u{1F4C4}</span>' +
                '<span class="fb-entry-name">' + escapeHtml(e.name) + '</span>' +
                '<span class="fb-entry-size">' + escapeHtml(e.size_display) + '</span></div>';
        }
    }).join('');
}

function fileBrowserUp() {
    if (fbCurrentPath && fbCurrentPath !== '/') {
        const parts = fbCurrentPath.split('/');
        parts.pop();
        fileBrowserGo(parts.join('/') || '/');
    }
}

function fileBrowserSelect(path) {
    document.getElementById(fbTargetId).value = path || fbCurrentPath;
    document.getElementById(fbTargetId).dispatchEvent(new Event('input', { bubbles: true }));
    closeFileBrowser();
}

let pendingBuild = 'vulkan';

// --- Device picker (preset editor) ---

// Keyed by backend: a separate CUDA build lists different devices.
const deviceListCache = {};

function selectedDevices() {
    const ticked = Array.from(document.querySelectorAll('#modal-devices input[type="checkbox"]:checked')).map(i => i.value);
    if (ticked.length) return ticked.join(',');
    // Devices the preset names that the list did not offer (binary changed,
    // list failed) are kept verbatim so saving never silently drops them.
    return document.getElementById('modal-devices').dataset.unlisted || '';
}

async function loadDevicePicker(selected) {
    const host = document.getElementById('modal-devices');
    const chosen = selected.split(',').map(s => s.trim()).filter(Boolean);
    const backend = document.getElementById('modal-backend').value || 'vulkan';
    host.dataset.unlisted = '';
    if (!deviceListCache[backend]) {
        host.innerHTML = '<span class="help-text">Loading devices from llama-server\u2026</span>';
        try {
            const res = await fetch('/api/devices?backend=' + encodeURIComponent(backend));
            const data = await res.json();
            if (data.error) throw new Error(data.error);
            deviceListCache[backend] = data.devices || [];
        } catch (err) {
            host.innerHTML = '<span class="help-text">Could not list devices: ' + escapeHtml(err.message) + (chosen.length ? ' \u00b7 keeping: ' + escapeHtml(chosen.join(', ')) : '') + '</span>';
            host.dataset.unlisted = chosen.join(',');
            return;
        }
    }
    const devices = deviceListCache[backend];
    if (devices.length === 0) {
        host.innerHTML = '<span class="help-text">llama-server reports no offload devices (CPU build?).</span>';
        host.dataset.unlisted = chosen.join(',');
        return;
    }
    const known = new Set(devices.map(d => d.id));
    host.dataset.unlisted = chosen.filter(c => !known.has(c)).join(',');
    host.innerHTML = devices.map(d =>
        '<label class="device-option"><input type="checkbox" value="' + escapeHtml(d.id) + '"' + (chosen.includes(d.id) ? ' checked' : '') + '>' +
        '<span class="mono">' + escapeHtml(d.id) + '</span><span>' + escapeHtml(d.name) + '</span>' +
        (d.total_mib ? '<span class="help-text">' + (d.total_mib / 1024).toFixed(1) + ' GB</span>' : '') + '</label>'
    ).join('') + '<span class="help-text">None ticked = all devices. Tick one card to keep a model off the other.</span>' +
    (host.dataset.unlisted ? '<span class="help-text">Not offered by this binary but kept: ' + escapeHtml(host.dataset.unlisted) + '</span>' : '');
}

function onBackendChange() {
    const isCuda = document.getElementById('modal-backend').value === 'cuda';
    const ts = document.getElementById('modal-tensor-split');
    ts.disabled = isCuda;
    ts.title = isCuda ? 'Not used -- a CUDA build only sees the NVIDIA GPU' : '';
    // The device list belongs to the selected build; keep the ticks.
    loadDevicePicker(selectedDevices());
}

// --- Optimize / Benchmark ---

let benchRunning = false;
let lastServerError = null;

async function populateBenchModels() {
    const splitsEl = document.getElementById('bench-splits');
    if (splitsEl && !splitsEl.value) {
        splitsEl.value = '50/50, 55/45, 61/39, 65/35, 70/30';
    }
    const sel = document.getElementById('bench-model-select');
    if (!sel) return;
    const prev = sel.value;
    await loadModelsCache();
    if (allModelsCache.length === 0) {
        sel.innerHTML = '<option value="">No models found -- download one first</option>';
        return;
    }
    sel.innerHTML = allModelsCache.filter(m => !m.is_mmproj).map(m =>
        '<option value="' + escapeHtml(m.path) + '">' + escapeHtml(m.model_name || m.filename) +
        (m.quant_type ? ' (' + escapeHtml(m.quant_type) + ')' : '') + ' \u2014 ' + escapeHtml(m.size_display) + '</option>'
    ).join('');
    if (prev) sel.value = prev;
    if (!sel.value && sel.options.length > 0) sel.selectedIndex = 0;
}

async function toggleBenchmark() {
    if (benchRunning) {
        const proceed = await showConfirm('Stop benchmark',
            'Stop the running benchmark? Results collected so far will be kept.');
        if (!proceed) return;
        try {
            const resp = await fetch('/api/bench/cancel', { method: 'POST' });
            const data = await resp.json();
            if (!data.ok) showToast('Could not stop: ' + (data.error || 'unknown'), 'error');
            else showToast('Stopping benchmark...', 'success');
        } catch (err) {
            showToast('Could not stop: ' + err.message, 'error');
        }
        return;
    }

    if (serverRunning) {
        showToast('Stop the llama.cpp server first -- benchmarking needs the GPUs', 'error');
        return;
    }

    const modelPath = document.getElementById('bench-model-select').value;
    if (!modelPath) {
        showToast('No model selected', 'error');
        return;
    }
    const splits = document.getElementById('bench-splits').value
        .split(/[\n,]/)
        .map(s => s.trim())
        .filter(s => s.length > 0);
    if (splits.length === 0) {
        showToast('Enter at least one tensor split ratio', 'error');
        return;
    }

    const parseList = id => document.getElementById(id).value
        .split(/[\n,]/)
        .map(s => parseInt(s.trim()))
        .filter(n => !isNaN(n));

    const batchSizes = parseList('bench-batch');
    const ubatchSizes = parseList('bench-ubatch');
    const threads = parseList('bench-threads');

    const ngl = parseInt(document.getElementById('bench-ngl').value) || 999;

    // Every field multiplies the sweep; make the size explicit before a long run.
    const runs = splits.length * Math.max(1, batchSizes.length) * Math.max(1, ubatchSizes.length) * Math.max(1, threads.length);
    if (runs > 6) {
        const proceed = await showConfirm('Run ' + runs + ' benchmarks?',
            'This sweep is ' + splits.length + ' split' + (splits.length === 1 ? '' : 's') +
            ' × ' + Math.max(1, batchSizes.length) + ' batch × ' + Math.max(1, ubatchSizes.length) + ' micro-batch × ' + Math.max(1, threads.length) + ' thread settings = ' + runs +
            ' llama-bench runs, each loading the model from scratch. Expect several minutes per run for large models.', 'Start ' + runs + ' runs');
        if (!proceed) return;
    }

    try {
        const payload = { model_path: modelPath, splits: splits, gpu_layers: ngl, backend: getConfig().backend || '' };
        if (batchSizes.length > 0) payload.batch_sizes = batchSizes;
        if (ubatchSizes.length > 0) payload.ubatch_sizes = ubatchSizes;
        if (threads.length > 0) payload.threads = threads;

        const resp = await fetch('/api/bench/run', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload),
        });
        const data = await resp.json();
        if (!data.ok) {
            showToast('Benchmark failed: ' + (data.error || 'unknown'), 'error');
            return;
        }
        document.getElementById('bench-panel').hidden = false;
        showToast('Benchmark started -- this will take a few minutes', 'success');
    } catch (err) {
        showToast('Benchmark failed: ' + err.message, 'error');
    }
}

async function applyBenchResult(split, batch, ubatch, threads) {
    const id = document.getElementById('preset-select').value;
    const p = presets.find(pr => pr.id === id);
    if (!p) {
        showToast('No preset selected to apply this to', 'error');
        return;
    }
    const proceed = await showConfirm('Apply benchmark result',
        'Set tensor split "' + split + '", batch ' + batch + ', micro-batch ' + ubatch + ' and ' + threads + ' threads on preset "' + p.name + '"?');
    if (!proceed) return;

    const updated = Object.assign({}, p, { 
        tensor_split: split,
        batch_size: batch,
        ubatch_size: ubatch,
        threads: threads
    });
    try {
        const resp = await fetch('/api/presets/' + encodeURIComponent(p.id), {
            method: 'PUT',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(updated),
        });
        if (!resp.ok) {
            showToast('Failed to update preset', 'error');
            return;
        }
        showToast('Applied best settings to ' + p.name, 'success');
        loadPresets();
    } catch (err) {
        showToast('Failed to update preset: ' + err.message, 'error');
    }
}

let benchLastDone = false;

function updateBenchProgress(b) {
    if (!b) return;
    const panel = document.getElementById('bench-panel');
    const statusEl = document.getElementById('bench-status');
    const barEl = document.getElementById('bench-bar');
    const resultsEl = document.getElementById('bench-results');

    // Button and badge state must update even when idle, so they don't
    // stay stuck showing a stale label from a previous run.
    benchRunning = b.running;
    const toggleBtn = document.getElementById('btn-bench-toggle');
    if (toggleBtn) {
        toggleBtn.textContent = b.running ? 'Stop benchmark' : 'Start benchmark';
        toggleBtn.className = 'btn ' + (b.running ? 'btn-danger' : 'btn-primary');
    }
    const benchBadge = document.getElementById('nav-count-bench');
    if (benchBadge) {
        benchBadge.textContent = b.running
            ? b.completed + '/' + b.total
            : (b.results.length > 0 ? String(b.results.length) : '');
    }

    if (!b.running && !b.done && b.results.length === 0) {
        return;
    }

    panel.hidden = false;

    const pct = b.total > 0 ? (b.completed / b.total) * 100 : 0;
    barEl.style.width = pct.toFixed(1) + '%';

    if (b.running) {
        statusEl.textContent = 'Benchmarking ' + (b.current_split || '...') +
            ' (b:' + b.current_batch + ' ub:' + b.current_ubatch + ' t:' + b.current_threads + ') ' +
            '  (' + b.completed + ' of ' + b.total + ' complete)';
        benchLastDone = false;
    } else if (b.done) {
        statusEl.textContent = b.cancelled
            ? 'Stopped. ' + b.results.length + ' of ' + b.total + ' runs completed.'
            : (b.best_result ? 'Done. Fastest: ' + b.best_result.tensor_split + ' (b:' + b.best_result.batch_size + ' t:' + b.best_result.threads + ')' : 'Done.');
        if (!benchLastDone) {
            benchLastDone = true;
            if (b.error) showToast('Benchmark error: ' + b.error, 'error');
            else showToast('Benchmark complete', 'success');
        }
    }

    resultsEl.innerHTML = b.results.map(r => {
        // Find if this is the best result by comparing its values exactly, or by reference if available, but here it's serialized.
        const isBest = b.best_result && b.best_result.tensor_split === r.tensor_split 
                        && b.best_result.batch_size === r.batch_size 
                        && b.best_result.ubatch_size === r.ubatch_size 
                        && b.best_result.threads === r.threads;
        
        return '<div class="bench-grid-row' + (isBest ? ' bench-best' : '') + '">' +
            '<span>' + escapeHtml(r.tensor_split) + (isBest ? ' \u2605' : '') + '</span>' +
            '<span>' + r.batch_size + '</span>' +
            '<span>' + r.ubatch_size + '</span>' +
            '<span>' + r.threads + '</span>' +
            '<span>' + r.prompt_tps.toFixed(1) + '</span>' +
            '<span>' + r.gen_tps.toFixed(1) + '</span>' +
            '<span>' + (b.done ? '<button class="btn btn-xs" onclick="applyBenchResult(\'' + jsStr(r.tensor_split) + '\', ' + r.batch_size + ', ' + r.ubatch_size + ', ' + r.threads + ')">Apply</button>' : '') + '</span>' +
            '</div>';
    }).join('');
}


// --- Generic Confirm Modal ---

let confirmResolve = null;

function showConfirm(title, message, okLabel = 'OK', danger = false) {
    document.getElementById('confirm-title').textContent = title;
    document.getElementById('confirm-message').textContent = message;
    const ok = document.getElementById('confirm-ok');
    ok.textContent = okLabel;
    ok.className = 'btn ' + (danger ? 'btn-danger' : 'btn-primary');
    openModal('confirm-modal');
    setTimeout(() => ok.focus(), 30);
    return new Promise(resolve => {
        confirmResolve = resolve;
    });
}

function closeConfirmModal(result) {
    closeModal('confirm-modal');
    if (confirmResolve) {
        confirmResolve(result);
        confirmResolve = null;
    }
}

document.getElementById('confirm-modal').addEventListener('click', e => {
    if (e.target === e.currentTarget) closeConfirmModal(false);
});

// --- Hugging Face Download ---

let hfCurrentRepo = '';
let hfDownloading = false;
let hfLastHandledDone = false;

function openHfModal() {
    openModal('hf-modal');
    document.getElementById('hf-search-input').value = '';
    document.getElementById('hf-repo-list').innerHTML = '<div class="fb-empty">Search for a model above.</div>';
    hfShowRepos();
    setTimeout(() => document.getElementById('hf-search-input').focus(), 50);
}

function closeHfModal() { closeModal('hf-modal'); }

document.getElementById('hf-modal').addEventListener('click', e => {
    if (e.target === e.currentTarget) closeHfModal();
});

function hfShowRepos() {
    document.getElementById('hf-companion-bar').classList.add('hidden');
    document.getElementById('hf-repo-list').classList.remove('hidden');
    document.getElementById('hf-file-list').classList.add('hidden');
    document.getElementById('hf-repo-header').classList.remove('hidden');
    document.getElementById('hf-file-header').classList.add('hidden');
    document.getElementById('hf-back-btn').classList.add('hidden');
    document.getElementById('hf-hint').textContent = 'Click a model to view its available files.';
}

async function hfSearch() {
    const q = document.getElementById('hf-search-input').value.trim();
    const listEl = document.getElementById('hf-repo-list');
    if (!q) return;
    listEl.innerHTML = '<div class="fb-empty">Searching...</div>';
    hfShowRepos();
    try {
        const resp = await fetch('/api/hf/search?q=' + encodeURIComponent(q));
        const data = await resp.json();
        if (data.error) {
            listEl.innerHTML = '<div class="fb-empty">' + escapeHtml(data.error) + '</div>';
            return;
        }
        if (!data.results || data.results.length === 0) {
            listEl.innerHTML = '<div class="fb-empty">No results</div>';
            return;
        }
        hfRepoData = data.results;
        renderHfRepos();
    } catch (err) {
        listEl.innerHTML = '<div class="fb-empty">Error: ' + escapeHtml(err.message) + '</div>';
    }
}

let hfRepoData = [];
let hfRepoSortKey = 'downloads';
let hfRepoSortDir = 'desc';

function sortHfRepos(key) {
    if (hfRepoSortKey === key) {
        hfRepoSortDir = hfRepoSortDir === 'asc' ? 'desc' : 'asc';
    } else {
        hfRepoSortKey = key;
        hfRepoSortDir = key === 'downloads' ? 'desc' : 'asc';
    }
    renderHfRepos();
}

function renderHfRepos() {
    const listEl = document.getElementById('hf-repo-list');
    const headerEl = document.getElementById('hf-repo-header');
    if (headerEl) {
        headerEl.innerHTML =
            '<span class="sortable" onclick="sortHfRepos(\'id\')">Model' +
            sortIndicator(hfRepoSortKey, 'id', hfRepoSortDir) + '</span>' +
            '<span class="sortable" onclick="sortHfRepos(\'downloads\')">Downloads' +
            sortIndicator(hfRepoSortKey, 'downloads', hfRepoSortDir) + '</span>';
    }
    if (!hfRepoData || hfRepoData.length === 0) {
        listEl.innerHTML = '<div class="fb-empty">No results</div>';
        return;
    }
    const sorted = hfRepoData.slice().sort((x, y) =>
        compareValues(x[hfRepoSortKey], y[hfRepoSortKey], hfRepoSortDir));
    listEl.innerHTML = sorted.map(r =>
            '<div class="fb-entry fb-entry-file fb-match" onclick="hfShowFiles(\'' + jsStr(r.id) + '\')">' +
            '<span class="fb-entry-icon">\u{1F4E6}</span>' +
            '<span class="fb-entry-name">' + escapeHtml(r.id) + '</span>' +
            '<span class="fb-entry-size" title="Total downloads">' + r.downloads.toLocaleString() + ' downloads</span></div>'
        ).join('');
}

async function hfShowFiles(repoId) {
    hfCurrentRepo = repoId;
    document.getElementById('hf-repo-list').classList.add('hidden');
    document.getElementById('hf-repo-header').classList.add('hidden');
    document.getElementById('hf-file-header').classList.remove('hidden');
    const fileListEl = document.getElementById('hf-file-list');
    fileListEl.classList.remove('hidden');
    document.getElementById('hf-back-btn').classList.remove('hidden');
    document.getElementById('hf-hint').textContent = 'Click a file to start downloading it to your models directory.';
    fileListEl.innerHTML = '<div class="fb-empty">Loading files...</div>';
    try {
        const resp = await fetch('/api/hf/files?repo=' + encodeURIComponent(repoId));
        const data = await resp.json();
        if (data.error) {
            fileListEl.innerHTML = '<div class="fb-empty">' + escapeHtml(data.error) + '</div>';
            return;
        }
        if (!data.files || data.files.length === 0) {
            fileListEl.innerHTML = '<div class="fb-empty">No .gguf files found</div>';
            return;
        }
        hfFileData = data.files;
        renderHfCompanionBar();
        renderHfFiles();
    } catch (err) {
        fileListEl.innerHTML = '<div class="fb-empty">Error: ' + escapeHtml(err.message) + '</div>';
    }
}

let hfFileData = [];

// Offers the repo's projector files as a companion for whichever model the
// user picks. Only shown when the repo actually ships one.
function renderHfCompanionBar() {
    const bar = document.getElementById('hf-companion-bar');
    const sel = document.getElementById('hf-companion-select');
    const projectors = hfFileData.filter(f => f.is_mmproj);
    if (projectors.length === 0) {
        bar.classList.add('hidden');
        sel.innerHTML = '<option value="">None</option>';
        return;
    }
    sel.innerHTML = '<option value="">None</option>' + projectors.map(f =>
        '<option value="' + escapeHtml(f.filename) + '">' + escapeHtml(f.filename) + ' (' + escapeHtml(f.size_display) + ')</option>').join('');
    // Preselect the smallest projector: it is the usual pairing and the
    // choice is one click away if a bigger one is wanted.
    const smallest = projectors.slice().sort((a, b) => a.size_bytes - b.size_bytes)[0];
    sel.value = smallest.filename;
    bar.classList.remove('hidden');
}

let hfFileSortKey = 'filename';
let hfFileSortDir = 'asc';

function sortHfFiles(key) {
    if (hfFileSortKey === key) {
        hfFileSortDir = hfFileSortDir === 'asc' ? 'desc' : 'asc';
    } else {
        hfFileSortKey = key;
        hfFileSortDir = key === 'size_bytes' ? 'desc' : 'asc';
    }
    renderHfFiles();
}

function renderHfFiles() {
    const fileListEl = document.getElementById('hf-file-list');
    const headerEl = document.getElementById('hf-file-header');
    if (headerEl) {
        headerEl.innerHTML =
            '<span class="sortable" onclick="sortHfFiles(\'filename\')">File (click to download)' +
            sortIndicator(hfFileSortKey, 'filename', hfFileSortDir) + '</span>' +
            '<span class="sortable" onclick="sortHfFiles(\'fits_vram\')">Fits VRAM?' +
            sortIndicator(hfFileSortKey, 'fits_vram', hfFileSortDir) + '</span>' +
            '<span class="sortable" onclick="sortHfFiles(\'size_bytes\')">Size' +
            sortIndicator(hfFileSortKey, 'size_bytes', hfFileSortDir) + '</span>';
    }
    if (!hfFileData || hfFileData.length === 0) {
        fileListEl.innerHTML = '<div class="fb-empty">No .gguf files found</div>';
        return;
    }
    const sorted = hfFileData.slice().sort((x, y) => {
        let a, b;
        if (hfFileSortKey === 'fits_vram') {
            a = vramFitCheck(x.size_bytes).cls === 'vram-fit-ok';
            b = vramFitCheck(y.size_bytes).cls === 'vram-fit-ok';
        } else {
            a = x[hfFileSortKey];
            b = y[hfFileSortKey];
        }
        return compareValues(a, b, hfFileSortDir);
    });
    fileListEl.innerHTML = sorted.map(f => {
        const fit = vramFitCheck(f.size_bytes);
        return '<div class="fb-entry fb-entry-file fb-match" onclick="hfDownload(\'' + jsStr(f.filename) + '\', \'' + jsStr(f.size_display) + '\', ' + (f.is_mmproj ? 'true' : 'false') + ')">' +
            '<span class="fb-entry-icon">' + (f.is_mmproj ? '\u{1F5BC}' : '\u{1F4C4}') + '</span>' +
            '<span class="fb-entry-name">' + escapeHtml(f.filename) + (f.is_mmproj ? ' <span class="chip-projector">projector</span>' : '') + '</span>' +
            '<span class="fb-entry-size ' + fit.cls + '" title="' + escapeHtml(fit.title) + '">' + fit.label + '</span>' +
            '<span class="fb-entry-size">' + escapeHtml(f.size_display) + '</span></div>';
    }).join('');
}

async function hfDownload(filename, sizeDisplay, isProjector) {
    if (hfDownloading) {
        showToast('A download is already in progress', 'error');
        return;
    }
    // A projector clicked directly downloads alone; a model brings the
    // selected companion along.
    const companion = isProjector ? '' : (document.getElementById('hf-companion-select').value || '');
    let message = 'Download ' + filename + ' (' + (sizeDisplay || 'unknown size') + ') to your models directory?';
    if (companion) message += '\n\nThe companion projector ' + companion + ' will be downloaded right after it.';
    const proceed = await showConfirm(isProjector ? 'Download projector' : 'Download model', message, 'Download');
    if (!proceed) {
        return;
    }
    hfDownloading = true;
    hfLastHandledDone = false;
    document.getElementById('hf-download-progress').hidden = false;
    document.getElementById('hf-progress-filename').textContent = filename;
    document.getElementById('hf-progress-pct').textContent = '0%';
    document.getElementById('hf-progress-bar').style.width = '0%';
    try {
        const resp = await fetch('/api/hf/download', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ repo: hfCurrentRepo, filename: filename, companion: companion }),
        });
        const data = await resp.json();
        if (!data.ok) {
            showToast('Download failed: ' + (data.error || 'unknown'), 'error');
            hfDownloading = false;
            document.getElementById('hf-download-progress').hidden = true;
        }
    } catch (err) {
        showToast('Download failed: ' + err.message, 'error');
        hfDownloading = false;
        document.getElementById('hf-download-progress').hidden = true;
    }
}

function vramFitCheck(sizeBytes) {
    if (totalVramMb <= 0) {
        return { label: '\u2014', cls: '', title: 'GPU VRAM data not yet available' };
    }
    const sizeMb = sizeBytes / 1024 / 1024;
    // Reserve headroom for compute buffers and KV cache -- this checks
    // model weights only, not a running-context estimate.
    const overheadMb = 1536;
    const usableMb = totalVramMb - overheadMb;
    const totalGb = (totalVramMb / 1024).toFixed(1);
    const sizeGb = (sizeMb / 1024).toFixed(1);
    if (sizeMb <= usableMb) {
        return {
            label: '\u2713 Fits (' + sizeGb + 'GB / ' + totalGb + 'GB)',
            cls: 'vram-fit-ok',
            title: 'Model weights fit within combined VRAM (' + totalGb + 'GB). Actual usable context depends on remaining headroom.',
        };
    }
    return {
        label: '\u2717 Too large (' + sizeGb + 'GB / ' + totalGb + 'GB)',
        cls: 'vram-fit-bad',
        title: 'Model weights exceed combined VRAM (' + totalGb + 'GB). Would need a smaller quantization or CPU offload.',
    };
}

async function refreshModels() {
    try {
        await fetch('/api/models/refresh', { method: 'POST' });
    } catch (err) {
        // Non-critical -- model discovery is best-effort
    }
    await loadModelsCache();
    if (activeSection === 'models') {
        loadModelsTab();
    }
}

async function loadModelsCache() {
    try {
        const resp = await fetch('/api/models');
        allModelsCache = await resp.json();
    } catch (err) {
        // Non-critical
    }
}

// --- GPU vendor detection (colours are theme tokens, see tokens.css) ---

function setTemp(el, cls, value, [warn, bad]) {
    el.textContent = Math.round(value) + ' \u00b0C';
    el.className = 'monitor-metric-reading ' + cls + (value >= bad ? ' is-bad' : value >= warn ? ' is-warn' : '');
}

function vendorInfo(cardName) {
    const n = cardName.toLowerCase();
    if (n.includes('amd') || n.includes('radeon')) return { key: 'amd', color: 'var(--vendor-amd)', label: 'AMD' };
    if (n.includes('nvidia') || n.includes('geforce') || n.includes('quadro') || n.includes('tesla')) return { key: 'nvidia', color: 'var(--vendor-nvidia)', label: 'NVIDIA' };
    if (n.includes('intel') || n.includes('arc')) return { key: 'intel', color: 'var(--vendor-intel)', label: 'Intel' };
    return { key: 'other', color: 'var(--vendor-other)', label: cardName };
}

// The card kicker already names the vendor, so the title drops it:
// "AMD Radeon AI PRO R9700" reads as "Radeon AI PRO R9700".
function shortCardName(cardName) {
    return cardName.replace(/^(AMD|NVIDIA|Intel(\(R\))?)\s+/i, '');
}

const VRAM_CONTEXT_COLOR = 'var(--vram-context)';

function renderVramBar(d) {
    const barEls = document.querySelectorAll('.vram-bar');
    const legendEls = document.querySelectorAll('.vram-legend');
    const setBars = html => barEls.forEach(el => { el.innerHTML = html; });
    const setLegends = html => legendEls.forEach(el => { el.innerHTML = html; });
    const gpuList = Object.entries(d.gpu);
    const totalBadge = document.getElementById('vram-total-badge');

    if (gpuList.length === 0 || totalVramMb <= 0) {
        setBars('');
        setLegends('<span>No GPU data yet</span>');
        if (totalBadge) totalBadge.textContent = '\u2014';
        return;
    }
    if (totalBadge) {
        const pct = Math.round((usedVramMb / totalVramMb) * 100);
        totalBadge.textContent = (usedVramMb / 1024).toFixed(1) + ' / ' + (totalVramMb / 1024).toFixed(1) + ' GB \u00b7 ' + pct + '%';
    }

    // Try to attribute used VRAM to model weights vs context/overhead,
    // using the loaded model's file size as an estimate of weight usage.
    // This is a proportional approximation split evenly across GPUs by
    // their share of total used VRAM -- it does not assume any specific
    // tensor-split device ordering, since that isn't reliably knowable
    // from GPU telemetry alone.
    let modelSizeMb = 0;
    if (d.server_running && d.model_path && allModelsCache.length > 0) {
        const match = allModelsCache.find(m => m.path === d.model_path);
        if (match) modelSizeMb = match.size_bytes / 1024 / 1024;
    }
    const contextMb = modelSizeMb > 0 ? Math.max(0, usedVramMb - modelSizeMb) : 0;

    const segments = [];
    const legendVendors = new Map();

    gpuList.forEach(([card, m]) => {
        const vendor = vendorInfo(card);
        legendVendors.set(vendor.label, vendor.color);
        const gpuUsedMb = m.vram_used || 0;
        if (gpuUsedMb <= 0) return;

        let weightMb = gpuUsedMb;
        let ctxMb = 0;
        if (contextMb > 0 && usedVramMb > 0) {
            ctxMb = gpuUsedMb * (contextMb / usedVramMb);
            weightMb = gpuUsedMb - ctxMb;
        }

        const gpuTotalGb = ((m.vram_total || 0) / 1024).toFixed(1);

        if (weightMb > 0) {
            segments.push({
                widthPct: (weightMb / totalVramMb) * 100,
                color: vendor.color,
                label: (weightMb / 1024).toFixed(1) + ' GB',
                title: card + ': ' + (weightMb / 1024).toFixed(1) + 'GB of ' + gpuTotalGb + 'GB used (weights/other)',
            });
        }
        if (ctxMb > 0.1) {
            segments.push({
                widthPct: (ctxMb / totalVramMb) * 100,
                color: VRAM_CONTEXT_COLOR,
                label: (ctxMb / 1024).toFixed(1) + ' GB',
                title: card + ': ' + (ctxMb / 1024).toFixed(1) + 'GB context (est.)',
            });
        }
    });

    const freePct = Math.max(0, 100 - segments.reduce((s, seg) => s + seg.widthPct, 0));
    const freeGb = ((totalVramMb * freePct / 100) / 1024).toFixed(1);

    // Only render text inside a segment when it is wide enough to fit,
    // otherwise the label overflows into neighbouring segments. The exact
    // figures live in the per-device rows and tooltips.
    const segText = (seg) => seg.widthPct >= 20 ? seg.label : '';

    setBars(segments.map(seg =>
        '<div class="vram-seg" style="width:' + seg.widthPct.toFixed(2) + '%; background:' + seg.color + ';" title="' + escapeHtml(seg.title) + '">' +
        '<span class="vram-seg-label">' + segText(seg) + '</span></div>'
    ).join('') +
        '<div class="vram-seg vram-seg-free" style="width:' + freePct.toFixed(2) + '%;" title="Free: ' + freeGb + 'GB">' +
        '<span class="vram-seg-label vram-seg-label-free">' + (freePct >= 20 ? freeGb + ' GB free' : '') + '</span></div>');

    const legendItems = Array.from(legendVendors.entries()).map(([label, color]) =>
        '<span class="vram-legend-item"><span class="vram-legend-swatch" style="background:' + color + ';"></span>' + escapeHtml(label) + '</span>');
    legendItems.push('<span class="vram-legend-item"><span class="vram-legend-swatch" style="background:' + VRAM_CONTEXT_COLOR + ';"></span>Context/KV (est.)</span>');
    legendItems.push('<span class="vram-legend-item"><span class="vram-legend-swatch vram-legend-swatch-free"></span>Free (' + ((totalVramMb - usedVramMb) / 1024).toFixed(1) + 'GB)</span>');
    setLegends(legendItems.join(''));

    // Per-device breakdown under the pooled bar, in the same vendor colours.
    const perGpu = document.getElementById('vram-per-gpu');
    if (perGpu) {
        perGpu.innerHTML = gpuList.map(([card, m]) => {
            const vendor = vendorInfo(card);
            const used = m.vram_used || 0;
            const total = m.vram_total || 0;
            const pct = total > 0 ? Math.round((used / total) * 100) : 0;
            return '<div>' +
                '<div class="monitor-metric-row"><span class="monitor-metric-label" title="' + escapeHtml(card) + '">' + escapeHtml(shortCardName(card)) + '</span>' +
                '<span class="monitor-metric-reading" title="' + pct + '% used">' + (used / 1024).toFixed(1) + ' / ' + (total / 1024).toFixed(1) + ' GB</span></div>' +
                '<div class="progress-bar"><div class="progress-fill" style="width:' + pct + '%; background:' + vendor.color + ';"></div></div>' +
                '</div>';
        }).join('');
    }
}

// --- GPU cards (one per device) ---

const GPU_ICON = '<svg viewBox="0 0 24 24"><rect x="2" y="6" width="20" height="12" rx="2"/><circle cx="9" cy="12" r="3"/><line x1="15" y1="10" x2="18" y2="10"/><line x1="15" y1="14" x2="18" y2="14"/><line x1="6" y1="18" x2="6" y2="21"/><line x1="10" y1="18" x2="10" y2="21"/></svg>';

function barClass(pct) {
    if (pct >= 95) return 'progress-fill progress-fill-critical';
    if (pct >= 80) return 'progress-fill progress-fill-warning';
    return 'progress-fill';
}

function cardTools(key, label) {
    return '<div class="monitor-card-tools">' +
        '<button class="btn btn-sm btn-ghost monitor-drag-handle" type="button" draggable="true" title="Drag to reorder; use arrow keys to move" aria-label="Move ' + escapeHtml(label) + ' card; use arrow keys">\u283f</button>' +
        '<button class="btn btn-sm btn-ghost monitor-hide-btn" type="button" data-monitor-hide="' + escapeHtml(key) + '" aria-label="Hide ' + escapeHtml(label) + ' card">Hide</button>';
}

function renderGpuCards(gpuList) {
    const grid = document.getElementById('monitor-card-grid');
    if (!grid) return;
    if (gpuList.length === 0) {
        if (lastGpuCount !== 0) {
            grid.querySelectorAll('.monitor-gpu-card, .monitor-empty').forEach(el => el.remove());
            const empty = document.createElement('div');
            empty.className = 'card monitor-empty';
            empty.innerHTML = '<div class="empty-state"><div class="empty-state-title">No GPU telemetry</div><p>Install rocm-smi (AMD) or nvidia-smi (NVIDIA), or force a backend with --gpu-backend.</p></div>';
            grid.appendChild(empty);
            lastGpuCount = 0;
        }
        return;
    }
    // Rebuild the DOM only when the device list changes; otherwise update
    // values in place so hover states and text selection survive each tick.
    if (lastGpuCount !== gpuList.length) {
        grid.querySelectorAll('.monitor-gpu-card, .monitor-empty').forEach(el => el.remove());
        grid.insertAdjacentHTML('beforeend', gpuList.map(([card], i) => {
            const vendor = vendorInfo(card);
            return '<div class="card monitor-gpu-card" data-gpu-index="' + i + '" data-monitor-key="gpu:' + i + '" data-monitor-label="GPU ' + i + '">' +
                '<div class="card-header">' +
                    '<div><div class="card-kicker">GPU ' + i + ' \u00b7 ' + escapeHtml(vendor.label) + '</div>' +
                    '<div class="card-title gpu-name" title="' + escapeHtml(card) + '">' + escapeHtml(shortCardName(card)) + '</div></div>' +
                    cardTools('gpu:' + i, 'GPU ' + i) +
                    '<div class="card-icon icon-' + vendor.key + '"><span class="icon icon-lg">' + GPU_ICON + '</span></div></div>' +
                '</div>' +
                '<div class="monitor-metric-block">' +
                    '<div class="monitor-metric-row"><span class="monitor-metric-label">Utilization</span><span class="monitor-metric-reading gpu-load">\u2014</span></div>' +
                    '<div class="progress-bar"><div class="progress-fill gpu-load-bar"></div></div>' +
                '</div>' +
                '<div class="monitor-metric-block">' +
                    '<div class="monitor-metric-row"><span class="monitor-metric-label">VRAM</span><span class="monitor-metric-reading gpu-vram">\u2014</span></div>' +
                    '<div class="progress-bar"><div class="progress-fill gpu-vram-bar"></div></div>' +
                '</div>' +
                '<div class="monitor-gpu-meta">' +
                    '<div class="monitor-metric-row"><span class="monitor-metric-label gpu-temp-label">Temperature</span><span class="monitor-metric-reading gpu-temp">\u2014</span></div>' +
                    '<div class="monitor-metric-row gpu-temp-edge-row" hidden><span class="monitor-metric-label">Edge temp</span><span class="monitor-metric-reading gpu-temp-edge">\u2014</span></div>' +
                    '<div class="monitor-metric-row gpu-temp-mem-row" hidden><span class="monitor-metric-label">Memory temp</span><span class="monitor-metric-reading gpu-temp-mem">\u2014</span></div>' +
                    '<div class="monitor-metric-row"><span class="monitor-metric-label">Power</span><span class="monitor-metric-reading gpu-power">\u2014</span></div>' +
                    '<div class="monitor-metric-row"><span class="monitor-metric-label">Core clock</span><span class="monitor-metric-reading gpu-sclk">\u2014</span></div>' +
                    '<div class="monitor-metric-row"><span class="monitor-metric-label">Memory clock</span><span class="monitor-metric-reading gpu-mclk">\u2014</span></div>' +
                '</div>' +
            '</div>';
        }).join(''));
        lastGpuCount = gpuList.length;
        applyCardLayout();
    }

    gpuList.forEach(([card, m], i) => {
        const el = grid.querySelector('[data-gpu-index="' + i + '"]');
        if (!el) return;
        const nameEl = el.querySelector('.gpu-name');
        if (nameEl.title !== card) {
            nameEl.title = card;
            nameEl.textContent = shortCardName(card);
            const vendor = vendorInfo(card);
            el.querySelector('.card-kicker').textContent = 'GPU ' + i + ' \u00b7 ' + vendor.label;
            el.querySelector('.card-icon').className = 'card-icon icon-' + vendor.key;
        }
        const load = Math.max(0, Math.min(100, m.load || 0));
        el.querySelector('.gpu-load').textContent = load + '%';
        const loadBar = el.querySelector('.gpu-load-bar');
        loadBar.style.width = load + '%';
        loadBar.className = barClass(load) + ' gpu-load-bar';

        const vpct = m.vram_total > 0 ? Math.round((m.vram_used / m.vram_total) * 100) : 0;
        el.querySelector('.gpu-vram').textContent = ((m.vram_used || 0) / 1024).toFixed(1) + ' / ' + ((m.vram_total || 0) / 1024).toFixed(1) + ' GB \u00b7 ' + vpct + '%';
        const vramBar = el.querySelector('.gpu-vram-bar');
        vramBar.style.width = vpct + '%';
        vramBar.className = barClass(vpct) + ' gpu-vram-bar';

        // AMD cards report edge, junction (hotspot) and memory sensors; the
        // headline value is junction when present. Junction and memory run
        // hotter by design (RDNA throttles around 110 C), so they get looser
        // thresholds than edge / NVIDIA's single sensor.
        const hasJunction = m.temp_junction != null;
        el.querySelector('.gpu-temp-label').textContent = hasJunction ? 'Hotspot temp' : 'Temperature';
        setTemp(el.querySelector('.gpu-temp'), 'gpu-temp', m.temp, hasJunction ? [95, 105] : [80, 90]);
        const edgeRow = el.querySelector('.gpu-temp-edge-row');
        edgeRow.hidden = !(hasJunction && m.temp_edge != null);
        if (!edgeRow.hidden) setTemp(el.querySelector('.gpu-temp-edge'), 'gpu-temp-edge', m.temp_edge, [80, 90]);
        const memRow = el.querySelector('.gpu-temp-mem-row');
        memRow.hidden = m.temp_memory == null;
        if (!memRow.hidden) setTemp(el.querySelector('.gpu-temp-mem'), 'gpu-temp-mem', m.temp_memory, [95, 105]);

        const capped = m.power_consumption >= m.power_limit && m.power_limit > 0;
        const powerEl = el.querySelector('.gpu-power');
        powerEl.textContent = capped
            ? m.power_consumption.toFixed(1) + ' W (at limit)'
            : m.power_consumption.toFixed(1) + ' W / ' + m.power_limit + ' W';
        powerEl.className = 'monitor-metric-reading gpu-power' + (capped ? ' is-warn' : '');

        el.querySelector('.gpu-sclk').textContent = m.sclk_mhz + ' MHz';
        el.querySelector('.gpu-mclk').textContent = m.mclk_mhz + ' MHz';
    });
}

// --- Monitor card layout (order + hidden), remembered per browser ---

const CARD_LAYOUT_KEY = 'llama_admin_monitor_cards';
let cardLayout = { order: [], hidden: [] };
try {
    const stored = JSON.parse(localStorage.getItem(CARD_LAYOUT_KEY) || '{}');
    if (Array.isArray(stored.order)) cardLayout.order = stored.order.filter(k => typeof k === 'string');
    if (Array.isArray(stored.hidden)) cardLayout.hidden = stored.hidden.filter(k => typeof k === 'string');
} catch (_) {}

function saveCardLayout() {
    try { localStorage.setItem(CARD_LAYOUT_KEY, JSON.stringify(cardLayout)); } catch (_) {}
}

function monitorCards() {
    return Array.from(document.querySelectorAll('#monitor-card-grid > .card[data-monitor-key]'));
}

// Reorders the grid to the saved order (unknown cards keep their DOM order
// at the end) and applies the hidden set, then rebuilds the restore bar.
function applyCardLayout() {
    const grid = document.getElementById('monitor-card-grid');
    if (!grid) return;
    const cards = monitorCards();
    const byKey = new Map(cards.map(c => [c.dataset.monitorKey, c]));
    const ordered = [];
    cardLayout.order.forEach(k => { if (byKey.has(k)) { ordered.push(byKey.get(k)); byKey.delete(k); } });
    byKey.forEach(c => ordered.push(c));
    ordered.forEach(c => grid.appendChild(c));
    cards.forEach(c => c.classList.toggle('card-hidden', cardLayout.hidden.includes(c.dataset.monitorKey)));
    renderHiddenBar();
}

function renderHiddenBar() {
    const bar = document.getElementById('monitor-hidden-controls');
    if (!bar) return;
    const hiddenCards = monitorCards().filter(c => cardLayout.hidden.includes(c.dataset.monitorKey));
    bar.hidden = hiddenCards.length === 0;
    document.getElementById('monitor-hidden-count').textContent =
        hiddenCards.length + (hiddenCards.length === 1 ? ' card hidden' : ' cards hidden');
    document.getElementById('monitor-restore-items').innerHTML = hiddenCards.map(c =>
        '<div class="monitor-restore-row"><span>' + escapeHtml(c.dataset.monitorLabel || c.dataset.monitorKey) + '</span>' +
        '<button class="btn btn-sm" type="button" onclick="showCard(\'' + jsStr(c.dataset.monitorKey) + '\')">Show</button></div>'
    ).join('');
}

function hideCard(key) {
    if (!cardLayout.hidden.includes(key)) cardLayout.hidden.push(key);
    saveCardLayout();
    applyCardLayout();
}

function showCard(key) {
    cardLayout.hidden = cardLayout.hidden.filter(k => k !== key);
    saveCardLayout();
    applyCardLayout();
}

function showAllCards() {
    cardLayout.hidden = [];
    saveCardLayout();
    applyCardLayout();
}

function commitCardOrder() {
    cardLayout.order = monitorCards().map(c => c.dataset.monitorKey);
    saveCardLayout();
    renderHiddenBar();
}

function moveCard(card, delta) {
    const visible = monitorCards().filter(c => !c.classList.contains('card-hidden'));
    const i = visible.indexOf(card);
    const j = i + delta;
    if (i < 0 || j < 0 || j >= visible.length) return;
    const grid = document.getElementById('monitor-card-grid');
    if (delta < 0) grid.insertBefore(card, visible[j]);
    else grid.insertBefore(card, visible[j].nextSibling);
    commitCardOrder();
    card.querySelector('.monitor-drag-handle').focus();
}

(function initCardDragging() {
    const grid = document.getElementById('monitor-card-grid');
    if (!grid) return;
    let dragging = null;

    grid.addEventListener('click', e => {
        const hide = e.target.closest('[data-monitor-hide]');
        if (hide) hideCard(hide.dataset.monitorHide);
    });

    grid.addEventListener('keydown', e => {
        if (!e.target.classList.contains('monitor-drag-handle')) return;
        const card = e.target.closest('.card');
        if (e.key === 'ArrowLeft' || e.key === 'ArrowUp') { e.preventDefault(); moveCard(card, -1); }
        if (e.key === 'ArrowRight' || e.key === 'ArrowDown') { e.preventDefault(); moveCard(card, 1); }
    });

    grid.addEventListener('dragstart', e => {
        const handle = e.target.closest('.monitor-drag-handle');
        if (!handle) { e.preventDefault(); return; }
        dragging = handle.closest('.card');
        dragging.classList.add('dragging');
        e.dataTransfer.effectAllowed = 'move';
        try { e.dataTransfer.setData('text/plain', dragging.dataset.monitorKey); } catch (_) {}
        try { e.dataTransfer.setDragImage(dragging, 24, 24); } catch (_) {}
    });

    const clearMarkers = () => grid.querySelectorAll('.drop-before, .drop-after').forEach(c => c.classList.remove('drop-before', 'drop-after'));

    grid.addEventListener('dragover', e => {
        if (!dragging) return;
        const target = e.target.closest('.card');
        if (!target || target === dragging) return;
        e.preventDefault();
        e.dataTransfer.dropEffect = 'move';
        const r = target.getBoundingClientRect();
        const before = e.clientX < r.left + r.width / 2;
        clearMarkers();
        target.classList.add(before ? 'drop-before' : 'drop-after');
    });

    grid.addEventListener('dragleave', e => {
        if (!grid.contains(e.relatedTarget)) clearMarkers();
    });

    grid.addEventListener('drop', e => {
        if (!dragging) return;
        const target = e.target.closest('.card');
        if (!target || target === dragging) return;
        e.preventDefault();
        const r = target.getBoundingClientRect();
        const before = e.clientX < r.left + r.width / 2;
        grid.insertBefore(dragging, before ? target : target.nextSibling);
        commitCardOrder();
    });

    grid.addEventListener('dragend', () => {
        if (dragging) dragging.classList.remove('dragging');
        dragging = null;
        clearMarkers();
    });
})();
applyCardLayout();

// --- System cards (CPU / memory / disk) ---

function fmtBytes(b) {
    if (b == null) return '\u2014';
    if (b >= 1024 ** 4) return (b / 1024 ** 4).toFixed(2) + ' TB';
    if (b >= 1024 ** 3) return (b / 1024 ** 3).toFixed(1) + ' GB';
    if (b >= 1024 ** 2) return (b / 1024 ** 2).toFixed(0) + ' MB';
    if (b >= 1024) return (b / 1024).toFixed(0) + ' KB';
    return b.toFixed(0) + ' B';
}

function fmtRate(bps) {
    if (bps == null) return '\u2014';
    if (bps >= 1024 ** 3) return (bps / 1024 ** 3).toFixed(2) + ' GB/s';
    if (bps >= 1024 ** 2) return (bps / 1024 ** 2).toFixed(1) + ' MB/s';
    if (bps >= 1024) return (bps / 1024).toFixed(0) + ' KB/s';
    return bps.toFixed(0) + ' B/s';
}

function setMetricValue(id, value, unit) {
    const el = document.getElementById(id);
    if (value == null) {
        el.className = 'monitor-metric-value monitor-not-available';
        el.textContent = 'Not available';
        return;
    }
    el.className = 'monitor-metric-value';
    el.innerHTML = escapeHtml(value) + (unit ? '<span class="monitor-metric-unit">' + unit + '</span>' : '');
}

function setBar(id, pct) {
    const bar = document.getElementById(id);
    const p = pct == null ? 0 : Math.max(0, Math.min(100, pct));
    bar.style.width = p.toFixed(1) + '%';
    bar.className = barClass(p);
}

function renderSystemCards(sys) {
    if (!sys || !sys.available) {
        setMetricValue('sys-cpu-value', null);
        setMetricValue('sys-mem-value', null);
        setBar('sys-cpu-bar', null);
        setBar('sys-mem-bar', null);
        document.getElementById('sys-cpu-sub').textContent = 'Host telemetry needs /proc (Linux).';
        document.getElementById('sys-mem-sub').textContent = '';
        document.getElementById('sys-disk-read').textContent = '\u2014';
        document.getElementById('sys-disk-write').textContent = '\u2014';
        document.getElementById('sys-disk-space').textContent = '\u2014';
        document.getElementById('sys-disk-sub').textContent = 'Host telemetry needs /proc (Linux).';
        return;
    }

    // CPU
    setMetricValue('sys-cpu-value', sys.cpu_percent == null ? null : sys.cpu_percent.toFixed(1), '%');
    setBar('sys-cpu-bar', sys.cpu_percent);
    const cpuBits = [];
    if (sys.cpu_cores) cpuBits.push(sys.cpu_cores + ' cores');
    if (sys.load_avg_1m != null) cpuBits.push('load ' + sys.load_avg_1m.toFixed(2));
    if (sys.sample_secs) cpuBits.push(sys.sample_secs.toFixed(1) + ' s sample');
    document.getElementById('sys-cpu-sub').textContent = cpuBits.join(' \u00b7 ');
    document.getElementById('sys-cpu-model').textContent = sys.cpu_model || '';
    const cpuTemp = document.getElementById('sys-cpu-temp');
    if (sys.cpu_temp_c == null) {
        cpuTemp.textContent = 'Not available';
        cpuTemp.className = 'monitor-metric-reading monitor-not-available';
        cpuTemp.title = 'No CPU sensor found under /sys/class/hwmon (k10temp, coretemp, cpu_thermal)';
    } else {
        cpuTemp.textContent = Math.round(sys.cpu_temp_c) + ' \u00b0C';
        cpuTemp.className = 'monitor-metric-reading' + (sys.cpu_temp_c >= 95 ? ' is-bad' : sys.cpu_temp_c >= 85 ? ' is-warn' : '');
        cpuTemp.title = '';
    }

    const coresGrid = document.getElementById('sys-cpu-cores');
    if (sys.core_percent && sys.core_percent.length > 0) {
        coresGrid.innerHTML = sys.core_percent.map(p => {
            const pct = p != null ? p : 0;
            return '<div class="cpu-core-cell" title="' + pct.toFixed(1) + '%" style="opacity: ' + Math.max(0.15, pct / 100) + ';"></div>';
        }).join('');
    } else {
        coresGrid.innerHTML = '';
    }

    // Memory
    const memPct = sys.mem_total_bytes > 0 ? (sys.mem_used_bytes / sys.mem_total_bytes) * 100 : null;
    setMetricValue('sys-mem-value', memPct == null ? null : memPct.toFixed(1), '%');
    setBar('sys-mem-bar', memPct);
    let memSub = fmtBytes(sys.mem_used_bytes) + ' used of ' + fmtBytes(sys.mem_total_bytes);
    if (sys.swap_total_bytes > 0) memSub += ' \u00b7 swap ' + fmtBytes(sys.swap_used_bytes) + ' / ' + fmtBytes(sys.swap_total_bytes);
    document.getElementById('sys-mem-sub').textContent = memSub;

    let dimmHtml = '';
    if (sys.dimms && sys.dimms.length > 0) {
        const usedSlots = sys.dimms.filter(d => d.size_bytes > 0).length;
        dimmHtml = '<div style="display: flex; flex-direction: column; gap: 4px; align-items: flex-start; margin-top: 8px;">' +
            '<span class="monitor-metric-label">' + usedSlots + '/' + sys.dimm_slots_total + ' slots populated:</span>' + 
            sys.dimms.filter(d => d.size_bytes > 0).map(d => {
                const speed = d.configured_speed_mts || d.speed_mts;
                return '<span class="badge badge-dim">' + fmtBytes(d.size_bytes) + ' ' + d.mem_type + (speed ? ' ' + speed + 'MT/s' : '') + '</span>';
            }).join('') + '</div>';
    } else if (sys.dimm_error) {
        dimmHtml = '<div style="margin-top: 8px;"><span class="help-text" title="' + escapeHtml(sys.dimm_error) + '" style="cursor: help;">DIMM slots info unavailable \u24d8</span></div>';
    }
    document.getElementById('sys-mem-dimm').innerHTML = dimmHtml;

    // Disk
    document.getElementById('sys-disk-read').textContent = fmtRate(sys.disk_read_bytes_per_sec);
    document.getElementById('sys-disk-write').textContent = fmtRate(sys.disk_write_bytes_per_sec);
    if (sys.disk_total_bytes != null && sys.disk_free_bytes != null && sys.disk_total_bytes > 0) {
        const used = sys.disk_total_bytes - sys.disk_free_bytes;
        const pct = (used / sys.disk_total_bytes) * 100;
        document.getElementById('sys-disk-mount').textContent = 'Free' + (sys.disk_mount ? ' \u00b7 ' + sys.disk_mount : '');
        document.getElementById('sys-disk-space').textContent = fmtBytes(sys.disk_free_bytes) + ' of ' + fmtBytes(sys.disk_total_bytes);
        setBar('sys-disk-bar', pct);
    } else {
        document.getElementById('sys-disk-space').textContent = '\u2014';
        setBar('sys-disk-bar', null);
    }
    document.getElementById('sys-disk-sub').textContent = 'All physical disks \u00b7 includes other applications';
}

// --- Sorting ---

// Compares two values for sorting. Nulls always sort last regardless of
// direction, so rows with missing data don't crowd the top.
function compareValues(a, b, dir) {
    const aMissing = a === null || a === undefined || a === '';
    const bMissing = b === null || b === undefined || b === '';
    if (aMissing && bMissing) return 0;
    if (aMissing) return 1;
    if (bMissing) return -1;
    let result;
    if (typeof a === 'number' && typeof b === 'number') {
        result = a - b;
    } else if (typeof a === 'boolean' && typeof b === 'boolean') {
        result = (a === b) ? 0 : (a ? -1 : 1);
    } else {
        result = String(a).localeCompare(String(b), undefined, { numeric: true, sensitivity: 'base' });
    }
    return dir === 'desc' ? -result : result;
}

function sortIndicator(activeKey, key, dir) {
    if (activeKey !== key) return '';
    return dir === 'asc' ? ' \u25b2' : ' \u25bc';
}

// --- Escaping helpers ---

function escapeHtml(s) {
    return String(s ?? '')
        .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;').replace(/'/g, '&#39;');
}

// For values interpolated into inline onclick single-quoted strings.
function jsStr(s) {
    return String(s ?? '').replace(/\\/g, '\\\\').replace(/'/g, "\\'").replace(/</g, '\\x3c');
}

// --- Models tab ---

let modelsData = [];
let modelsSortKey = 'model_name';
let modelsSortDir = 'asc';

function sortModels(key) {
    if (modelsSortKey === key) {
        modelsSortDir = modelsSortDir === 'asc' ? 'desc' : 'asc';
    } else {
        modelsSortKey = key;
        // Sizes, counts and dates are most useful largest/newest first.
        modelsSortDir = ['size_bytes', 'hf_downloads', 'downloaded_at', 'hf_last_modified'].includes(key) ? 'desc' : 'asc';
    }
    renderModelsTab();
}

function renderModelsHeader() {
    const headerEl = document.getElementById('models-header');
    if (!headerEl) return;
    const cols = [
        ['model_name', 'Model'],
        ['quant_type', 'Quant'],
        ['size_bytes', 'Size'],
        ['fits_vram', 'Fits VRAM?'],
        ['hf_downloads', 'Downloads'],
        ['downloaded_at', 'Downloaded'],
        ['hf_last_modified', 'HF updated'],
    ];
    headerEl.innerHTML = cols.map(([key, label]) =>
        '<span class="sortable" onclick="sortModels(\'' + key + '\')">' +
        label + sortIndicator(modelsSortKey, key, modelsSortDir) + '</span>'
    ).join('') + '<span></span>';
}

function renderModelsTab() {
    const listEl = document.getElementById('models-list');
    renderModelsHeader();

    if (!modelsData || modelsData.length === 0) {
        listEl.innerHTML = '<div class="empty-state"><div class="empty-state-title">No models found</div><p>Download one from Hugging Face, or set your models directory in Settings.</p></div>';
        return;
    }

    const sorted = modelsData.slice().sort((x, y) => {
        let a, b;
        if (modelsSortKey === 'fits_vram') {
            a = vramFitCheck(x.size_bytes).cls === 'vram-fit-ok';
            b = vramFitCheck(y.size_bytes).cls === 'vram-fit-ok';
        } else if (modelsSortKey === 'model_name') {
            a = x.model_name || x.filename;
            b = y.model_name || y.filename;
        } else if (modelsSortKey === 'hf_last_modified') {
            a = x.hf_last_modified ? Date.parse(x.hf_last_modified) : null;
            b = y.hf_last_modified ? Date.parse(y.hf_last_modified) : null;
        } else {
            a = x[modelsSortKey];
            b = y[modelsSortKey];
        }
        return compareValues(a, b, modelsSortDir);
    });

    listEl.innerHTML = sorted.map(m => {
            const downloads = m.hf_downloads ? m.hf_downloads.toLocaleString() : '\u2014';
            const downloadedOn = m.downloaded_at
                ? new Date(m.downloaded_at * 1000).toLocaleDateString()
                : '\u2014';
            const hfUpdated = m.hf_last_modified
                ? new Date(m.hf_last_modified).toLocaleDateString()
                : '\u2014';
            const fit = vramFitCheck(m.size_bytes);
            return '<div class="model-grid-row">' +
                '<span class="model-name" title="' + escapeHtml(m.filename) + '"><span class="model-name-main">\u{1F4C4} ' + escapeHtml(m.model_name || m.filename) + '</span>' + modelPairChip(m) + '</span>' +
                '<span class="model-cell">' + escapeHtml(m.quant_type || '\u2014') + '</span>' +
                '<span class="model-cell">' + escapeHtml(m.size_display) + '</span>' +
                '<span class="model-cell ' + fit.cls + '" title="' + escapeHtml(fit.title) + '">' + fit.label + '</span>' +
                '<span class="model-cell">' + downloads + '</span>' +
                '<span class="model-cell">' + downloadedOn + '</span>' +
                '<span class="model-cell">' + hfUpdated + '</span>' +
                '<span class="model-delete-cell"><button class="btn btn-xs btn-danger" onclick="deleteModel(\'' + jsStr(m.filename) + '\')">Delete</button></span>' +
                '</div>';
        }).join('');
}

// What a row pairs with: a model shows its projector, a projector the
// model(s) it belongs to.
function modelPairChip(m) {
    if (m.is_mmproj) {
        const names = (m.pairs_with || []).map(f => (allModelsCache.find(x => x.filename === f) || {}).model_name || f);
        return names.length
            ? '<span class="model-pair" title="Projector for: ' + escapeHtml(m.pairs_with.join(', ')) + '"><span class="chip-projector">projector</span> for ' + escapeHtml(names.join(', ')) + '</span>'
            : '<span class="model-pair" title="No model in this directory matches this projector by repo or name"><span class="chip-projector is-unpaired">projector</span> not matched to any model here</span>';
    }
    if (m.projector) {
        return '<span class="model-pair" title="Vision model: launches with --mmproj ' + escapeHtml(m.projector) + '"><span class="chip-projector">vision</span> mmproj: ' + escapeHtml(m.projector) + '</span>';
    }
    return '';
}

async function loadModelsTab() {
    const listEl = document.getElementById('models-list');
    listEl.innerHTML = '<div class="fb-empty">Loading...</div>';
    try {
        await fetch('/api/models/refresh', { method: 'POST' });
        const resp = await fetch('/api/models');
        modelsData = await resp.json();
        allModelsCache = modelsData;
        document.getElementById('nav-count-models').textContent = modelsData.length || '';
        renderModelsTab();
    } catch (err) {
        listEl.innerHTML = '<div class="fb-empty">Error: ' + escapeHtml(err.message) + '</div>';
    }
}

async function deleteModel(filename) {
    const proceed = await showConfirm('Delete model', 'Delete ' + filename + ' from disk? This cannot be undone.', 'Delete', true);
    if (!proceed) return;
    try {
        const resp = await fetch('/api/models/delete', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ filename: filename }),
        });
        const data = await resp.json();
        if (!data.ok) {
            showToast('Delete failed: ' + (data.error || 'unknown'), 'error');
            return;
        }
        showToast('Deleted: ' + filename, 'success');
        loadModelsTab();
    } catch (err) {
        showToast('Delete failed: ' + err.message, 'error');
    }
}

function updateHfProgress(p) {
    const badge = document.getElementById('hf-badge');
    if (!p) {
        badge.classList.add('hidden');
        return;
    }
    if (p.done) {
        hfDownloading = false;
        badge.classList.add('hidden');
        document.getElementById('hf-download-progress').hidden = true;
        document.getElementById('models-hf-banner').hidden = true;
        if (hfLastHandledDone) return;
        hfLastHandledDone = true;
        if (p.error) {
            showToast('Download failed: ' + p.error, 'error');
        } else {
            const n = (p.completed_paths || []).length;
            showToast(n > 1 ? 'Downloaded ' + n + ' files: ' + p.filename + ' and companion' : 'Downloaded: ' + p.filename, 'success');
            refreshModels();
            fillPresetFromDownload(p.completed_paths || []);
        }
        return;
    }
    if (p.total_bytes > 0) {
        const pct = ((p.downloaded_bytes / p.total_bytes) * 100).toFixed(1);
        const seq = p.file_count > 1 ? ' (' + p.file_index + '/' + p.file_count + ')' : '';
        document.getElementById('hf-progress-filename').textContent = p.filename + seq;
        document.getElementById('hf-progress-pct').textContent = pct + '%';
        document.getElementById('hf-progress-bar').style.width = pct + '%';
        badge.classList.remove('hidden');
        badge.textContent = '(' + pct + '%' + seq + ')';

        const banner = document.getElementById('models-hf-banner');
        banner.hidden = false;
        document.getElementById('models-hf-banner-name').textContent = 'Downloading: ' + p.filename + seq;
        document.getElementById('models-hf-banner-pct').textContent = pct + '%';
        document.getElementById('models-hf-banner-bar').style.width = pct + '%';
    }
}

// When the download was started from the preset editor, point the editor
// at what just arrived: the model path if it is still empty, and the
// projector whenever one came along.
function fillPresetFromDownload(paths) {
    if (!modalIsOpen('preset-modal') || paths.length === 0) return;
    const isProj = path => /mmproj/i.test(path.split('/').pop() || '');
    const model = paths.find(p => !isProj(p));
    const proj = paths.find(isProj);
    const modelField = document.getElementById('modal-model-path');
    if (model && !modelField.value.trim()) modelField.value = model;
    if (proj) document.getElementById('modal-mmproj').value = proj;
    if (model || proj) showToast('Preset editor updated with the downloaded file' + (model && proj ? 's' : ''), 'success');
}

// --- Escape closes the topmost open modal ---

document.addEventListener('keydown', e => {
    if (e.key !== 'Escape') return;
    const order = [
        ['confirm-modal', () => closeConfirmModal(false)],
        ['file-browser-modal', closeFileBrowser],
        ['hf-modal', closeHfModal],
        ['config-modal', closeConfigModal],
        ['preset-modal', closePresetModal],
    ];
    for (const [id, close] of order) {
        if (modalIsOpen(id)) {
            close();
            e.preventDefault();
            e.stopImmediatePropagation();
            return;
        }
    }
}, true);

// --- Preset Selection ---

document.getElementById('preset-select').addEventListener('change', () => {
    saveSettings();
    renderPresetsPage();
    renderRuntime();
});

// --- Toast Notifications ---

const TOAST_ICONS = {
    success: '<svg viewBox="0 0 24 24"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/></svg>',
    error: '<svg viewBox="0 0 24 24"><circle cx="12" cy="12" r="10"/><line x1="15" y1="9" x2="9" y2="15"/><line x1="9" y1="9" x2="15" y2="15"/></svg>',
    warn: '<svg viewBox="0 0 24 24"><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>',
};

// `action` is an optional { label, onClick }; toasts with one stay longer.
function showToast(message, type = 'error', action = null) {
    const container = document.getElementById('toast-container');
    const toast = document.createElement('div');
    toast.className = 'toast toast-' + type;
    toast.innerHTML = '<span class="icon icon-sm toast-icon">' + (TOAST_ICONS[type] || TOAST_ICONS.error) + '</span><span class="toast-message"></span>';
    toast.querySelector('.toast-message').textContent = message;
    const dismiss = () => {
        toast.classList.remove('show');
        setTimeout(() => toast.remove(), 300);
    };
    if (action) {
        const btn = document.createElement('button');
        btn.type = 'button';
        btn.className = 'toast-action';
        btn.textContent = action.label;
        btn.addEventListener('click', e => { e.stopPropagation(); dismiss(); action.onClick(); });
        toast.appendChild(btn);
    }
    container.appendChild(toast);
    requestAnimationFrame(() => { toast.classList.add('show'); });
    toast.addEventListener('click', dismiss);
    setTimeout(dismiss, action ? 9000 : type === 'error' ? 5000 : 3500);
}

// --- Presets page ---

function presetChips(p) {
    const chips = [];
    if (p.context_size) chips.push('ctx ' + p.context_size.toLocaleString());
    if (p.gpu_layers != null) chips.push('ngl ' + p.gpu_layers);
    if (p.backend && p.backend.startsWith('build:')) chips.push(p.backend.slice(6));
    else if (p.backend === 'cuda') chips.push('build-cuda');
    if (p.devices) chips.push('dev ' + p.devices);
    if (p.tensor_split) chips.push('ts ' + p.tensor_split);
    if (p.backend) chips.push(p.backend);
    if (p.ctk || p.ctv) chips.push('kv ' + (p.ctk || 'f16') + '/' + (p.ctv || 'f16'));
    if (p.flash_attn) chips.push('fa ' + p.flash_attn);
    if (p.parallel_slots > 1) chips.push('np ' + p.parallel_slots);
    if (p.ngram_spec) chips.push('ngram-spec');
    if (p.mmproj) chips.push('mmproj');
    return chips.map(c => '<span class="preset-chip">' + escapeHtml(c) + '</span>').join('');
}

function renderPresetsPage() {
    const listEl = document.getElementById('presets-list');
    if (!listEl) return;
    if (!presets || presets.length === 0) {
        listEl.innerHTML = '<div class="empty-state"><div class="empty-state-title">No presets yet</div><p>Create one to describe how llama-server should be launched.</p></div>';
        return;
    }
    const activeId = document.getElementById('preset-select').value;
    listEl.innerHTML = presets.map(p => {
        const active = p.id === activeId;
        const modelName = (p.model_path || '').split('/').pop() || '\u2014';
        return '<div class="preset-row' + (active ? ' is-active' : '') + '">' +
            '<div class="preset-row-main">' +
                '<div class="preset-row-title"><span class="preset-row-name">' + escapeHtml(p.name) + '</span>' +
                (active ? '<span class="badge badge-accent">Active</span>' : '') + '</div>' +
                '<div class="preset-row-model" title="' + escapeHtml(p.model_path || '') + '">' + escapeHtml(modelName) + '</div>' +
                '<div class="preset-chips">' + presetChips(p) + '</div>' +
            '</div>' +
            '<div class="preset-row-actions">' +
                (active ? '' : '<button class="btn btn-sm btn-primary" onclick="activatePreset(\'' + jsStr(p.id) + '\')">Use</button>') +
                '<button class="btn btn-sm" onclick="openPresetModal(\'edit\', \'' + jsStr(p.id) + '\')">Edit</button>' +
                '<button class="btn btn-sm" onclick="copyPreset(\'' + jsStr(p.id) + '\')">Copy</button>' +
                '<button class="btn btn-sm btn-ghost preset-delete" onclick="deletePreset(\'' + jsStr(p.id) + '\')">Delete</button>' +
            '</div>' +
        '</div>';
    }).join('');
}

function activatePreset(id) {
    const sel = document.getElementById('preset-select');
    if (!presets.find(p => p.id === id)) return;
    sel.value = id;
    sel.dispatchEvent(new Event('change', { bubbles: true }));
    const p = presets.find(pr => pr.id === id);
    showToast('Active preset: ' + p.name, 'success');
}

// --- Preset Modal ---

function setVal(id, v) { document.getElementById(id).value = v ?? ''; }
function setChk(id, v) { document.getElementById(id).checked = !!v; }
function setOpt(id, v) { document.getElementById(id).value = v || ''; }
function numOrEmpty(id, v) { document.getElementById(id).value = v != null ? v : ''; }

function clearFieldErrors() {
    document.querySelectorAll('#preset-form .field-error').forEach(el => el.classList.remove('field-error'));
}

function selectedPreset(id) {
    const targetId = id || document.getElementById('preset-select').value;
    return presets.find(pr => pr.id === targetId) || null;
}

// Options for the Build select: the configured binary, every installed
// build, and the legacy sibling build-cuda tree when a preset uses it.
async function populateBuildSelect(current) {
    const sel = document.getElementById('modal-backend');
    let builds = [];
    try {
        const res = await fetch('/api/builds/installed');
        builds = (await res.json()).builds || [];
    } catch (_) {}
    const options = ['<option value="vulkan">Configured binary (Settings)</option>'];
    builds.forEach(b => {
        options.push('<option value="build:' + escapeHtml(b.id) + '">' + escapeHtml(buildLabel(b)) + ' (installed)</option>');
    });
    // Legacy sibling build-cuda tree: only offered to presets already using it.
    // New CUDA setups come from the Install page instead.
    if (current === 'cuda') {
        options.push('<option value="cuda">Separate CUDA build (build-cuda/bin)</option>');
    }
    if (current && current.startsWith('build:') && !builds.some(b => 'build:' + b.id === current)) {
        options.push('<option value="' + escapeHtml(current) + '">' + escapeHtml(current.slice(6)) + ' (not installed)</option>');
    }
    sel.innerHTML = options.join('');
    sel.value = current || 'vulkan';
    if (sel.value !== (current || 'vulkan')) sel.value = 'vulkan';
}

function buildLabel(b) {
    const names = { vulkan: 'Vulkan', 'cuda-12.8': 'CUDA 12.8', 'cuda-13.3': 'CUDA 13.3', 'rocm-10.0': 'ROCm 10.0', cpu: 'CPU', metal: 'Metal' };
    return (names[b.backend] || b.backend) + ' ' + b.tag;
}

function openPresetModal(mode, id) {
    const modal = document.getElementById('preset-modal');
    const title = document.getElementById('modal-title');
    const form = document.getElementById('preset-form');
    form.reset();
    clearFieldErrors();

    if (mode === 'edit') {
        const p = selectedPreset(id);
        if (!p) { showToast('No preset selected', 'warn'); return; }
        title.textContent = 'Edit preset';
        setVal('modal-preset-id', p.id);
        // Model & Memory
        setVal('modal-name', p.name);
        setVal('modal-model-path', p.model_path);
        setVal('modal-mmproj', p.mmproj);
        numOrEmpty('modal-gpu-layers', p.gpu_layers);
        setChk('modal-no-mmap', p.no_mmap);
        setChk('modal-mlock', p.mlock);
        // Context & KV
        setVal('modal-context-size', p.context_size || 128000);
        setVal('modal-ctk', p.ctk || 'q8_0');
        setVal('modal-ctv', p.ctv || 'f16');
        setOpt('modal-flash-attn', p.flash_attn);
        // Batching
        setVal('modal-batch-size', p.batch_size || 2048);
        setVal('modal-ubatch-size', p.ubatch_size || p.batch_size || 2048);
        setVal('modal-parallel-slots', p.parallel_slots || 1);
        // GPU
        setVal('modal-tensor-split', p.tensor_split);
        pendingBuild = p.backend || 'vulkan';
        const dp = document.getElementById('modal-devices'); dp.innerHTML = ''; dp.dataset.unlisted = p.devices || '';
        setOpt('modal-split-mode', p.split_mode);
        numOrEmpty('modal-main-gpu', p.main_gpu);
        // Threading
        numOrEmpty('modal-threads', p.threads);
        numOrEmpty('modal-threads-batch', p.threads_batch);
        // Rope
        setOpt('modal-rope-scaling', p.rope_scaling);
        numOrEmpty('modal-rope-freq-base', p.rope_freq_base);
        numOrEmpty('modal-rope-freq-scale', p.rope_freq_scale);
        // Spec decoding
        setChk('modal-ngram-spec', p.ngram_spec);
        numOrEmpty('modal-spec-ngram-size', p.spec_ngram_size);
        numOrEmpty('modal-draft-min', p.draft_min);
        numOrEmpty('modal-draft-max', p.draft_max);
        setVal('modal-draft-model', p.draft_model);
        // Advanced
        numOrEmpty('modal-seed', p.seed);
        setVal('modal-system-prompt-file', p.system_prompt_file);
        setVal('modal-extra-args', p.extra_args);
    } else {
        title.textContent = 'New preset';
        setVal('modal-preset-id', '');
        setVal('modal-context-size', 128000);
        setVal('modal-ctk', 'q8_0');
        setVal('modal-ctv', 'f16');
        setVal('modal-batch-size', 2048);
        setVal('modal-ubatch-size', 2048);
        setVal('modal-parallel-slots', 1);
        pendingBuild = 'vulkan';
        const dp = document.getElementById('modal-devices'); dp.innerHTML = ''; dp.dataset.unlisted = '';
    }
    // The Build list comes from the server; the device picker follows it.
    populateBuildSelect(pendingBuild).then(onBackendChange);

    modal.classList.add('open');
    // Scroll modal body to top
    const body = modal.querySelector('.modal-body');
    if (body) body.scrollTop = 0;
    setTimeout(() => document.getElementById('modal-name').focus(), 50);
}

function closePresetModal() { closeModal('preset-modal'); }

// Close modal on overlay click
document.getElementById('preset-modal').addEventListener('click', e => {
    if (e.target === e.currentTarget) closePresetModal();
});

function intOrNull(id) { const v = document.getElementById(id).value; return v !== '' ? parseInt(v) : null; }
function floatOrNull(id) { const v = document.getElementById(id).value; return v !== '' ? parseFloat(v) : null; }
function strVal(id) { return document.getElementById(id).value.trim(); }

async function savePreset(event) {
    event.preventDefault();
    clearFieldErrors();

    const id = document.getElementById('modal-preset-id').value;
    const preset = {
        // Model & Memory
        name: strVal('modal-name'),
        model_path: strVal('modal-model-path'),
        mmproj: strVal('modal-mmproj'),
        gpu_layers: intOrNull('modal-gpu-layers'),
        no_mmap: document.getElementById('modal-no-mmap').checked,
        mlock: document.getElementById('modal-mlock').checked,
        // Context & KV
        context_size: parseInt(document.getElementById('modal-context-size').value) || 128000,
        ctk: strVal('modal-ctk') || 'q8_0',
        ctv: strVal('modal-ctv') || 'f16',
        flash_attn: strVal('modal-flash-attn'),
        // Batching
        batch_size: parseInt(document.getElementById('modal-batch-size').value) || 2048,
        ubatch_size: parseInt(document.getElementById('modal-ubatch-size').value) || 2048,
        parallel_slots: parseInt(document.getElementById('modal-parallel-slots').value) || 1,
        // GPU
        tensor_split: strVal('modal-tensor-split'),
        devices: selectedDevices(),
        backend: strVal('modal-backend') || 'vulkan',
        split_mode: strVal('modal-split-mode'),
        main_gpu: intOrNull('modal-main-gpu'),
        // Threading
        threads: intOrNull('modal-threads'),
        threads_batch: intOrNull('modal-threads-batch'),
        // Rope
        rope_scaling: strVal('modal-rope-scaling'),
        rope_freq_base: floatOrNull('modal-rope-freq-base'),
        rope_freq_scale: floatOrNull('modal-rope-freq-scale'),
        // Spec decoding
        ngram_spec: document.getElementById('modal-ngram-spec').checked,
        spec_ngram_size: intOrNull('modal-spec-ngram-size'),
        draft_min: intOrNull('modal-draft-min'),
        draft_max: intOrNull('modal-draft-max'),
        draft_model: strVal('modal-draft-model'),
        // Advanced
        seed: intOrNull('modal-seed'),
        system_prompt_file: strVal('modal-system-prompt-file'),
        extra_args: strVal('modal-extra-args'),
    };

    // Inline validation
    let valid = true;
    if (!preset.name) {
        document.getElementById('modal-name').classList.add('field-error');
        valid = false;
    }
    if (!preset.model_path) {
        document.getElementById('modal-model-path').classList.add('field-error');
        valid = false;
    }
    if (!valid) {
        showToast('Please fill in all required fields', 'error');
        return;
    }

    const saveBtn = document.getElementById('btn-modal-save');
    saveBtn.classList.add('saving');
    saveBtn.textContent = 'Saving...';

    try {
        let resp;
        let savedId;
        if (id) {
            resp = await fetch('/api/presets/' + encodeURIComponent(id), {
                method: 'PUT',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(preset),
            });
            if (!resp.ok) {
                const err = await resp.text().catch(() => 'Unknown error');
                showToast('Save failed: ' + err, 'error');
                return;
            }
            savedId = id;
        } else {
            resp = await fetch('/api/presets', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(preset),
            });
            if (!resp.ok) {
                const err = await resp.text().catch(() => 'Unknown error');
                showToast('Save failed: ' + err, 'error');
                return;
            }
            const data = await resp.json();
            savedId = data.preset?.id || null;
        }
        closePresetModal();
        // Editing keeps the current active preset; a new preset becomes active.
        await loadPresets(id ? document.getElementById('preset-select').value : savedId);
        showToast('Preset saved', 'success');
    } catch (err) {
        showToast('Save failed: ' + err.message, 'error');
    } finally {
        saveBtn.classList.remove('saving');
        saveBtn.textContent = 'Save';
    }
}

async function copyPreset(id) {
    const p = selectedPreset(id);
    if (!p) { showToast('No preset selected', 'warn'); return; }

    const copy = Object.assign({}, p);
    delete copy.id;
    copy.name = p.name + ' (copy)';

    try {
        const resp = await fetch('/api/presets', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(copy),
        });
        if (!resp.ok) {
            const err = await resp.text().catch(() => 'Unknown error');
            showToast('Copy failed: ' + err, 'error');
            return;
        }
        const data = await resp.json();
        await loadPresets(data.preset?.id || data.id || document.getElementById('preset-select').value);
        showToast('Preset copied', 'success');
    } catch (err) {
        showToast('Copy failed: ' + err.message, 'error');
    }
}

async function deletePreset(id) {
    const p = selectedPreset(id);
    if (!p) { showToast('No preset selected', 'warn'); return; }
    const proceed = await showConfirm('Delete preset', 'Delete preset "' + p.name + '"?', 'Delete', true);
    if (!proceed) return;

    try {
        const resp = await fetch('/api/presets/' + encodeURIComponent(p.id), { method: 'DELETE' });
        if (!resp.ok) {
            const err = await resp.text().catch(() => 'Unknown error');
            showToast('Delete failed: ' + err, 'error');
            return;
        }
        await loadPresets();
        showToast('Preset deleted', 'success');
    } catch (err) {
        showToast('Delete failed: ' + err.message, 'error');
    }
}

async function resetPresets() {
    const proceed = await showConfirm('Reset presets', 'Reset all presets to built-in defaults? Custom presets will be removed.', 'Reset', true);
    if (!proceed) return;
    try {
        const resp = await fetch('/api/presets/reset', { method: 'POST' });
        if (!resp.ok) {
            const err = await resp.text().catch(() => 'Unknown error');
            showToast('Reset failed: ' + err, 'error');
            return;
        }
        await loadPresets();
        showToast('Presets reset to defaults', 'success');
    } catch (err) {
        showToast('Reset failed: ' + err.message, 'error');
    }
}

// Clear field errors on input
// Choosing a model that has a paired projector fills --mmproj in, unless
// the field already holds something.
document.getElementById('modal-model-path').addEventListener('input', () => {
    const path = document.getElementById('modal-model-path').value.trim();
    const field = document.getElementById('modal-mmproj');
    const m = allModelsCache.find(x => x.path === path);
    if (m && m.projector && !field.value.trim()) {
        const proj = allModelsCache.find(x => x.filename === m.projector);
        if (proj) {
            field.value = proj.path;
            showToast('Projector set to ' + m.projector, 'success');
        }
    }
});

['modal-name', 'modal-model-path'].forEach(id => {
    document.getElementById(id).addEventListener('input', function() {
        this.classList.remove('field-error');
    });
});

// --- End Preset Modal ---

function getConfig() {
    const id = document.getElementById('preset-select').value;
    const p = presets.find(pr => pr.id === id) || {};
    return {
        model_path: p.model_path || '',
        mmproj: p.mmproj || '',
        context_size: p.context_size || 128000,
        ctk: p.ctk || 'q8_0',
        ctv: p.ctv || 'f16',
        tensor_split: p.tensor_split || '',
        devices: p.devices || '',
        backend: p.backend || '',
        batch_size: p.batch_size || 2048,
        ubatch_size: p.ubatch_size || p.batch_size || 2048,
        no_mmap: !!p.no_mmap,
        port: parseInt(document.getElementById('port').value) || 8080,
        ngram_spec: !!p.ngram_spec,
        parallel_slots: p.parallel_slots || 1,
        gpu_layers: p.gpu_layers ?? null,
        mlock: !!p.mlock,
        flash_attn: p.flash_attn || '',
        split_mode: p.split_mode || '',
        main_gpu: p.main_gpu ?? null,
        threads: p.threads ?? null,
        threads_batch: p.threads_batch ?? null,
        rope_scaling: p.rope_scaling || '',
        rope_freq_base: p.rope_freq_base ?? null,
        rope_freq_scale: p.rope_freq_scale ?? null,
        draft_model: p.draft_model || '',
        draft_min: p.draft_min ?? null,
        draft_max: p.draft_max ?? null,
        spec_ngram_size: p.spec_ngram_size ?? null,
        seed: p.seed ?? null,
        system_prompt_file: p.system_prompt_file || '',
        extra_args: p.extra_args || '',
    };
}

let runtimePhase = 'idle';

async function doToggle() {
    const btn = document.getElementById('btn-toggle');
    btn.disabled = true;
    if (serverRunning) {
        runtimePhase = 'stopping';
        renderRuntime();
        await fetch('/api/stop', { method: 'POST' });
    } else {
        const config = getConfig();
        if (!config.model_path) {
            showToast('No model path set. Edit the preset to select a model.', 'error');
            btn.disabled = false;
            return;
        }
        runtimePhase = 'starting';
        renderRuntime();
        const resp = await fetch('/api/start', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify(config),
        });
        const data = await resp.json();
        if (!data.ok) {
            showToast('Start failed: ' + (data.error || 'unknown'), 'error');
            runtimePhase = 'idle';
            renderRuntime();
        }
    }
    // The next WebSocket tick reflects the new state; this only guards
    // against a tick that never reports a change.
    setTimeout(() => {
        if (runtimePhase !== 'idle') {
            runtimePhase = 'idle';
            renderRuntime();
        }
    }, 4000);
}

function openLlamaUi() {
    const port = document.getElementById('port').value || '8080';
    window.open('http://' + location.hostname + ':' + port, '_blank');
}

// --- Runtime summary (sidebar + Monitor strip) ---

const PLAY_ICON = '<svg viewBox="0 0 24 24"><polygon points="5 3 19 12 5 21 5 3"/></svg>';
const STOP_ICON = '<svg viewBox="0 0 24 24"><rect x="6" y="6" width="12" height="12"/></svg>';
let lastModelPath = null;
let wsConnected = false;

function renderRuntime() {
    const phase = !wsConnected ? 'disconnected'
        : (runtimePhase === 'starting' || runtimePhase === 'stopping') ? runtimePhase
        : serverRunning ? 'running' : 'idle';
    const labels = { idle: 'Stopped', starting: 'Starting', stopping: 'Stopping', running: 'Running', disconnected: 'Disconnected' };

    const activePreset = presets.find(p => p.id === document.getElementById('preset-select').value);
    const modelPath = serverRunning && lastModelPath ? lastModelPath : (activePreset?.model_path || '');
    const modelName = modelPath ? modelPath.split('/').pop() : '';
    const port = document.getElementById('port').value || '8080';

    // Sidebar
    const state = document.getElementById('sidebar-runtime-state');
    state.textContent = labels[phase];
    state.dataset.phase = phase;
    const model = document.getElementById('sidebar-runtime-model');
    model.textContent = serverRunning ? (modelName || 'Model unavailable') : (modelName ? 'Next: ' + modelName : 'No server running');
    model.title = modelPath;
    document.getElementById('sidebar-runtime-endpoint').textContent = serverRunning ? 'Endpoint: ' + location.hostname + ':' + port : '';

    // Launch button
    const toggleBtn = document.getElementById('btn-toggle');
    const busy = phase === 'starting' || phase === 'stopping';
    toggleBtn.disabled = busy || !wsConnected;
    toggleBtn.className = 'btn sidebar-launch-btn ' + (serverRunning ? 'btn-danger' : 'btn-primary');
    document.getElementById('btn-toggle-icon').innerHTML = serverRunning ? STOP_ICON : PLAY_ICON;
    document.getElementById('btn-toggle-label').textContent =
        phase === 'starting' ? 'Starting…' : phase === 'stopping' ? 'Stopping…' : serverRunning ? 'Stop server' : 'Start server';
    document.querySelectorAll('.btn-open-ui').forEach(b => b.classList.toggle('hidden', !serverRunning));

    // Monitor strip
    const badge = document.getElementById('monitor-runtime-state');
    badge.textContent = labels[phase];
    badge.className = 'badge ' + (phase === 'running' ? 'badge-green' : phase === 'disconnected' ? 'badge-red' : busy ? 'badge-yellow' : 'badge-neutral');
    document.getElementById('monitor-runtime-model').textContent = serverRunning
        ? (modelName || 'Model unavailable')
        : (activePreset ? 'Ready to launch: ' + activePreset.name : 'No preset selected');
    const details = [];
    if (serverRunning) details.push('llama-server', 'Endpoint: ' + location.hostname + ':' + port);
    else details.push('Port ' + port);
    if (activePreset) details.push('Preset: ' + activePreset.name);
    document.getElementById('monitor-runtime-details').innerHTML = details.map(d => '<span>' + escapeHtml(d) + '</span>').join('');

    document.getElementById('monitor-nav-live').classList.toggle('hidden', !serverRunning);
    document.getElementById('monitor-process-tool').classList.toggle('hidden', !serverRunning);
    document.getElementById('monitor-process-state').classList.toggle('hidden', !serverRunning);

    const chatBadge = document.getElementById('chat-server-badge');
    chatBadge.textContent = serverRunning ? 'Server running on :' + port : 'Server stopped';
    chatBadge.className = 'badge ' + (serverRunning ? 'badge-green' : 'badge-dim');
}

// --- Process output ---

let logClearedAt = 0;

function clearOutput(e) {
    if (e) {
        e.preventDefault();
        e.stopPropagation();
    }
    logClearedAt = prevLogLen;
    document.getElementById('log-panel').textContent = '';
    setLogCounts(0);
}

function setLogCounts(shown) {
    document.getElementById('monitor-output-lines').textContent = shown + (shown === 1 ? ' line' : ' lines');
}

// --- Monitor event log (Logs page) ---

let appLog = [];
let appLogSeq = 0;
let appLogFetching = false;
let logsUnread = 0;

function renderLogsNav() {
    const el = document.getElementById('nav-count-logs');
    el.textContent = logsUnread > 0 ? String(logsUnread) : '';
    el.classList.toggle('is-alert', logsUnread > 0);
}

function fmtLogTime(ms) {
    const d = new Date(ms);
    return d.toLocaleDateString(undefined, { month: 'short', day: '2-digit' }) + ' ' + d.toLocaleTimeString([], { hour12: false });
}

function renderAppLog() {
    const host = document.getElementById('app-log');
    const problemsOnly = document.getElementById('logs-problems-only').checked;
    const rows = problemsOnly ? appLog.filter(e => e.level !== 'info') : appLog;
    const wasAtBottom = host.scrollHeight - host.scrollTop - host.clientHeight < 40;
    if (rows.length === 0) {
        host.innerHTML = '<div class="empty-state"><div class="empty-state-title">' + (problemsOnly ? 'No warnings or errors' : 'Nothing logged yet') + '</div><p>Launches, failures, updates and telemetry problems appear here.</p></div>';
    } else {
        host.innerHTML = rows.map(e =>
            '<div class="app-log-row is-' + e.level + '">' +
            '<span class="app-log-time">' + fmtLogTime(e.ts_ms) + '</span>' +
            '<span class="app-log-level ' + e.level + '">' + e.level + '</span>' +
            '<span class="app-log-msg">' + escapeHtml(e.message) + '</span></div>'
        ).join('');
    }
    document.getElementById('logs-count').textContent = rows.length + (rows.length === 1 ? ' entry' : ' entries');
    if (wasAtBottom) host.scrollTop = host.scrollHeight;
}

// Pulls entries newer than what we have. Driven by the WebSocket's
// app_log_seq, so idle pages never poll.
async function fetchAppLog() {
    if (appLogFetching) return;
    appLogFetching = true;
    try {
        const res = await fetch('/api/app/logs' + (appLogSeq > 0 ? '?after=' + appLogSeq : ''));
        const data = await res.json();
        const fresh = data.entries || [];
        if (fresh.length) {
            appLog = appLog.concat(fresh).slice(-2000);
            appLogSeq = fresh[fresh.length - 1].seq;
            if (activeSection !== 'logs') {
                logsUnread += fresh.filter(e => e.level !== 'info').length;
                renderLogsNav();
            }
            renderAppLog();
        }
        if (data.latest_seq > appLogSeq) appLogSeq = data.latest_seq;
    } catch (_) { /* next tick retries */ }
    appLogFetching = false;
}

function downloadAppLog() {
    if (appLog.length === 0) { showToast('Nothing logged yet', 'warn'); return; }
    const text = appLog.map(e => new Date(e.ts_ms).toISOString() + ' [' + e.level + '] ' + e.message).join('\n') + '\n';
    const blob = new Blob([text], { type: 'text/plain' });
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = 'llama-admin-monitor-' + new Date().toISOString().replace(/[:T]/g, '-').slice(0, 19) + '.log';
    document.body.appendChild(a);
    a.click();
    a.remove();
    setTimeout(() => URL.revokeObjectURL(a.href), 1000);
}

function renderLogs(logs) {
    if (logs.length === prevLogLen) return;
    // The backlog shrinks when the server restarts; drop the cleared offset
    // so the new run's output is visible.
    if (logs.length < logClearedAt) logClearedAt = 0;
    const el = document.getElementById('log-panel');
    const wasAtBottom = el.scrollHeight - el.scrollTop - el.clientHeight < 40;
    el.textContent = logs.slice(logClearedAt).join('\n');
    if (wasAtBottom) el.scrollTop = el.scrollHeight;
    prevLogLen = logs.length;
    setLogCounts(logs.length - logClearedAt);
}

// WebSocket
loadModelsCache();
const ws = new WebSocket((location.protocol === 'https:' ? 'wss://' : 'ws://') + location.host + '/ws');
ws.onopen = () => { wsConnected = true; renderRuntime(); scheduleUpdateChecks(); fetchAppLog(); };
ws.onmessage = e => {
    const d = JSON.parse(e.data);

    // Server state
    const wasRunning = serverRunning;
    serverRunning = d.server_running;
    lastModelPath = d.model_path || null;
    if (serverRunning !== wasRunning) runtimePhase = 'idle';
    updateHfProgress(d.hf_download);
    updateBenchProgress(d.bench);

    // Surface startup/crash failures once, rather than on every tick.
    const errBox = document.getElementById('monitor-error');
    if (d.server_error && d.server_error !== lastServerError) {
        lastServerError = d.server_error;
        showToast(d.server_error, 'error', { label: 'Process output', onClick: () => { switchTab('monitor'); const card = document.getElementById('monitor-output-card'); card.open = true; card.scrollIntoView({ behavior: 'smooth', block: 'start' }); } });
        runtimePhase = 'idle';
    } else if (!d.server_error) {
        lastServerError = null;
    }
    errBox.hidden = !d.server_error;
    if (d.server_error) errBox.textContent = d.server_error;
    if ((d.app_log_seq || 0) > appLogSeq) fetchAppLog();

    renderRuntime();

    // Inference
    const l = d.llama;
    document.getElementById('m-prompt').textContent = l.prompt_tokens_per_sec > 0 ? l.prompt_tokens_per_sec.toFixed(1) + ' t/s' : '\u2014';
    document.getElementById('m-gen').textContent = l.generation_tokens_per_sec > 0 ? l.generation_tokens_per_sec.toFixed(1) + ' t/s' : '\u2014';
    const ctxBar = document.getElementById('m-ctx-bar');
    if (l.kv_cache_max > 0) {
        const pctNum = (l.kv_cache_tokens / l.kv_cache_max) * 100;
        document.getElementById('m-ctx').textContent = l.kv_cache_tokens.toLocaleString() + ' / ' + l.kv_cache_max.toLocaleString() + ' (' + pctNum.toFixed(1) + '%)';
        ctxBar.style.width = Math.min(100, pctNum).toFixed(1) + '%';
        ctxBar.className = barClass(pctNum);
    } else {
        document.getElementById('m-ctx').textContent = '\u2014';
        ctxBar.style.width = '0%';
    }
    document.getElementById('m-slots').textContent = l.slots_idle + l.slots_processing > 0 ? l.slots_idle + ' idle \u00b7 ' + l.slots_processing + ' busy' : '\u2014';

    const statusEl = document.getElementById('m-status');
    statusEl.textContent = l.status || (serverRunning ? 'waiting' : 'offline');
    statusEl.className = 'badge ' + (l.status === 'ok' ? 'status-ok' : l.status === 'no slot available' ? 'status-busy' : l.status ? 'status-err' : 'badge-dim');
    document.getElementById('monitor-inference-kicker').textContent = 'Llama server \u00b7 ' + (serverRunning ? (l.status || 'starting') : 'idle');

    // GPU
    const gpuList = Object.entries(d.gpu);
    totalVramMb = gpuList.reduce((sum, [, m]) => sum + (m.vram_total || 0), 0);
    usedVramMb = gpuList.reduce((sum, [, m]) => sum + (m.vram_used || 0), 0);
    renderVramBar(d);
    renderGpuCards(gpuList);
    renderSystemCards(d.system);
    renderUpdatePhase(d.app_update || null);
    renderInstallProgress(d.build_install || null);

    const telemetry = document.getElementById('monitor-telemetry-badge');
    telemetry.textContent = gpuList.length > 0 ? 'GPU telemetry \u00b7 Live \u00b7 ' + gpuList.length + ' device' + (gpuList.length === 1 ? '' : 's') : 'GPU telemetry \u00b7 Unavailable';
    telemetry.className = 'badge ' + (gpuList.length > 0 ? 'badge-green' : 'badge-dim');
    document.getElementById('monitor-last-updated').textContent = new Date().toLocaleTimeString([], { hour12: false });

    // Logs
    renderLogs(d.logs || []);

    // Nav counts
    document.getElementById('nav-count-chat').textContent = chatHistory.length > 0 ? String(chatHistory.length) : '';
};
ws.onerror = e => console.error('WebSocket error:', e);
ws.onclose = () => {
    wsConnected = false;
    renderRuntime();
    showToast('Lost connection to Llama Admin Monitor. Reload the page to reconnect.', 'warn');
};

// Markdown
if (typeof marked !== 'undefined') {
    marked.setOptions({ breaks: true, gfm: true });
}
function renderMd(src) {
    if (typeof marked !== 'undefined') {
        try { return marked.parse(src); } catch(_) {}
    }
    return escapeHtml(src).replace(/\n/g, '<br>');
}

// Chat
let chatHistory = [];
let chatBusy = false;

document.getElementById('chat-input').addEventListener('keydown', e => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendChat(); }
});

function clearChat() {
    chatHistory = [];
    const box = document.getElementById('chat-messages');
    box.innerHTML = '<div class="empty-state" id="chat-empty"><div class="empty-state-title">No messages yet</div><p>Start the server, then send a message below.</p></div>';
    document.getElementById('nav-count-chat').textContent = '';
}

function chatScroll() {
    const c = document.getElementById('chat-messages');
    c.scrollTop = c.scrollHeight;
}

function appendMsg(role, text) {
    const empty = document.getElementById('chat-empty');
    if (empty) empty.remove();
    const el = document.createElement('div');
    el.className = 'msg msg-' + role;
    el.textContent = text;
    document.getElementById('chat-messages').appendChild(el);
    chatScroll();
    return el;
}

async function sendChat() {
    if (chatBusy) return;
    const input = document.getElementById('chat-input');
    const text = input.value.trim();
    if (!text) return;
    if (!serverRunning) {
        showToast('Start the server before chatting', 'warn');
        return;
    }
    input.value = '';

    chatHistory.push({ role: 'user', content: text });
    appendMsg('user', text);
    document.getElementById('nav-count-chat').textContent = String(chatHistory.length);

    const url = '/v1/chat/completions';

    chatBusy = true;
    document.getElementById('btn-send').disabled = true;

    let thinkEl = null;
    let thinkContent = '';
    const msgEl = appendMsg('assistant', '');
    let msgContent = '';

    try {
        const resp = await fetch(url, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                messages: chatHistory,
                stream: true,
                temperature: 1.0,
                top_p: 0.95,
                top_k: 40,
                min_p: 0.01,
                repeat_penalty: 1.0,
            }),
        });

        const reader = resp.body.getReader();
        const decoder = new TextDecoder();
        let buf = '';

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;
            buf += decoder.decode(value, { stream: true });

            const lines = buf.split('\n');
            buf = lines.pop() || '';

            for (const line of lines) {
                if (!line.startsWith('data: ')) continue;
                const payload = line.slice(6).trim();
                if (payload === '[DONE]') continue;
                try {
                    const obj = JSON.parse(payload);
                    const delta = obj.choices && obj.choices[0] && obj.choices[0].delta;
                    if (!delta) continue;

                    // Reasoning / thinking content
                    const rc = delta.reasoning_content || '';
                    if (rc) {
                        thinkContent += rc;
                        if (!thinkEl) {
                            thinkEl = document.createElement('details');
                            thinkEl.className = 'msg msg-thinking';
                            thinkEl.innerHTML = '<summary>Reasoning</summary><span></span>';
                            document.getElementById('chat-messages').insertBefore(thinkEl, msgEl);
                        }
                        thinkEl.querySelector('span').textContent = thinkContent;
                    }

                    // Regular content
                    const c = delta.content || '';
                    if (c) {
                        msgContent += c;
                        msgEl.innerHTML = renderMd(msgContent);
                    }
                } catch (_) {}
            }
            chatScroll();
        }
    } catch (err) {
        msgEl.textContent = '[error] ' + err.message;
        msgEl.classList.add('msg-error');
    }

    if (msgContent) {
        chatHistory.push({ role: 'assistant', content: msgContent });
    }
    chatBusy = false;
    document.getElementById('btn-send').disabled = false;
    document.getElementById('nav-count-chat').textContent = String(chatHistory.length);
}
if ('serviceWorker' in navigator) {
    navigator.serviceWorker.register('/sw.js').catch(() => {});
}

// --- In-app updates ---

let updateStatus = null;

function releaseLabel(r) {
    if (!r) return 'none published';
    const when = r.published_at ? ' \u00b7 ' + new Date(r.published_at).toLocaleDateString() : '';
    return r.tag + when;
}

async function checkAppUpdates(force = false) {
    const note = document.getElementById('updates-note');
    note.hidden = true;
    document.getElementById('updates-available-row').hidden = true;
    document.getElementById('updates-switch-row').hidden = true;
    try {
        const res = await fetch('/api/app/update/check' + (force ? '?force=1' : ''));
        const data = await res.json();
        if (data.error) throw new Error(data.error);
        updateStatus = data;

        document.getElementById('update-current-version').textContent = 'v' + data.current_version;
        const trackBadge = document.getElementById('update-current-track');
        trackBadge.textContent = data.current_track === 'dev' ? 'local build' : data.current_track + ' track';
        trackBadge.className = 'badge ' + (data.current_track === 'beta' ? 'badge-accent' : data.current_track === 'main' ? 'badge-green' : 'badge-dim');

        if (!data.platform_asset) {
            note.textContent = 'In-app updates are available for Linux and macOS builds only.';
            note.hidden = false;
            return;
        }
        if (data.current_track === 'dev') {
            // A local cargo build has no release track, so there is nothing to
            // "update" to; but a GitHub release can replace it, after which
            // the normal update flow applies.
            note.textContent = 'This is a local cargo build. Installing a release below replaces this binary with the GitHub build for that track and enables automatic update checks.';
            note.hidden = false;
            if (data.stable && data.stable.asset_url) {
                document.getElementById('update-available-title').textContent = 'Install the stable release';
                document.getElementById('update-available-desc').textContent = 'Recommended for most users · ' + releaseLabel(data.stable);
                const btn = document.getElementById('btn-update-current');
                btn.textContent = 'Install ' + data.stable.tag;
                btn.onclick = () => applyAppUpdate('main', data.stable.tag);
                document.getElementById('updates-available-row').hidden = false;
            }
            if (data.beta && data.beta.asset_url) {
                document.getElementById('update-switch-title').textContent = 'Install the beta release';
                document.getElementById('update-switch-desc').textContent = 'Newer features, less tested · ' + releaseLabel(data.beta);
                const btn = document.getElementById('btn-update-switch');
                btn.textContent = 'Install ' + data.beta.tag;
                btn.onclick = () => applyAppUpdate('beta', data.beta.tag);
                document.getElementById('updates-switch-row').hidden = false;
            }
            return;
        }

        const onBeta = data.current_track === 'beta';
        const current = onBeta ? data.beta : data.stable;
        const other = onBeta ? data.stable : data.beta;

        if (data.update_available && current && current.asset_url) {
            document.getElementById('update-available-title').textContent = (onBeta ? 'Beta' : 'Stable') + ' update available';
            document.getElementById('update-available-desc').textContent = releaseLabel(current);
            const btn = document.getElementById('btn-update-current');
            btn.textContent = 'Install ' + current.tag;
            btn.onclick = () => applyAppUpdate(onBeta ? 'beta' : 'main', current.tag);
            document.getElementById('updates-available-row').hidden = false;
        } else {
            note.textContent = 'You are on the latest ' + data.current_track + ' release.';
            note.hidden = false;
        }

        if (other && other.asset_url) {
            document.getElementById('update-switch-title').textContent = onBeta ? 'Switch to stable' : 'Switch to beta';
            document.getElementById('update-switch-desc').textContent = (onBeta ? 'Recommended for most users \u00b7 ' : 'Newer features, less tested \u00b7 ') + releaseLabel(other);
            const btn = document.getElementById('btn-update-switch');
            btn.textContent = 'Switch to ' + other.tag;
            btn.onclick = () => applyAppUpdate(onBeta ? 'main' : 'beta', other.tag);
            document.getElementById('updates-switch-row').hidden = false;
        }
    } catch (err) {
        note.textContent = 'Could not check for updates: ' + err.message;
        note.hidden = false;
    }
}

let updateWatch = null;

async function applyAppUpdate(track, tag) {
    const proceed = await showConfirm('Install ' + tag,
        'This downloads the ' + track + ' release ' + tag + ', stops llama-server, replaces the running binary and restarts. The page reconnects on its own.', 'Install');
    if (!proceed) return;

    document.getElementById('updates-available-row').hidden = true;
    document.getElementById('updates-switch-row').hidden = true;
    document.getElementById('updates-note').hidden = true;
    const box = document.getElementById('updates-installing');
    box.hidden = false;
    document.getElementById('updates-phase').textContent = 'Starting\u2026';

    try {
        const res = await fetch('/api/app/update/apply', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ track }),
        });
        const data = await res.json();
        if (!data.ok) throw new Error(data.error || 'unknown error');
    } catch (err) {
        box.hidden = true;
        showToast('Update failed to start: ' + err.message, 'error');
        return;
    }

    // Progress arrives over the WebSocket (`app_update`). Once the socket
    // drops the process is restarting; poll until it answers, then reload.
    clearInterval(updateWatch);
    updateWatch = setInterval(async () => {
        if (wsConnected) return;
        try {
            const res = await fetch('/', { cache: 'no-store' });
            if (res.ok) {
                clearInterval(updateWatch);
                window.location.reload();
            }
        } catch (_) { /* still restarting */ }
    }, 2000);
}

// Background check: an indicator in the sidebar and one toast per newer
// release, dismissable per tag. Release builds only; dev builds never nag.
const UPDATE_SEEN_KEY = 'llama_admin_monitor_update_seen';
let updateChecksScheduled = false;

async function quietUpdateCheck() {
    try {
        const res = await fetch('/api/app/update/check');
        const data = await res.json();
        if (data.error || data.current_track === 'dev' || !data.platform_asset) return;
        const latest = data.current_track === 'beta' ? data.beta : data.stable;
        const indicator = document.getElementById('update-indicator');
        if (!data.update_available || !latest || !latest.asset_url) {
            indicator.classList.add('hidden');
            return;
        }
        indicator.classList.remove('hidden');
        indicator.title = latest.tag + ' is available on the ' + data.current_track + ' track';
        let seen = '';
        try { seen = localStorage.getItem(UPDATE_SEEN_KEY) || ''; } catch (_) {}
        if (seen !== latest.tag) {
            try { localStorage.setItem(UPDATE_SEEN_KEY, latest.tag); } catch (_) {}
            showToast('Update available: ' + latest.tag + ' (' + data.current_track + ' track)', 'success',
                { label: 'Open updates', onClick: () => openConfigModal('updates') });
        }
    } catch (_) { /* offline or GitHub unreachable: try again next time */ }
}

function scheduleUpdateChecks() {
    if (updateChecksScheduled) return;
    updateChecksScheduled = true;
    setTimeout(quietUpdateCheck, 3000);
    setInterval(quietUpdateCheck, 6 * 60 * 60 * 1000);
}

// Called from the WebSocket handler with the backend's update phase.
let lastUpdatePhase = null;
function renderUpdatePhase(phase) {
    if (phase === lastUpdatePhase) return;
    lastUpdatePhase = phase;
    const box = document.getElementById('updates-installing');
    if (!phase) {
        box.hidden = true;
        return;
    }
    if (phase.startsWith('Error')) {
        box.hidden = true;
        showToast('Update failed: ' + phase.slice(7), 'error');
        return;
    }
    box.hidden = false;
    document.getElementById('updates-phase').textContent = phase + '\u2026';
}

document.addEventListener('DOMContentLoaded', () => {
    // Populate OpenAI API Base URL dynamically
    const apiUrlEls = document.querySelectorAll('.api-base-url-display');
    apiUrlEls.forEach(el => {
        el.textContent = window.location.protocol + '//' + window.location.host + '/v1';
    });

    const outCard = document.getElementById('monitor-output-card');
    if (outCard) {
        const stored = localStorage.getItem('llama_admin_monitor_output_open');
        if (stored !== null) outCard.open = stored === 'true';
        outCard.addEventListener('toggle', () => {
            localStorage.setItem('llama_admin_monitor_output_open', outCard.open);
        });
    }
});


// --- Install page: prebuilt llama.cpp builds ---

let installCatalog = null;
let installedBuilds = [];

async function loadInstallPage(force) {
    if (force || !installCatalog) {
        document.getElementById('install-release').innerHTML = '<option value="">Loading\u2026</option>';
        try {
            const res = await fetch('/api/builds/catalog');
            installCatalog = await res.json();
        } catch (err) {
            installCatalog = { backends: [], releases: [], error: err.message };
        }
    }
    await refreshInstalledBuilds();
    renderInstallCatalog();
}

async function refreshInstalledBuilds() {
    try {
        const res = await fetch('/api/builds/installed');
        installedBuilds = (await res.json()).builds || [];
    } catch (_) { installedBuilds = []; }
    renderInstalledBuilds();
}

function renderInstalledBuilds() {
    const host = document.getElementById('builds-installed');
    document.getElementById('builds-root').textContent = installCatalog && installCatalog.root ? installCatalog.root : '';
    if (installedBuilds.length === 0) {
        host.innerHTML = '<div class="empty-state"><div class="empty-state-title">No builds installed yet</div><p>Install one below, or keep using the binary configured in Settings.</p></div>';
        return;
    }
    const latest = installCatalog && installCatalog.releases && installCatalog.releases[0] ? installCatalog.releases[0].tag : '';
    const tagNum = t => parseInt(String(t || '').replace(/^b/, ''), 10) || 0;
    const stale = installedBuilds.filter(b => tagNum(latest) > tagNum(b.tag)).length;
    const summary = document.getElementById('builds-update-summary');
    if (summary) {
        summary.textContent = latest
            ? (stale ? stale + ' of ' + installedBuilds.length + ' build(s) behind the newest release, ' + latest : 'All builds are on the newest release, ' + latest)
            : '';
    }
    host.innerHTML = installedBuilds.map(b => {
        const behind = tagNum(latest) > tagNum(b.tag);
        const updateBtn = behind
            ? '<button class="btn btn-sm btn-primary" type="button" onclick="updateBuild(\'' + jsStr(b.id) + '\', \'' + jsStr(latest) + '\')" title="Install ' + escapeHtml(latest) + ' for this backend, move presets and Settings to it, and remove ' + escapeHtml(b.tag) + '">Update to ' + escapeHtml(latest) + '</button>'
            : '';
        const devices = b.devices && b.devices.length
            ? b.devices.map(d => '<span class="preset-chip" title="' + escapeHtml(d.name) + '">' + escapeHtml(d.id) + ' \u00b7 ' + escapeHtml(d.name) + '</span>').join('')
            : '<span class="preset-chip" title="' + escapeHtml(b.device_check_error || '') + '">' + (b.device_check_error ? 'device check failed' : 'no GPU devices') + '</span>';
        return '<div class="build-row">' +
            '<div class="build-row-main">' +
                '<div class="build-row-title">' + escapeHtml(buildLabel(b)) + '<span class="badge badge-neutral mono">build:' + escapeHtml(b.id) + '</span>' + (behind ? '<span class="badge badge-accent">' + escapeHtml(latest) + ' available</span>' : '<span class="badge badge-green">up to date</span>') + '</div>' +
                '<div class="build-row-path" title="' + escapeHtml(b.server_path) + '">' + escapeHtml(b.server_path) + '</div>' +
                '<div class="build-row-devices">' + devices + '</div>' +
            '</div>' +
            '<div class="build-row-actions">' + updateBtn +
                '<button class="btn btn-sm" type="button" onclick="useBuildInSettings(\'' + jsStr(b.id) + '\')" title="Point Settings at this build so presets on Configured binary use it">Use in Settings</button>' +
                '<button class="btn btn-sm btn-ghost preset-delete" type="button" onclick="removeBuild(\'' + jsStr(b.id) + '\')">Remove</button>' +
            '</div></div>';
    }).join('');
}

function renderInstallCatalog() {
    const sel = document.getElementById('install-release');
    const note = document.getElementById('install-note');
    const releases = installCatalog.releases || [];
    sel.innerHTML = releases.length
        ? releases.map(r => '<option value="' + escapeHtml(r.tag) + '">' + escapeHtml(r.tag) + (r.published_at ? ' \u00b7 ' + new Date(r.published_at).toLocaleDateString() : '') + '</option>').join('')
        : '<option value="">No releases found</option>';
    const backends = installCatalog.backends || [];
    const host = document.getElementById('install-backends');
    if (backends.length === 0) {
        host.innerHTML = '<span class="help-text">No prebuilt llama.cpp binaries are published for this OS/CPU.</span>';
    } else {
        host.innerHTML = backends.map((b, i) => {
            const have = installedBuilds.some(x => x.backend === b.id);
            return '<label class="backend-option' + (have ? ' is-installed' : '') + '"><input type="radio" name="install-backend" value="' + escapeHtml(b.id) + '"' + (i === 0 ? ' checked' : '') + '>' +
                '<span class="backend-option-label">' + escapeHtml(b.label) + '</span>' +
                '<span class="backend-option-req">' + escapeHtml(b.requires) + '</span></label>';
        }).join('');
    }
    if (installCatalog.error) {
        note.textContent = 'Could not list llama.cpp releases: ' + installCatalog.error;
        note.hidden = false;
    } else {
        note.hidden = true;
    }
    document.getElementById('btn-install-build').disabled = releases.length === 0 || backends.length === 0;
}

async function startInstall() {
    const tag = document.getElementById('install-release').value;
    const backend = (document.querySelector('input[name="install-backend"]:checked') || {}).value;
    if (!tag || !backend) return;
    const id = backend + '-' + tag;
    if (installedBuilds.some(b => b.id === id)) {
        showToast(id + ' is already installed', 'warn');
        return;
    }
    const proceed = await showConfirm('Install ' + id,
        'Download llama.cpp ' + tag + ' (' + backend + ') from GitHub into the builds folder and check which devices it can see?', 'Install');
    if (!proceed) return;
    try {
        const res = await fetch('/api/builds/install', {
            method: 'POST', headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ backend, tag }),
        });
        const data = await res.json();
        if (!data.ok) throw new Error(data.error || 'unknown error');
        showToast('Installing ' + id + '\u2026', 'success');
    } catch (err) {
        showToast('Install failed to start: ' + err.message, 'error');
    }
}

let lastInstallKey = '';

function renderInstallProgress(p) {
    const box = document.getElementById('install-progress');
    const nav = document.getElementById('nav-count-install');
    if (!p) {
        box.hidden = true;
        nav.textContent = '';
        return;
    }
    const pct = p.total_bytes > 0 ? Math.min(100, p.downloaded_bytes / p.total_bytes * 100) : 0;
    if (!p.done) {
        box.hidden = false;
        document.getElementById('install-progress-title').textContent = 'Installing ' + p.id;
        document.getElementById('install-progress-phase').textContent = p.phase;
        document.getElementById('install-progress-bar').style.width = (p.total_bytes > 0 ? pct : 100).toFixed(0) + '%';
        document.getElementById('install-progress-bytes').textContent = p.total_bytes > 0 ? fmtBytes(p.downloaded_bytes) + ' of ' + fmtBytes(p.total_bytes) : '';
        nav.textContent = p.total_bytes > 0 ? pct.toFixed(0) + '%' : '\u2026';
        return;
    }
    nav.textContent = '';
    // Announce completion once per install.
    const key = p.id + ':' + (p.error || 'ok');
    if (key !== lastInstallKey) {
        lastInstallKey = key;
        box.hidden = true;
        if (p.error) showToast('Install of ' + p.id + ' failed: ' + p.error, 'error', { label: 'Open Install', onClick: () => switchTab('install') });
        else showToast(p.phase || ('Installed ' + p.id), 'success', { label: 'Open Install', onClick: () => switchTab('install') });
        refreshInstalledBuilds();
        loadPresets(document.getElementById('preset-select').value || undefined);
    }
}

async function updateBuild(id, latest) {
    const proceed = await showConfirm('Update ' + id,
        'Install llama.cpp ' + latest + ' for this backend, switch presets and Settings that use ' + id + ' over to it, then remove the old build?', 'Update');
    if (!proceed) return;
    try {
        const res = await fetch('/api/builds/update', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ id }) });
        const data = await res.json();
        if (!data.ok) throw new Error(data.error || 'unknown error');
        showToast('Updating ' + id + ' to ' + latest + '\u2026', 'success');
    } catch (err) {
        showToast('Update failed to start: ' + err.message, 'error');
    }
}

async function removeBuild(id) {
    const proceed = await showConfirm('Remove ' + id, 'Delete this build from disk? Presets that use it will fail to start until they pick another build.', 'Remove', true);
    if (!proceed) return;
    try {
        const res = await fetch('/api/builds/remove', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ id }) });
        const data = await res.json();
        if (!data.ok) throw new Error(data.error || 'unknown error');
        showToast('Removed ' + id, 'success');
        refreshInstalledBuilds();
    } catch (err) {
        showToast('Remove failed: ' + err.message, 'error');
    }
}

async function useBuildInSettings(id) {
    try {
        const res = await fetch('/api/builds/use', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ id }) });
        const data = await res.json();
        if (!data.ok) throw new Error(data.error || 'unknown error');
        const settings = await (await fetch('/api/settings')).json();
        applySettings(settings);
        showToast('Settings now use ' + id + ' for presets on "Configured binary"', 'success');
    } catch (err) {
        showToast('Could not update Settings: ' + err.message, 'error');
    }
}
