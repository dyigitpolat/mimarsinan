/* ══════════════════════════════════════════════════════════════
   MIMARSINAN CONFIGURATION WIZARD — Controller
   ══════════════════════════════════════════════════════════════ */

// Populated from API (GET /api/model_types, GET /api/model_config_schema/:id)
let MODEL_TYPES = [];
let MODEL_CONFIG_SCHEMAS = {};

function loadFromAPI() {
  const select = document.getElementById('dataProvider');
  return fetch('/api/data_providers')
    .then(function (r) { return r.json(); })
    .then(function (list) {
      select.innerHTML = '';
      list.forEach(function (p) {
        select.appendChild(new Option(p.label, p.id));
      });
      if (select.options.length) select.value = select.options[0].value;
      return fetch('/api/model_types');
    })
    .then(function (r) { return r.json(); })
    .then(function (list) {
      MODEL_TYPES = list.map(function (m) {
        return { id: m.id, label: m.label, cat: m.category || 'native' };
      });
      if (MODEL_TYPES.length) state.modelType = MODEL_TYPES[0].id;
      var schemaId = state.modelType || 'mlp_mixer';
      return fetch('/api/model_config_schema/' + schemaId);
    })
    .then(function (r) { return r.json(); })
    .then(function (schema) {
      var id = state.modelType || 'mlp_mixer';
      MODEL_CONFIG_SCHEMAS[id] = schema;
    })
    .catch(function (err) {
      console.error('Wizard API load failed', err);
      select.innerHTML = '<option value="">Failed to load</option>';
    });
}

function ensureModelConfigSchema(modelTypeId) {
  if (MODEL_CONFIG_SCHEMAS[modelTypeId]) return Promise.resolve();
  return fetch('/api/model_config_schema/' + modelTypeId)
    .then(function (r) { return r.json(); })
    .then(function (schema) {
      MODEL_CONFIG_SCHEMAS[modelTypeId] = schema;
    });
}

// ── Hardcoded fallbacks (used only if API unavailable) ───────

// ── State ──────────────────────────────────────────────────
let state = {
  modelType: "mlp_mixer",
  coreTypes: [{ max_axons: 1025, max_neurons: 257, count: 100 }],
};

// ── Init ───────────────────────────────────────────────────
function init() {
  loadFromAPI().then(function () {
    renderModelChips();
    renderModelConfigFields();
    renderCoreTypes();
    applySpikingDeps();
    applyHwDeps();
    updateSearchVisibility();
    update();
    var showJson = document.getElementById('showJsonToggle');
    var copyBtn = document.getElementById('copyJsonBtn');
    if (copyBtn && showJson) copyBtn.style.display = showJson.classList.contains('on') ? '' : 'none';
  });
}

// ── Section toggle ─────────────────────────────────────────
function toggleSection(header) { header.parentElement.classList.toggle('open'); }

// ── Segmented control ──────────────────────────────────────
function setSegment(btn) {
  const parent = btn.parentElement;
  parent.querySelectorAll('.seg-btn').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
  handleSegmentChange(parent.id, btn.dataset.val);
  update();
}
function getSegVal(id) { return document.getElementById(id)?.querySelector('.seg-btn.active')?.dataset.val || ''; }
function setSegVal(containerId, val) {
  const c = document.getElementById(containerId); if (!c) return;
  c.querySelectorAll('.seg-btn').forEach(b => b.classList.toggle('active', b.dataset.val === val));
}

// ── Toggle helpers ─────────────────────────────────────────
function toggleClick(el) {
  if (el.classList.contains('forced')) return;
  el.classList.toggle('on');
  handleToggleChange(el.id);
  update();
}
function isToggleOn(id) { return document.getElementById(id)?.classList.contains('on') || false; }
function setToggle(id, val, forced = false) {
  const el = document.getElementById(id); if (!el) return;
  el.classList.toggle('on', val);
  el.classList.toggle('forced', forced);
}

// ── Segment change handlers ────────────────────────────────
function handleSegmentChange(controlId, val) {
  if (controlId === 'configMode') {
    document.getElementById('userModelConfig').className = 'cond ' + (val === 'user' ? 'visible' : 'hidden');
    updateSearchVisibility();
  }
  if (controlId === 'weightSourceMode') {
    document.getElementById('trainFromScratch').className = 'cond ' + (val === 'train' ? 'visible' : 'hidden');
    document.getElementById('pretrainedConfig').className = 'cond ' + (val !== 'train' ? 'visible' : 'hidden');
  }
  if (controlId === 'spikingMode') { applySpikingDeps(); }
  if (controlId === 'hwMode') {
    document.getElementById('hwFixed').className = 'cond ' + (val === 'user' ? 'visible' : 'hidden');
    document.getElementById('hwAuto').className = 'cond ' + (val !== 'user' ? 'visible' : 'hidden');
    updateSearchVisibility();
  }
  if (controlId === 'optimizer') {
    document.getElementById('kediConfig').className = 'cond ' + (val === 'kedi' ? 'visible' : 'hidden');
  }
}

function handleToggleChange(id) {
  if (id === 'pruningToggle') {
    document.getElementById('pruningFraction').disabled = !isToggleOn('pruningToggle');
  }
}

// ══════════════════════════════════════════════════════════
// DEPENDENCY LOGIC
// ══════════════════════════════════════════════════════════

// ── Spiking mode → deps ────────────────────────────────────
function applySpikingDeps() {
  const mode = getSegVal('spikingMode');
  const depsEl = document.getElementById('spikingDeps');
  let deps = [];

  if (mode === 'rate') {
    // Rate-coded defaults: Uniform / Default / <=
    document.getElementById('firingMode').value = 'Default';
    document.getElementById('spikeGenMode').value = 'Uniform';
    document.getElementById('thresholdMode').value = '<=';
    // Quantization: NOT forced — user controls
    setToggle('actQuantToggle', isToggleOn('actQuantToggle'), false);
    // Weight quant: controlled by HW (applyHwDeps handles forcing)
    applyHwDeps();
    deps.push({ text: 'Firing: Default', active: true });
    deps.push({ text: 'Spike Gen: Uniform', active: true });
    deps.push({ text: 'Threshold: ≤', active: true });
    deps.push({ text: 'CoreFlow Tuning: included', active: true });
  } else if (mode === 'ttfs') {
    document.getElementById('firingMode').value = 'TTFS';
    document.getElementById('spikeGenMode').value = 'TTFS';
    document.getElementById('thresholdMode').value = '<=';
    setToggle('actQuantToggle', isToggleOn('actQuantToggle'), false);
    applyHwDeps();
    deps.push({ text: 'Firing: TTFS', forced: true });
    deps.push({ text: 'Spike Gen: TTFS', forced: true });
    deps.push({ text: 'Threshold: ≤', forced: true });
    deps.push({ text: 'CoreFlow Tuning: skipped', active: true });
  } else if (mode === 'ttfs_quantized') {
    document.getElementById('firingMode').value = 'TTFS';
    document.getElementById('spikeGenMode').value = 'TTFS';
    document.getElementById('thresholdMode').value = '<=';
    // TTFS Quantized forces ACTIVATION quant only (not weight quant)
    setToggle('actQuantToggle', true, true);
    applyHwDeps(); // weight quant still driven by HW bits
    deps.push({ text: 'Firing: TTFS', forced: true });
    deps.push({ text: 'Spike Gen: TTFS', forced: true });
    deps.push({ text: 'Activation Quant: forced ON', forced: true });
    deps.push({ text: 'CoreFlow Tuning: skipped', active: true });
  }

  depsEl.innerHTML = deps.map(d =>
    `<span class="dep-chip ${d.forced ? 'forced' : d.active ? 'triggered' : ''}"><span class="dot"></span>${d.text}</span>`
  ).join('');

  const isTTFS = mode.startsWith('ttfs');
  document.getElementById('firingMode').disabled = isTTFS;
  document.getElementById('spikeGenMode').disabled = isTTFS;
  document.getElementById('thresholdMode').disabled = isTTFS;
}

// ── Hardware weight_bits → force weight quantization ───────
function applyHwDeps() {
  const bits = parseInt(v('weightBits'));
  const depsEl = document.getElementById('hwDeps');
  // If weight_bits is specified (any positive integer), hardware requires quantized weights
  // → force weight_quantization ON
  if (bits > 0 && bits < 32) {
    setToggle('wtQuantToggle', true, true);
    depsEl.innerHTML = `<span class="dep-chip forced"><span class="dot"></span>Weight Quantization: forced ON (${bits}-bit hardware)</span>`;
  } else {
    setToggle('wtQuantToggle', false, false);
    depsEl.innerHTML = '';
  }
}

function onWeightBitsChange() {
  applyHwDeps();
}

// ── Cycles ↔ Target Tq validation ──────────────────────────
function onCyclesChange() {
  // Auto-set target_tq = cycles
  const cycles = parseInt(v('simCycles'));
  if (!isNaN(cycles) && cycles > 0) {
    document.getElementById('targetTq').value = cycles;
  }
  validateTq();
}

function onTqChange() { validateTq(); }

function validateTq() {
  const cycles = parseInt(v('simCycles'));
  const tq = parseInt(v('targetTq'));
  const warnEl = document.getElementById('tqWarn');

  if (isNaN(cycles) || isNaN(tq) || cycles <= 0 || tq <= 0) {
    warnEl.classList.add('hide');
    return;
  }

  // Valid if cycles / tq is a positive integer (tq divides cycles evenly)
  if (cycles % tq !== 0) {
    warnEl.textContent = `⚠ target_tq (${tq}) should evenly divide cycles (${cycles}). Valid values: ${getDivisors(cycles).join(', ')}`;
    warnEl.classList.remove('hide');
  } else {
    warnEl.classList.add('hide');
  }
}

function getDivisors(n) {
  const divs = [];
  for (let i = 1; i <= n; i++) { if (n % i === 0) divs.push(i); }
  return divs;
}

// ── Search section visibility ──────────────────────────────
function updateSearchVisibility() {
  const needsSearch = getSegVal('configMode') === 'nas' || getSegVal('hwMode') === 'auto';
  const sec = document.getElementById('searchSection');

  if (needsSearch) {
    sec.classList.remove('section-hidden');   // display:none → visible
    sec.classList.remove('section-hiding');   // in case we're mid-fade
    sec.classList.remove('search-entering');  // reset if already present
    void sec.offsetHeight;                    // force reflow between display change and animation
    sec.classList.add('search-entering');     // trigger entrance animation

    // After entrance (0.5s), remove class so section-body expand/collapse uses normal transition
    var entranceTimeout = setTimeout(function () {
      sec.classList.remove('search-entering');
    }, 500);
    var onEntranceDone = function () {
      sec.classList.remove('search-entering');
      sec.removeEventListener('animationend', onEntranceDone);
      clearTimeout(entranceTimeout);
    };
    sec.addEventListener('animationend', onEntranceDone);

    if (!sec.classList.contains('open')) sec.classList.add('open');
  } else {
    // Fade out, then remove from flow
    if (sec.classList.contains('section-hiding')) return; // already hiding
    sec.classList.add('section-hiding');
    sec.classList.remove('open');
    sec.classList.remove('search-entering');
    var onFadeDone = function (e) {
      if (e.propertyName !== 'opacity') return;
      sec.classList.remove('section-hiding');
      sec.classList.add('section-hidden');
      sec.removeEventListener('transitionend', onFadeDone);
    };
    sec.addEventListener('transitionend', onFadeDone);
    // Fallback if transitionend doesn't fire (e.g. opacity already 0)
    setTimeout(function () {
      if (sec.classList.contains('section-hiding')) {
        sec.classList.remove('section-hiding');
        sec.classList.add('section-hidden');
        sec.removeEventListener('transitionend', onFadeDone);
      }
    }, 400);
  }
}

// ── Model type chips ───────────────────────────────────────
function renderModelChips() {
  document.getElementById('modelTypeChips').innerHTML = MODEL_TYPES.map(m =>
    `<div class="chip ${m.id === state.modelType ? 'active' : ''}" onclick="selectModelType('${m.id}')">${m.label}</div>`
  ).join('');
}

function selectModelType(id) {
  state.modelType = id;
  renderModelChips();
  ensureModelConfigSchema(id).then(function () {
    renderModelConfigFields();
    if (['torch_vgg16', 'torch_squeezenet11'].includes(id)) {
      setSegVal('weightSourceMode', 'pretrained');
      handleSegmentChange('weightSourceMode', 'pretrained');
    }
    update();
  });
}

// ── Model config fields ────────────────────────────────────
function renderModelConfigFields() {
  const container = document.getElementById('modelConfigFields');
  const schema = MODEL_CONFIG_SCHEMAS[state.modelType] || [];
  if (schema.length === 0) {
    container.innerHTML = '<div class="note">No configurable parameters for this model type.</div>';
    return;
  }
  container.innerHTML = schema.map(f => {
    if (f.type === 'select') {
      return `<div class="field"><label class="field-label">${f.label}</label>
        <select id="mc_${f.key}" onchange="update()">${f.options.map(o => `<option value="${o}" ${o === f.default ? 'selected' : ''}>${o}</option>`).join('')}</select></div>`;
    } else if (f.type === 'toggle') {
      return `<div class="field"><div class="toggle-row ${f.default ? 'on' : ''}" id="mc_${f.key}" onclick="toggleClick(this)">
        <span class="toggle-label">${f.label}</span><div class="toggle-switch"></div></div></div>`;
    } else {
      return `<div class="field"><label class="field-label">${f.label}</label>
        <input type="${f.type}" id="mc_${f.key}" value="${f.default}" ${f.step ? 'step="' + f.step + '"' : ''} oninput="update()"></div>`;
    }
  }).join('');
}

// ── Core types ─────────────────────────────────────────────
function renderCoreTypes() {
  document.getElementById('coreTypesList').innerHTML = state.coreTypes.map((ct, i) => `
    <div class="field-grid cols-3" style="margin-bottom:8px;align-items:end">
      <div class="field">
        <label class="field-label">Max Axons</label>
        <input type="number" value="${ct.max_axons}" min="1" onchange="state.coreTypes[${i}].max_axons=+this.value;update()">
      </div>
      <div class="field">
        <label class="field-label">Max Neurons</label>
        <input type="number" value="${ct.max_neurons}" min="1" onchange="state.coreTypes[${i}].max_neurons=+this.value;update()">
      </div>
      <div class="field">
        <label class="field-label">Count ${state.coreTypes.length > 1 ? '<span style="cursor:pointer;color:var(--accent-rose)" onclick="removeCoreType(' + i + ')">✕ remove</span>' : ''}</label>
        <input type="number" value="${ct.count}" min="1" onchange="state.coreTypes[${i}].count=+this.value;update()">
      </div>
    </div>
  `).join('');
}

function addCoreType() {
  state.coreTypes.push({ max_axons: 257, max_neurons: 1025, count: 100 });
  renderCoreTypes(); update();
}
function removeCoreType(i) {
  if (state.coreTypes.length > 1) { state.coreTypes.splice(i, 1); renderCoreTypes(); update(); }
}

// ══════════════════════════════════════════════════════════
// JSON GENERATION
// ══════════════════════════════════════════════════════════
function buildConfig() {
  const spikingMode = getSegVal('spikingMode');
  const configMode = getSegVal('configMode');
  const hwMode = getSegVal('hwMode');
  const weightMode = getSegVal('weightSourceMode');
  const optimizerType = getSegVal('optimizer');

  const actQuant = isToggleOn('actQuantToggle');
  const wtQuant = isToggleOn('wtQuantToggle');
  const pruning = isToggleOn('pruningToggle');
  const pipelineMode = (actQuant || wtQuant) ? "phased" : "vanilla";

  const lr = weightMode === 'train' ? parseFloat(v('lr')) : parseFloat(v('lrPretrained'));
  const targetTq = parseInt(v('targetTq'));
  const simCycles = parseInt(v('simCycles'));
  const weightBits = parseInt(v('weightBits'));

  // Model config from dynamic fields
  const schema = MODEL_CONFIG_SCHEMAS[state.modelType] || [];
  const modelConfig = {};
  schema.forEach(f => {
    const el = document.getElementById('mc_' + f.key); if (!el) return;
    if (f.type === 'number') modelConfig[f.key] = parseFloat(el.value);
    else if (f.type === 'toggle') modelConfig[f.key] = el.classList.contains('on');
    else if (f.type === 'text' && f.key === 'hidden_dims')
      modelConfig[f.key] = el.value.split(',').map(s => parseInt(s.trim())).filter(n => !isNaN(n));
    else modelConfig[f.key] = el.value;
  });

  // deployment_parameters
  const dp = {
    lr,
    training_epochs: weightMode === 'train' ? parseInt(v('trainingEpochs')) : undefined,
    tuner_epochs: parseInt(v('tunerEpochs')),
    degradation_tolerance: parseFloat(v('degradationTolerance')),
    configuration_mode: configMode,
    model_type: state.modelType,
    model_config: modelConfig,
    spiking_mode: spikingMode,
    firing_mode: v('firingMode'),
    spike_generation_mode: v('spikeGenMode'),
    thresholding_mode: v('thresholdMode'),
  };

  // Quantization flags
  dp.activation_quantization = actQuant;
  dp.weight_quantization = wtQuant;

  // Pruning
  if (pruning) {
    dp.pruning = true;
    dp.pruning_fraction = parseFloat(v('pruningFraction'));
  }

  // Weight preloading
  if (weightMode === 'pretrained') {
    dp.weight_source = v('weightSource');
    dp.finetune_epochs = parseInt(v('finetuneEpochs'));
    const ftLr = v('finetuneLr');
    if (ftLr) dp.finetune_lr = parseFloat(ftLr);
    delete dp.training_epochs;
  }

  // NAS / search config
  const needsSearch = configMode === 'nas' || hwMode === 'auto';
  if (needsSearch) {
    dp.arch_search = {
      optimizer: optimizerType,
      pop_size: parseInt(v('popSize')),
      generations: parseInt(v('generations')),
      seed: parseInt(v('searchSeed')),
      warmup_fraction: parseFloat(v('warmupFraction')),
      accuracy_evaluator: v('accuracyEvaluator'),
      extrapolation_num_train_epochs: parseInt(v('extrapTrainEpochs')),
      extrapolation_num_checkpoints: parseInt(v('extrapCheckpoints')),
      extrapolation_target_epochs: parseInt(v('extrapTargetEpochs')),
    };
    const sbs = v('searchBatchSize');
    if (sbs) dp.arch_search.training_batch_size = parseInt(sbs);
    if (optimizerType === 'kedi') {
      dp.arch_search.kedi_model = v('kediModel');
      dp.arch_search.kedi_adapter = v('kediAdapter');
      dp.arch_search.candidates_per_batch = parseInt(v('kediCandidates'));
      dp.arch_search.max_regen_rounds = parseInt(v('kediRegenRounds'));
    }
  }

  // platform_constraints
  let platformConstraints;
  const allowTiling = isToggleOn('axonTilingToggle');

  if (hwMode === 'user') {
    platformConstraints = {
      cores: state.coreTypes.map(ct => ({ ...ct })),
      max_axons: Math.max(...state.coreTypes.map(c => c.max_axons)),
      max_neurons: Math.max(...state.coreTypes.map(c => c.max_neurons)),
      target_tq: targetTq,
      simulation_steps: simCycles,
      weight_bits: weightBits,
    };
    if (allowTiling) platformConstraints.allow_axon_tiling = true;
  } else {
    platformConstraints = {
      mode: "auto",
      auto: {
        fixed: {
          target_tq: targetTq,
          simulation_steps: simCycles,
          weight_bits: weightBits,
          allow_axon_tiling: allowTiling,
        },
        search_space: {
          num_core_types: parseInt(v('numCoreTypes')),
          core_type_counts: v('coreTypeCounts').split(',').map(s => parseInt(s.trim())).filter(n => !isNaN(n)),
          core_axons_bounds: v('coreAxonBounds').split(',').map(s => parseInt(s.trim())).filter(n => !isNaN(n)),
          core_neurons_bounds: v('coreNeuronBounds').split(',').map(s => parseInt(s.trim())).filter(n => !isNaN(n)),
          max_threshold_groups: parseInt(v('maxThresholdGroups')),
        },
      },
    };
  }

  const config = {
    seed: parseInt(v('seed')),
    pipeline_mode: pipelineMode,
    experiment_name: v('experimentName'),
    generated_files_path: "./generated",
    data_provider_name: v('dataProvider'),
    platform_constraints: platformConstraints,
    deployment_parameters: dp,
    target_metric_override: null,
    start_step: null,
    stop_step: null,
  };

  // Clean undefined
  Object.keys(config.deployment_parameters).forEach(k => {
    if (config.deployment_parameters[k] === undefined) delete config.deployment_parameters[k];
  });

  return config;
}

function v(id) { const el = document.getElementById(id); return el ? el.value : ''; }

// ── Syntax-highlighted JSON ────────────────────────────────
function renderJson(obj) {
  return JSON.stringify(obj, null, 2)
    .replace(/("(?:\\.|[^"\\])*")\s*:/g, (m, key) => `<span class="key">${key}</span>:`)
    .replace(/:\s*("(?:\\.|[^"\\])*")/g, (m, val) => `: <span class="str">${val}</span>`)
    .replace(/:\s*(-?\d+\.?\d*(?:e[+-]?\d+)?)/gi, (m, val) => `: <span class="num">${val}</span>`)
    .replace(/:\s*(true|false)/g, (m, val) => `: <span class="bool">${val}</span>`)
    .replace(/:\s*(null)/g, (m, val) => `: <span class="null">${val}</span>`);
}

function update() {
  document.getElementById('jsonOutput').innerHTML = renderJson(buildConfig());
}

// ── Actions ────────────────────────────────────────────────
function copyJson() {
  navigator.clipboard.writeText(JSON.stringify(buildConfig(), null, 2)).then(() => {
    const btn = document.querySelector('.btn-sm.primary');
    const orig = btn.textContent; btn.textContent = 'Copied!';
    setTimeout(() => btn.textContent = orig, 1500);
  });
}
function downloadJson() {
  const cfg = buildConfig();
  const a = document.createElement('a');
  a.href = URL.createObjectURL(new Blob([JSON.stringify(cfg, null, 2)], { type: 'application/json' }));
  a.download = (cfg.experiment_name || 'config') + '.json'; a.click();
}
function resetAll() { if (confirm('Reset all configuration to defaults?')) location.reload(); }

// ── Show JSON toggle & RUN ──────────────────────────────────
function toggleShowJson(el) {
  el.classList.toggle('on');
  const show = el.classList.contains('on');
  document.getElementById('appShell').classList.toggle('show-json', show);
  const copyBtn = document.getElementById('copyJsonBtn');
  if (copyBtn) copyBtn.style.display = show ? '' : 'none';
}

function runPipeline() {
  const runBtn = document.getElementById('runBtn');
  if (runBtn) runBtn.disabled = true;
  const config = buildConfig();
  fetch('/api/run', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(config),
  }).then(function (res) {
    if (res.status === 202 || res.ok) {
      window.location.href = '/';
      return;
    }
    return res.json().then(function (body) {
      alert(body.error || 'Run failed');
    });
  }).catch(function (err) {
    alert(err.message || 'Run failed');
  }).finally(function () {
    if (runBtn) runBtn.disabled = false;
  });
}

// ── Boot ───────────────────────────────────────────────────
init();
