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
  coreTypes: [{ max_axons: 1025, max_neurons: 257, count: 100, has_bias: true }],
};

let pipelineStepsDebounceTimer = null;
let lastPipelineSteps = [];
let lastPipelineGroups = [];
let pipelineStepsLoading = false;

// Hardware auto-fill state
let _hwAutoMode = false;   // true after user clicks Auto; triggers re-fill on param changes
let _hwRefilling = false;  // true while a fetch is in-flight (prevents re-entrant refills)
let _hwAutoRefillTimer = null;

// Stats panel state — true once full verified stats have been rendered at least once.
// Enables the overlay-on-reload approach instead of full content replacement.
let _hwStatsHasContent = false;

// Edit & continue: names of steps completed in the previous run.
// Persists for the session so _renderWizardPipelineSteps can colour steps.
let _ecPrevCompleted = null;   // Set<string> | null
// True after the one-shot auto-suggestion has fired (prevents re-suggesting on re-renders).
let _ecSuggestionDone = false;
let _pipelineStepsEcBound = false;

// ── Pipeline step bar (edit & continue): delegated click — inline onclick with
// JSON.stringify(stepName) breaks HTML attributes (inner quotes close the attribute).
function onPipelineStepsListClick(ev) {
  if (!window.__isEditContinueMode) return;
  var col = ev.target.closest('.psb-col[data-ec-start-step]');
  if (!col) return;
  var enc = col.getAttribute('data-ec-start-step');
  if (enc == null || enc === '') return;
  try {
    selectStartStep(decodeURIComponent(enc));
  } catch (e) { /* ignore malformed */ }
}

// ── Wizard URL params: deployment config (same shape as edit & continue) ──
/** Load deployment JSON for ``?run_id=`` or ``?template_id=`` (flat config like GET /api/runs/.../config). */
function fetchDeploymentConfigForWizardUrlParams(runId, templateId) {
  if (runId) {
    return fetch('/api/runs/' + encodeURIComponent(runId) + '/config').then(function (r) { return r.ok ? r.json() : null; });
  }
  if (templateId) {
    return fetch('/api/templates/' + encodeURIComponent(templateId)).then(function (r) { return r.ok ? r.json() : null; });
  }
  return Promise.resolve(null);
}

// ── Init ───────────────────────────────────────────────────
function init() {
  _renderHwStats(null); // show placeholder immediately, before API loads
  var _psList = document.getElementById('pipelineStepsList');
  if (_psList && !_pipelineStepsEcBound) {
    _pipelineStepsEcBound = true;
    _psList.addEventListener('click', onPipelineStepsListClick);
  }
  loadFromAPI().then(function () {
    var params = new URLSearchParams(location.search);
    var runId = params.get('run_id');
    var templateId = params.get('template_id');

    renderModelChips();
    renderModelConfigFields();
    renderCoreTypes();
    setToggle('hardwareBiasToggle', true);
    // Reuse ON by default — matches the "Apply recipe to tuning" checkbox
    // behaviour the old checkbox started in. ``loadStateFromConfig`` below
    // will override this when a template supplies a distinct tuning_recipe.
    setToggle('recipeApplyToTuningToggle', true);
    applyRecipeReuseDeps();
    applySpikingDeps();
    applyHwDeps();
    onPruningFractionChange();
    // Avoid running hide animation on #searchSection before hydrate — it can race with
    // transitionend and leave the panel hidden when loading ?template_id= / ?run_id=.
    if (!runId && !templateId) updateSearchVisibility();

    window.__isEditContinueMode = !!runId;
    window.__editContinueSourceRunId = runId || null;
    if (window.__isEditContinueMode) document.body.classList.add('edit-continue-mode');

    function done() {
      update();
      schedulePipelineStepsUpdate();
      // Same condition as updateSearchVisibility: do not run fixed-HW verify/auto-fill
      // during architecture search — update() already shows search placeholders; these
      // paths would overwrite loaded config and show "Auto-configured" / mapping stats.
      var needsSearch = getSegVal('configMode') === 'search' || getSegVal('hwMode') === 'search';
      if (!needsSearch) {
        if (isHwAutoSuggestOn()) autoFillHardware(); else scheduleHwValidation();
      }
      var showJson = document.getElementById('showJsonToggle');
      var copyBtn = document.getElementById('copyJsonBtn');
      if (copyBtn && showJson) copyBtn.style.display = showJson.classList.contains('on') ? '' : 'none';
    }

    // Same load path for run config and template: flat deployment dict → loadStateFromConfig.
    // For edit & continue only, also fetch pipeline to suggest the restart step.
    var pipelinePrefetch = runId
      ? fetch('/api/runs/' + encodeURIComponent(runId) + '/pipeline')
        .then(function (r) { return r.ok ? r.json() : null; })
        .catch(function () { return null; })
      : Promise.resolve(null);

    // Load NAS objective schema before hydrating so loadStateFromConfig can restore chips
    // and a late fetch cannot rebuild chips with reset selection.
    return loadObjectiveOptions().then(function () {
      return fetchDeploymentConfigForWizardUrlParams(runId, templateId)
        .then(function (c) {
          return (c ? loadStateFromConfig(c) : Promise.resolve()).then(function () { return pipelinePrefetch; });
        })
        .then(function (pipelineState) {
          if (runId && pipelineState && pipelineState.steps) {
            _ecPrevCompleted = new Set(
              pipelineState.steps
                .filter(function (s) { return s.status === 'completed'; })
                .map(function (s) { return s.name; })
            );
          }
          done();
        })
        .catch(function () { done(); });
    });
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
  if (val) {
    el.classList.add('on');
  } else {
    el.classList.remove('on');
  }
  el.classList.toggle('forced', forced);
}

// ── Segment change handlers ────────────────────────────────
function handleSegmentChange(controlId, val) {
  if (controlId === 'configMode') {
    document.getElementById('userModelConfig').className = 'cond ' + (val === 'user' ? 'visible' : 'hidden');
    updateSearchVisibility();
    updateObjectiveCheckboxes();
  }
  if (controlId === 'weightSourceMode') {
    document.getElementById('trainFromScratch').className = 'cond ' + (val === 'train' ? 'visible' : 'hidden');
    document.getElementById('pretrainedConfig').className = 'cond ' + (val !== 'train' ? 'visible' : 'hidden');
  }
  if (controlId === 'spikingMode') { applySpikingDeps(); }
  if (controlId === 'hwMode') {
    document.getElementById('hwFixed').className = 'cond ' + (val === 'fixed' ? 'visible' : 'hidden');
    document.getElementById('hwAuto').className = 'cond ' + (val === 'search' ? 'visible' : 'hidden');
    var ast = document.getElementById('hwAutoSuggestToggle');
    if (ast) ast.style.display = (val === 'search' ? 'none' : '');
    updateSearchVisibility();
    updateObjectiveCheckboxes();
  }
  if (controlId === 'optimizer') {
    document.getElementById('agentEvolveConfig').className = 'cond ' + (val === 'agent_evolve' ? 'visible' : 'hidden');
  }
}

function onPruningFractionChange() {
  const el = document.getElementById('pruningFraction');
  const valueEl = document.getElementById('pruningFractionValue');
  const warnEl = document.getElementById('pruningWarn');
  if (!el || !valueEl || !warnEl) return;
  const val = parseFloat(el.value);
  valueEl.textContent = val.toFixed(2);
  if (val >= 0.8) {
    warnEl.textContent = 'This much pruning may not be feasible.';
    warnEl.classList.remove('hide');
    el.classList.add('pruning-slider-warn');
  } else {
    warnEl.classList.add('hide');
    el.classList.remove('pruning-slider-warn');
  }
}

function handleToggleChange(id) {
  if (id === 'pruningToggle') {
    const slider = document.getElementById('pruningFraction');
    if (slider) slider.disabled = !isToggleOn('pruningToggle');
    onPruningFractionChange();
  }
  if (id === 'actQuantToggle') {
    const targetTqEl = document.getElementById('targetTq');
    if (targetTqEl) targetTqEl.disabled = !isToggleOn('actQuantToggle');
  }
  if (id === 'floatWeightsToggle') {
    applyHwDeps();
  }
  if (id === 'coreCoalescingToggle') {
    applyCoalescingDeps();
  }
  if (id === 'neuronSplittingToggle') {
    applyHwDeps();
  }
  if (id === 'scheduledMappingToggle') {
    applyHwDeps();
  }
  if (id === 'recipeApplyToTuningToggle') {
    applyRecipeReuseDeps();
  }
}

// When the "reuse training recipe" toggle is OFF, expose the dedicated
// tuning-recipe drawer so the user can edit it inline. When ON, the
// drawer is collapsed and ``generate()`` just mirrors training_recipe
// into ``dp.tuning_recipe``.
function applyRecipeReuseDeps() {
  const reuse = isToggleOn('recipeApplyToTuningToggle');
  const drawer = document.getElementById('tuningRecipeDrawer');
  if (drawer) drawer.classList.toggle('open', !reuse);
}

// Read the dedicated tuning-recipe fields. Emits only the keys the user
// actually supplied; absent keys stay absent so backend defaults apply
// (matches how the training-recipe builder above behaves).
function buildTuningRecipeFromFields() {
  const v = (id) => document.getElementById(id)?.value;
  const recipe = {};
  const optimizer = v('tuningRecipeOptimizer');
  const scheduler = v('tuningRecipeScheduler');
  if (optimizer) recipe.optimizer = optimizer;
  if (scheduler) recipe.scheduler = scheduler;
  const asFloat = (raw) => (raw !== '' && raw != null ? parseFloat(raw) : null);
  const wd = asFloat(v('tuningRecipeWeightDecay'));
  if (wd != null) recipe.weight_decay = wd;
  const wr = asFloat(v('tuningRecipeWarmupRatio'));
  if (wr != null) recipe.warmup_ratio = wr;
  const gc = asFloat(v('tuningRecipeGradClipNorm'));
  if (gc != null) recipe.grad_clip_norm = gc;
  const llrd = asFloat(v('tuningRecipeLayerwiseLrDecay'));
  if (llrd != null) recipe.layer_wise_lr_decay = llrd;
  const ls = asFloat(v('tuningRecipeLabelSmoothing'));
  if (ls != null) recipe.label_smoothing = ls;
  const b1 = asFloat(v('tuningRecipeBeta1'));
  const b2 = asFloat(v('tuningRecipeBeta2'));
  if (b1 != null || b2 != null) {
    recipe.betas = [b1 != null ? b1 : 0.9, b2 != null ? b2 : 0.999];
  }
  return recipe;
}

// Populate the drawer inputs from a recipe dict. Leaves inputs blank
// when the source recipe omits a key so the placeholders (showing
// backend defaults) remain visible.
//
// Uses a local ``set`` helper because the sibling ``setVal`` in
// ``loadStateFromConfig`` is a closure — calling it from module scope
// raises ``ReferenceError`` and silently aborts the rest of
// ``loadStateFromConfig`` (regressing every toggle set after this call).
function loadTuningRecipeFields(recipe) {
  const r = recipe || {};
  const set = (id, val) => {
    const el = document.getElementById(id);
    if (!el) return;
    if (el.tagName === 'SELECT') {
      el.value = val != null ? String(val) : '';
    } else if (el.getAttribute('type') === 'number') {
      el.value = val != null && val !== '' ? Number(val) : '';
    } else {
      el.value = val != null ? String(val) : '';
    }
  };
  set('tuningRecipeOptimizer', r.optimizer || '');
  set('tuningRecipeScheduler', r.scheduler || 'cosine');
  set('tuningRecipeWeightDecay', r.weight_decay != null ? r.weight_decay : '');
  set('tuningRecipeWarmupRatio', r.warmup_ratio != null ? r.warmup_ratio : '');
  set('tuningRecipeGradClipNorm', r.grad_clip_norm != null ? r.grad_clip_norm : '');
  set('tuningRecipeLayerwiseLrDecay', r.layer_wise_lr_decay != null ? r.layer_wise_lr_decay : '');
  set('tuningRecipeLabelSmoothing', r.label_smoothing != null ? r.label_smoothing : '');
  if (Array.isArray(r.betas) && r.betas.length === 2) {
    set('tuningRecipeBeta1', r.betas[0]);
    set('tuningRecipeBeta2', r.betas[1]);
  } else {
    set('tuningRecipeBeta1', '');
    set('tuningRecipeBeta2', '');
  }
}

// Semantic equality between two recipe dicts. Order-insensitive,
// numerically tolerant (keeps JSON float round-trips from being
// read as "different"), and treats missing keys on one side as
// equal when the other side stores ``undefined``.
function recipesAreEquivalent(a, b) {
  const toMap = (r) => (r && typeof r === 'object' ? r : {});
  const A = toMap(a);
  const B = toMap(b);
  const keys = new Set([...Object.keys(A), ...Object.keys(B)]);
  for (const k of keys) {
    const av = A[k];
    const bv = B[k];
    if (av == null && bv == null) continue;
    if (Array.isArray(av) || Array.isArray(bv)) {
      if (!Array.isArray(av) || !Array.isArray(bv) || av.length !== bv.length) return false;
      for (let i = 0; i < av.length; i++) {
        if (typeof av[i] === 'number' && typeof bv[i] === 'number') {
          if (Math.abs(av[i] - bv[i]) > 1e-9) return false;
        } else if (av[i] !== bv[i]) {
          return false;
        }
      }
      continue;
    }
    if (typeof av === 'number' && typeof bv === 'number') {
      if (Math.abs(av - bv) > 1e-9) return false;
      continue;
    }
    if (av !== bv) return false;
  }
  return true;
}

// ══════════════════════════════════════════════════════════
// DEPENDENCY LOGIC
// ══════════════════════════════════════════════════════════

// ── Spiking mode → deps ────────────────────────────────────
function applySpikingDeps() {
  const mode = getSegVal('spikingMode');
  const depsEl = document.getElementById('spikingDeps');
  let deps = [];

  // Weight quantization: locked to OFF when Float is on, ON when Float is off
  const floatWeights = isToggleOn('floatWeightsToggle');
  function setWtQuantFromHw() {
    if (floatWeights) setToggle('wtQuantToggle', false, true);
    else setToggle('wtQuantToggle', true, true);
  }

  if (mode === 'rate') {
    // Rate-coded defaults: Uniform / Default / <=
    document.getElementById('firingMode').value = 'Default';
    document.getElementById('spikeGenMode').value = 'Uniform';
    document.getElementById('thresholdMode').value = '<=';
    // Rate-coded deployment requires activation quantization
    setToggle('actQuantToggle', true, true);
    setWtQuantFromHw();
    applyHwDeps();
    deps.push({ text: 'Firing: Default', active: true });
    deps.push({ text: 'Spike Gen: Uniform', active: true });
    deps.push({ text: 'Threshold: ≤', active: true });
    deps.push({ text: 'CoreFlow Tuning: included', active: true });
    deps.push({ text: 'Activation Quant: forced ON', forced: true });
  } else if (mode === 'ttfs') {
    document.getElementById('firingMode').value = 'TTFS';
    document.getElementById('spikeGenMode').value = 'TTFS';
    document.getElementById('thresholdMode').value = '<=';
    setToggle('actQuantToggle', false, true);
    setWtQuantFromHw();
    applyHwDeps();
    deps.push({ text: 'Firing: TTFS', forced: true });
    deps.push({ text: 'Spike Gen: TTFS', forced: true });
    deps.push({ text: 'Threshold: ≤', forced: true });
    deps.push({ text: 'CoreFlow Tuning: skipped', active: true });
    deps.push({ text: 'Cycles: not used (analytical TTFS)', forced: true });
  } else if (mode === 'ttfs_quantized') {
    document.getElementById('firingMode').value = 'TTFS';
    document.getElementById('spikeGenMode').value = 'TTFS';
    document.getElementById('thresholdMode').value = '<=';
    // TTFS Quantized forces ACTIVATION quant only (not weight quant)
    setToggle('actQuantToggle', true, true);
    setWtQuantFromHw();
    applyHwDeps();
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

  const simCyclesEl = document.getElementById('simCycles');
  if (simCyclesEl) simCyclesEl.disabled = (mode === 'ttfs');
  const targetTqEl = document.getElementById('targetTq');
  if (targetTqEl) targetTqEl.disabled = !isToggleOn('actQuantToggle');
  validateTq();
}

// ── Hardware weight_bits → force weight quantization ───────
function applyHwDeps() {
  const floatWeights = isToggleOn('floatWeightsToggle');
  const weightBitsEl = document.getElementById('weightBits');
  const depsEl = document.getElementById('hwDeps');

  if (floatWeights) {
    if (weightBitsEl) weightBitsEl.disabled = true;
    depsEl.innerHTML = '<span class="dep-chip forced"><span class="dot"></span>Float weights: weight quantization off</span>';
  } else {
    if (weightBitsEl) weightBitsEl.disabled = false;
    depsEl.innerHTML = '';
  }
  syncWtQuantToggle();
}

function applyCoalescingDeps() {
  applyHwDeps();
}

function syncWtQuantToggle() {
  const floatWeights = isToggleOn('floatWeightsToggle');
  if (floatWeights) setToggle('wtQuantToggle', false, true);
  else setToggle('wtQuantToggle', true, true);
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
  const simCyclesEl = document.getElementById('simCycles');

  if (simCyclesEl && simCyclesEl.disabled) {
    warnEl.classList.add('hide');
    return;
  }

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
  const needsSearch = getSegVal('configMode') === 'search' || getSegVal('hwMode') === 'search';
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
  const locked = isHwAutoSuggestOn();
  document.getElementById('coreTypesList').innerHTML = state.coreTypes.map((ct, i) => `
    <div class="field-grid cols-3" style="margin-bottom:8px;align-items:end">
      <div class="field">
        <label class="field-label">Max Axons</label>
        <input id="ct_${i}_max_axons" type="number" value="${ct.max_axons}" min="1"
          ${locked ? 'disabled' : `onchange="state.coreTypes[${i}].max_axons=+this.value;_hwAutoMode=false;update()"`}>
      </div>
      <div class="field">
        <label class="field-label">Max Neurons</label>
        <input id="ct_${i}_max_neurons" type="number" value="${ct.max_neurons}" min="1"
          ${locked ? 'disabled' : `onchange="state.coreTypes[${i}].max_neurons=+this.value;_hwAutoMode=false;update()"`}>
      </div>
      <div class="field">
        <label class="field-label">Count ${!locked && state.coreTypes.length > 1 ? `<span style="cursor:pointer;color:var(--accent-rose)" onclick="removeCoreType(${i})">✕ remove</span>` : ''}</label>
        <input id="ct_${i}_count" type="number" value="${ct.count}" min="1"
          ${locked ? 'disabled' : `onchange="state.coreTypes[${i}].count=+this.value;_hwAutoMode=false;update()"`}>
      </div>
    </div>
  `).join('');

  const addBtn = document.getElementById('addCoreTypeBtn');
  const autoLabel = document.getElementById('coreTypesAutoLabel');
  if (addBtn) addBtn.style.display = locked ? 'none' : '';
  if (autoLabel) autoLabel.style.display = locked ? '' : 'none';
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

  const floatWeights = isToggleOn('floatWeightsToggle');
  const wtQuant = !floatWeights;
  const actQuant = (spikingMode === 'rate' || spikingMode === 'ttfs_quantized');
  const pruning = isToggleOn('pruningToggle');
  let pipelineMode;
  if (floatWeights) {
    pipelineMode = "vanilla";
  } else {
    pipelineMode = (actQuant || wtQuant) ? "phased" : "vanilla";
  }

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
    tuning_budget_scale: parseFloat(v('tuningBudgetScale')),
    degradation_tolerance: parseFloat(v('degradationTolerance')),
    model_config_mode: configMode,
    hw_config_mode: hwMode,
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

  // Batch size (applies to both train-from-scratch and fine-tune paths)
  const batchSizeVal = v('batchSize');
  if (batchSizeVal !== '' && batchSizeVal != null) {
    const bs = parseInt(batchSizeVal);
    if (!isNaN(bs) && bs > 0) dp.batch_size = bs;
  }

  // Input preprocessing (resize + normalize)
  const resizeToVal = v('preprocessResizeTo');
  const normalizeVal = v('preprocessNormalize');
  const interpolationVal = v('preprocessInterpolation') || 'bicubic';
  const hasResize = resizeToVal !== '' && resizeToVal != null && parseInt(resizeToVal) > 0;
  const hasNormalize = normalizeVal !== '' && normalizeVal != null;
  if (hasResize || hasNormalize) {
    const preproc = { interpolation: interpolationVal };
    if (hasResize) preproc.resize_to = parseInt(resizeToVal);
    if (hasNormalize) preproc.normalize = normalizeVal;
    dp.preprocessing = preproc;
  }

  // Training recipe (fully optional: any non-empty field triggers emission)
  const recipeFields = {
    optimizer: v('recipeOptimizer'),
    scheduler: v('recipeScheduler'),
    weight_decay: v('recipeWeightDecay'),
    warmup_ratio: v('recipeWarmupRatio'),
    grad_clip_norm: v('recipeGradClipNorm'),
    layer_wise_lr_decay: v('recipeLayerwiseLrDecay'),
    label_smoothing: v('recipeLabelSmoothing'),
    beta1: v('recipeBeta1'),
    beta2: v('recipeBeta2'),
  };
  const hasAnyRecipeField = Object.entries(recipeFields)
    .filter(([k]) => k !== 'scheduler') // scheduler has a default so ignore for "set" detection
    .some(([, val]) => val !== '' && val != null);
  if (hasAnyRecipeField) {
    const recipe = {};
    if (recipeFields.optimizer) recipe.optimizer = recipeFields.optimizer;
    if (recipeFields.scheduler) recipe.scheduler = recipeFields.scheduler;
    if (recipeFields.weight_decay !== '' && recipeFields.weight_decay != null)
      recipe.weight_decay = parseFloat(recipeFields.weight_decay);
    if (recipeFields.warmup_ratio !== '' && recipeFields.warmup_ratio != null)
      recipe.warmup_ratio = parseFloat(recipeFields.warmup_ratio);
    if (recipeFields.grad_clip_norm !== '' && recipeFields.grad_clip_norm != null)
      recipe.grad_clip_norm = parseFloat(recipeFields.grad_clip_norm);
    if (recipeFields.layer_wise_lr_decay !== '' && recipeFields.layer_wise_lr_decay != null)
      recipe.layer_wise_lr_decay = parseFloat(recipeFields.layer_wise_lr_decay);
    if (recipeFields.label_smoothing !== '' && recipeFields.label_smoothing != null)
      recipe.label_smoothing = parseFloat(recipeFields.label_smoothing);
    const b1 = recipeFields.beta1, b2 = recipeFields.beta2;
    if ((b1 !== '' && b1 != null) || (b2 !== '' && b2 != null)) {
      recipe.betas = [
        b1 !== '' && b1 != null ? parseFloat(b1) : 0.9,
        b2 !== '' && b2 != null ? parseFloat(b2) : 0.999,
      ];
    }
    dp.training_recipe = recipe;
    // Adaptation tuner recipe: either reused from training_recipe (the
    // common case — deep-clone so backend consumers can mutate without
    // leaking back) or read from the dedicated drawer when the user has
    // asked for a separate recipe. ``buildTuningRecipeFromFields`` emits
    // only the fields the user actually supplied, preserving the legacy
    // behaviour where an absent key falls back to the recipe default.
    if (isToggleOn('recipeApplyToTuningToggle')) {
      dp.tuning_recipe = { ...recipe };
    } else {
      const tuningRecipe = buildTuningRecipeFromFields();
      if (Object.keys(tuningRecipe).length > 0) {
        dp.tuning_recipe = tuningRecipe;
      }
    }
  }

  // NAS / search config
  const needsSearch = configMode === 'search' || hwMode === 'search';
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
    // Collect selected objectives
    var selectedObjectives = getSelectedObjectives();
    if (selectedObjectives.length > 0) dp.arch_search.objectives = selectedObjectives;
    const sbs = v('searchBatchSize');
    const sbsNum = sbs !== '' ? parseInt(sbs) : null;
    dp.arch_search.training_batch_size = (sbsNum !== null && sbsNum > 0) ? sbsNum : null;
    if (optimizerType === 'agent_evolve') {
      dp.arch_search.agent_model = v('agentModel');
      dp.arch_search.candidates_per_batch = parseInt(v('agentCandidates'));
      dp.arch_search.max_regen_rounds = parseInt(v('agentRegenRounds'));
    }
  }

  const maxSim = parseInt(v('maxSimulationSamples'), 10);
  if (maxSim > 0) dp.max_simulation_samples = maxSim;

  dp.allow_scheduling = isToggleOn('scheduledMappingToggle');

  // platform_constraints
  let platformConstraints;
  const allowCoalescing = isToggleOn('coreCoalescingToggle');

  if (hwMode === 'fixed') {
    const hardwareBias = isToggleOn('hardwareBiasToggle');
    platformConstraints = {
      cores: state.coreTypes.map(ct => ({ ...ct, has_bias: hardwareBias })),
      max_axons: Math.max(...state.coreTypes.map(c => c.max_axons)),
      max_neurons: Math.max(...state.coreTypes.map(c => c.max_neurons)),
      target_tq: targetTq,
      simulation_steps: simCycles,
      weight_bits: weightBits,
      has_bias: hardwareBias,
    };
    platformConstraints.allow_coalescing = allowCoalescing;
    if (isToggleOn('neuronSplittingToggle')) platformConstraints.allow_neuron_splitting = true;
  } else {
    // HW search mode: fixed values + search_space for architecture search
    platformConstraints = {
      target_tq: targetTq,
      simulation_steps: simCycles,
      weight_bits: weightBits,
      has_bias: isToggleOn('hardwareBiasToggle'),
      allow_coalescing: allowCoalescing,
      allow_neuron_splitting: isToggleOn('neuronSplittingToggle'),
      search_space: {
        num_core_types: parseInt(v('numCoreTypes')),
        core_type_counts: v('coreTypeCounts').split(',').map(s => parseInt(s.trim())).filter(n => !isNaN(n)),
        core_axons_bounds: v('coreAxonBounds').split(',').map(s => parseInt(s.trim())).filter(n => !isNaN(n)),
        core_neurons_bounds: v('coreNeuronBounds').split(',').map(s => parseInt(s.trim())).filter(n => !isNaN(n)),
        max_threshold_groups: parseInt(v('maxThresholdGroups')),
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
    start_step: (typeof window.__wizardStartStep !== 'undefined' && window.__wizardStartStep != null) ? window.__wizardStartStep : null,
    stop_step: (typeof window.__wizardStopStep !== 'undefined' && window.__wizardStopStep != null) ? window.__wizardStopStep : null,
  };

  if (window.__isEditContinueMode && window.__editContinueSourceRunId) {
    config._continue_from_run_id = window.__editContinueSourceRunId;
  }

  // Clean undefined
  Object.keys(config.deployment_parameters).forEach(k => {
    if (config.deployment_parameters[k] === undefined) delete config.deployment_parameters[k];
  });

  return config;
}

// ── Load config from run or template (inverse of buildConfig) ─────
function loadStateFromConfig(config) {
  if (!config || typeof config !== 'object') return Promise.resolve();
  const dp = config.deployment_parameters || {};
  const pc = config.platform_constraints || {};

  function setVal(id, val) {
    const el = document.getElementById(id);
    if (!el) return;
    if (el.tagName === 'SELECT') {
      el.value = val != null ? String(val) : '';
    } else if (el.getAttribute('type') === 'number') {
      el.value = val != null ? Number(val) : '';
    } else {
      el.value = val != null ? String(val) : '';
    }
  }
  function setToggleFromConfig(id, val) {
    setToggle(id, !!val, false);
  }

  setVal('experimentName', config.experiment_name);
  setVal('dataProvider', config.data_provider_name);
  setVal('seed', config.seed);

  const configMode = dp.model_config_mode || 'user';
  setSegVal('configMode', configMode);
  handleSegmentChange('configMode', configMode);

  const modelType = dp.model_type || state.modelType;
  var modelConfigPromise = Promise.resolve();
  if (MODEL_TYPES.some(function (m) { return m.id === modelType; })) {
    state.modelType = modelType;
    renderModelChips();
    modelConfigPromise = ensureModelConfigSchema(state.modelType).then(function () {
      renderModelConfigFields();
      const modelConfig = dp.model_config || {};
      const schema = MODEL_CONFIG_SCHEMAS[state.modelType] || [];
      schema.forEach(function (f) {
        const v = modelConfig[f.key];
        if (v === undefined) return;
        const el = document.getElementById('mc_' + f.key);
        if (!el) return;
        if (f.type === 'toggle') setToggleFromConfig('mc_' + f.key, v);
        else if (f.type === 'text' && f.key === 'hidden_dims') setVal('mc_' + f.key, Array.isArray(v) ? v.join(', ') : v);
        else setVal('mc_' + f.key, v);
      });
    });
  }

  const weightSourceMode = dp.weight_source ? 'pretrained' : 'train';
  setSegVal('weightSourceMode', weightSourceMode);
  handleSegmentChange('weightSourceMode', weightSourceMode);
  setVal('lr', dp.lr);
  setVal('trainingEpochs', dp.training_epochs);
  setVal('weightSource', dp.weight_source);
  setVal('lrPretrained', dp.finetune_lr != null ? dp.finetune_lr : dp.lr);
  setVal('finetuneEpochs', dp.finetune_epochs);

  // Batch size + preprocessing restoration
  setVal('batchSize', dp.batch_size != null ? dp.batch_size : '');
  const pp = dp.preprocessing || {};
  setVal('preprocessResizeTo', pp.resize_to != null ? pp.resize_to : '');
  setVal('preprocessInterpolation', pp.interpolation || 'bicubic');
  setVal('preprocessNormalize', pp.normalize != null ? pp.normalize : '');

  // Training recipe restoration
  const tr = dp.training_recipe || {};
  setVal('recipeOptimizer', tr.optimizer || '');
  setVal('recipeScheduler', tr.scheduler || 'cosine');
  setVal('recipeWeightDecay', tr.weight_decay != null ? tr.weight_decay : '');
  setVal('recipeWarmupRatio', tr.warmup_ratio != null ? tr.warmup_ratio : '');
  setVal('recipeGradClipNorm', tr.grad_clip_norm != null ? tr.grad_clip_norm : '');
  setVal('recipeLayerwiseLrDecay', tr.layer_wise_lr_decay != null ? tr.layer_wise_lr_decay : '');
  setVal('recipeLabelSmoothing', tr.label_smoothing != null ? tr.label_smoothing : '');
  if (Array.isArray(tr.betas) && tr.betas.length === 2) {
    setVal('recipeBeta1', tr.betas[0]);
    setVal('recipeBeta2', tr.betas[1]);
  } else {
    setVal('recipeBeta1', '');
    setVal('recipeBeta2', '');
  }
  // "Reuse training recipe for SNN tuners" toggle. Default ON when:
  //   - no ``tuning_recipe`` is present (nothing separate to honour), or
  //   - ``tuning_recipe`` is byte-identical to ``training_recipe`` (the
  //     config explicitly duplicates it; no editing intent).
  // Switch OFF when a distinct tuning recipe is provided so the drawer
  // opens with the loaded values ready to edit. Fields are always
  // populated from ``dp.tuning_recipe`` (when present) so toggling the
  // reuse switch never silently discards user-loaded values.
  const reuseToggleEl = document.getElementById('recipeApplyToTuningToggle');
  const tuningRecipePresent = dp.tuning_recipe != null && typeof dp.tuning_recipe === 'object';
  const tuningRecipeMatches = tuningRecipePresent && recipesAreEquivalent(dp.training_recipe, dp.tuning_recipe);
  const reuseTuning = !tuningRecipePresent || tuningRecipeMatches;
  if (reuseToggleEl) setToggle('recipeApplyToTuningToggle', reuseTuning);
  loadTuningRecipeFields(tuningRecipePresent ? dp.tuning_recipe : (dp.training_recipe || {}));
  applyRecipeReuseDeps();

  setVal('tuningBudgetScale', dp.tuning_budget_scale);
  setVal('degradationTolerance', dp.degradation_tolerance);

  const hwMode = dp.hw_config_mode || 'fixed';
  setSegVal('hwMode', hwMode);
  handleSegmentChange('hwMode', hwMode);

  if (hwMode === 'fixed') {
    const cores = pc.cores || [{ max_axons: 256, max_neurons: 256, count: 100, has_bias: true }];
    state.coreTypes = cores.map(function (c) {
      return { max_axons: c.max_axons || 256, max_neurons: c.max_neurons || 256, count: c.count || 100, has_bias: c.has_bias !== false };
    });
    // Preserve loaded core types — disable auto-suggest so init()'s done() won't overwrite them
    const autoToggleEl = document.getElementById('hwAutoSuggestToggle');
    if (autoToggleEl) autoToggleEl.classList.remove('on');
    _hwAutoMode = false;
    renderCoreTypes();
    setToggleFromConfig('hardwareBiasToggle', pc.has_bias !== false);
    setToggleFromConfig('coreCoalescingToggle', !!pc.allow_coalescing);
    setToggleFromConfig('neuronSplittingToggle', pc.allow_neuron_splitting);
    setVal('weightBits', pc.weight_bits != null ? pc.weight_bits : dp.weight_bits);
    setVal('targetTq', pc.target_tq);
    setVal('simCycles', pc.simulation_steps);
  } else {
    // HW search mode: fixed values + search_space in flat structure
    setVal('targetTq', pc.target_tq);
    setVal('simCycles', pc.simulation_steps);
    setVal('weightBits', pc.weight_bits);
    setToggleFromConfig('hardwareBiasToggle', pc.has_bias !== false);
    setToggleFromConfig('coreCoalescingToggle', !!pc.allow_coalescing);
    setToggleFromConfig('neuronSplittingToggle', pc.allow_neuron_splitting);
    const ss = pc.search_space || {};
    setVal('numCoreTypes', ss.num_core_types);
    setVal('coreTypeCounts', Array.isArray(ss.core_type_counts) ? ss.core_type_counts.join(', ') : ss.core_type_counts);
    setVal('coreAxonBounds', Array.isArray(ss.core_axons_bounds) ? ss.core_axons_bounds.join(', ') : ss.core_axons_bounds);
    setVal('coreNeuronBounds', Array.isArray(ss.core_neurons_bounds) ? ss.core_neurons_bounds.join(', ') : ss.core_neurons_bounds);
    setVal('maxThresholdGroups', ss.max_threshold_groups);
  }

  setToggleFromConfig('floatWeightsToggle', dp.weight_quantization === false);
  setSegVal('spikingMode', dp.spiking_mode || 'rate');
  setVal('firingMode', dp.firing_mode);
  setVal('spikeGenMode', dp.spike_generation_mode);
  setVal('thresholdMode', dp.thresholding_mode);
  setToggleFromConfig('actQuantToggle', dp.activation_quantization);
  setToggleFromConfig('wtQuantToggle', dp.weight_quantization);
  setToggleFromConfig('pruningToggle', dp.pruning);
  setVal('pruningFraction', dp.pruning_fraction != null ? dp.pruning_fraction : 0.5);
  setVal('maxSimulationSamples', dp.max_simulation_samples != null ? dp.max_simulation_samples : '');
  (function setScheduledMappingFromConfig() {
    let allowSched;
    if (dp.allow_scheduling !== undefined && dp.allow_scheduling !== null) {
      allowSched = !!dp.allow_scheduling;
    } else if (hwMode === 'fixed') {
      allowSched = !!pc.allow_scheduling;
    } else {
      const fixedLegacy = (pc.auto || {}).fixed || {};
      allowSched = !!fixedLegacy.allow_scheduling;
    }
    setToggleFromConfig('scheduledMappingToggle', allowSched);
  })();
  applySpikingDeps();
  applyHwDeps();
  onPruningFractionChange();

  const arch = dp.arch_search || {};
  var optimizerVal = arch.optimizer || 'nsga2';
  setSegVal('optimizer', optimizerVal);
  handleSegmentChange('optimizer', optimizerVal);
  setVal('popSize', arch.pop_size);
  setVal('generations', arch.generations);
  setVal('searchSeed', arch.seed);
  setVal('warmupFraction', arch.warmup_fraction);
  (function setAccuracyEvaluatorFromArch() {
    var raw = arch.accuracy_evaluator;
    if (raw != null && String(raw) === 'direct') raw = 'fast';
    setVal('accuracyEvaluator', raw);
  })();
  setVal('extrapTrainEpochs', arch.extrapolation_num_train_epochs);
  setVal('extrapCheckpoints', arch.extrapolation_num_checkpoints);
  setVal('extrapTargetEpochs', arch.extrapolation_target_epochs);
  setVal('searchBatchSize', arch.training_batch_size != null ? arch.training_batch_size : '');
  setVal('agentModel', arch.agent_model);
  setVal('agentCandidates', arch.candidates_per_batch);
  setVal('agentRegenRounds', arch.max_regen_rounds);

  window.__wizardStartStep = config.start_step != null ? config.start_step : null;
  window.__wizardStopStep = config.stop_step != null ? config.stop_step : null;

  // In edit & continue mode, always turn off Auto-suggest so loaded hw settings
  // are editable immediately and done() does not trigger an unexpected auto-fill.
  if (window.__isEditContinueMode) {
    const autoToggleEl = document.getElementById('hwAutoSuggestToggle');
    if (autoToggleEl) autoToggleEl.classList.remove('on');
    _hwAutoMode = false;
    renderCoreTypes();
  }

  updateSearchVisibility();
  updateObjectiveCheckboxes(arch.objectives);
  return modelConfigPromise;
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
  syncWtQuantToggle();
  document.getElementById('jsonOutput').innerHTML = renderJson(buildConfig());
  schedulePipelineStepsUpdate();

  // NAS architecture search: mapping stats are meaningless until the search settles.
  if (getSegVal('configMode') === 'search') {
    const banner = document.getElementById('hwValidationBanner');
    if (banner) banner.classList.add('hide');
    _renderHwStats('search');
    return;
  }

  if (getSegVal('hwMode') === 'fixed') {
    if (isHwAutoSuggestOn() && _hwAutoMode) {
      scheduleHwAutoRefill();
    } else {
      scheduleHwValidation();
    }
  } else {
    const banner = document.getElementById('hwValidationBanner');
    if (banner) banner.classList.add('hide');
    _renderHwStats('search');
  }
}

// ── Pipeline steps preview ──────────────────────────────────
function schedulePipelineStepsUpdate() {
  if (pipelineStepsDebounceTimer) clearTimeout(pipelineStepsDebounceTimer);
  pipelineStepsDebounceTimer = setTimeout(function () {
    pipelineStepsDebounceTimer = null;
    updatePipelineStepsBar();
  }, 250);
}

function _renderWizardPipelineSteps(steps, groups) {
  var selectable = !!window.__isEditContinueMode;

  // Find the suggested step index (first step not completed in the previous run).
  var ecSuggestedIdx = -1;
  if (selectable && _ecPrevCompleted) {
    for (var j = 0; j < steps.length; j++) {
      if (!_ecPrevCompleted.has(steps[j])) { ecSuggestedIdx = j; break; }
    }
  }

  var cols = steps.map(function (name, i) {
    var group = (groups && groups[i]) || 'other';
    var isStart = selectable && window.__wizardStartStep === name;
    var dataStatus, ecStatus, isClickable;

    if (selectable && _ecPrevCompleted) {
      // EC mode with previous-run data: assign meaningful states.
      if (isStart) {
        dataStatus = 'running';
        ecStatus = _ecPrevCompleted.has(name) ? 'cache' : 'suggested';
        isClickable = true;
      } else if (_ecPrevCompleted.has(name)) {
        dataStatus = 'completed';
        ecStatus = 'cache';
        isClickable = true;
      } else if (i === ecSuggestedIdx) {
        dataStatus = 'pending';
        ecStatus = 'suggested';
        isClickable = true;
      } else {
        dataStatus = 'pending';
        ecStatus = 'future';
        isClickable = false;
      }
    } else if (selectable) {
      // EC mode but no previous-run data (API failed or run has no steps yet): fall back
      // to position-based colouring so the bar is at least somewhat informative.
      var isPast = window.__wizardStartStep && steps.indexOf(window.__wizardStartStep) > i;
      dataStatus = isStart ? 'running' : (isPast ? 'completed' : 'pending');
      ecStatus = null;
      isClickable = true;
    } else {
      // Plain preview (not EC): all bars uniform, none clickable.
      dataStatus = 'pending';
      ecStatus = null;
      isClickable = false;
    }

    var selectedCls = isStart ? ' selected' : '';
    var ecAttr = ecStatus ? ' data-ec-status="' + ecStatus + '"' : '';
    var selCls = isClickable ? ' psb-col--ec-selectable' : '';
    var dataStart = isClickable ? ' data-ec-start-step="' + encodeURIComponent(name) + '"' : '';
    var title = (!isClickable && selectable) ? ' title="Cannot restart from this step — earlier steps must run first"' : '';
    return '<div class="psb-col' + selectedCls + selCls + '" data-status="' + dataStatus +
      '" data-group="' + escapeHtml(group) + '"' + ecAttr + dataStart + title + '>' +
      '<div class="psb-bar"></div>' +
      '<span class="psb-label">' + escapeHtml(name) + '</span>' +
      '</div>';
  });
  return '<div class="psb-list psb-list--preview">' + cols.join('') + '</div>';
}

function updatePipelineStepsBar() {
  var listEl = document.getElementById('pipelineStepsList');
  var barEl = document.getElementById('pipelineStepsBar');
  if (!listEl || !barEl) return;
  if (pipelineStepsLoading) return;
  pipelineStepsLoading = true;
  barEl.classList.add('pipeline-steps-loading');
  fetch('/api/pipeline_steps', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(buildConfig()),
  })
    .then(function (res) {
      if (!res.ok) return res.json().then(function (b) { throw new Error(b.error || 'Failed'); });
      return res.json();
    })
    .then(function (data) {
      var steps = data.steps || [];
      var semanticGroups = data.semantic_groups || [];

      var stepChanged = false;
      // Stale selection guard: clear if the selected step no longer exists in the new pipeline.
      if (window.__wizardStartStep && steps.indexOf(window.__wizardStartStep) < 0) {
        window.__wizardStartStep = null;
        stepChanged = true;
      }
      // Edit & continue: on first pipeline load, suggest the first incomplete step.
      if (window.__isEditContinueMode && window.__wizardStartStep == null && _ecPrevCompleted && !_ecSuggestionDone) {
        for (var i = 0; i < steps.length; i++) {
          if (!_ecPrevCompleted.has(steps[i])) {
            window.__wizardStartStep = steps[i];
            stepChanged = true;
            break;
          }
        }
        _ecSuggestionDone = true;
      }

      lastPipelineSteps = steps;
      lastPipelineGroups = semanticGroups;
      barEl.classList.remove('pipeline-steps-loading', 'pipeline-steps-error');
      listEl.classList.add('pipeline-steps-updating');
      setTimeout(function () {
        listEl.innerHTML = _renderWizardPipelineSteps(steps, semanticGroups);
        listEl.classList.remove('pipeline-steps-updating');
        if (stepChanged) update(); // refresh JSON preview to reflect new start_step
      }, 150);
    })
    .catch(function (err) {
      barEl.classList.remove('pipeline-steps-loading');
      barEl.classList.add('pipeline-steps-error');
      if (lastPipelineSteps.length) {
        listEl.innerHTML = _renderWizardPipelineSteps(lastPipelineSteps, lastPipelineGroups);
      } else {
        listEl.innerHTML = '<span class="pipeline-steps-error-msg">Could not load steps</span>';
      }
    })
    .finally(function () {
      pipelineStepsLoading = false;
    });
}

function selectStartStep(name) {
  // Guard: in EC mode with previous-run data, only completed (cache) and the
  // suggested step are valid restart points; future steps are not selectable.
  if (window.__isEditContinueMode && _ecPrevCompleted) {
    var suggestedIdx = -1;
    for (var j = 0; j < lastPipelineSteps.length; j++) {
      if (!_ecPrevCompleted.has(lastPipelineSteps[j])) { suggestedIdx = j; break; }
    }
    var nameIdx = lastPipelineSteps.indexOf(name);
    if (suggestedIdx >= 0 && nameIdx > suggestedIdx) return;
  }
  window.__wizardStartStep = (window.__wizardStartStep === name) ? null : name;
  var listEl = document.getElementById('pipelineStepsList');
  if (listEl && lastPipelineSteps.length) {
    listEl.innerHTML = _renderWizardPipelineSteps(lastPipelineSteps, lastPipelineGroups);
  }
  update();
}

function escapeHtml(s) {
  var div = document.createElement('div');
  div.textContent = s;
  return div.innerHTML;
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

function saveAsTemplate() {
  var name = window.prompt('Template name:', buildConfig().experiment_name || 'config');
  if (name == null || !name.trim()) return;
  var btn = document.getElementById('saveTemplateBtn');
  if (btn) btn.disabled = true;
  fetch('/api/templates', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ name: name.trim(), config: buildConfig() }),
  }).then(function (r) {
    if (!r.ok) return r.json().then(function (b) { throw new Error(b.error || 'Save failed'); });
    return r.json();
  }).then(function () {
    alert('Template saved.');
  }).catch(function (err) {
    alert(err.message || 'Failed to save template');
  }).finally(function () {
    if (btn) btn.disabled = false;
  });
}

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
      return res.json().then(function (body) {
        var rid = body && body.run_id;
        window.location.href = rid ? '/monitor?run_id=' + encodeURIComponent(rid) : '/';
      });
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

// ══════════════════════════════════════════════════════════
// HARDWARE AUTO-FILL & VALIDATION
// ══════════════════════════════════════════════════════════

function isHwAutoSuggestOn() {
  const el = document.getElementById('hwAutoSuggestToggle');
  return el ? el.classList.contains('on') : true;
}

function toggleHwAutoSuggest(el) {
  if (!el || el.classList.contains('loading')) return;
  el.classList.toggle('on');
  renderCoreTypes(); // re-render to apply/remove locked state
  if (isHwAutoSuggestOn()) {
    autoFillHardware();
  } else {
    _hwAutoMode = false;
    scheduleHwValidation();
  }
  update();
}

let _hwValidateTimer = null;
let _hwValidating = false;

// Build the request body for both auto and verify endpoints
function _buildHwApiBody() {
  const schema = MODEL_CONFIG_SCHEMAS[state.modelType] || [];
  const modelConfig = {};
  schema.forEach(f => {
    const el = document.getElementById('mc_' + f.key);
    if (!el) return;
    if (f.type === 'number') modelConfig[f.key] = parseFloat(el.value);
    else if (f.type === 'toggle') modelConfig[f.key] = el.classList.contains('on');
    else if (f.type === 'text' && f.key === 'hidden_dims')
      modelConfig[f.key] = el.value.split(',').map(s => parseInt(s.trim())).filter(n => !isNaN(n));
    else modelConfig[f.key] = el.value;
  });

  const maxAx = state.coreTypes.length ? Math.max(...state.coreTypes.map(c => c.max_axons)) : 1024;
  const maxNeu = state.coreTypes.length ? Math.max(...state.coreTypes.map(c => c.max_neurons)) : 1024;
  const pruningOn = isToggleOn('pruningToggle');
  const pruningFrac = pruningOn ? parseFloat(document.getElementById('pruningFraction')?.value || '0') : 0;
  const thresholdGroups = parseInt(document.getElementById('maxThresholdGroups')?.value || '1') || 1;
  const targetTq = parseInt(document.getElementById('targetTq')?.value || '32') || 32;

  return {
    model_type: state.modelType,
    input_shape: (function () {
      const dp = v('dataProvider') || '';
      // Honour the preprocessing resize_to so the wizard builds the model at
      // the same spatial resolution as the deployment pipeline.  Without this
      // the wizard would build a ViT for 32x32 while the deployment pipeline
      // (which applies the configured resize_to=224 via the data provider
      // preprocessing) builds a ViT for 224x224 — producing wildly different
      // softcore counts and a falsely-feasible mapping report.
      const resizeRaw = v('preprocessResizeTo');
      const resizeVal = (resizeRaw !== '' && resizeRaw != null) ? parseInt(resizeRaw) : NaN;
      const resize = (!isNaN(resizeVal) && resizeVal > 0) ? resizeVal : null;
      if (dp.includes('CIFAR')) return [3, resize || 32, resize || 32];
      if (dp.includes('IMAGENET') || dp.includes('ImageNet') || dp.includes('IMGNET')) {
        return [3, resize || 224, resize || 224];
      }
      if (dp.includes('ECG')) return [1, 140];
      if (resize) return [1, resize, resize];
      return [1, 28, 28];
    })(),
    num_classes: 10,
    model_config: modelConfig,
    max_axons: maxAx,
    max_neurons: maxNeu,
    threshold_groups: thresholdGroups,
    pruning_fraction: pruningFrac,
    threshold_seed: 0,
    allow_coalescing: isToggleOn('coreCoalescingToggle'),
    hardware_bias: isToggleOn('hardwareBiasToggle'),
    allow_neuron_splitting: isToggleOn('neuronSplittingToggle'),
    allow_scheduling: isToggleOn('scheduledMappingToggle'),
    target_tq: targetTq,
  };
}

// Show/hide a translucent loading overlay on the hw-config fixed panel.
function _setHwConfigLoading(isLoading) {
  var el = document.getElementById('hwFixed');
  if (!el) return;
  if (isLoading) {
    if (!el.querySelector('.hw-auto-loading-overlay')) {
      var overlay = document.createElement('div');
      overlay.className = 'hw-auto-loading-overlay';
      overlay.innerHTML = '<div class="hw-spinner"></div>';
      el.appendChild(overlay);
    }
  } else {
    var existing = el.querySelector('.hw-auto-loading-overlay');
    if (existing) existing.remove();
  }
}

// Auto-fill hardware configuration using the greedy algorithm
function autoFillHardware() {
  if (_hwRefilling) return;
  _hwRefilling = true;
  _setHwConfigLoading(true);

  const toggleEl = document.getElementById('hwAutoSuggestToggle');
  if (toggleEl) toggleEl.classList.add('loading');

  const body = _buildHwApiBody();

  fetch('/api/hw_config_auto', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  }).then(r => r.json()).then(data => {
    if (data.error) {
      _hwAutoMode = false;
      _showHwValidation(false, [data.error], {});
      return;
    }
    if (data.core_types && data.core_types.length) {
      _hwAutoMode = true;
      state.coreTypes = data.core_types.map(ct => ({
        max_axons: ct.max_axons,
        max_neurons: ct.max_neurons,
        count: ct.count,
        has_bias: ct.has_bias !== undefined ? ct.has_bias : isToggleOn('hardwareBiasToggle'),
      }));
      renderCoreTypes();
      // Flash the hardware config pane for 1 second
      const pane = document.getElementById('hwSection');
      if (pane) {
        pane.classList.remove('hw-auto-flash');
        void pane.offsetWidth; // reflow to restart animation
        pane.classList.add('hw-auto-flash');
        setTimeout(() => pane.classList.remove('hw-auto-flash'), 1000);
      }
      // Verify the auto-suggestion is actually feasible (catches under-estimated counts)
      let rationaleMsg = '\u2713 Auto-configured: ' + (data.rationale || '');
      if (data.num_passes && data.num_passes > 1) {
        rationaleMsg += ' (' + data.num_passes + ' schedule passes)';
      }
      _doHwValidation(rationaleMsg);
    } else {
      // No core types returned (e.g. no softcores, or layout failed) — clear Recalculating state
      _hwAutoMode = false;
      const msg = data.rationale || 'Auto-config returned no core types.';
      _showHwValidation(false, [msg], {});
    }
  }).catch(err => {
    _hwAutoMode = false;
    _showHwValidation(false, ['Auto-config request failed: ' + err.message], {});
  }).finally(() => {
    _hwRefilling = false;
    _setHwConfigLoading(false);
    const toggleEl = document.getElementById('hwAutoSuggestToggle');
    if (toggleEl) toggleEl.classList.remove('loading');
  });
}

// Schedule a re-fill when _hwAutoMode is active and params changed
function scheduleHwAutoRefill() {
  if (_hwRefilling) return;
  if (_hwAutoRefillTimer) clearTimeout(_hwAutoRefillTimer);
  // Show a brief "updating" hint in the banner and loading state in stats
  const banner = document.getElementById('hwValidationBanner');
  if (banner && !banner.classList.contains('hide')) {
    banner.textContent = 'Recalculating\u2026';
    banner.className = 'hw-validation-banner hw-updating';
  }
  _renderHwStats('loading');
  _hwAutoRefillTimer = setTimeout(() => {
    _hwAutoRefillTimer = null;
    if (isHwAutoSuggestOn() && _hwAutoMode && !_hwRefilling) autoFillHardware();
  }, 1200);
}

// Debounced validation: called on every update() when hwMode=user
function scheduleHwValidation() {
  if (_hwValidateTimer) clearTimeout(_hwValidateTimer);
  _renderHwStats('loading');
  _hwValidateTimer = setTimeout(_runHwValidation, 800);
}

function _runHwValidation() {
  if (getSegVal('hwMode') !== 'fixed') return;
  _doHwValidation();
}

function _doHwValidation(successPrefix) {
  if (!state.coreTypes || !state.coreTypes.length) return;

  const body = _buildHwApiBody();
  const coreTypes = state.coreTypes.map(ct => ({
    max_axons: ct.max_axons,
    max_neurons: ct.max_neurons,
    count: ct.count,
  }));

  fetch('/api/hw_config_verify', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      model_repr_json: body,
      core_types: coreTypes,
      allow_coalescing: body.allow_coalescing || false,
      allow_neuron_splitting: body.allow_neuron_splitting || false,
      allow_scheduling: body.allow_scheduling || false,
    }),
  }).then(r => r.json()).then(data => {
    if (data.error) return;  // server error, skip silently
    if (data.feasible && successPrefix) {
      let msg = successPrefix;
      if (data.schedule_info && data.schedule_info.scheduled_feasible) {
        msg += ' (scheduled: ' + data.schedule_info.num_passes + ' passes)';
      }
      _showHwValidation(true, [msg], {});
    } else if (data.schedule_info && data.schedule_info.scheduled_feasible) {
      _showHwValidation(true, [data.schedule_info.message], {});
    } else {
      _showHwValidation(data.feasible, data.errors || [], data.field_errors || {});
    }
    _renderHwStats(data.feasible ? data.stats : null);
  }).catch(() => {});  // network errors: skip silently
}

function _showHwValidation(feasible, errors, fieldErrors) {
  const banner = document.getElementById('hwValidationBanner');
  const section = document.getElementById('hwSection');
  if (!banner) return;

  // Clear previous per-field highlights
  document.querySelectorAll('.hw-field-invalid').forEach(el => el.classList.remove('hw-field-invalid'));

  if (feasible) {
    const msg = errors && errors.length ? errors[0] : '\u2713 Hardware configuration is sufficient.';
    banner.textContent = msg;
    banner.className = 'hw-validation-banner ok';
    if (section) section.classList.remove('hw-invalid');
    // Auto-hide after 3s for simple success; keep visible for scheduling info
    const isScheduleInfo = msg.toLowerCase().includes('scheduled') || msg.toLowerCase().includes('passes');
    if (!isScheduleInfo) {
      setTimeout(() => { if (banner.className.includes('ok')) banner.classList.add('hide'); }, 3000);
    }
    return;
  }

  // Show error banner and flash section header
  if (section) {
    section.classList.add('hw-invalid');
    void section.offsetWidth;
    section.classList.remove('hw-invalid');
    section.classList.add('hw-invalid');
  }

  banner.className = 'hw-validation-banner';
  banner.innerHTML = '<strong>Hardware configuration issues:</strong><ul style="margin:4px 0 0 16px">' +
    (errors || []).map(e => '<li>' + _escHtml(e) + '</li>').join('') +
    '</ul>';

  // Highlight invalid inputs with a red border (no extra DOM elements that break grid layout)
  Object.keys(fieldErrors || {}).forEach(key => {
    const m = key.match(/^core_type_(\d+)_(max_axons|max_neurons)$/);
    if (m) {
      const inputEl = document.getElementById('ct_' + m[1] + '_' + m[2]);
      if (inputEl) inputEl.classList.add('hw-field-invalid');
    }
    // Insufficient total core count: highlight all count inputs
    if (key === 'total_count') {
      state.coreTypes.forEach((_, i) => {
        const inputEl = document.getElementById('ct_' + i + '_count');
        if (inputEl) inputEl.classList.add('hw-field-invalid');
      });
    }
  });
}

function _renderHwStats(statsOrState) {
  var panel = document.getElementById('hwStatsPanel');
  if (!panel) return;

  // ── Loading: overlay on existing content to avoid janky blank-then-repopulate ──
  if (statsOrState === 'loading') {
    if (_hwStatsHasContent) {
      if (!panel.querySelector('.hw-stats-overlay')) {
        var overlay = document.createElement('div');
        overlay.className = 'hw-stats-overlay';
        overlay.innerHTML = '<div class="hw-spinner"></div>';
        panel.appendChild(overlay);
      }
      var badge = panel.querySelector('.hw-stats-badge');
      if (badge) { badge.className = 'hw-stats-badge loading'; badge.textContent = 'Verifying\u2026'; }
      return;  // keep existing content intact underneath the overlay
    }
    // First load — no data yet, show a placeholder
    _hwStatsHasContent = false;
    panel.className = 'hw-stats-panel hw-stats-state-loading';
    panel.innerHTML =
      '<div class="hw-stats-header">' +
        '<span class="hw-stats-title">Mapping Performance</span>' +
        '<span class="hw-stats-badge loading">Verifying\u2026</span>' +
      '</div>' +
      '<div class="hw-stats-empty-msg">Computing mapping statistics\u2026</div>';
    return;
  }

  // For all non-loading transitions: remove any existing overlay and reset flag.
  var existingOverlay = panel.querySelector('.hw-stats-overlay');
  if (existingOverlay) existingOverlay.remove();
  _hwStatsHasContent = false;

  if (statsOrState === 'nas') {
    panel.className = 'hw-stats-panel hw-stats-state-empty';
    panel.innerHTML =
      '<div class="hw-stats-header">' +
        '<span class="hw-stats-title">Mapping Performance</span>' +
        '<span class="hw-stats-badge empty">NAS Search</span>' +
      '</div>' +
      '<div class="hw-stats-empty-msg">Architecture is being searched \u2014 mapping stats depend on the final architecture.</div>';
    return;
  }

  if (statsOrState === 'search') {
    panel.className = 'hw-stats-panel hw-stats-state-empty';
    panel.innerHTML =
      '<div class="hw-stats-header">' +
        '<span class="hw-stats-title">Mapping Performance</span>' +
        '<span class="hw-stats-badge empty">HW Search</span>' +
      '</div>' +
      '<div class="hw-stats-empty-msg">Hardware is being searched \u2014 run the pipeline to see mapping stats.</div>';
    return;
  }

  if (!statsOrState) {
    panel.className = 'hw-stats-panel hw-stats-state-empty';
    panel.innerHTML =
      '<div class="hw-stats-header">' +
        '<span class="hw-stats-title">Mapping Performance</span>' +
        '<span class="hw-stats-badge empty">Not Verified</span>' +
      '</div>' +
      '<div class="hw-stats-empty-msg">Verify the hardware configuration to see mapping performance statistics.</div>';
    return;
  }

  var stats = statsOrState;

  if (!stats.feasible) {
    panel.className = 'hw-stats-panel hw-stats-state-error';
    panel.innerHTML =
      '<div class="hw-stats-header">' +
        '<span class="hw-stats-title">Mapping Performance</span>' +
        '<span class="hw-stats-badge error">Infeasible</span>' +
      '</div>' +
      '<div class="hw-stats-empty-msg">Hardware configuration cannot fit all soft cores \u2014 adjust core types.</div>';
    return;
  }

  // Scheduled mapping: show pass count in the stats header badge
  var scheduleBadge = '';
  if (stats.schedule_pass_count && stats.schedule_pass_count > 1) {
    scheduleBadge = ' \u2022 Scheduled';
  }

  // ── Helpers ─────────────────────────────────────────────
  function fmt(v) { return v != null ? v.toFixed(1) : '\u2014'; }
  function fmtInt(v) { return v != null ? String(v) : '\u2014'; }
  function fmtNum(v) {
    if (v == null) return '\u2014';
    var n = Number(v);
    return Number.isInteger(n) ? String(n) : n.toFixed(1);
  }

  function barColor(pct) {
    if (pct >= 70) return 'green';
    if (pct >= 40) return 'amber';
    return 'rose';
  }

  function wasteBarColor(pct) {
    if (pct <= 30) return 'green';
    if (pct <= 60) return 'amber';
    return 'rose';
  }

  function healthBar(label, pct, colorFn) {
    var p = Math.max(0, Math.min(100, pct != null ? pct : 0));
    var cls = colorFn(p);
    return '<div class="hw-health-bar">' +
      '<span class="hw-health-bar-label">' + _escHtml(label) + '</span>' +
      '<div class="hw-health-bar-track"><div class="hw-health-bar-fill ' + cls + '" style="width:' + p.toFixed(1) + '%"></div></div>' +
      '<span class="hw-health-bar-value">' + fmt(pct) + '%</span>' +
      '</div>';
  }

  function miniBar(pct, colorFn) {
    var p = Math.max(0, Math.min(100, pct != null ? pct : 0));
    var cls = colorFn(p);
    return '<div class="hw-per-core-cell">' +
      '<div class="hw-per-core-bar-row">' +
        '<div class="hw-per-core-track"><div class="hw-per-core-fill ' + cls + '" style="width:' + p.toFixed(1) + '%"></div></div>' +
        '<span class="hw-per-core-value">' + fmt(pct) + '%</span>' +
      '</div></div>';
  }

  function perCoreRow(label, min, avg, max, colorFn) {
    return '<div class="hw-per-core-label">' + _escHtml(label) + '</div>' +
      miniBar(min, colorFn) + miniBar(avg, colorFn) + miniBar(max, colorFn);
  }

  function detailChip(lbl, val) {
    return '<span class="hw-detail-chip"><span class="hw-detail-chip-lbl">' +
      _escHtml(lbl) + '</span>' + _escHtml(fmtNum(val)) + '</span>';
  }

  function renderLayoutPreview(preview) {
    if (!preview || !Array.isArray(preview.flow) || preview.flow.length === 0) return '';

    var hasScheduling = preview.schedule_sync_count > 0;

    var items = preview.flow.map(function (item) {
      if (item.kind === 'input' || item.kind === 'output') {
        return '<div class="hw-layout-mini-endcap ' + item.kind + '">' +
          '<span>' + item.kind.toUpperCase() + '</span>' +
          '</div>';
      }
      if (item.kind === 'host') {
        if (item.schedule_sync) {
          // Schedule sync barrier — same visual weight as host-ops, red accent
          return '<div class="hw-layout-mini-host hw-layout-mini-schedule-sync">' +
            '<div class="hw-layout-mini-host-box">' +
            '<div class="hw-layout-mini-host-label">\u21bb sync</div>' +
            '<div class="hw-layout-mini-host-sub">pass</div>' +
            '</div>' +
            '</div>';
        }
        return '<div class="hw-layout-mini-host">' +
          '<div class="hw-layout-mini-host-box">' +
          '<div class="hw-layout-mini-host-label">' + fmtInt(item.compute_op_count) + ' ops</div>' +
          '<div class="hw-layout-mini-host-sub">host</div>' +
          '</div>' +
          '</div>';
      }
      if (item.kind === 'neural') {
        return '<div class="hw-layout-mini-lat-group">' +
          '<div class="hw-layout-mini-lat-label">' + fmtInt(item.latency_group_index) + '</div>' +
          '<div class="hw-layout-mini-lat-bar">' + fmtInt(item.softcore_count) + '</div>' +
          '</div>';
      }
      return '';
    }).filter(Boolean);

    var flowHtml = items.map(function (itemHtml, idx) {
      if (idx === items.length - 1) return itemHtml;
      return itemHtml + '<div class="hw-layout-mini-arrow">\u2192</div>';
    }).join('');

    var title = hasScheduling ? 'Mapping Miniview (Scheduled)' : 'Mapping Miniview';
    return '<div class="hw-layout-mini-wrap">' +
      '<div class="hw-stats-section-label">' + title + '</div>' +
      '<div class="hw-layout-mini-flow">' + flowHtml + '</div>' +
      '</div>';
  }

  // ── Build HTML ──────────────────────────────────────────
  var html =
    '<div class="hw-stats-header">' +
      '<span class="hw-stats-title">Mapping Performance</span>' +
      '<span class="hw-stats-badge ok">Verified' + scheduleBadge + '</span>' +
    '</div>';

  // Count cards — always-present topology overview
  var segmentCard = (stats.neural_segment_count > 0)
    ? '<div class="hw-stat-card"><div class="hw-stat-card-value">' + fmtInt(stats.neural_segment_count) + '</div><div class="hw-stat-card-label">Neural Segments</div></div>'
    : '';
  var scheduleSyncs = stats.schedule_sync_count
    || (stats.layout_preview && stats.layout_preview.schedule_sync_count)
    || 0;
  var totalBarriers = (stats.host_side_segment_count || 0) + scheduleSyncs;
  var syncBarrierCard = (totalBarriers > 0)
    ? '<div class="hw-stat-card"><div class="hw-stat-card-value">' + fmtInt(totalBarriers) + '</div><div class="hw-stat-card-label">Sync Barriers</div></div>'
    : '';
  var coresLabel = (stats.schedule_pass_count > 1) ? 'Cores / Pass' : 'Cores Used';

  html +=
    '<div class="hw-stats-cards">' +
      '<div class="hw-stat-card"><div class="hw-stat-card-value">' + fmtInt(stats.total_cores) + '</div><div class="hw-stat-card-label">' + coresLabel + '</div></div>' +
      '<div class="hw-stat-card"><div class="hw-stat-card-value">' + fmtInt(stats.total_softcores) + '</div><div class="hw-stat-card-label">Softcores</div></div>' +
      segmentCard + syncBarrierCard +
    '</div>';

  // Two-column body: total (left) + per-core (right)
  html += '<div class="hw-stats-body">';

  // Left: total health bars
  html +=
    '<div>' +
      '<div class="hw-stats-section-label">Total</div>' +
      '<div class="hw-stats-bars">' +
        healthBar('Param Utilization', stats.mapped_params_pct, barColor) +
        healthBar('Wasted Axons', stats.total_wasted_axons_pct, wasteBarColor) +
        healthBar('Wasted Neurons', stats.total_wasted_neurons_pct, wasteBarColor) +
      '</div>' +
    '</div>';

  // Right: per-core min/avg/max health bars
  html +=
    '<div>' +
      '<div class="hw-stats-section-label">Per-Core</div>' +
      '<div class="hw-per-core-grid">' +
        '<div></div>' +
        '<div class="hw-per-core-header">Min</div>' +
        '<div class="hw-per-core-header">Avg</div>' +
        '<div class="hw-per-core-header">Max</div>' +
        perCoreRow('Wasted Axons',
          stats.per_core_wasted_axons_pct_min,
          stats.per_core_wasted_axons_pct_avg,
          stats.per_core_wasted_axons_pct_max,
          wasteBarColor) +
        perCoreRow('Wasted Neurons',
          stats.per_core_wasted_neurons_pct_min,
          stats.per_core_wasted_neurons_pct_avg,
          stats.per_core_wasted_neurons_pct_max,
          wasteBarColor) +
        perCoreRow('Param Usage',
          stats.per_core_mapped_params_pct_min,
          stats.per_core_mapped_params_pct_avg,
          stats.per_core_mapped_params_pct_max,
          barColor) +
      '</div>' +
    '</div>';

  html += '</div>'; // end hw-stats-body

  var detailRowsHtml = '';

  // ── Coalescing detail ──────────────────────────────────
  if (stats.coalescing_group_count > 0) {
    detailRowsHtml +=
      '<div class="hw-detail-row">' +
        '<span class="hw-detail-title">Coalescing</span>' +
        '<span class="hw-detail-count">' + fmtInt(stats.coalescing_group_count) + ' groups</span>' +
        '<span class="hw-detail-stat-label">Frags/group:</span>' +
        detailChip('Min', stats.coalescing_frags_per_group_min) +
        detailChip('Mdn', stats.coalescing_frags_per_group_median) +
        detailChip('Max', stats.coalescing_frags_per_group_max) +
      '</div>';
  }

  // ── Segment latency detail ───────────────────────────────
  if (stats.neural_segment_count > 0) {
    detailRowsHtml +=
      '<div class="hw-detail-row">' +
        '<span class="hw-detail-title">Segment Latency</span>' +
        '<span class="hw-detail-count">' + fmtInt(stats.neural_segment_count) + ' segments</span>' +
        '<span class="hw-detail-stat-label">Latency groups/segment:</span>' +
        detailChip('Min', stats.segment_latency_min) +
        detailChip('Mdn', stats.segment_latency_median) +
        detailChip('Max', stats.segment_latency_max) +
      '</div>';
  }

  // ── Scheduling detail ──────────────────────────────────
  if (stats.schedule_pass_count > 1) {
    var perSegDetail = '';
    if (stats.per_segment_passes) {
      var segEntries = Object.keys(stats.per_segment_passes).sort(function(a, b) { return a - b; });
      perSegDetail = segEntries.map(function(sid) {
        return 'seg ' + sid + ': ' + stats.per_segment_passes[sid] + 'p';
      }).join(', ');
    }
    detailRowsHtml +=
      '<div class="hw-detail-row">' +
        '<span class="hw-detail-title">Scheduling</span>' +
        '<span class="hw-detail-count">' + fmtInt(stats.schedule_pass_count) + ' total passes</span>' +
        '<span class="hw-detail-stat-label">' + (perSegDetail || 'Cores reused between passes.') + '</span>' +
      '</div>';
  }

  // ── Splitting detail ───────────────────────────────────
  if (stats.split_softcore_count > 0) {
    detailRowsHtml +=
      '<div class="hw-detail-row">' +
        '<span class="hw-detail-title">Splitting</span>' +
        '<span class="hw-detail-count">' + fmtInt(stats.split_softcore_count) + ' softcores split</span>' +
        '<span class="hw-detail-stat-label">Splits/SC:</span>' +
        detailChip('Min', stats.splits_per_softcore_min) +
        detailChip('Mdn', stats.splits_per_softcore_median) +
        detailChip('Max', stats.splits_per_softcore_max) +
      '</div>';
  }

  var layoutPreviewHtml = renderLayoutPreview(stats.layout_preview);
  if (detailRowsHtml || layoutPreviewHtml) {
    html += '<div class="hw-stats-bottom">';
    html += '<div class="hw-stats-bottom-left">' + detailRowsHtml + '</div>';
    if (layoutPreviewHtml) {
      html += '<div class="hw-stats-bottom-right">' + layoutPreviewHtml + '</div>';
    }
    html += '</div>';
  }

  panel.className = 'hw-stats-panel';
  panel.innerHTML = html;
  _hwStatsHasContent = true;

  // Always mask the content swap to prevent the "bars growing from 0%" flash.
  // We write the final-state HTML, then immediately place an opaque overlay on
  // top (suppressing its own fade-in animation so it appears at full opacity
  // instantly), then fade it out — revealing content that is already at its
  // final state.  This applies on first load AND on every subsequent update.
  (function () {
    var m = document.createElement('div');
    m.className = 'hw-stats-overlay';
    // Override the class animation/transition so the overlay appears solid
    // immediately (no fade-in).  We will trigger our own fade-out below.
    m.style.cssText = 'animation:none;opacity:1;transition:none;';
    m.innerHTML = '<div class="hw-spinner"></div>';
    panel.appendChild(m);
    m.offsetHeight;  // force reflow — browser commits opacity:1 before our next change
    m.style.transition = 'opacity 0.22s ease';
    m.style.opacity = '0';
    setTimeout(function () { if (m.parentNode) m.parentNode.removeChild(m); }, 280);
  }());
}

function _escHtml(s) {
  return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}

// ── Objective checkboxes ────────────────────────────────────
var _objectiveOptions = [];

function loadObjectiveOptions() {
  return fetch('/api/wizard/schema').then(function(r) { return r.json(); }).then(function(data) {
    var nas = data.nas || {};
    _objectiveOptions = nas.objective_options || [];
    updateObjectiveCheckboxes();
  }).catch(function() {
    _objectiveOptions = [];
  });
}

/** @param {string[]|null|undefined} selectedObjectiveIds When undefined/null, all objectives are active (new session). When an array, only those ids are active. */
function updateObjectiveCheckboxes(selectedObjectiveIds) {
  var container = document.getElementById('objectiveCheckboxes');
  if (!container) return;
  var modelSearch = getSegVal('configMode') === 'search';
  var hwSearch = getSegVal('hwMode') === 'search';
  var selectionSet = selectedObjectiveIds == null ? null : new Set(selectedObjectiveIds);
  container.innerHTML = '';
  _objectiveOptions.forEach(function(opt) {
    if (opt.requires_training && !modelSearch && hwSearch) return;
    var chip = document.createElement('div');
    var active = selectionSet === null ? true : selectionSet.has(opt.id);
    chip.className = 'objective-chip' + (active ? ' active' : '');
    chip.setAttribute('data-objective', opt.id);
    var nameSpan = document.createElement('span');
    nameSpan.textContent = opt.label;
    var badge = document.createElement('span');
    badge.className = 'goal-badge';
    badge.textContent = opt.goal;
    chip.appendChild(nameSpan);
    chip.appendChild(badge);
    chip.onclick = function() {
      chip.classList.toggle('active');
      update();
    };
    container.appendChild(chip);
  });
}

function getSelectedObjectives() {
  var container = document.getElementById('objectiveCheckboxes');
  if (!container) return [];
  var chips = container.querySelectorAll('.objective-chip.active');
  return Array.from(chips).map(function(c) { return c.getAttribute('data-objective'); });
}

// Hook into update() to trigger debounced validation
const _originalUpdate = update;

// ── Boot ───────────────────────────────────────────────────
init();
