# WS2 scope — model-operator support map + A6 keystone enable (C3)

This is a planning + small-validation artifact. No `src/` changes. It answers two
questions for downstream WS2:

1. **(a) MODEL-STRUCTURE SUPPORT MAP** — exactly which torch ops the ANN→SNN
   conversion path currently supports, with file:line evidence, plus the concrete
   gap list for a CNN (Conv2d) and a transformer block (attention / LayerNorm /
   GELU / softmax).
2. **(b) KEYSTONE ENABLE** — the exact config value that turns on the A6
   conversion-policy keystone, and a validation that the keystone config parses
   and builds a real `CascadeCharacterizer`.

---

## (a) Model-structure support map

### The two conversion fronts

There are **two** ANN→SNN front-ends that both terminate in the same Mapper DAG /
`Perceptron` representation:

- **Hand-written `PerceptronFlow` models** (`simple_mlp`, `mlp_mixer`,
  `skip_perceptron_mixer`) — build the Mapper graph directly in
  `get_mapper_repr()`. They can only use the Mapper node types they import.
- **Native `nn.Module` torch models** (`mlp_mixer_core`, `torch_vgg16`,
  `torch_vit`, `torch_squeezenet11`, `torch_custom`, `torch_sequential_*`) — are
  FX-traced and converted to the Mapper DAG by
  `src/mimarsinan/torch_mapping/` after pretraining (the `TorchMappingStep`).
  This is the general path and the SSOT for "what op is supported".

The **decision tables** for the FX path live in
`representability_analyzer.py` (classify) and `mapper_graph_converter.py` (emit).

### What "supported" means — on-chip vs host-side

The critical distinction for an SNN deployment is **NOT** representable-or-not
(the analyzer marks essentially everything representable — see below); it is
**where the op runs**:

- **on-chip NeuralCore** — a spiking `Perceptron` (rate / TTFS-timing code). Only
  `nn.Linear`, `nn.Conv1d`, `nn.Conv2d` (each followed by an optional BN +
  nonlinearity) become these.
  Evidence: `representability_analyzer.py:61` `_NEURAL_CORE_MODULES = {nn.Linear,
  nn.Conv2d, nn.Conv1d}`; emit at `mapper_graph_converter.py:125-130`
  (`_convert_linear` / `_convert_conv2d` / `_convert_conv1d`).
- **host-side ComputeOp** — everything else becomes a `ComputeOpMapper` wrapping a
  `ComputeAdapter` (a picklable callable run **off-chip in floating point**,
  rate↔absolute rescaled by `ScaleNormalizingWrapper`). It is *deployed* but the
  arithmetic happens on the host CPU, not on neuromorphic cores.
  Evidence: generic path `mapper_graph_converter.py:158-159`
  `_emit_generic_compute_op(node, fn)`; payload
  `mapping/support/compute_modules.py:17` `class ComputeAdapter` ("Generic
  host-side ComputeOp payload"); host rescale `compute_modules.py:72`
  `class ScaleNormalizingWrapper`.

So "supported" below is graded **on-chip** (genuine spiking), **host** (runs, but
off-chip float), **approximated** (deployed only after a behavior change), or
**unsupported** (conversion refuses / errors).

### The support map

| torch op | status | where it lands | file:line evidence |
|----------|--------|----------------|--------------------|
| `nn.Linear` | **on-chip** | `Perceptron` NeuralCore | `representability_analyzer.py:61`; `mapper_graph_converter.py:125` `_convert_linear` |
| `nn.Conv2d` (groups=1) | **on-chip** | shared-weight `Conv2DPerceptronMapper` (im2col + tiled matmul) | `representability_analyzer.py:63`; `mapper_graph_converter.py:127` `_convert_conv2d`; `mapping/mappers/conv2d_mapper.py:19` |
| `nn.Conv1d` (groups=1) | **on-chip** | `Conv1DPerceptronMapper` | `mapper_graph_converter.py:129` `_convert_conv1d`; `mapping/mappers/conv1d_mapper.py:19` |
| `nn.Conv*` with `groups>1` | **unsupported** | conversion refuses | `representability_analyzer.py:129-135` ("Grouped convolution … not supported") |
| `nn.BatchNorm1d/2d` | **on-chip (folded)** | absorbed into the preceding Linear/Conv | `representability_analyzer.py:70-77` `_ABSORBABLE_MODULES`; fold in `graph_normalization.normalize_fx_graph` |
| `nn.ReLU` / `nn.LeakyReLU` | **on-chip** | the Perceptron's nonlinearity (rate / TTFS code) | `representability_analyzer.py:73-74` (absorbable); `perceptron.py:25-29` `ACTIVATION_REGISTRY` |
| `nn.GELU` (as a module) | **approximated** | absorbed as the Perceptron base-activation, then **adapted to ReLU** before deployment | `representability_analyzer.py:75` (absorbable); `activation_adaptation_tuner.py:1-4,82-83` ("non-ReLU → ReLU" blend) |
| `nn.Identity` / `nn.Dropout` | **passthrough** | dropped (eval) | `mapper_graph_converter.py:109-110,137-138` `_PASSTHROUGH_MODULES` |
| `nn.Flatten` / `torch.flatten` / `.view` / `.reshape` / `.flatten` | **structural** | shape-only `ReshapeMapper` (no runtime op) | `mapper_graph_converter.py:139-140,154-155,165-195` |
| `.permute` / `.transpose` | **structural** | `PermuteMapper` (batch-leading) or `ReshapeMapper` | `mapper_graph_converter.py:199-221` |
| `torch.cat` | **structural** | `ConcatMapper` | `mapper_graph_converter.py:152-153` `_convert_cat` |
| `operator.getitem` (subscript) | **structural** | `SubscriptMapper` / passthrough | `mapper_graph_converter.py:156-157` |
| `.mean` / `torch.mean` (pool) | **host** | generic `ComputeOpMapper` | `mapper_graph_converter.py:223-224` |
| residual / `+` / `operator.add` / `.add` | **host** | generic `ComputeOpMapper(operator.add)` (multi-input) | `mapper_graph_converter.py:232-233`; multi-input emit in `mapper_graph_fx.py`; hand-written twin `skip_perceptron_mixer.py:123,144` `ComputeOpMapper([out, skip], ComputeAdapter(operator.add))` |
| `nn.LayerNorm` | **host** | `ComputeOpMapper(deepcopy(LayerNorm))` (off-chip float) | `mapper_graph_converter.py:131-134` |
| `nn.MultiheadAttention` | **host** | `ComputeOpMapper(deepcopy(MHA), need_weights=False)` (off-chip float; QK/softmax/AV all on host) | `mapper_graph_converter.py:135-136,248-266` `_convert_multihead_attention` |
| `F.softmax` / `F.gelu` / any other `F.*` / custom `nn.Module` | **host** | generic `ComputeOpMapper(fn)` | `mapper_graph_converter.py:158-159`; classifier marks all functions/methods supported `representability_analyzer.py:145-164` |

**Key consequence:** the representability analyzer (`analyze()`) returns
`is_representable=True` for almost any graph — the *only* hard refusals are
grouped convolution (`representability_analyzer.py:129-135`) and a missing module
target (`:119-123`). Everything else is "representable" because it falls back to a
host-side ComputeOp. **Representable ≠ on-chip-spiking.** For a genuine SNN
deployment the on-chip surface is exactly `{Linear, Conv1d, Conv2d}` (+ folded
BN + ReLU/LeakyReLU); LayerNorm, attention, softmax, residual-add, mean-pool, and
GELU-as-function all execute **off-chip in floating point** (and `forward` parity
is preserved because the host runs the real op).

### The matrix_6 deployment cell uses the clean subset

The `mlp_mixer_core` model behind every `matrix_6` config
(`src/mimarsinan/models/torch_mlp_mixer_core.py`) uses ONLY on-chip / structural /
single-host-mean ops: `Conv2d` patch-embed (`:84`), `BatchNorm2d` (`:87`, folded),
`ReLU` (`:89,107`), `Linear` token/channel mixers + classifier (`:25-26,44-45,102`),
`permute`/`flatten`/`reshape` (structural, `:31,34,108`), and a single
`x.mean(dim=1)` host pool (`:112`). **No** residual-add, LayerNorm, attention, or
softmax. So the keystone cell (mlp_mixer_core, cascaded TTFS) is a clean,
fully-on-chip cascade — the right vehicle for the A6 keystone.

### Concrete WS2 gap list

**To add a CNN (Conv2d-heavy, e.g. VGG/SqueezeNet on chip):**

- Conv2d **already on-chip** (`conv2d_mapper.py`), incl. tiling
  (`_chunk_sizes`), padding (off-source pad, `:201-208`), stride/dilation.
  `torch_vgg16` / `torch_squeezenet11` builders already exist.
- **Gap G1 — pooling.** `nn.MaxPool2d` / `nn.AvgPool2d` / `F.max_pool2d` flow
  through the **generic host ComputeOp** path (not in `_NEURAL_CORE_MODULES`,
  no dedicated handler). They run off-chip float. To deploy on-chip they need a
  spiking pooling primitive (or rewrite avg-pool as a strided shared-weight Conv).
- **Gap G2 — grouped / depthwise conv.** Hard-refused
  (`representability_analyzer.py:129-135`). SqueezeNet fire modules and most
  efficient CNNs are fine (groups=1), but any depthwise/grouped block must be
  un-grouped or a grouped-conv mapper added.
- **Gap G3 — residual adds at scale.** Each skip `+` is a multi-input host
  ComputeOp; functionally correct but off-chip. ResNet-style nets deploy with all
  adds on host. For a genuinely-on-chip residual, a spiking add primitive +
  per-source-scale alignment is needed (the `per_source_scales` seam exists at
  `mapping/support/per_source_scales.py`).
- **Gap G4 — cascade depth.** Even with everything on-chip, the cascaded
  single-spike TTFS depth budget is `d_max(S) ≈ 0.56·√S`
  (`torch_mlp_mixer_core.py:91-95`). A deep CNN exceeds it → use the
  synchronized schedule or residual relief, not the cascaded schedule.

**To add a transformer block (attention + LN + GELU + softmax):**

- **Gap G5 — attention is host-only.** `nn.MultiheadAttention` →
  `ComputeOpMapper` (`mapper_graph_converter.py:135-136,248-266`); QK^T, the
  softmax, and AV all run off-chip in float. There is **no** spiking attention
  primitive. `torch_vit` deploys with the entire attention block on host.
- **Gap G6 — LayerNorm is host-only.** `nn.LayerNorm` → `ComputeOpMapper`
  (`mapper_graph_converter.py:131-134`). Unlike BatchNorm (folded into the
  preceding Linear), LayerNorm's per-token mean/var cannot be folded into static
  weights, so it stays an off-chip op. On-chip would need a spiking
  normalization approximation.
- **Gap G7 — softmax is host-only.** As a function it hits the generic
  ComputeOp path (`:158-159`). No spiking softmax.
- **Gap G8 — GELU is approximated to ReLU.** GELU-as-module is absorbed and the
  `ActivationAdaptationTuner` blends it to ReLU before deployment
  (`activation_adaptation_tuner.py:82-83`); GELU-as-function (`F.gelu`) would be
  a host ComputeOp. Either way the deployed nonlinearity is ReLU-equivalent, an
  approximation of GELU.
- **Net:** a ViT block converts and the `forward` is faithful, but only its
  Linear projections (q/k/v/o, MLP fc1/fc2) land on-chip; attention scoring, LN,
  and softmax are host float. A genuinely-on-chip transformer is the largest WS2
  lift (needs G5–G7 spiking primitives), and is depth-bound by G4 under cascaded
  TTFS.

---

## (b) Keystone enable (A6)

### The exact enable value

The A6 keystone is the **conversion-policy** layer
(`src/mimarsinan/tuning/orchestration/conversion_policy.py`: propose → confirm →
escalate). It is **DEFAULT-OFF** and gated by a single config key:

```
conversion_policy : true        # a plain JSON boolean (the literal True)
```

- **Shape:** a boolean. Not a dict — the code reads it as
  `bool(config.get("conversion_policy", False))`
  (`conversion_policy.py:278`, `deployment_plan.py:319-340`,
  `ttfs_cycle_adaptation_tuner.py:103`). Any truthy value enables; the canonical
  on-value is JSON `true`.
- **Location:** inside `deployment_parameters` in the experiment JSON. The
  pipeline flattens `deployment_parameters` into `self.config`
  (`deployment_pipeline.py:65-66`), so `config["conversion_policy"]` reads it.
- **Default-off invariant:** unset (or `false`) ⇒ `ConversionDecision(enabled=
  False, driver=controller, characterized=False)` ⇒ byte-identical current
  behavior; the characterizer is never constructed
  (`conversion_policy.py:278-286`; `ttfs_cycle_adaptation_tuner.py:103-104`).

### What enabling it does on a cascaded mmixcore run

The live tuner path (`ttfs_cycle_adaptation_tuner.py`):

1. `_conversion_characterizer_for(tuner)` (`:92-109`) returns `None` unless
   `conversion_policy` is set AND the cell is **not synchronized**. For a
   cascaded mmixcore cell (`ttfs_cycle_schedule="cascaded"`, `_synchronized=
   False`) with the flag on, it **constructs the real `CascadeCharacterizer`**
   (`context=tuner.trainer`, `S=tuner._T`).
2. `contract.conversion_policy(config, model=…, characterizer=…, context=…)`
   (`:374-379`) → `ConversionPolicy.resolve` (`conversion_policy.py:266-312`):
   - **PROPOSE** `propose_recipe(mode_policy)` — for the cascaded fire-once cell
     (`does_conversion_health_calibration=True`) the recipe carries the
     two-residual hints (`train_s_hint=16`, `deploy_s_hint=32`,
     `staircase_ste=True`, `ste_mix=0.5`) and the four R1 probe assumptions
     (`conversion_policy.py:205-221`).
   - **CONFIRM** the `CascadeCharacterizer` runs the four forward-only probes
     (cold-cascade liveness / ramp monotonicity / staircase-vs-LIF ceiling /
     firing-gain) on the model (`characterization.py:122-329`).
   - **ESCALATE** on a probe mismatch → `escalate_to_controller` flips the
     driver to `controller` and carries the reason
     (`conversion_policy.py:235-245,303-311`).

So **yes**: enabling `conversion_policy: true` on a cascaded mmixcore run
constructs the `CascadeCharacterizer` and can escalate.

### The keystone-on validation config

`experiments/campaign/keystone_on_matrix_6.json` = the canonical matrix_6
cascaded template (`cert_fast_matrix_6_ttfs_cycle_cascaded.json`, model
`mlp_mixer_core`, `spiking_mode=ttfs_cycle_based`, `ttfs_cycle_schedule=cascaded`)
**with `"conversion_policy": true` added** inside `deployment_parameters`. It is
the only delta vs the proven matrix_6 template.

### Validation (parse + real characterizer builds) — PASSED

A no-full-run python check (config merged exactly as `DeploymentPipeline` does:
defaults + `deployment_parameters` + platform defaults + `platform_constraints`)
confirms:

- `[1]` JSON parses; `conversion_policy` is the literal `True`, located inside
  `deployment_parameters`; flattens to `config["conversion_policy"]=True`.
- `[2]` `SpikingDeploymentContract.from_pipeline_config(config).mode_policy()` →
  `TtfsCascadeModePolicy` (`is_cascaded=True`, `is_synchronized=False`,
  `needs_health=True`).
- `[2]` `contract.conversion_policy(config)` → `enabled=True`,
  `characterized=True`.
- `[2]` the real `CascadeCharacterizer(S=4)` constructs and `resolve(...)` RUNS
  its confirm pass (`enabled=True`, `characterized=True`).
- `[2]` the proposed recipe = `ttfs_cycle_based/cascaded`, `expects_health=True`,
  `train_s_hint=16`, `deploy_s_hint=32`, `staircase_ste=True`, `ste_mix=0.5`,
  assumptions `(cold_cascade_live, ramp_monotone, staircase_lif_ceiling,
  firing_gain)`.
- Toggle proof: with `conversion_policy=False` the same cell returns
  `enabled=False, characterized=False` (inert, byte-identical).
- Escalation proof: a deliberately-mismatching characterizer →
  `escalated=True`, `driver=controller`, `escalation_reason` carried.
- Gating proof: the `_conversion_characterizer_for` guards build a
  `CascadeCharacterizer` for the cascaded/non-sync cell with the flag on, and
  `None` with the flag off.

### How to run the validation

```bash
source env/bin/activate
PYTHONPATH="$PWD/src:$PYTHONPATH" python - <<'PY'
import json
from mimarsinan.config_schema.defaults import (
    get_default_deployment_parameters, get_default_platform_constraints)
raw = json.load(open("experiments/campaign/keystone_on_matrix_6.json"))
config = {}
config.update(get_default_deployment_parameters()); config.update(raw["deployment_parameters"])
config.update(get_default_platform_constraints()); config.update(raw["platform_constraints"])
from mimarsinan.chip_simulation.deployment_contract import SpikingDeploymentContract
from mimarsinan.tuning.orchestration.characterization import CascadeCharacterizer
from mimarsinan.tuning.orchestration.conversion_policy import ConversionPolicy
contract = SpikingDeploymentContract.from_pipeline_config(config)
mp = contract.mode_policy()
char = CascadeCharacterizer(context=None, n_batches=1, S=int(config["simulation_steps"]))
dec = ConversionPolicy.resolve(config, mode_policy=mp, model=None, characterizer=char, context=None)
assert config["conversion_policy"] is True
assert dec.enabled and dec.characterized
print("keystone ENABLED; real CascadeCharacterizer built+ran:", dec.recipe.name,
      "| assumptions=", dec.recipe.assumptions)
PY
```

The full pipeline run (the actual keystone-on deployment) is:

```bash
source env/bin/activate
python run.py --headless experiments/campaign/keystone_on_matrix_6.json
```
