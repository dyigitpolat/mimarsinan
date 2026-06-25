# AC Evidence — golden-standard absolute-AC table, populated from the campaign ledger

This file collects **per-(model, dataset, regime) absolute-AC evidence** harvested
from the autonomous GPU campaign (`runs/campaign/ledger.jsonl`), in the form the
§8 / §10 roadmap of `docs/mimarsinan_closeout_analysis_v2.md` demands: every cell
carries a **measured absolute verdict** against its **own** ANN reference, with the
deployment-validity gate and confounds stated inline.

**AC semantics** (per `src/mimarsinan/chip_simulation/certification.py`):
- **AC1** — absolute deployed-accuracy goal (`ac1_target`).
- **AC2** — *lossless*: deployed forward within `ε` of the ANN reference (`ac2_reference`).
  Here AC2 is read as the **deployed→ANN gap** (smaller = closer to lossless).
- **AC5** — per-fine-tuning-PASS wall budget (not the subject of these cells).

**Validity gate** (`src/mimarsinan/mapping/verification/onchip_majority.py`,
default-on): a deployment is VALID only when `on-chip params / total params ≥ 0.5`.
`deep_mlp` (all depths, w64) is **INVALID host-majority** (d4 19.7% / d8 36.4% on-chip)
and its rows are evidence of *phenomena*, not of valid deployments. `lenet5`
(99.1% on-chip), `deep_cnn` (98.5–99.5%), and `mlp_mixer` (90.1%) are **VALID**.

---

## 1. AC2 (lossless vs ANN) on the VALID `lenet5` CNN — cascaded TTFS S=4 (2026-06-24)

The `lenet5` cell is the **VALID on-chip-majority** vehicle for the WS3 cascaded
firing-gain question (vs the retired host-majority `deep_mlp`). n=1000 re-measure,
3 seeds, `ttfs_cycle_based` S=4. Ledger: `cluster:"WS3"`, `kind:"arch_dataset"`,
`model:"lenet5"`.

| model | dataset | regime | deployed (cascaded, 3-seed mean ± sd_pp) | ANN ref (AC2 target) | AC2 deployed→ANN gap | validity | AC2 verdict |
|:------|:--------|:-------|:-----------------------------------------|---------------------:|---------------------:|:---------|:------------|
| lenet5 | MNIST | cascaded TTFS S=4 | **0.9873** (±0.31) | 0.9912 | **0.39pp** | VALID (99.1% on-chip) | **near-lossless** (gap ≈ seed noise) |
| lenet5 | FashionMNIST | cascaded TTFS S=4 | **0.8397** (±1.03) | 0.9183 | **7.86pp** | VALID (99.1% on-chip) | **lossy** (real firing-gain residual) |

- **MNIST** — the apparent cascade deficit was 50-sample rounding noise; at n=1000 the
  cascaded forward tracks the ANN within 0.39pp (cascaded→synchronized gap 0.18pp <
  seed std 0.31pp). **AC2 effectively MET** on a VALID CNN.
- **FashionMNIST** — the cascade carries a **real 7.86pp deployed→ANN gap** that barely
  moved (−0.03pp) from the n=50 baseline. **AC2 NOT MET**; a genuine firing-gain residual
  on a VALID CNN, hardening the architecture×dataset-dependence of the death cascade.

Run ids: `csr_lenet_{MNIST,FashionMNIST}_DataProvider_cascaded_n1000_s{0,1,2}`.
**Confound:** the synchronized comparison arm is the recorded n=50 `sync_full` tag (no
paired n=1000 synchronized re-run), so the cascaded→synchronized gap mixes resolutions;
the cleanest within-arm check is the cascaded n50→n1000 shift (MNIST +0.73pp washes the
gap to noise; FMNIST −0.03pp stays). The cascaded n=1000 deployed→ANN gaps above are
clean (all rc=0, ANN ≫ chance).

---

## 1b. Paired cascaded-vs-synchronized lenet5/MNIST — full-test SCM closes the §1 mixed-resolution confound (2026-06-24)

§1 left one open confound: the cascaded→synchronized comparison mixed an n=1000
cascaded arm against an n=50 synchronized arm. This **paired** batch removes it —
both modes run on the **same** lenet5/MNIST/`ttfs_cycle_based`/S=4 vehicle, 3
seeds each, and both pipelines log the **full-test-set SCM identity-mapped
accuracy** (all 10000 samples, identical instrument). That is the apples-to-apples
**AC2 firing-gain** quantity.

| model | dataset | regime | metric | cascaded (3-seed mean) | synchronized (3-seed mean) | cascaded→sync gap | cascaded→ANN gap (AC2) | validity | AC2 verdict |
|:------|:--------|:-------|:-------|-----------------------:|---------------------------:|------------------:|-----------------------:|:---------|:------------|
| lenet5 | MNIST | cascaded vs sync, S=4 | full-test SCM identity | **0.9835** | **0.9891** | **0.56pp** | **+0.78pp** | VALID (99.1% on-chip) | **near-lossless** (no death-cascade) |

- **AC2 effectively MET on a VALID CNN.** Cascaded deploys at 0.9835 full-test SCM,
  only 0.56pp below synchronized and within 0.78pp of its own ANN (~0.991). Cascaded
  full-test **SCM == HCM (0.9846, s0)** ⇒ mapping lossless, the sub-pp loss is
  mode-intrinsic. This is the clean paired control §3 asked for, on the firing-gain
  axis: **no depth-driven firing-gain collapse on a real, well-trained convnet.**

Run ids: `ws3cnn_lenet5_{cascaded,synchronized}_s{0,1,2}` (ledger `cluster:"WS3"`,
`kind:"cnn_mode_compare"`). **Confound:** the bare `__target_metric.json` floats are
asymmetric — cascaded's (0.98/1.0/0.96) is a genuine cascaded nevresim sim on only
**50/10000** subsampled samples, synchronized's is the full-test SCM value; the
naïve raw-target gap (0.91pp) is ±2% subsample-quantization noise and is **not** the
headline. depth = 5 trainable layers but IR max-latency=3 / 2 neural segments, so the
depth-axis stress is modest (a deeper convnet would test depth harder).

---

## 1c. AC2 on the VALID `deep_cnn` (d5) and the lenet5 KMNIST cell — the no-collapse law extends, KMNIST is mild (2026-06-24)

Two more VALID-vehicle AC2 cells. **`deep_cnn` d5** extends the within-CNN depth
ladder one rung past the §4c d4 cell; **`lenet5`/KMNIST** completes the 4-dataset CNN
table at n=1000. Both are paired cascaded-vs-synchronized (deep_cnn) or
cascaded-vs-(n50)-synchronized (lenet5). Ledger: `cluster:"WS3"`, `kind:"arch_dataset"`.

| model | dataset | regime | deployed (cascaded, 3-seed mean) | ANN ref (AC2 target) | AC2 deployed→ANN gap | casc→sync gap | validity | AC2 verdict |
|:------|:--------|:-------|:---------------------------------|---------------------:|---------------------:|--------------:|:---------|:------------|
| deep_cnn | MNIST (d5) | cascaded TTFS S=4 | **0.9917** (.99/.99/.995) | 0.9913 | **−0.04pp** | **+0.07pp** | VALID (98.5–99.5% on-chip) | **near-lossless** (no within-CNN cascade through d5) |
| lenet5 | KMNIST | cascaded TTFS S=4 (n=1000) | **0.934** (±0.73) | 0.9646 | **3.06pp** | +1.45pp | VALID (99.1% on-chip) | **mild** (small firing-gain residual, dataset-stable) |

- **deep_cnn d5** — cascaded tracks its **own** ANN to −0.04pp and the cascaded→sync
  gap is +0.07pp (within seed sd). **AC2 effectively MET.** With the §4c d4 cell
  (−0.15pp) the within-CNN gap is flat/near-zero at **both** rungs — the no-collapse
  law extends d4→d5 on a VALID convnet, in sharp contrast to the (INVALID) deep_mlp
  death-cascade. d6/d7 still untested (prior d6 crashed rc=1) so a *deeper* within-CNN
  cascade is unproven beyond d5.
- **lenet5/KMNIST** — a **3.06pp deployed→ANN gap > seed sd 0.73pp**: a small-but-real
  firing-gain residual that places KMNIST **between** the lossless MNIST (§1) and the
  lossy FMNIST (§1, 7.86pp) on the dataset-hardness axis. **AC2 partially MET (mild).**
  The n=1000 re-measure LIFTED +4.07pp over the round-1 n=50 cascaded mean, so most of
  the round-1 5.52pp gap was subsample quantization, not real loss.

Run ids: `pdcnnladder_d5_{cascaded,synchronized}_s{0,1,2}` (deep_cnn),
`csr_lenet_KMNIST_DataProvider_cascaded_n1000_s{0,1,2}` paired with the finalized n=50
`sch_lenet_KMNIST_DataProvider_synchronized_s{0,1,2}` (lenet5). **Confounds:** deep_cnn
`max_simulation_samples=200` (read the ~0pp gap, not third decimals; cascaded s2=1.0
is small-N variance); lenet5 KMNIST mixes n1000-cascaded against n50-synchronized (the
sync arm is the finalized n=50 run), and the companion SVHN cascaded@n1000 arm crashed
rc=1 this round. All cascaded runs rc=0, ANN ≫ chance (not untrained).

---

## 1d. AC2 on the VALID `deep_cnn` depth ladder — the death-cascade IS real and depth-driven on a CNN (d6→d12) (2026-06-24)

§1c reported the within-CNN cascade *flat* through d5. The deeper ladder (d6, d8, d10,
d12) now **closes** the §3 open gap and **reverses** the §1c "no within-CNN cascade"
reading: the cascaded AC2 deployed→ANN gap is depth-driven and blows out by d10. All
cells: `deep_cnn` (width 16), MNIST, `ttfs_cycle_based` S=4, 3 seeds, paired by seed,
`max_simulation_samples=200`. Ledger: `cluster:"WS3"`, `kind:"depth_firing_gain"`.

| model | dataset (depth) | deployed (cascaded, 3-seed mean) | ANN ref (AC2 target) | **AC2 casc→ANN gap** | casc→sync gap | sync→ANN gap | AC2 verdict |
|:------|:----------------|:---------------------------------|---------------------:|---------------------:|--------------:|-------------:|:------------|
| deep_cnn | MNIST (d6)  | **0.9664** (.976/.977/.946) | 0.9917 | **2.69pp** | +2.49 | −0.11 | mild (gap emerges) |
| deep_cnn | MNIST (d8)  | **0.9564** (.964/.965/.940) | 0.9928 | **3.80pp** | +3.40 | +0.08 | mild (widens) |
| deep_cnn | MNIST (d10) | **0.8531** (.928/.856/.775) | 0.9897 | **13.66pp** | +13.86 | −0.20 | **NOT MET (death-cascade)** |
| deep_cnn | MNIST (d12) | **0.8780** (.884/.848/.902) | 0.9921 | **11.41pp** | +11.43 | −0.02 | **NOT MET (death-cascade)** |

- **AC2 on cascaded breaks with depth on a VALID convnet.** The deployed→ANN gap widens
  monotonically (d4 0.47 → d5 −0.04 → d6 2.69 → d8 3.80 → **d10 13.66 → d12 11.41pp**):
  near-lossless through d5, mild through d8, then a sharp **death-cascade** at d10/d12
  where cascaded drops to 0.85/0.88 with high seed variance while **synchronized holds
  the ANN ceiling at every depth** (sync→ANN ≤0.20pp, sd ≤0.15pp). This is the
  closeout-v2 §6 depth × firing-gain risk **observed on a real, well-trained CNN** —
  *correcting* §1b/§1c, which only reached the shallow rungs where the cascade had not
  yet onset. ANN ≫ chance at every depth ⇒ genuine firing-gain, not untrained-floor.
- **Synchronized AC2 is effectively MET at every depth** (sync→ANN gap ≤0.20pp), so the
  schedule recommendation is unchanged: synchronized is the unconditional deep CNN
  default; cascaded is depth-risky and should carry the gate-fix or be retired deep.

Run ids: `dcnn_d{6,8}_{cascaded,synchronized}_s{0,1,2}` (d6/d8),
`pdcnndeep_d{10,12}_{cascaded,synchronized}_s{0,1,2}` (d10/d12). **Confound (dominant —
validity):** *all* queue-recorded runs finalized `returncode==1` (in `q/failed/`, none in
`done/`), so by the strict `returncode==0` rule **none is a formally valid deployment**.
The crash is a downstream **`HardCoreMappingStep` "No more hard cores available"** chip
**capacity/packing** failure, raised **after** SoftCoreMapping wrote `__target_metric.json`
and **after** its parity gates passed (NF↔SCM cascaded agreement 1.0, torch↔sim parity
0.9961–1.0). The deployed values are the genuine full-test-set SCM accuracies captured
pre-crash (each matches the log "Test accuracy" line) — a real firing-gain result on a
trainable vehicle, but **not a clean finalized on-chip deployment**. The `d12_cascaded`
seeds have no queue JSON at all (`returncode==None`). `max_simulation_samples=200` → read
the 10+pp d10/d12 gaps (robust), not third decimals; cascaded d6 s2=0.946 / d8 s2=0.94 are
small-N outliers. **A `cores_config`-enlarged (or coalescing-on) re-run to clear the
packing crash and finalize `rc=0` is the one step that turns this into clean VALID
evidence** (proposed: WS3 `plan_stage:14`).

---

## 1e. AC2 on the VALID `deep_cnn` dataset axis (d4, d8 × FMNIST/KMNIST) — the cascade re-opens off MNIST and widens with task margin (2026-06-24)

§1d closed the deep_cnn cascade *depth* ladder on **MNIST**. This adds the **dataset
axis** at two depths: `deep_cnn` (width 16, S=4, `ttfs_cycle_based`) at **d4 and d8** on
**FashionMNIST and KMNIST**, paired cascaded-vs-synchronized, 3 seeds/arm. Ledger:
`cluster:"WS3"`, `kind:"arch_dataset"`, `model:"deep_cnn"`.

| model | dataset (depth) | deployed (cascaded, 3-seed mean) | ANN ref (AC2 target) | **AC2 casc→ANN gap** | casc→sync gap | sync→ANN gap | validity | AC2 verdict |
|:------|:----------------|:---------------------------------|---------------------:|---------------------:|--------------:|-------------:|:---------|:------------|
| deep_cnn | FashionMNIST (d4) | **0.8700** (±0.71) | 0.9276 | **5.76pp** | +3.90 | 1.86 | VALID `rc=0` | lossy (degraded) |
| deep_cnn | KMNIST (d4)       | **0.8867** (±1.31) | 0.9684 | **8.17pp** | +6.19 | 1.99 | VALID `rc=0` | lossy (degraded) |
| deep_cnn | KMNIST (d8)       | **0.9153** (±1.59) | 0.9684 | **5.23pp** | +4.96 | 0.42 | `rc=1` pre-crash SCM | lossy (degraded) |
| deep_cnn | FashionMNIST (d8) | **0.7802** (±1.07) | 0.9328 | **15.36pp** | +11.98 | 3.18 | `rc=1` pre-crash SCM | **NOT MET (collapse)** |

- **The d4 "no-collapse" MNIST corner (§1c, −0.04pp) does NOT generalize.** On the *same*
  shallow `deep_cnn` d4 the cascaded AC2 deployed→ANN gap re-opens to **5.76pp (FMNIST)**
  and **8.17pp (KMNIST)** — the cascade is gated by **dataset margin**, not just depth.
  The d4 cells are clean (`rc=0`, ANN ≫ chance), so this is firmly VALID AC2 evidence.
- **At d8 the cascade widens with dataset margin** (MNIST §1d 3.80 < KMNIST 5.23 <
  **FMNIST 15.36pp**), and **synchronized AC2 stays effectively MET on every cell**
  (sync→ANN ≤3.18pp, ≤0.42pp on KMNIST). The depth × dataset compound is the worst case
  (deep × hard = FMNIST d8, 15.36pp), exactly as §4b's law predicts — now on a CNN.

Run ids: `pdcnndata_{FashionMNIST,KMNIST}_DataProvider_{cascaded,synchronized}_s{0,1,2}`
(d4), `pdcnn_{FashionMNIST,KMNIST}_DataProvider_{cascaded,synchronized}_s{0,1,2}` (d8).
**Confound (validity split by depth):** the **d4** runs all finalized `rc=0` (clean). The
**d8** runs carry the **same `NON_FINALIZED_rc1` infra-crash confound as the §1d MNIST CNN
cells** — all finalized `returncode==1` at the downstream `HardCoreMappingStep` "No more
hard cores available" (`features_13`) *after* SoftCoreMapping wrote `__target_metric.json`
and its parity gates passed, so d8 deployed = genuine pre-crash full-test SCM accuracy
(each == its log "Test accuracy" line), **not** a clean finalized deployment. The
enlarged-`cores_config` re-run (`plan_stage:14`) would lift the d8 cells to clean `rc=0`.
`max_simulation_samples=200` → read the gaps (3.9–12pp robust), not third decimals; the
cascaded arm carries wide seed sd (1.07–1.59pp) vs synchronized's 0.24–0.53pp.

---

## 1f. AC2 on a CLEAN-FINALIZED (`rc=0`) `deep_cnn` depth ladder — the death-cascade is a depth-THRESHOLD (lossless ≤d5, ~5pp deficit ≥d6) (2026-06-24)

§1d reported the deep_cnn cascade depth law but on a **`rc=1`-confounded** vehicle (all
`dcnn_`/`pdcnndeep_` d6–d12 runs crashed at `HardCoreMappingStep` "No more hard cores
available" *after* the SCM metric was written). This batch lands the same law on a
**VALID, clean-finalized** `deep_cnn` vehicle — and sharpens it: the onset is a
**threshold**, not the deep_mlp smooth widening. All cells: `deep_cnn` (width 16), MNIST,
`ttfs_cycle_based` S=4, 3 seeds (d8 cascaded: 2), paired by seed; cascaded
`max_simulation_samples=200`, synchronized FULL 10k test set. Ledger: `cluster:"WS3"`,
`kind:"depth"`, `model:"deep_cnn"`.

| model | dataset (depth) | vehicle | deployed (cascaded, mean) | ANN ref (AC2 target) | **AC2 casc→ANN gap** | casc→sync gap | sync→ANN gap | validity | AC2 verdict |
|:------|:----------------|:--------|:--------------------------|---------------------:|---------------------:|--------------:|-------------:|:---------|:------------|
| deep_cnn | MNIST (d4) | `dcnn_`        | **0.9883** | 0.9931 | **0.47pp** | −0.15 | 0.33 | VALID `rc=0` | near-lossless (tied) |
| deep_cnn | MNIST (d5) | `pdcnnladder_` | **0.9917** | 0.9913 | **−0.04pp** | −0.07 | −0.11 | VALID `rc=0` | near-lossless (tied) |
| deep_cnn | MNIST (d6) | `pdcnnbc_`     | **0.9383** | 0.9923 | **5.39pp** | −5.21 | 0.18 | VALID `rc=0` | **NOT MET (sharp onset)** |
| deep_cnn | MNIST (d8) | `pdcnnbc_`     | **0.9425** | 0.9925 | **5.00pp** | −5.10 | −0.10 | VALID `rc=0` (n=2) | **NOT MET (~5pp plateau)** |

- **AC2 on cascaded is a depth-THRESHOLD on a clean `rc=0` convnet.** Cascaded tracks its
  own ANN within 0.47pp (d4) / −0.04pp (d5) — **AC2 effectively MET through d5** — then
  the deployed→ANN gap jumps to **5.39pp (d6)** and **5.00pp (d8)**: a sharp d5→d6 onset
  followed by a **~5pp plateau**, qualitatively unlike the deep_mlp **smooth**
  d4(4.3)→d8(9.3pp) widening, and *milder* than §1d's `rc=1` d10/d12 (~11–14pp). This is
  the **closeout-v2 §6 depth × firing-gain risk on a VALID, clean-finalized vehicle**:
  the risk is REAL but its severity is **bounded to ~5pp** over d6–d8 on deep_cnn.
- **Synchronized AC2 is effectively MET at every depth** (sync→ANN ≤0.18pp, sd ≤0.24pp),
  so the schedule recommendation is unchanged: synchronized is the unconditional deep CNN
  default; cascaded carries a bounded ~5pp depth-risk past d5.

Run ids: `dcnn_d4_{cascaded,synchronized}_s{0,1,2}` (d4), `pdcnnladder_d5_{...}_s{0,1,2}`
(d5), `pdcnnbc_d{6,8}_{cascaded,synchronized}_s{0,1,2}` (d6/d8; d8 cascaded = s0,s2 only).
**Confounds:** (1) **EVAL-SET MISMATCH** — cascaded subsamples to n=200 (0.005 grid,
~1.5–3.5pp/seed noise) while synchronized reports FULL 10k (read gaps, not third decimals;
the d6/d8 ~5pp gaps are >2× the noise band, the d4/d5 sub-0.2pp gaps are within noise). (2)
**`rc=1` runs EXCLUDED** — the `dcnn_`/`pdcnnladder_` d6–d8 runs crashed in hard-core
packing *before* deploying (stale `__target_metric.json` = pre-deployment training metric,
NOT used), so valid d6/d8 evidence is from `pdcnnbc_`; the valid ladder is a mild
cross-vehicle composite (3 deep_cnn families, all MNIST/w16, on-chip 98.9–99.5%). (3) **d8
cascaded = 2 finalized seeds** (s1 still in `q/running/`), sd 2.75pp. (4) **No at-chance
confound** — all ANN ~0.99 (chance 0.10), parity gates NF↔SCM=1.0 / torch↔sim=1.0 ⇒
genuine firing-gain deficit. **This is the clean `rc=0` upgrade of §1d's depth law.** Next:
finalize d8 s1 and push d10/d12 on the `pdcnnbc_` bigger-cores vehicle (proposed: WS3
`plan_stage:19`).

---

## 1g. The clean `rc=0` ladder reaches d10/d12 + genuine n=1000 resolution — the deficit is a BOUNDED ~4–5pp plateau, synchronized lossless through d12 (2026-06-24)

This batch closes the §1f/§3 "d10/d12 still only on the `rc=1` vehicle" open gap by landing
the deep rungs on the **clean `rc=0` `pdcnnbc_` bigcores (480/480, 4×-enlarged cores)**
vehicle, and adds a **genuine high-resolution** read (nevresim **n=1000**, 5× the n=200
ladder). It also **completes the §1f d8 cascaded cell to 3 seeds** (this batch's
`pdcnnbc_d8_cascaded_s1` = 0.95, `rc=0` → mean 0.9450, sd 2.27pp, superseding the n=2
0.9425). All cells: `deep_cnn` (w16), MNIST, `ttfs_cycle_based` S=4, 3 seeds, paired by
seed. Ledger: `cluster:"WS3"`, `kind:"depth"`, `model:"deep_cnn"`.

| model | dataset (depth) | vehicle | res | deployed (cascaded, mean) | ANN ref | casc→sync gap | sync→ANN | validity | AC2 verdict |
|:------|:----------------|:--------|:----|:--------------------------|--------:|--------------:|---------:|:---------|:------------|
| deep_cnn | MNIST (d8)  | `pdcnnbc_`       | n200  | **0.9450** (.97/.95/.915) | 0.9929 | **−4.85** | −0.06 | VALID `rc=0` (n=3) | NOT MET (~5pp plateau) |
| deep_cnn | MNIST (d10) | `pdcnnbc_`       | n200  | **0.9517** (.925/.945/.985) | 0.9923 | **−4.00** | −0.06 | VALID `rc=0` | NOT MET (bounded plateau) |
| deep_cnn | MNIST (d12) | `pdcnnbc_`       | n200  | 0.98 (s1 only) | 0.9887 | −1.17 † | −0.30 | sync `rc=0`; casc **UNMEASURED n=1** | sync MET; casc open |
| deep_cnn | MNIST (d8)  | `pdcnndeeppair_` | **n1000** | **0.9066** (.930/.964/.826) | 0.9918 | **+8.51** | −0.01 | `rc=1` ‡ | NOT MET (confounded) |
| deep_cnn | MNIST (d10) | `pdcnndeeppair_` | **n1000** | **0.8807** (.738/.926/.978) | 0.9932 | **+11.14** | −0.11 | `rc=1` ‡ | NOT MET (confounded) |

- **The d10 death-cascade gap SHRINKS ~10pp on the clean vehicle.** §1d's `rc=1`
  `pdcnndeep_d10` read was ~13.86pp; the clean `rc=0` `pdcnnbc_d10` read is **−4.00pp** —
  the prior gap was inflated by the post-metric packing crash + cross-vehicle comparison.
  The real cascaded deficit is a **BOUNDED ~4–5pp plateau, NOT a deepening collapse**
  (d6 5.39 / d8 4.85 / d10 4.00pp).
- **Synchronized AC2 is LOSSLESS through d12** (d12 sync 0.9917 vs ANN 0.9887, +0.30pp,
  3 seeds `rc=0`, sd 0.07pp) — synchronized owns deep deployment at every measured depth.
- **(†) d12 cascaded is UNMEASURED:** only `pdcnnbc_d12_cascaded_s1` finalized `rc=0`
  (0.98); s0/s2 are `returncode=-9` (killed). The −1.17pp gap is an n=1 point — **OPEN:
  re-run d12 cascaded s0/s2.**
- **(‡) Resolution HARDENS the law, it does not shrink it.** The genuine n=1000 reads
  (8.51 / 11.14pp) are LARGER than the clean n=200 reads — the gap is not a grid artifact.
  The n=1000 `pdcnndeeppair_` runs are `rc=1` (same documented post-metric
  `HardCoreMappingStep` crash AFTER `__target_metric.json` + NF↔SCM 1.0 + torch↔sim 1.0
  were written) → read as **CONFIRMED-WITH-CONFOUND** under the `pdcnndeep_`/`dcnn_`
  precedent. The d10 s0=0.7375 log shows a genuine mid-pipeline SCM collapse
  (0.9939 ANN → 0.1873 → 0.7375) — death-cascade fragility, not noise.

Run ids: `pdcnnbc_d{8,10}_{cascaded,synchronized}_s{0,1,2}`, `pdcnnbc_d12_cascaded_s1` +
`pdcnnbc_d12_synchronized_s{0,1,2}` (n200); `pdcnndeeppair_d{8,10}_{cascaded,synchronized}_n1000_s{0,1,2}`
(n1000). **Confounds:** (1) cascaded n200/n1000 vs sync FULL 10k — read gaps not third
decimals; all reported gaps are >2× the per-seed binomial band. (2) the n1000 vehicle is
`rc=1` (genuine pre-crash SCM, CONFIRMED-WITH-CONFOUND). (3) no at-chance confound — all
ANN ~0.99 ≫ chance 0.10. **This is the clean `rc=0` upgrade + high-resolution cross-check
of §1d's deep rungs; the only remaining open cell is d12 cascaded (n=1).**

---

## 1h. AC2 on the VALID `deep_cnn` deep × hard CORNER (d10 × FMNIST/KMNIST, `rc=0`) — the death-cascade WORST-CASE: 16–18pp deployed→ANN gaps (2026-06-24)

§1e opened the deep_cnn dataset axis at d4/d8 but left the **d10 collapse rung on the
harder datasets OPEN** — the deep × hard compound's worst case. This lands it on the
**clean `rc=0` enlarged-`bigcores` (count=480, `plan_stage:14`) vehicle**: `deep_cnn`
(width 16, S=4, `ttfs_cycle_based`) at **d10** on **FashionMNIST and KMNIST**, paired
cascaded-vs-synchronized. Ledger: `cluster:"WS3"`, `kind:"arch_dataset"`,
`model:"deep_cnn"`.

| model | dataset (depth) | deployed (cascaded mean) | ANN ref (AC2 target) | **AC2 casc→ANN gap** | casc→sync gap | sync→ANN gap | validity | AC2 verdict |
|:------|:----------------|:-------------------------|---------------------:|---------------------:|--------------:|-------------:|:---------|:------------|
| deep_cnn | FashionMNIST (d10) | **0.7250** (±1.50, n=2) | 0.9347 | **20.97pp** | +17.91 | 3.06 | VALID `rc=0` | **NOT MET (collapse)** |
| deep_cnn | KMNIST (d10)       | **0.8025** (±3.25, n=2) | 0.9663 | **16.38pp** | +15.98 | 0.40 | VALID `rc=0` | **NOT MET (collapse)** |

- **The deep × hard worst case is the largest cascaded AC2 deficit in the whole deep_cnn
  table.** On FMNIST the cascaded deployed→ANN gap widens **monotonically with depth** —
  d4 5.76pp (§1e) → d8 15.36pp → **d10 20.97pp** — and the casc→sync gap (the firing-gain
  signal) climbs **+3.90 → +11.98 → +17.91pp**. On KMNIST the casc→sync gap has a d8 dip
  (+4.96) but **blows out to +15.98pp at d10**.
- **Synchronized AC2 stays MET at d10** (sync→ANN 0.40pp KMNIST, 3.06pp FMNIST) — it owns
  deep × hard deployment. The closeout §6 "prefer synchronized for deep models" ruling is
  **reinforced** on the worst corner: cascaded is **strictly dominated by ~16–18pp**.

Run ids: `pdcnnd10data_{FashionMNIST,KMNIST}_DataProvider_{cascaded,synchronized}_s{0,1,2}`.
**Confounds.** (1) **cascaded n=2 (not 3):** FMNIST cascaded s1 `rc=-9` (killed) and KMNIST
cascaded s0 `rc=1`, both `q/failed/` → excluded; synchronized arms are full 3 seeds; KMNIST
cascaded sd is wide (3.25pp). (2) **eval-set mismatch:** cascaded `max_simulation_samples=200`
(deployed = exact 1/200 multiples, e.g. FMNIST casc_s0 0.71 ≈ HCM 0.7087) vs synchronized
FULL 10k test set → **read the 16–18pp gaps, not 3rd decimals** (>4–5× the per-seed binomial
band). (3) **no at-chance confound:** ANN refs ~0.935/0.966 ≫ chance 0.10 → genuine
firing-gain death-cascade. (4) **VALIDITY upgrade:** all 10 done runs are `rc=0` reaching
`HardCoreMappingStep` *without* the "No more hard cores" crash — the `bigcores` config clears
the §1e d8 `NON_FINALIZED_rc1` confound, so these are **CLEAN FINALIZED `rc=0`** deployments
(`FINALIZED_rc0`). **The §3 open gap "d10 collapse rung on the harder datasets is still
open" is now CLOSED.** Next: a 3rd cascaded seed + the firing-gain gate-fix on this d10
deep × hard cell (`plan_stage:24`).

---

## 1s. The genuine n=1000 deep ladder is CLOSED on a clean `rc=0` vehicle d6→d12 — the §1g `rc=1`-n1000 confound AND the §1g d12-cascaded-`n=1` gap are both resolved; deficit is depth-MONOTONE (extends §1r to d6/d12) (`item_id=dcnn_mnist_depth_deathcascade_n1000`, 2026-06-25)

§1g landed the genuine n=1000 d8/d10 reads only on the `rc=1`-confounded `pdcnndeeppair_`
vehicle and left **d12 cascaded UNMEASURED (n=1, †)**. This item closes both gaps on the
clean `rc=0` `pdcnnbcn1000*` vehicle (`cores.count=480`): n=1000 d6/d8/d10 paired arms +
a FIRMED 3-seed d12 cascaded (`pdcnnbcd12fin`). `deep_cnn` (w16), MNIST, `ttfs_cycle_based`
S=4. Ledger: `cluster:"WS3"`, `kind:"depth"` (4 rungs) + `kind:"synthesis"`.

| model | dataset (depth) | res | deployed (cascaded mean ± sd) | ANN ref (AC2 target) | **casc→sync gap** | sync→ANN | n | validity | AC2 verdict |
|:------|:----------------|:----|:------------------------------|---------------------:|------------------:|---------:|--:|:---------|:------------|
| deep_cnn | MNIST (d6)  | n1000 | **0.9563** ± 1.16pp | 0.9914 | **+3.61** | −0.10 | 3 | VALID `rc=0` | NOT MET (onset) |
| deep_cnn | MNIST (d8)  | n1000 | **0.9450** ± 2.09pp | 0.9914 | **+4.62** | −0.06 | 6 | VALID `rc=0` | NOT MET (widens) |
| deep_cnn | MNIST (d10) | n1000 | **0.9328** ± 3.76pp | 0.9924 | **+5.84** | +0.08 | 6 | VALID `rc=0` | NOT MET (widens) |
| deep_cnn | MNIST (d12) | n200  | **0.8967** ± 4.40pp | 0.9916 | **+9.48** | −0.15 | 3 | VALID `rc=0` | **NOT MET (collapse)** |

- **The §1g d12-cascaded `n=1` open cell is CLOSED.** d12 cascaded is now a FIRMED 3-seed
  cell (0.835/0.92/0.935, mean 0.8967) on the clean `rc=0` `pdcnnbcd12fin` family — the
  `rc=−9` OOM-killed `pdcnnbcclean_d12_cascaded` seeds are NOT used.
- **The §1g `rc=1`-n1000 confound is CLOSED.** d6/d8/d10 are clean `rc=0` at n=1000 on BOTH
  arms; the deficit is **depth-MONOTONE** (3.61→4.62→5.84→9.48pp), reconciling §1g's two-point
  `pdcnndeeppair_` d10>d8 read by pooling the `plat_*`+`seed_*` families (which agree).
- **Synchronized AC2 is LOSSLESS at every rung** (`|sync→ANN| ≤ 0.15pp`, sd ≤ 0.15pp) — it
  owns deep deployment through d12. Cascaded is strictly dominated and the deficit deepens.

Run ids: `pdcnnbcn1000plat_d{6,8,10}_{cascaded,synchronized}_s{0,1,2}`,
`pdcnnbcn1000seed_d{8,10}_{cascaded,synchronized}_s{3,4,5}`,
`pdcnnbcd12fin_cascaded_s{0,1,2}`, `pdcnnbcclean_d12_synchronized_s{0,1,2}` (+ n1000 d12 sync
cross-check `pdcnnbcn1000seed_d12_synchronized_s{3,4,5}`, mean 0.9920 → +9.54pp). **Confounds.**
(1) **d12 resolution split:** d12 is n200 on both arms (coarser than the d6–d10 n1000 rungs)
— read the +9.48pp as a gap; the n1000 sync cross-check gives a near-identical +9.54pp
(resolution-robust). (2) **pooled 6-seed d8/d10:** the `plat`/`seed` families agree, only
strengthening n. (3) **high cascaded variance at deep rungs** (d10 sd 3.76pp incl. an s2=0.99
outlier; d12 sd 4.40pp) — every gap is ≫ 2× the per-seed band. (4) **no at-chance confound**
— every ANN ≈ 0.99 ≫ chance 0.1135 → genuine firing-gain. **This is the clean `rc=0` +
high-resolution upgrade that turns §1g's `rc=1`-n1000 and d12-`n=1` cells into formally VALID
AC2 evidence; the deep_cnn depth death-cascade is now fully closed d4→d12 at n1000.** The
firing-gain gate-fix remains REFUTED as a deep auto-rescue (§2d/§2e) — synchronized is the
unconditional deep default.

---

## 1j. AC2 on the VALID `deep_cnn` d8 dataset cells — CLEAN `rc=0` `bigcores` re-run closes the §1e confound (2026-06-24)

§1e opened the deep_cnn dataset axis at d4/d8 but the **d8 cells carried a
`NON_FINALIZED_rc1` infra-crash confound** (all finalized `returncode==1` at
`HardCoreMappingStep` "No more hard cores available" *after* the SCM metric was written —
genuine pre-crash reads, not formally valid deployments). This lands the **same d8
FMNIST/KMNIST cells on the enlarged `bigcores` (count=480, 4×-enlarged cores,
`plan_stage:17`) vehicle** so they finalize **`rc=0`** — the d8 analog of the §1g MNIST
ladder and §1h d10 dataset rung. `deep_cnn` (width 16, S=4, `ttfs_cycle_based`), paired
cascaded-vs-synchronized. Ledger: `cluster:"WS3"`, `kind:"arch_dataset"`, `model:"deep_cnn"`.

| model | dataset (depth) | deployed (cascaded mean) | ANN ref (AC2 target) | **AC2 casc→ANN gap** | casc→sync gap | sync→ANN gap | validity | AC2 verdict |
|:------|:----------------|:-------------------------|---------------------:|---------------------:|--------------:|-------------:|:---------|:------------|
| deep_cnn | FashionMNIST (d8) | **0.7900** (±2.86, n=3) | 0.933 | **14.28pp** | +11.34 | 2.98 | VALID `rc=0` | **NOT MET (collapse)** |
| deep_cnn | KMNIST (d8)       | **0.8900** (±1.0, n=2)  | 0.9689 | **8.11pp** | +7.19 | 0.55 | VALID `rc=0` | lossy (degraded) |

- **The §1e d8 `rc=1` reads are CONFIRMED on the clean vehicle.** FMNIST casc→sync
  +11.98 → **+11.34pp**; KMNIST (the §1e +4.96/+5.23pp `rc=1` 3-seed) → **+7.19pp** on
  the clean 2-seed cascaded arm. The dataset-margin ordering at d8 is
  **MNIST +3.40 (§1f) < KMNIST +7.19 < FMNIST +11.34pp** — the closeout §10.1 dataset-margin
  death-cascade law, now off the INVALID deep_mlp onto a VALID, clean-finalized convnet.
- **Synchronized AC2 stays effectively MET** (sync→ANN 2.98pp FMNIST, 0.55pp KMNIST) — the
  unconditional deep-model default holds. With §1j (d8) + §1h (d10) both `rc=0`, the entire
  **d4/d8/d10 × {FMNIST,KMNIST} dataset-axis cube is now VALID `rc=0`**, and FMNIST widens
  monotonically with depth (+3.90 → +11.34 → +17.91pp).

Run ids: `pdcnnd8databc_{FashionMNIST,KMNIST}_DataProvider_{cascaded,synchronized}_s{0,1,2}`.
**Confounds.** (1) **KMNIST cascaded n=2 (not 3):** the third seed
`pdcnnd8databc_KMNIST_DataProvider_cascaded_s2` is still in `q/running/` (NON-FINALIZED)
and excluded per the strict `rc==0` rule; FMNIST cascaded + both sync arms are full 3-seed.
(2) **eval-set mismatch:** cascaded `max_simulation_samples=200` (FMNIST 0.83/0.765/0.775
carries small-N variance, sd 2.86pp) vs synchronized FULL 10k → **read the 7–11pp gaps,
not 3rd decimals**; deployed = the bare float in `__target_metric.json` (200-sample SCM),
slightly off each log's final "Test accuracy" line (FMNIST_casc_s0 target 0.83 vs
log-last 0.7989) — the `__target_metric.json` convention governs. (3) **no at-chance
confound:** ANN refs ~0.933/0.969 ≫ chance 0.10 → genuine firing-gain. (4) **VALIDITY
upgrade:** all 11 finalized runs `rc=0` reaching `HardCoreMappingStep` *without* the
crash — `bigcores` clears the §1e d8 `NON_FINALIZED_rc1` confound (`VALID_on_chip_majority_rc0`).
**The §3 open gap "the d8 cells share the §1d `rc=1` pre-crash-SCM confound (need the
`plan_stage:14` enlarged-cores re-run)" is now CLOSED via `plan_stage:17`.** Next:
finalize KMNIST cascaded s2; the d6/d8 FMNIST/KMNIST firing-gain gate-fix (`plan_stage:25`).

---

## 1k. AC2 on the VALID `deep_cnn` d6 dataset cells — the MISSING rung is filled, FMNIST AC2 ladder is now CONTINUOUS `rc=0` d4→d10 (2026-06-24)

§1j closed the d8 dataset confound but the **d4/d8/d10 cube still skipped d6** — exactly
the inflection where §1f located the within-CNN onset threshold (lossless ≤d5, ~5pp
deficit ≥d6). This lands the **d6 FMNIST/KMNIST cells on the enlarged `bigcores`
(count=480, `plan_stage:14`) vehicle** so they finalize **`rc=0`**. `deep_cnn` (width
16, S=4, `ttfs_cycle_based`), paired cascaded-vs-synchronized by seed. Ledger:
`cluster:"WS3"`, `kind:"arch_dataset"`, `model:"deep_cnn"`.

| model | dataset (depth) | deployed (cascaded mean) | ANN ref (AC2 target) | **AC2 casc→ANN gap** | casc→sync gap | sync→ANN gap | validity | AC2 verdict |
|:------|:----------------|:-------------------------|---------------------:|---------------------:|--------------:|-------------:|:---------|:------------|
| deep_cnn | FashionMNIST (d6) | **0.8400** (±2.00, n=3) | 0.930 | **9.11pp** | +6.11 | 2.78 | VALID `rc=0` | lossy (degraded) |
| deep_cnn | KMNIST (d6)       | **0.9100** (n=1) ‡ | 0.9753 | **6.28pp** | +5.85 | 0.80 | VALID `rc=0` | lossy (degraded, n=1 prov.) |

- **The d6 rung CONFIRMS the FMNIST monotone-widening AC2 law and closes the last cube
  gap.** FMNIST casc→sync now reads a smooth, gapless **+3.90 → +6.11 → +11.34 →
  +17.91pp** ladder (d4→d6→d8→d10) on a single clean-finalized convnet; the d6 cell
  slots cleanly between the §1e d4 (+3.90) and §1j d8 (+11.34) anchors. KMNIST's d6
  (+5.85) sits between its d4 (+6.19) and d8 (+7.19), consistent with the gentler KMNIST
  ladder (n=1 caveat below).
- **Synchronized AC2 stays effectively MET** (sync→ANN 2.78pp FMNIST, 0.80pp KMNIST) —
  the unconditional deep-model default holds at the inflection depth. With §1k (d6) +
  §1j (d8) + §1h (d10) all `rc=0`, the **d4/d6/d8/d10 × {FMNIST,KMNIST} dataset-axis cube
  is now CONTINUOUS and VALID `rc=0`**.

Run ids: `pdcnnd6databc_{FashionMNIST,KMNIST}_DataProvider_{cascaded,synchronized}_s{0,1,2}`.
**Confounds.** (1) **KMNIST cascaded n=1 (PROVISIONAL):** only
`pdcnnd6databc_KMNIST_DataProvider_cascaded_s0` finalized `rc=0`; s1/s2 are still in
`q/running/` (NON-FINALIZED) and excluded per the strict `rc==0` rule, and the 3rd sync
seed (`synchronized_s2`) is also still running → KMNIST is a single-seed cascaded point
vs a 2-seed sync arm (the +5.85pp gap is provisional). FMNIST d6 is full 3-seed on both
arms. (2) **eval-set mismatch:** cascaded `max_simulation_samples=200` (deployed bare
floats are exact 1/200 multiples, e.g. FMNIST .84/.82/.86, KMNIST .91) vs synchronized
FULL 10k (no subsample line, per commit 5568518) → **read the ~6pp gaps, not 3rd
decimals**; deployed = the bare float in `__target_metric.json` (200-sample SCM). (3)
**no at-chance confound:** ANN refs ~0.930/0.973 ≫ chance 0.10 → genuine firing-gain.
(4) **VALIDITY:** all 9 done runs `rc=0` reaching `HardCoreMappingStep` *without* the
§1e crash (`VALID_on_chip_majority_rc0`, `plan_stage:14`); the only paired-arm config
diff is `ttfs_cycle_schedule` cascaded-vs-synchronized. **Next:** finalize KMNIST
cascaded s1/s2 to firm the +5.85pp d6 read; the firing-gain gate-fix grid
(`plan_stage:25`) now spans the continuous 3.9–18pp d6/d8/d10 ladder.

---

## 1p. The §1k d6 KMNIST n=1 PROVISIONAL is UPGRADED to full 3-seed — d6 dataset-margin AC2 ordering CONFIRMED on a first-fully-finalized `rc=0` vehicle (`item_id=ws3_dcnn_d6_onset_dataset_axis`, 2026-06-24)

§1k filled the d6 rung but left **KMNIST cascaded at n=1 (PROVISIONAL)**. This batch
lands the **first fully-finalized (12/12 `rc=0`) d6 dataset axis** on the
`pdcnnbcd6data_*` `bigcores` family (`count=480`, `plan_stage:14`): `deep_cnn` (width 16,
S=4, `ttfs_cycle_based`), **3 seeds/arm** on FMNIST and KMNIST, paired by seed. Ledger:
`cluster:"WS3"`, `kind:"arch_dataset"`, `model:"deep_cnn"`.

| model | dataset (depth) | cascaded deployed (3-seed mean ± sd) | ANN ref | **AC2 casc→ANN gap** | casc→sync gap | sync→ANN gap | AC2 verdict |
|:------|:----------------|:-------------------------------------|--------:|---------------------:|--------------:|-------------:|:------------|
| deep_cnn | FashionMNIST (d6) | **0.8183** (±1.89) | 0.9293 | **11.09pp** | +7.79 | 3.31 | lossy (degraded) |
| deep_cnn | KMNIST (d6)       | **0.9167** (±1.61) | 0.9647 | **4.81pp** | +4.53 | 0.28 | mild (degraded, dataset-stable) |

- **The d6 dataset-margin ordering is CONFIRMED and the KMNIST n=1 provisional UPGRADED.**
  At matched d6 the harder dataset carries the larger cascade deficit: **FMNIST casc→sync
  +7.79pp ≫ KMNIST +4.53pp** (ANN ~0.929 vs ~0.965) — the §1e/§1n dataset-margin law on
  a clean `rc=0` vehicle. The full-seed KMNIST cell stays **MILD/dataset-stable**
  (sync→ANN 0.28pp, sync sd 0.23pp).
- **Synchronized AC2 stays effectively MET** (sync→ANN 3.31pp FMNIST / 0.28pp KMNIST) —
  deep-model default reinforced at the onset depth.

Run ids: `pdcnnbcd6data_{FashionMNIST,KMNIST}_DataProvider_{cascaded,synchronized}_s{0,1,2}`.
**Confounds.** (1) **TWO KMNIST run-id families DISAGREE on the absolute gap:** the
authoritative `pdcnnbcd6data_KMNIST_*` family reads **+4.53pp** (sync sd 0.23pp, the
cleaner cell, seeds matching the item .91/.935/.905); a **separate** `pdcnnd6databc_KMNIST_*`
family (the §1k n=1 source) also finalized 3-seed but reads **WIDER +7.94pp** (casc 0.8900
sd 3.04pp, s1=0.855 outlier). Both ledgered; they agree on the mild/dataset-stable
conclusion and FMNIST>KMNIST ordering, not the 3rd-decimal gap. (2) the FMNIST d6 here
(+7.79pp) reads WIDER than the §1k FMNIST d6 (+6.11pp, earlier batch). (3) **eval-set
mismatch:** cascaded `max_simulation_samples=200`, synchronized FULL 10k → read gaps not
3rd decimals. (4) **no at-chance confound:** ANN ~0.929/0.965 ≫ 0.10. (5) **VALIDITY:**
all 12 runs `rc=0` (`VALID_on_chip_majority_rc0`, `plan_stage:14`). **Next:** the
firing-gain gate-fix on the d6 FMNIST/KMNIST onset cells (`plan_stage:26`), gated on the
WS7 §2d convnet θ-cotrain `rc=1` fix.

---

## 1m. AC2 on the CLEAN `pdcnnbcclean_` `deep_cnn` MNIST ladder — d8 FULL 3-seed plateau (+4.16pp), synchronized LOSSLESS through d10 (`item_id=dcnn_clean_depth_ladder_d8_d10`, 2026-06-24)

The explicitly clean-named `pdcnnbcclean_` `bigcores` vehicle (`cores.count = 480`,
MNIST, w16, S=4, `ttfs_cycle_based`, paired by seed) UPGRADES the §1f/§1g d8 read from
n=2 to a **full 3-seed `rc=0`** anchor and lands a synchronized-lossless d10 rung.
ANN refs (0.9926 d8 / 0.992 d10) ≫ 10-class chance (0.1135) → genuine firing-gain, not
an untrained floor.

| model | dataset (depth) | deployed (cascaded mean) | ANN ref (AC2 target) | **AC2 casc→ANN gap** | casc→sync gap | sync→ANN gap | validity | AC2 verdict |
|:------|:----------------|:-------------------------|---------------------:|---------------------:|--------------:|-------------:|:---------|:------------|
| deep_cnn | MNIST (d8)  | **0.9517** (.925/.975/.955, n=3) | 0.9926 | **4.07pp** | +4.16 | −0.06 | VALID `rc=0` (3-seed both arms) | **NOT MET (~4–5pp plateau)** |
| deep_cnn | MNIST (d10) | *0.9555 prov.* (.9427/.9545/.9694, **NON-FINAL**) | 0.992 | *3.65pp prov.* | +3.77 prov. | **−0.12** | sync VALID `rc=0`; casc NON-FINAL | **sync MET (lossless); casc provisional ⇒ plateau** |

- **The d8 plateau is FIRM at full 3 seeds `rc=0`** (+4.16pp casc→sync), squarely in the
  ~5pp band of §1f (d6 −5.21, d8 −5.10) and FAR from the §1g `rc=1`-confounded 11–14pp
  collapse — the within-MNIST death-cascade **plateaus, it does not widen** on a valid
  convnet.
- **Synchronized AC2 stays MET (lossless) at BOTH d8 (−0.06pp) and d10 (−0.12pp)** —
  the deep-model-default verdict extends one rung past d8. The d10 cascaded provisional
  (+3.77pp) is consistent with the d8 plateau but NON-FINALIZED (uncountable).

Run ids: `pdcnnbcclean_d{8,10}_{cascaded,synchronized}_s{0,1,2}`.
**Confounds.** (1) **d10 cascaded NON-FINALIZED** at the verifier snapshot (`q/running/`,
`result=NONE`) → excluded per the strict `rc==0` rule; **progress 2026-06-24:** s0+s2 now
finalized `rc=0` (0.95/0.96), only s1 running → 2/3 done, plateau strengthening; lock when
s1 hits `rc=0`. (2) **eval-set mismatch:** cascaded `max_simulation_samples=200` (d8
.925/.975/.955 are exact 1/200 multiples) vs synchronized FULL 10k → read the +4.16pp gap,
not 3rd decimals. (3) **seed spread:** cascaded d8 sd 2.04pp vs sync sd 0.01pp; every
cascaded seed (≥0.925) is far above the collapse band. (4) **no at-chance confound:** ANN
0.989–0.994 ≫ 0.1135. (5) **d12:** absent at the snapshot, now `pdcnnbcclean_d12_*` is in
`q/running/` (in flight). **Next:** finalize d10 cascaded s1 + the d12 arms; layer the
firing-gain gate-fix (`plan_stage:25`) on the d8 plateau anchor.

---

## 1n. AC2 on the VALID `deep_cnn` d5 dataset axis — the d5 MNIST no-collapse corner RE-OPENS off MNIST (`item_id=dcnn_d5_dataset_axis`, 2026-06-24)

§1c reported the within-CNN cascade *flat* through d5 **on MNIST** (−0.04pp,
lossless-tied). This adds the **dataset axis at d5** — the rung the §1e (d4) / §1k (d6) /
§1j (d8) / §1h (d10) dataset cube skipped — and shows the MNIST d5 no-collapse corner is
**MNIST-bounded**: it re-opens the moment the dataset margin tightens. `deep_cnn` (width
16, S=4, `ttfs_cycle_based`), paired cascaded-vs-synchronized by seed, 3 seeds/arm. All
12 runs `rc=0` (`q/done/`). Ledger: `cluster:"WS3"`, `kind:"arch_dataset"`, `model:"deep_cnn"`.

| model | dataset (depth) | deployed (cascaded mean) | ANN ref (AC2 target) | **AC2 casc→ANN gap** | casc→sync gap | sync→ANN gap | validity | AC2 verdict |
|:------|:----------------|:-------------------------|---------------------:|---------------------:|--------------:|-------------:|:---------|:------------|
| deep_cnn | MNIST (d5) *(ref §1c)* | 0.9917 | 0.9913 | −0.04pp | +0.07 | −0.11 | VALID `rc=0` | near-lossless (no collapse) |
| deep_cnn | FashionMNIST (d5) | **0.8383** (±1.89, n=3) | 0.9273 | **9.00pp** | +6.03 | 2.76 | VALID `rc=0` | lossy (degraded) |
| deep_cnn | KMNIST (d5)       | **0.9167** (±1.03, n=3) | 0.9696 | **5.16pp** | +4.62 | 0.80 | VALID `rc=0` | lossy (degraded) |

- **The §1c d5 MNIST no-collapse corner does NOT generalize.** On the *same* shallow
  `deep_cnn` d5 the cascaded casc→sync firing-gain gap re-opens to **+6.03pp (FMNIST)**
  and **+4.62pp (KMNIST)** — both in the **degraded (2–8pp) band** — versus the +0.07pp
  MNIST tie. The cascade is gated by **dataset margin**, not just depth, even at the
  shallowest CNN rung where MNIST is lossless. This mirrors the §1e d4 result
  (FMNIST +3.90, KMNIST +6.19pp) one rung deeper, and **fills the d5 rung the dataset cube
  skipped** — the deep_cnn FMNIST/KMNIST dataset axis is now continuous at d4/**d5**/d6/d8/d10.
- **Synchronized AC2 stays effectively MET at d5** (sync→ANN 2.76pp FMNIST, 0.80pp KMNIST)
  — the unconditional deep-model default holds even at the shallow dataset-margin rung.
  The FMNIST d5 casc→sync (+6.03) slots cleanly between the §1e d4 (+3.90) and §1k d6
  (+6.11) anchors; KMNIST d5 (+4.62) sits below its d4 (+6.19) and d6 (+5.85), consistent
  with the gentler KMNIST ladder.

Run ids: `pdcnnd5data_{FashionMNIST,KMNIST}_DataProvider_{cascaded,synchronized}_s{0,1,2}`.
**Confounds.** (1) **eval-set mismatch:** cascaded `max_simulation_samples=200` (FMNIST
.825/.865/.825, KMNIST .905/.93/.915 are exact 1/200 multiples) vs synchronized FULL 10k
→ **read the ~4.6–6.0pp gaps, not 3rd decimals** (>2× the per-seed binomial band); cascaded
carries wider seed sd (FMNIST 1.89pp, KMNIST 1.03pp) than synchronized (0.23pp / 0.85pp).
(2) the cascaded and synchronized arms differ **only** in `deployment_parameters.ttfs_cycle_schedule`
(all other config byte-matched, paired by seed). (3) **no at-chance confound:** every ANN
≫ chance (min 0.9225) → genuine firing-gain, not untrained-floor. (4) **VALIDITY:** all 12
runs finalized `rc=0` (`q/done/`) — no `rc=1` packing-crash confound at this depth (unlike
the §1e d8 cells). **The previously-unconsolidated d5 dataset rung is now ledgered with
run-id provenance.** Next: layer the firing-gain gate-fix (`plan_stage:25`) across the now
continuous 4.6–18pp d5/d6/d8/d10 dataset ladder.

---

## 2. The WS7 keystone is an AC-recovery lever scoped to the AC2 deficit — `lenet5` negative control (2026-06-24)

The WS7 keystone (`conversion_policy:true`) recovers the cascaded firing-gain deficit
on the *deficient* `deep_mlp d8` cells. The **VALID `lenet5`** runs are the clean
negative control: the lift must vanish where the cascade is already near-lossless (AC2
near-MET). 3 seeds/arm, n=1000, `ttfs_cycle_based` S=4. Ledger: `cluster:"WS7"`,
`kind:"escalation"`, `model:"lenet5"`.

| model | dataset | cpFalse deployed | cpTrue deployed | cp lift | cpFalse→ANN gap (AC2) | deficit regime | verdict |
|:------|:--------|-----------------:|----------------:|--------:|----------------------:|:---------------|:--------|
| lenet5 | MNIST | 0.9847 | 0.9840 | **−0.07pp** | −0.63pp (near-lossless) | none | MATCH no-op (negative control) |
| lenet5 | FashionMNIST | 0.8460 | 0.8577 | **+1.17pp** | −7.09pp (mild) | mild | small deficit-proportional lift |
| *(ref, INVALID)* deep_mlp d8 | MNIST | 0.887 | 0.948 | +6.1pp | severe (host-majority) | severe | large rescue |
| *(ref, INVALID)* deep_mlp d8 | FashionMNIST | 0.756 | 0.858 | +10.2pp | severe (host-majority) | severe | large rescue |

**Verdict — CONFIRMED: the keystone is a firing-gain-deficit-SPECIFIC lever, not a
blanket accuracy boost.** The lift scales with the AC2 deficit: **none → −0.07pp**
(near-lossless MNIST), **mild → +1.17pp** (FashionMNIST), **severe → +6/+10pp**
(deep_mlp d8). The clean MNIST/lenet5 no-op is the negative control the WS7 claim needed.

Run ids: `cp_lenet_{MNIST,FashionMNIST}_DataProvider_cp{False,True}_s{0,1,2}`.
**Confound:** this is a `conversion_policy` true-vs-false control, NOT a
cascaded-vs-synchronized pairing — all 12 runs are cascaded; the relevant quantity is
`cp_lift = cpTrue − cpFalse`. ANN refs healthy (MNIST 0.991, FMNIST 0.917) ⇒ genuine
deployment, not a chance/untrained artifact. The deep_mlp d8 reference rows are quoted
from the §0/§10 records (INVALID host-majority vehicle).

---

## 1i. AC2 lenet5 arch×dataset breadth at n=1000, ordered by dataset margin + a byte-identical KMNIST re-pair (2026-06-24)

Consolidates §1/§1c into the full n=1000 cascaded breadth: the deployed→ANN (AC2)
gap on the VALID `lenet5` CNN orders **monotonically by dataset margin**. 3 seeds/cell,
`ttfs_cycle_based` S=4. Ledger: `cluster:"WS3"`, `kind:"arch_dataset"`, `model:"lenet5"`.

| dataset | ANN ref | cascaded deployed (3-seed ± sd_pp) | AC2 deployed→ANN gap | AC2 verdict |
|:--------|--------:|:-----------------------------------|---------------------:|:------------|
| MNIST | 0.9912 | **0.9873** (±0.25) | **0.39pp** | near-lossless (AC2 MET) |
| KMNIST | 0.9646 | **0.9340** (±0.73) | **3.06pp** | mild |
| KMNIST re-pair (`plncpair`) | 0.9657 | **0.9303** (±0.78) | **3.54pp** | reproduces csr KMNIST |
| FashionMNIST | 0.9183 | **0.8397** (±0.84) | **7.86pp** | lossy (largest, hardest dataset) |

**Verdict — the cascaded CNN AC2 deficit is MILD and dataset-STABLE; it orders by
dataset margin** (easier dataset / higher ANN → smaller gap: 0.39 < 3.06 < 7.86pp),
all far from the MLP-style death cascade (deep_mlp d8 was 10.8/16.0pp). Seed psd
≤ 0.84pp ⇒ low-variance, not a fragile collapse. The `plncpair` KMNIST set is
byte-identical to `csr_lenet` KMNIST except `experiment_name`, so 3.54pp vs 3.06pp
is a genuine **replicate** (combined 6-seed KMNIST: deployed 0.9322, ANN 0.9651,
gap 3.30pp).

Run ids: `csr_lenet_{MNIST,KMNIST,FashionMNIST}_DataProvider_cascaded_n1000_s{0,1,2}`,
`plncpair_lenet_KMNIST_DataProvider_cascaded_n1000_s{0,1,2}`.
**Confound:** the matched-resolution cascaded→synchronized AC2 gap is **NOT yet
computable** — all n1000 synchronized counterparts (`plnsync_lenet_{MNIST,FashionMNIST}_
synchronized_n1000`, `plncpair_lenet_KMNIST_synchronized_n1000`) remain PENDING
(0 finalized), so only the deployed→ANN gap is reported (the 50-sample `sync_full`
tags are unmatched-resolution context only). All 12 cascaded runs rc=0; ANN refs
≫ chance.

---

## 1q. The §1i matched-resolution confound is CLOSED for KMNIST — lenet5 KMNIST cascaded→sync gap +1.79pp at both arms n=1000; SVHN sync-only (`item_id=ws3_lenet_paired_n1000_kmnist_svhn`, 2026-06-24)

§1i reported the lenet5 n=1000 cascaded breadth but flagged the **matched-resolution
cascaded→synchronized AC2 gap as NOT yet computable** (all n1000 synchronized
counterparts were PENDING). This batch lands the **paired n=1000 synchronized arm** for
KMNIST and SVHN, so the cascaded→sync gap is finally read at **both arms n=1000**. Ledger:
`cluster:"WS3"`, `kind:"arch_dataset"`, `model:"lenet5"`. Pairing axis =
`deployment_parameters.ttfs_cycle_schedule`.

| dataset | cascaded n=1000 (3-seed ± sd) | synchronized n=1000 (3-seed ± sd) | **casc→sync gap (matched)** | prior §1i/§4d.1 (n50 sync) | ANN ref | AC2 verdict |
|:--------|:------------------------------|:----------------------------------|----------------------------:|:---------------------------|--------:|:------------|
| KMNIST | **0.934** (±0.73) | **0.9519** (±0.30) | **+1.79pp** | +1.45 (n50 sync 0.9485) | 0.9646 | mild, dataset-stable (CONFIRMED) |
| SVHN | *cascaded all `rc=1` — UNAVAILABLE* | **0.8593** (±0.36) | *null* | — | 0.8945 | sync-only baseline (sync→ANN 3.52pp) |

- **The §1i confound is CLOSED for KMNIST.** With both arms at n=1000 the cascaded→sync
  gap is **+1.79pp** — MILD and dataset-stable, in the ~1.4–1.8pp band, between
  MNIST-lossless and FMNIST ~6pp. The n=50 sync arm (0.9485) was only **0.34pp below**
  the n1000 sync (0.9519), so the §4d.1 resolution-mix confound was **SMALL** and moved
  the verdict **toward** MILD (1.45 → 1.79pp, both in band). A residual cascaded→ANN gap
  of **3.06pp > seed sd 0.73pp** persists ⇒ small-but-real firing-gain residual.
- **SVHN stays sync-only:** synchronized n1000 0.8593 (sync→ANN 3.52pp, ANN 0.8945 ≫ SVHN
  chance 0.196) is a valid AC2 sync baseline, but **all 6 SVHN cascaded n1000 seeds**
  (both `plncpair_*` and `csr_*` prefixes) finalized `rc=1` → no matched gap.

Run ids: `csr_lenet_KMNIST_DataProvider_cascaded_n1000_s{0,1,2}` paired with
`plncpair_lenet_KMNIST_DataProvider_synchronized_n1000_s{0,1,2}`;
`plncpair_lenet_SVHN_DataProvider_synchronized_n1000_s{0,1,2}` (sync-only).
**Confounds:** (1) SVHN cascaded UNAVAILABLE (all `rc=1`, both prefixes) → SVHN matched
gap remains open. (2) all 9 harvested runs `rc=0`, finalized, `max_simulation_samples=1000`
on both arms → the §1i / §4d.1 n=50 sync confound is removed for KMNIST. (3) both arms
share `model_type=lenet5`, TTFS, thresholding `<=`. **Next:** recover the SVHN cascaded
n=1000 arm (`plan_stage:27`) to complete the matched 4-dataset CNN cascaded→sync AC2 table.

---

## 2b. The WS7 lenet5/FMNIST rescue is PARTIAL — ~5.7pp AC2 floor remains; theta_cotrain=TRUE never run (2026-06-24)

§2 showed `conversion_policy` lifts the mild lenet5/FMNIST deficit +1.17pp. This adds
the `plnrescue` rescue baseline and reads the AC2 residual. 3 seeds/arm, n=1000.
Ledger: `cluster:"WS7"`, `kind:"escalation"`, `model:"lenet5"`.

| arm | conversion_policy | theta_cotrain | deployed | AC2 deployed→ANN gap | fraction of 7.12pp closed |
|:----|:-----------------:|:-------------:|---------:|---------------------:|:--------------------------|
| cpFalse (baseline) | false | — | **0.846** | 7.12pp | — |
| cpTrue (routing ON) | true | — | **0.8577** | 5.88pp | ~17% |
| plnrescue cotFalse | true | false | **0.8603** | 5.71pp | ~20% |

**Verdict — PARTIAL-RESCUE / AC2 FLOOR-REMAINS.** The deficit is real and
gradient-bearing (ANN ~0.917; logs show a `finalize_cliff` 0.177 that recovers to
~0.84 ⇒ non-dead). But the controller-routing lever closes only ~17% (+1.17pp) and
the `plnrescue` baseline reaches 0.8603 (residual **5.71pp**). A **hard ~5.6–5.9pp
AC2 floor remains** ⇒ the closeout-10.2 / WS3-4.2 "recoverable" claim is **NOT
validated to lossless on this VALID vehicle** (it holds only on the INVALID deep_mlp
d8). The **WS7 automatic-rescue-on-a-VALID-vehicle cell does NOT move to MET.**

Run ids: `cp_lenet_FashionMNIST_DataProvider_{cpFalse,cpTrue}_s{0,1,2}`,
`plnrescue_lenet_FashionMNIST_cotFalse_s{0,1,2}`.
**Confound:** the named rescue lever `theta_cotrain` was **NEVER turned ON** for this
cell — `plnrescue_*cotFalse` has `ttfs_theta_cotrain=false` and **no `plnrescue_*cotTrue`
run exists** (done/ or failed/), so the rescue's upper bound is UNTESTED. No
synchronized arm (all cascaded); ANN ≫ chance; n=1000 ⇒ read pp not 3rd decimals.

---

## 2d. The WS7 rescue is a NO-OP on the FIRST VALID `deep_cnn` d6 onset cell; θ-cotrain is BROKEN on the convnet (`item_id=dcnn_d6_onset_gatefix_rescue`, 2026-06-24)

§2/§2b established the keystone's deficit-proportional lift on `lenet5` (mild) and the
INVALID `deep_mlp d8` (severe). This is the AC-recovery test on the **§1f/§1m d6 onset
rung** — the first VALID `deep_cnn` cell with a real (~4–5pp) firing-gain deficit, where
a working rescue would move the cell toward AC2-MET. `deep_cnn` (w16), MNIST,
`ttfs_cycle_based` S=4, on-chip 99.41%, 3 seeds/arm, `max_simulation_samples=200`. Ledger:
`cluster:"WS7"`, `kind:"escalation"`, `model:"deep_cnn"`, `depth:6`.

| arm | conversion_policy | theta_cotrain | deployed (3-seed) | seeds | cp lift | casc→ANN gap (AC2) | rc | rescue verdict |
|:----|:-----------------:|:-------------:|------------------:|:------|--------:|-------------------:|:--:|:---------------|
| no-policy (baseline) | false | false | **0.9500** | .965/.955/.93 | — | 4.38pp | 0 | gradient-bearing deficit |
| controller routing | true | false | **0.8983** | .985/.94/**.77** | **−5.17pp** | 9.57pp | 0 | **REGRESSES** (high-variance) |
| θ-cotrain (any cp) | — | true | **n/a** | — | — | — | **1** | **BROKEN (rc=1 crash)** |

**Verdict — NO WORKING RESCUE LEVER on the valid convnet; closeout-10.2's controller-rescue
lift does NOT replicate.** `conversion_policy` does **not** auto-rescue the d6 deficit: the
cpFalse→cpTrue lift is **NEGATIVE** (−5.17pp mean, −1.50pp median) because the policy is
high-variance and catastrophically regresses one seed (cpTrue s2 = **0.77**, a genuine
rc=0 finalized collapse — on-chip 99.41%, NF↔SCM 1.0000, torch↔sim parity 1.0000). The
+2pp s0 lift is a single-seed artifact. And the per-channel θ-cotrain gain-trim knob —
the AC_EVIDENCE §2b "named rescue lever never run" — **WAS run here and crashes rc=1**
(`[ModelRepresentation] forward failed at node Conv2DPerceptronMapper(name='features_3')`,
all 6 cotTrue runs `q/failed/`). So the **d6 cell does NOT move to AC2-MET**, and the §1f
~5pp plateau has **no working rescue knob on the convnet**: synchronized stays the
unconditional deep_cnn default.

Run ids: `pdcnnd6fix_cotFalse_cp{False,True}_s{0,1,2}` (rc=0),
`pdcnnd6fix_cotTrue_cp{False,True}_s{0,1,2}` (rc=1, θ-cotrain broken); synchronized
ceiling `pdcnnbc_d6_synchronized_s{0,1,2}` (0.9904 FULL 10k). **Confounds:** (1) cascaded
n=200 vs sync FULL 10k → read pp gaps; (2) the 0.99+ `__target_metric.json` floats on the
cotTrue runs are **stale pre-deployment ANN-stage artifacts** (the runs crash before
deployment) — not valid metrics; (3) NOT chance (ANN ~0.994 ≫ 0.1135). **Open:** the d6
firing-gain gate-fix (`plan_stage:25`) — a θ-cotrain convnet-forward fix or a relative-gain
gate-fix is the only remaining route to a working rescue. (Detailed analysis:
`docs/research/findings/WS7_keystone_automatic.md` §9.)

---

## 2e. The convnet-compatible STE replacement for the broken θ-cotrain ALSO fails — staircase-STE REGRESSES the d6 onset on the DATASET axis (`item_id=dcnn_d6_dataset_ste_gatefix`, 2026-06-25)

§2d left the per-channel θ-cotrain `rc=1`-broken on the convnet (`Conv2DPerceptronMapper
features_3` tensor-shape break), so it was never measured as an AC2-recovery lever. This cell
tests its **convnet-compatible replacement** — `ttfs_staircase_ste` (`ste_mix=0.5`, the hedged
staircase-backward STE) — and extends the rescue question to the **harder datasets** at the
same d6 onset. `deep_cnn` (w16), `ttfs_cycle_based` S=4, VALID on-chip-majority, 3 seeds/arm,
`max_simulation_samples=200`. Levers gridded: `ttfs_staircase_ste` × `conversion_policy`.
Ledger: `cluster:"WS7"`, `kind:"escalation"`, `item_id:"dcnn_d6_dataset_ste_gatefix"` (2 rows).

| dataset | best steTrue arm | no-lever baseline | sync ceiling | ste lift (cpFalse) | best steTrue→baseline | best steTrue→sync (AC2) | rc | rescue verdict |
|:--------|-----------------:|------------------:|-------------:|-------------------:|----------------------:|------------------------:|:--:|:--------------|
| FashionMNIST | **0.7933** | 0.8433 | 0.8962 | **−6.66pp** | **−5.0pp** | **−10.29pp** | 0 | **STE REGRESSES** |
| KMNIST | **0.865** | 0.9083 | 0.9619 | **−5.0pp** | **−4.33pp** | **−9.69pp** | 0 | **STE REGRESSES** |

`conversion_policy` is itself net-negative on the convnet (cp lift −2.66pp FMNIST / −5.0pp
KMNIST at steFalse — same sign as the §2d MNIST −5.17pp cp regression); ste and cp do **not**
compose. Sync ceiling + cascaded baseline are the SAME d6/S=4/n=200 paired `pdcnnbcd6data_*`
cell (commensurable, unlike §2d's full-10k sync).

**Verdict — NO WORKING FIRING-GAIN RESCUE LEVER on the convnet d6 onset, dataset axis confirms
§2d.** Every steTrue arm sits **9.7–10.3pp under the synchronized ceiling** and **4.3–5.0pp
under the no-lever baseline** on both harder datasets. With θ-cotrain `rc=1`-broken (§2d, §1r),
`conversion_policy` net-regressing (§2d MNIST + here), and now staircase-STE regressing, the
config-level rescue space is **exhausted** at the convnet d6 onset. **Synchronized stays the
unconditional deep_cnn default**; the only remaining route is a **code fix to the per-channel
θ-cotrain convnet-forward path** (the `features_3` tensor-shape break), not a new schedule/lever.

**Confounds.** (1) All 24 grid runs `rc=0`, finalized, deployed `__target_metric.json` present;
`max_simulation_samples=200` on EVERY run (grid + paired baseline/sync) → 0.5% granularity →
read pp gaps, not 3rd decimals. (2) NOT chance: ANN healthy (FMNIST ~0.929–0.932, KMNIST
~0.963–0.974 ≫ 0.10) → genuine firing-gain regression, not untrained-floor. (3) Commensurable
pairing: sync/cascaded baseline from the SAME-cell `pdcnnbcd6data` n=200 batch; the KMNIST
steFalse/cpFalse baseline 0.9083 dips ~1pp below the bc-batch cascaded baseline 0.9167 within
seed/200-sample noise, both well under sync 0.9619. (4) θ-cotrain NOT run here (`rc=1` convnet
crash); this grid is its convnet-compatible replacement, which also fails. Run ids (lever, rc=0):
`pdcnnd6datastefix_{FashionMNIST,KMNIST}_DataProvider_steTrue_cp{False,True}_s{0,1,2}`;
baselines `…_steFalse_cp{False,True}_s{0,1,2}`; paired ceiling/baseline
`pdcnnbcd6data_{…}_{synchronized,cascaded}_s{0,1,2}`. (Detailed analysis:
`docs/research/findings/WS7_keystone_automatic.md` §11.)

---

## 2f. On the EASIER MNIST d6 onset the staircase-STE 2×2 DECOMPOSES — `ttfs_staircase_ste` is the DOMINANT knob and HALVES the AC2 deficit (unlike §2e's REGRESSION on the harder dataset axis), but the best combo still leaves +2.40pp ANN gap (`item_id=dcnn_d6_ste_gatefix_decomposition`, 2026-06-25)

§2e found `ttfs_staircase_ste` **regresses** the d6 onset on the harder FMNIST/KMNIST axis. This
cell tests the SAME convnet-compatible STE lever on the **easier MNIST** d6 onset and decomposes
the full 2×2 (`ttfs_staircase_ste` × `conversion_policy`). `deep_cnn` (w16), `ttfs_cycle_based`
S=4, VALID on-chip-majority, **all 12 runs `rc=0`**, 3 seeds/cell, `max_simulation_samples=200`.
Ledger: `cluster:"WS3"`, `kind:"escalation"`, `item_id:"dcnn_d6_ste_gatefix_decomposition"` (2 rows).
`conversion_policy=False=pure cascaded`, `True=synchronized/conversion route`.

| cell | ste | cp | deployed mean (3 seeds) | ANN mean | **ANN gap (AC2)** | verdict |
|:-----|:----|:---|------------------------:|---------:|------------------:|:--------|
| steTrue_cpTrue (**best of grid**) | T | T | **0.9683** (.96/.96/.985) | 0.9923 | **+2.40pp** | partial rescue, NOT lossless |
| steTrue_cpFalse | T | F | 0.9650 (.945/.99/.96) | 0.9907 | +2.57pp | partial rescue |
| steFalse_cpTrue | F | T | 0.9600 (.97/.97/.94) | 0.9930 | +3.30pp | weak partial |
| steFalse_cpFalse (**worst, pure cascaded**) | F | F | **0.9483** (.96/.955/.93) | 0.9920 | **+4.37pp** | full uncorrected deficit |

Knob lifts (pp): **STE +0.83 @cpTrue / +1.67 @cpFalse** (dominant); conversion_policy +0.33 @steTrue / +1.17 @steFalse.

**Verdict — `ttfs_staircase_ste` DOMINANT and PARTIAL on MNIST; AC2 still NOT MET.** Unlike the
§2e dataset-axis regression, on the easier MNIST onset STE is the **larger knob on both columns**
and roughly **halves** the AC2 deficit (worst pure-cascaded +4.37pp → best combo +2.40pp). But the
best combo (steTrue+cpTrue 0.9683) does **NOT** reach the ~0.992 ANN/sync ceiling — a **+2.40pp
residual remains, AC2 NOT MET**. With STE ON the conversion-route adds only +0.33pp (WITHIN n=200
noise), i.e. STE has already substituted for the conversion-route rescue. This is the **first
config-level lever that makes net-positive progress** on a VALID deep_cnn cascaded onset — but only
on MNIST, and only halfway; the dataset-axis (§2e) and θ-cotrain (§2d) routes still fail/crash.

**Confounds.** (1) **n=200 → 0.005 grid:** single-seed swings (e.g. steTrue_cpFalse .945/.99/.96)
~1–2 samples of noise; sub-pp cell differences (the casc→sync +0.33pp under STE) WITHIN-noise — but
the **+2.40pp best-combo ANN gap and +1.67pp STE-lift @cpFalse exceed** this resolution. (2) NOT
chance: ANN ~0.9915 (range 0.9875–0.9941) ≫ 0.10 → genuine firing-gain. (3) All 12 runs `rc=0`,
full 3 seeds/cell. (4) Eval-set asymmetry: deployed n=200 vs ANN n=10000. (5) θ-cotrain remains
`rc=1`-broken on this convnet (§2d, §2e); STE is the working swap. Run ids:
`pdcnnd6stefix_ste{True,False}_cp{True,False}_s{0,1,2}` (cpTrue=synchronized arm, cpFalse=cascaded
arm). (Detailed analysis: `docs/research/findings/WS3_depth_firing_gain.md` §4t.)

---

## 1v. AC2 on the VALID `deep_cnn` dataset×depth vehicle at n=200 — the death-cascade reproduces, replacing the retired INVALID deep_mlp §10.1 table: depth-law HOLDS on FMNIST, FLAT on KMNIST (`item_id=dcnn_dataset_depth_deathcascade_valid_vehicle`, 2026-06-25)

The headline depth × firing-gain death-cascade table was originally measured on the **INVALID
host-majority `deep_mlp`** (retired). This cell reproduces it on the VALID on-chip-majority
`deep_cnn` (w16) across {FMNIST,KMNIST}×{d6,d8}. `ttfs_cycle_based` S=4, cascaded schedule,
`ttfs_theta_cotrain=False`, **11 valid `rc=0` cascaded runs**, `max_simulation_samples=200`.
**NO synchronized arm in this batch** (only axis = cotrain T/F), so the AC2 gap is reported as
**cascaded→ANN** only. Ledger: `cluster:"WS3"`, `kind:"arch_dataset"`,
`item_id:"dcnn_dataset_depth_deathcascade_valid_vehicle"` (4 rows).

| dataset | depth | cascaded n200 (mean ± sd) | ANN ref | **casc→ANN GAP (AC2)** | verdict |
|:--------|:------|:--------------------------|:--------|-----------------------:|:--------|
| fmnist | 6 | 0.855 ± 3.19pp | 0.9304 | **+7.54pp** | firing-gain degraded |
| fmnist | 8 | 0.7675 ± 0.75pp (n=2†) | 0.9356 | **+16.81pp** | **COLLAPSE** (widens sharply) |
| kmnist | 6 | 0.8967 ± 1.93pp | 0.9698 | **+7.31pp** | firing-gain degraded |
| kmnist | 8 | 0.8967 ± 1.03pp | 0.9702 | **+7.35pp** | degraded (**FLAT** vs d6) |

† FMNIST d8 is n=2 (s2 failed `rc=1`); all other cells n=3.

**Verdict — SUPPORTED-WITH-CONFOUND; AC2 NOT MET, the death-cascade reproduces on the VALID
vehicle.** The cascaded→ANN AC2 gap widens **sharply** with depth on FMNIST (+7.54 → +16.81pp,
deployed 0.855 → 0.768) and with **dataset margin at d8** (FMNIST +16.81pp ≫ KMNIST +7.35pp). But
on KMNIST the depth axis is **FLAT** (+7.31 → +7.35pp) — the depth-widening law does NOT hold on
KMNIST. This **replaces the retired INVALID deep_mlp §10.1 death-cascade table** on a VALID
on-chip-majority CNN.

**Confounds.** (1) **NO synchronized arm** → AC2 gap is cascaded→ANN only (no cascaded→sync).
(2) The gate-fix cotTrue arm: **all 12 runs failed `rc=1`** (Conv2DPerceptronMapper crash). (3)
FMNIST d8 n=2 (s2 `rc=1`). (4) n=200 → 0.005 grid: read pp gaps, not 3rd decimals. (5) NOT chance:
ANN 0.93–0.97 ≫ 0.10 → genuine firing-gain. (6) 3 duplicate d8_KMNIST queue JSONs (id==filename)
excluded by strict rule. Run ids (cascaded, rc=0):
`pdcnndatafix_d{6,8}_{FashionMNIST,KMNIST}_DataProvider_cotFalse_s{0,1,2}` (FMNIST d8 = s0,s1 only).
(Detailed analysis: `docs/research/findings/WS3_depth_firing_gain.md` §4u.)

---

## 2c. SYNTHESIS — the two CONFIRMED `deep_cnn` AC2 items consolidated, with the corrected verdicts (2026-06-24)

Two `kind="synthesis"` ledger rows (`cluster:"WS3"`) roll the per-rung AC2 cells
(§1d/§1f/§1g/§1h/§1j/§1k) into two cross-rung findings and re-cite every run_id.
This subsection freezes the CORRECTED verdicts; the per-rung tables are above.

- **`item_id=deep_cnn_depth_ladder` (MNIST).** VALID vehicle (ANN ~0.99 each
  depth; synchronized AC2 MET — lossless `==`ANN each depth). Cascaded AC2 is
  LOSSLESS at d5 (0.07pp) then BREAKS at d6 (5.21pp) and holds a **bounded
  ~4–5pp plateau** through d10 (5.21 / 4.85 / 4.00pp at d6/d8/d10). **Verdict:
  CONFIRMED-WITH-CONFOUND — sharp d5→d6 onset, NOT the monotone-widening curve of
  closeout §6** (the gap shrinks d6→d10). **d12 cascaded INCONCLUSIVE (n=1;
  s0/s2 `rc=-9` timed_out at the 3600s wall).**
- **`item_id=deep_cnn_dataset_axis` (FMNIST+KMNIST).** VALID vehicle (ANN ~0.93
  FMNIST / ~0.97 KMNIST; synchronized AC2 MET, sync→ANN ≤~3pp). Cascaded AC2
  deficit WIDENS with depth on both datasets — **FMNIST strict monotone**
  (6.03→6.11→11.34→17.91pp at d5/d6/d8/d10), **KMNIST widens overall** with a
  d6→d8 dip (7.94→7.02pp) that is **within 200-sample noise**. **Verdict:
  SUPPORTED-with-caveats** (deepest cascaded rungs n=2: d10 FMNIST s1 `rc=-9`,
  d10 KMNIST s0 `rc=1`).

**AC2 read:** synchronized is the deep-model default (AC2 MET at every valid
`deep_cnn` rung in both items); cascaded carries a real, depth-onset,
dataset-amplified AC2 deficit whose honest shape is **threshold + margin
amplification**, with d12 MNIST still unmeasured.

---

## 1r. AC2 on the CLEAN `rc=0` genuine-n=1000 `deep_cnn` deep ladder — the §1g (‡) `rc=1` crash confound is CLOSED; death-cascade VALID at d8/d10 but NOT depth-monotone (`item_id=dcnn_n1000_deathcascade_finalize`, 2026-06-25)

§1g landed the genuine high-resolution (nevresim **n=1000**, 5× the n=200 grid) deep_cnn
d8/d10 AC2 reads, but only on the **`rc=1`-confounded `pdcnndeeppair_` vehicle** (the (‡)
rows: d8 casc→sync +8.51pp, d10 +11.14pp — CONFIRMED-WITH-CONFOUND because every run crashed
downstream at `HardCoreMappingStep` "No more hard cores available" *after* the SCM metric +
parity gates were written). This batch (the `plan_stage:23` proposal) re-runs the **same
genuine n=1000 d8/d10 paired ladder on the proven CLEAN `rc=0` `bigcores` (`cores.count=480`)
vehicle**. **ALL 12 runs finalized `rc=0`** (`q/done/`, `artifact_ok`, ZERO packing crashes).
`deep_cnn` (w16), MNIST, `ttfs_cycle_based` S=4, **`max_simulation_samples=1000`**, 3 seeds/arm
paired by seed. Ledger: `cluster:"WS3"`, `kind:"depth"`, `model:"deep_cnn"`.

| model | dataset (depth) | deployed (cascaded, 3-seed mean) | ANN ref (AC2 target) | **AC2 casc→ANN gap** | casc→sync gap | sync→ANN gap | validity | AC2 verdict |
|:------|:----------------|:---------------------------------|---------------------:|---------------------:|--------------:|-------------:|:---------|:------------|
| deep_cnn | MNIST (d8, n=1000)  | **0.898** (.814/.95/.93)    | 0.9923 | **9.43pp** | −9.48 | +0.05 | VALID `rc=0` (12/12) | **NOT MET (firing-gain degraded)** |
| deep_cnn | MNIST (d10, n=1000) | **0.9297** (.907/.918/.964) | 0.9907 | **6.10pp** | −6.06 | −0.04 | VALID `rc=0` (12/12) | **NOT MET (firing-gain degraded)** |

- **The §1g (‡) `rc=1` n=1000 crash confound is CLOSED.** The same genuine n=1000 reads
  now exist on a clean `rc=0` vehicle: cascaded AC2 deployed→ANN gaps are **9.43pp (d8)** and
  **6.10pp (d10)** with NF↔SCM agreement **1.0000** and torch↔sim parity **0.9922 (d8s0) /
  1.0000 (d10s0)** → a **genuine firing-gain deficit, not a parity/decode/crash artifact**.
  **Synchronized AC2 is effectively MET (lossless) at both depths** (sync→ANN +0.05pp d8,
  −0.04pp d10, sd ≤0.18pp) — the unconditional deep-model default holds.
- **The "d8 mild / d10 collapse" framing is REFUTED on the clean reads.** The clean rc=0
  ordering is **INVERTED**: d8 gap (−9.48pp) > d10 gap (−6.06pp), opposite to the §1g (‡)
  `rc=1` ordering (d10 +11.14 > d8 +8.51pp). The d8>d10 ordering is **dominated by the d8
  cascaded s0=0.814 outlier** (vs s1=0.95/s2=0.93; sd 6.00pp). Both depths degrade ~6–9pp;
  the death-cascade reproduces at both but its **magnitude is NOT depth-monotone** across
  vehicle/resolution. ANN ≫ chance ⇒ genuine firing-gain (not untrained-floor).

Run ids: `pdcnnbcn1000_d{8,10}_{cascaded,synchronized}_s{0,1,2}`. **Confounds:** (1) cascaded
subsamples **1000/10000** (~1pp/seed binomial; the d8 s0=0.814 drives the 6.00pp sd) → read
the GAPS (9.43/6.10pp ≫ ~1pp noise), not 3rd decimals; `__target_metric.json` tracks SCM/HCM
within ~1pp and the bare float is reported per convention. (2) no at-chance confound (ANN
0.988–0.9946). (3) 3 seeds/arm, paired, only `ttfs_cycle_schedule` differs. **This is the
clean `rc=0` + high-resolution upgrade of §1g's deep rungs.** Next: the firing-gain gate-fix
on this clean d8/d10 n=1000 anchor — **but the gate-fix is REFUTED as a deep auto-rescue**
(WS7 §10, `dcnn_d10_gatefix_rescue`: θ-cotrain crashes the convnet, `cp:true`-only deploys
~0.79 < the cascaded baseline) → synchronized stays the deep default. Next-round proposals:
seed-firm + d12 the clean n=1000 ladder (WS3 `plan_stage:38`); the gate-fix retest on the
clean n=1000 anchor is BLOCKED on a θ-cotrain convnet-forward fix (WS7 `plan_stage:39`).

---

## 1s. AC2 — the FULL clean `rc=0` `bigcores` MNIST depth ladder, both resolutions, d12 rung CLOSED: sharp d6 onset → BOUNDED ~4–7pp plateau, NOT monotone (`item_id=deep_cnn_depth_cascade_ladder_mnist`, 2026-06-25)

Consolidates **every clean `rc=0` `pdcnnbc*` `bigcores` (`cores.count=480`) rung** of
the within-CNN MNIST cascade ladder into one AC2 item at **both** the n200 (0.005-grid)
and the genuine **n1000** (0.001-grid) read, and **closes the d12 cascaded rung** §1g/§1r
left open at n=1. `deep_cnn` (w16), MNIST, `ttfs_cycle_based` S=4, paired
cascaded-vs-synchronized by seed. ANN ~0.99 at every depth (≫ chance 0.1135) ⇒ genuine
firing-gain. Ledger: `cluster:"WS3"`, `kind:"depth"`,
`item_id:"deep_cnn_depth_cascade_ladder_mnist"`.

| d | casc n200 | casc n1000 | sync | ANN | **casc→sync gap n200** | **casc→sync gap n1000** | AC2 (cascaded) | AC2 (sync) |
|--:|----------:|-----------:|-----:|----:|-----------------------:|------------------------:|:---------------|:-----------|
| 4  | 0.9883 | — | 0.9898 | 0.9931 | **+0.15** | — | **near-MET** | MET |
| 5  | 0.9917 | — | 0.9924 | 0.9913 | **+0.07** | — | **near-MET** | MET |
| 6  | 0.9383 | 0.9563 | 0.9904/0.9924 | 0.992 | **+5.21** | **+3.61** | NOT MET (onset) | MET |
| 8  | 0.9483 | 0.9293 | 0.9934/0.9918 | 0.992 | **+4.50** | **+6.24** | NOT MET | MET |
| 10 | 0.9525 | 0.9318 | 0.9925/0.9909 | 0.992 | **+4.00** | **+5.91** | NOT MET | MET |
| 12 | 0.9175 ‡ | 0.9353 | 0.9916/0.9920 | 0.992 | **+7.41** ‡ | **+5.67** | NOT MET (d12 CLOSED) | MET |

- **Verdict — `cascaded_firing_gain_degraded_bounded_plateau_sharp_d6_onset`.** Cascaded
  is **byte-tied to synchronized and the ANN ceiling at d4/d5** (+0.15/+0.07pp,
  near-lossless), then a **SHARP onset at d6** drops it to a **BOUNDED ~4–7pp plateau**
  that **does NOT widen monotonically through d12** (n200 5.21→4.50→4.00→7.41; n1000
  3.61→6.24→5.91→5.67pp). **Synchronized AC2 is MET (lossless) at EVERY depth** (≤0.3pp to
  ANN, sd ≤0.16pp) — the unconditional deep-model default holds on the CNN.
- **This is NOT the deep_mlp monotone-widening (pattern a)** and **NOT absent (pattern c)** —
  it CORRECTS §1c's shallow-only "no-collapse" reading and §1r's two-point d8>d10
  ordering: the d12 n1000 rung (+5.67pp) sits squarely IN the plateau, confirming
  bounded-not-runaway through the deepest rung. The closeout §6 "monotonically widening"
  framing is **REFUTED in literal form for the CNN MNIST ladder**.

‡ d12 **n200** cascaded is a **cross-vehicle pool** (1 `pdcnnbc_` s1=0.980 + 3
`pdcnnbcd12fin_` 0.835/0.920/0.935; the same-vehicle d12 cascaded OOM-crashed `rc=-9`).
The **d12 n1000** read (`pdcnnbcn1000seed_` s3/4/5, +5.67pp) is the clean same-vehicle
load-bearing d12 number. Run ids: `dcnn_d4_*`, `pdcnnladder_d5_*`, `pdcnnbc_d{6,8,10}_*`,
`pdcnnbcclean_d{8,10}_*`, `pdcnnbcn1000{,plat,seed}_d{6,8,10,12}_*`, `pdcnnbcd12fin_*`.
**Confounds:** (1) cascaded subsamples to `max_simulation_samples` (n200 0.005-grid /
n1000 0.001-grid) vs synchronized FULL 10k → read GAPS not 3rd decimals; cascaded sd
1–5pp vs sync ~0.1–0.2pp (fragile high-variance code). (2) higher resolution does NOT
shrink the gap (n1000 ≥ n200 at d8/d10) → the depth-law hardens. (3) EXCLUDED: the
`rc=1`/`rc=-9` ladders (`dcnn_d6/d8`, `pdcnnladder_d6/d7`, `pdcnnbc_d12 s0/s2`,
`pdcnnbcclean_d12`) and the WS7 gate-fix grids (no sync counterpart). All clean runs
passed parity (NF↔SCM 1.0, torch↔sim 0.9961–1.0).
- **POOLED-BATCH hardening (synthesis row, 2026-06-25):** pooling the d8/d10 n1000 rungs
  over THREE independent seed batches (`pdcnnbcn1000_` base + `pdcnnbcn1000plat_` plat +
  `pdcnnbcn1000seed_` seed, **n=9/rung**) re-confirms the plateau: pooled gaps
  d8 +6.24 / d10 +5.91 / d12 +5.67pp (`d10 ≤ d8`, `d12 ≤ d10`). The d8 pooled mean is
  INFLATED by one base-batch outlier (`pdcnnbcn1000_d8_cascaded_s0=0.814`, verified
  artifact); clean per-batch d8 gaps are **3.65 (plat) / 5.59 (seed)pp** and d10 agrees
  tightly across batches at **5.63–6.06pp** — monotone-widening is refuted regardless of
  outlier treatment. See WS3 findings §4x. Synthesis ledger row
  `kind:"synthesis", item_id:"deep_cnn_depth_cascade_ladder_mnist"` cites all 54 run_ids.

---

## 1t. AC2 on the VALID `deep_cnn` d8 DATASET cells at genuine n=1000 (BOTH arms paired) — the §1j n=200 dataset-margin death-cascade HARDENS, ordering holds at 4-decimal fidelity (`item_id=dcnn_d8_dataset_n1000`, 2026-06-25)

§1j closed the deep_cnn d8 dataset confound on the clean `rc=0` `bigcores` vehicle but at
**n=200** (cascaded 0.005-grid vs synchronized FULL 10k — a resolution mismatch). This batch
re-measures the **same d8 FMNIST/KMNIST cells with BOTH arms paired at
`max_simulation_samples=1000`** — the genuine high-resolution dataset read. All 12 runs
finalized `rc=0`. `deep_cnn` (w16), S=4, `ttfs_cycle_based`, 3 seeds/arm paired by seed.
Ledger: `cluster:"WS3"`, `kind:"arch_dataset"`, `model:"deep_cnn"`, `depth:8`.

| model | dataset (depth) | cascaded n1000 (3-seed mean ± sd) | synchronized n1000 (mean ± sd) | **AC2 casc→ANN gap** | casc→sync gap | sync→ANN gap | n=200 prior (§1j) | AC2 verdict |
|:------|:----------------|:----------------------------------|:-------------------------------|---------------------:|--------------:|-------------:|------------------:|:------------|
| deep_cnn | FashionMNIST (d8) | **0.7677** (±2.51) | 0.9015 (±0.21) | **16.40pp** | +13.38 | 3.14 | +11.34 | **NOT MET (collapse, HARDENS)** |
| deep_cnn | KMNIST (d8)       | **0.8903** (±1.58) | 0.9732 (±0.37) | **9.06pp** | +8.28 | 0.24 | +7.19 | **NOT MET (degraded, HARDENS)** |

- **VALID-CONFIRMED: the §1j n=200 dataset-margin read was NOT grid-noise inflation — the gap
  HARDENS at genuine resolution.** With both arms paired at n=1000 the cascaded→sync gap
  **grows**: FMNIST +11.34 → **+13.38pp**, KMNIST +7.19 → **+8.28pp** (matching the §1r/§1s
  depth-axis n=1000-hardens precedent). The dataset-margin ordering **KMNIST (+8.28) <
  FashionMNIST (+13.38)** holds at 4-decimal fidelity. Cascaded carries all the deficit and
  spread (FMNIST sd 2.51pp / AC2 gap 16.40pp; KMNIST sd 1.58pp / 9.06pp).
- **Synchronized AC2 stays effectively MET** (sync→ANN 3.14pp FMNIST, 0.24pp KMNIST, sd
  ≤0.37pp) — the unconditional deep-model default holds at the genuine-resolution dataset
  worst case. This is the §1h/§1j d8 dataset cube re-certified at n=1000.

Run ids: `pdcnnd8datan1000_{FashionMNIST,KMNIST}_DataProvider_{cascaded,synchronized}_s{0,1,2}`.
**Confounds.** (1) **PAIRING CLEAN — both arms n=1000:** unlike the §1q lenet5 precedent that
mixed n1000-cascaded vs n50-synchronized, both arms here subsample to
`max_simulation_samples=1000`, so the prior n200-vs-10000 mismatch is fully removed — only
~±0.1pp 1000-sample binomial noise remains, far below the 8–13pp gaps. (2) **n_seeds=3 per
arm** (full 2-dataset × 2-policy × 3-seed grid, all 12 `rc=0`). (3) **DEPTH cross-reference:**
this batch is d8-only — the depth-widening is cross-referenced against the §1s depth-axis
n=1000 precedent (d8 MNIST +4.62pp), and the d8 dataset gaps (+8.28/+13.38pp) exceed the
MNIST baseline → dataset-margin **amplifies** the cascade beyond MNIST. (4) **NO at-chance
confound:** ANN refs FMNIST 0.9317 / KMNIST 0.9809 ≫ 0.10 → genuine firing-gain (fully
trained). **Next:** the firing-gain gate-fix on the d8 FMNIST collapse cell (the worst
dataset-margin corner) — but the gate-fix is REFUTED as a deep auto-rescue (WS7 §2d/§2e:
θ-cotrain `rc=1`-broken, staircase-STE/`cp` regress), so synchronized stays the deep default.

---

## 1u. AC2 on the VALID `deep_cnn` d6 DATASET cells at genuine n=1000 — the d6 rung HOLDS and the continuous FMNIST monotone-widening AC2 ladder SURVIVES (`item_id=dcnn_d6_dataset_n1000`, 2026-06-25)

§1k/§1p read the d6 dataset cells at n=200 (with §1k's KMNIST an n=1 provisional). This
firms them at **genuine n=1000**: `deep_cnn` (w16), S=4, `ttfs_cycle_based`, full 3 seeds/arm
paired by seed, cascaded `max_simulation_samples=1000`, all 12 runs `rc=0`. Ledger:
`cluster:"WS3"`, `kind:"arch_dataset"`, `model:"deep_cnn"`, `depth:6`.

| model | dataset (depth) | cascaded n1000 (3-seed mean ± sd) | synchronized (mean ± sd) | **AC2 casc→ANN gap** | casc→sync gap | sync→ANN gap | prior read | AC2 verdict |
|:------|:----------------|:----------------------------------|:-------------------------|---------------------:|--------------:|-------------:|:-----------|:------------|
| deep_cnn | FashionMNIST (d6) | **0.8247** (±0.62) | 0.8979 (±0.36) | **10.46pp** | +7.33 | 3.25 | +6.11 (n200, §1k) | lossy (degraded, HARDENS) |
| deep_cnn | KMNIST (d6)       | **0.917** (±0.43)  | 0.9598 | **5.28pp** | +4.28 | −0.34 | +5.85 (n=1 prov., §1k) | lossy (degraded, FIRMED 3-seed) |

- **SUPPORTED at high resolution; the d6 rung HOLDS and the continuous FMNIST monotone-widening
  AC2 ladder SURVIVES.** FMNIST casc→sync +6.11 (n200) → **+7.33pp** (n1000) HARDENS; KMNIST
  +5.85 (n=1 provisional) → **+4.28pp** (full 3-seed) confirms the degraded-but-gentler KMNIST
  ladder. The **FMNIST casc→sync ladder stays continuous and gapless**:
  **d4 +3.90 → d6 +7.33 → d8 +11.34 → d10 +17.91pp**, with the d6 rung slotting cleanly between
  the §1e d4 and §1j d8 anchors at 4-decimal resolution. KMNIST d6 +4.28pp sits below FMNIST d6
  +7.33pp (**dataset-margin ordering preserved**) and is consistent with the gentler KMNIST
  ladder (d4 +6.19 / d8 +7.19).
- **Synchronized AC2 stays effectively MET** (sync→ANN 3.25pp FMNIST; KMNIST −0.34pp,
  statistically at/above its ANN) — the unconditional deep-model default is reinforced at the
  within-CNN cascade-onset depth.

Run ids: `pdcnnd6datan1000_{FashionMNIST,KMNIST}_DataProvider_{cascaded,synchronized}_s{0,1,2}`.
**Confounds.** (1) **0.005-grid noise deconfounded:** cascaded deployed floats are exact
1/1000 multiples (FMNIST .817/.832/.825, KMNIST .913/.923/.915), 5× finer than n=200. (2)
**RESIDUAL eval asymmetry (minor):** cascaded eval n=1000 (1/1000 grid) vs synchronized FULL
10000 test set (its floats e.g. 0.9025/0.9594 are 1/10000 multiples) — read at the multi-pp
gap scale (7.33/4.28pp ≫ 0.1pp sampling noise), far smaller than the prior n200-vs-10000
mismatch. (3) **All 12 runs `rc=0`, full 3-seed both arms** (the §1k KMNIST n=1 provisional is
retired). (4) **NO at-chance confound:** ANN refs FMNIST ~0.93 / KMNIST ~0.96 ≫ 0.10 →
genuine; distinguishing knob = `ttfs_cycle_schedule`. **Next:** the firing-gain gate-fix at
the d6 onset on FMNIST maps recovery across the now-continuous-at-n1000 3.9–18pp ladder
(gate-fix REFUTED as deep auto-rescue per WS7 §2d/§2e).

---

## 1w. CONSOLIDATED — the VALID `deep_cnn` dataset-breadth × depth death-cascade with the SYNCHRONIZED ceiling attached (closes the §1v "no-sync-arm" gap at d8/d10) (`item_id=dcnn_dataset_breadth_depth`, 2026-06-25)

§1v reproduced the death-cascade on the VALID `deep_cnn` vehicle but had **NO synchronized
arm** — the AC2 deficit was only readable as cascaded→ANN. This item attaches the
**matched-batch synchronized companions** at d8 (`pdcnnd8databc`) and d10 (`pdcnnd10data`),
pools the d6 cells (`pdcnnd6kmfin`, `pdcnndatafix_d6`) and the gate-fix grid
(`pdcnnd10datafix`), and rolls the whole dataset-breadth × depth corpus into one AC2 cell.
`deep_cnn` (w16), S=4, `ttfs_cycle_based`, `max_simulation_samples=200`. Ledger:
`cluster:"WS3"`, `kind:"arch_dataset"`, `item_id:"dcnn_dataset_breadth_depth"` (9 rows).

| model | dataset (depth) | cascaded (mean, n) | synchronized (mean, n) | **AC2 casc→sync gap** | casc→ANN | sync→ANN | validity | AC2 verdict |
|:------|:----------------|:-------------------|:-----------------------|----------------------:|---------:|---------:|:---------|:------------|
| deep_cnn | FMNIST (d6)  | 0.855 (n=3) | — | — | +7.54 | — | VALID `rc=0` | cascaded-only (no sync companion) |
| deep_cnn | KMNIST (d6)  | 0.9183 (n=3) | — | — | +5.58 | — | VALID `rc=0` | cascaded-only (smallest deficit) |
| deep_cnn | FMNIST (d8)  | 0.790 (n=3) | 0.9034 (n=3) | **+11.34** | +14.32 | 2.98 | VALID `rc=0` | **NOT MET (collapse)** |
| deep_cnn | KMNIST (d8)  | 0.8917 (n=3) | 0.9619 (n=3) | **+7.02** | +7.57 | 0.55 | VALID `rc=0` | lossy (sync near-lossless) |
| deep_cnn | FMNIST (d10) | 0.725 (n=2†) | 0.9041 (n=3) | **+17.91** | +20.86 | 2.95 | VALID `rc=0` | **NOT MET (WORST corner)** |
| deep_cnn | KMNIST (d10) | 0.8025 (n=2†) | 0.9623 (n=3) | **+15.98** | +15.91 | −0.07‡ | VALID `rc=0` | **NOT MET (collapse)** |

† d10 cascaded n=2 (FMNIST s1 `rc=-9`, KMNIST s0 `rc=1`; sync arms full n=3). ‡ KMNIST d10
sync 0.9623 vs 200-sample ANN 0.9616 = −0.07pp = sampling artifact of the coarse n=200 ANN
eval, not a super-ANN result.

**Gate-fix grid (cotFalse cpFalse, cascaded-only) does NOT close the deep × hard deficit:**
FMNIST d10 0.750 (n=2) → +15.41pp vs the d10data sync ceiling 0.9041; KMNIST d10 0.8625
(n=2) → +9.98pp vs 0.9623 (cpTrue s2=0.865 ≈ same). The firing-gain gate knobs sit at the
**same depressed cascaded level**, corroborating (not closing) the deficit — consistent with
WS7 §2d/§2e/§12.

**Verdict — CONFIRMED; the death-cascade law holds across FMNIST + KMNIST × d6/d8/d10, the
harder dataset-margin carries the LARGER cascaded AC2 deficit at every matched depth, and
synchronized stays ~lossless.** The cascaded→sync firing-gain gap grows **monotonically with
depth** on BOTH datasets (FMNIST +11.34@d8 → +17.91pp@d10; KMNIST +7.02@d8 → +15.98pp@d10)
and is consistently larger on the harder FMNIST margin than on KMNIST at every matched depth.
Synchronized AC2 stays MET within **0.55–2.98pp** of the ANN ceiling everywhere — the deficit
is a **cascaded-firing-gain pathology, not a capacity limit**. This **closes the §1v
"no-synchronized-arm" gap at d8/d10** and consolidates §1h (d10) + §1j (d8) + the d6 cells +
the gate-fix grid into one dataset-breadth × depth item; **FMNIST × d10 is the worst corner
in the whole deep_cnn table.**

**Confounds.** (1) **No d5 runs in this corpus** (depths present d6/d8/d10) — the d5 leg is
UNANSWERED here. (2) **d10 cascaded n=2** (one crash each). (3) **d6 cells cascaded-only** (no
sync companion) so the d6 casc→sync gap is not directly measurable — only casc→ANN
(~5.6–7.5pp); duplicate d6 KMNIST cell (`pdcnndatafix_d6 cotFalse` 0.8967) consistent with
`pdcnnd6kmfin` (0.9183). (4) **All 45 valid runs `mss=200`** → 0.005 grid; **read pp-gaps,
not 3rd decimals**. (5) **NO at-chance confound:** every ANN 0.93–0.98 ≫ 0.10. (6) Gate-fix
`pdcnnd10datafix` cot/cp grid is cascaded-only — corroborating, not closing. Run ids: see
ledger `item_id:"dcnn_dataset_breadth_depth"`. (Detailed analysis:
`docs/research/findings/WS3_depth_firing_gain.md` §4w.)
- **The §1w-(1) "no d5 leg" gap is CLOSED (synthesis row, 2026-06-25):** the
  `pdcnnd5data`/`pdcnnd6databc`/`pdcnnd8databc` dataset-axis corpus (36 paired runs, all
  `rc=0`, mss=200) extends the ordering law to **d5/d6/d8** with the **dataset-margin
  ordering CONFIRMED at every depth** — FMNIST cascaded→ANN > KMNIST at d5 (8.89>5.29),
  d6 (9.00>8.42), d8 (14.30>7.71pp); synchronized near-ANN throughout (FMNIST 2.87–2.96,
  KMNIST 0.48–0.68pp). Depth-growth is **clean/monotone on FMNIST** (8.89→9.00→14.30) but
  **non-monotone on KMNIST** (d6 8.42 > d8 7.71, inside the KMNIST cascaded seed-std up to
  2.48pp). See WS3 findings §4y; synthesis ledger row
  `kind:"synthesis", item_id:"deep_cnn_dataset_axis_death_cascade"` cites the 36 run_ids.

---

## 2g. NEITHER escalation rescues the n=1000 bigcores-gatefix `deep_cnn` d8 MNIST cell — θ-cotrain CRASHES `rc=1` (unmeasurable) and conversion_policy REGRESSES −2.47pp; the cell is already near-lossless so there is almost no deficit (`item_id=dcnn_deep_controller_escalation`, 2026-06-25)

§2d/§2e/§2f exhausted the d6 onset rescue space. This extends the rescue question one rung
deeper, on the **n=1000-trained big-cores-gatefix `deep_cnn` at d8 (MNIST, S=4, cascaded
`ttfs_cycle_based`)** — the controller-escalation grid
`pdcnnbcn1000fix_d8_cot{T,F}_cp{T,F}_s{0,1,2}` (12 runs). Ledger: `cluster:"WS7"`,
`kind:"escalation"`, `item_id:"dcnn_deep_controller_escalation"`.

| arm | cot | cp | deployed (3-seed) | seeds | rc | ANN ref | AC2 verdict |
|:----|:--:|:--:|------------------:|:------|:--:|--------:|:------------|
| baseline (pure cascaded) | false | false | **0.9723** | .96/.981/.976 | 0 | ~0.974 | near-lossless (casc→ANN ~0.2pp) |
| conversion_policy escalation | false | true | **0.9477** | .978/.954/**.911** | 0 | ~0.974 | **REGRESSES −2.47pp** |
| θ-cotrain escalation (any cp) | true | — | **n/a** | — | **1** | — | **BROKEN (rc=1, unmeasurable)** |

**Verdict — NEITHER ESCALATION RESCUES; the cell is already near-lossless so there is almost
nothing to rescue.** The **θ-cotrain (cot) escalation is unmeasurable** (all 6 cotTrue runs
`rc=1`, the `Conv2DPerceptronMapper features_3` tensor-shape break; their 0.99 floats are
stale pre-deployment ANN-stage artifacts). **conversion_policy (cp) does NOT rescue:** across
the 6 finalized rc=0 cotFalse runs, cp leaves accuracy unchanged-to-WORSE — cpFalse **0.9723**
vs cpTrue **0.9477**, a cp lift of **−2.47pp** (a regression dragged by one seed collapse to
0.911, NOT a lift toward the ~0.974 ANN ceiling). The d8 cell is already **near-lossless on
cpFalse** (cascaded→ANN gap only ~0.2pp at n=1000) — there is almost no firing-gain deficit
for an escalation to close. This is the **deeper-rung confirmation of the §2d MNIST d6 cp
NO-OP/REGRESSION and the §2d–§2f cotTrue `rc=1` crash**; synchronized remains the
unconditional deep_cnn default.

**Confounds.** (1) **cot unmeasurable** (all 6 cotTrue `rc=1`, `q/failed/`; 0.99 floats are
stale ANN-stage artifacts). (2) **cp regression one-seed-driven** (cpTrue 0.978/0.954/0.911;
sd 2.77pp vs cpFalse 0.90pp). (3) **`mss=1000`** → read pp-gaps. (4) **NO at-chance confound:**
in-log ANN 0.9704/0.9807/0.972 ≫ 0.1135. (5) **NO paired synchronized arm** in this batch.
(6) **Negative-rescue cell:** the near-lossless baseline (casc→ANN ~0.2pp) shows the
escalation levers do not HELP and can HURT even where the deficit is small. Run ids: baseline
`pdcnnbcn1000fix_d8_cotFalse_cpFalse_s{0,1,2}`; cp escalation
`…_cotFalse_cpTrue_s{0,1,2}`; cot-crashed `…_cotTrue_cp{False,True}_s{0,1,2}`. (Detailed
analysis: `docs/research/findings/WS7_keystone_automatic.md` §12.)

---

## 1x. The full 4-dataset lenet5 cascaded→sync AC2 table is COMPLETE at matched n=1000 — MNIST/FMNIST synchronized arms (§1i PENDING) FINALIZED; SVHN cascaded is a PARITY-GATE FAILURE, not a result (`item_id`s `lenet_sync_n1000_complete_cnn_gap` + `lenet_cascade_kmnist_rung_svhn_parityfail`, 2026-06-25)

§1i/§1q left the MNIST and FashionMNIST n=1000 synchronized arms PENDING (gap mixed
n1000-cascaded vs n50-synchronized), and §1q's KMNIST gap (+1.79pp) paired the `csr_lenet`
cascaded arm. This batch FINALIZES the missing `plnsync_lenet_{MNIST,FashionMNIST}_synchronized_n1000`
arms and re-reads KMNIST/SVHN on the `plnmargin` cascaded arm against `plncpair` synchronized
— so all four CNN datasets now carry a **paired same-resolution n=1000 cascaded→sync AC2 gap**.
Configs byte-identical except `ttfs_cycle_schedule`; TTFS `ttfs_cycle_based`, S=4, `mss=1000`.
Ledger: `cluster:"WS3"`, `kind:"arch_dataset"`, `model:"lenet5"`.

| dataset | ANN ref | cascaded n1000 (3-seed ± sd) | synchronized n1000 (3-seed ± sd) | **casc→sync gap (matched)** | casc→ANN | AC2 verdict |
|:--------|--------:|:-----------------------------|:---------------------------------|----------------------------:|---------:|:-----------|
| MNIST | 0.9922 | **0.9873** (±0.25) | **0.9894** (±0.11) | **+0.21pp** | 0.48pp | near-lossless / MILD (gap < seed sd) |
| KMNIST | 0.9600 | **0.9310** (±0.10) | **0.9519** (±0.37) | **+2.09pp** | 2.90pp | mild firing-gain residual |
| FashionMNIST | 0.9176 | **0.8397** (±0.84) | **0.8911** (±0.48) | **+5.14pp** | 7.79pp | real MODERATE residual (not mild) |
| SVHN | 0.8945 | *cascaded all `rc=1` — PARITY-GATE FAIL* | **0.8593** (±0.44) | *null* | — | cascaded sync-only / UNAVAILABLE |

- **§1i confound CLOSED for MNIST/FMNIST.** With the paired n1000 sync arms finalized,
  MNIST casc→sync = **+0.21pp** (near-lossless, below seed sd) and FashionMNIST = **+5.14pp**.
  The FMNIST figure TIGHTENS the §1i n50-context 6.02pp, because true n1000 sync (0.8911)
  sits below the n50 sync baseline (0.8999) — and at +5.14pp FMNIST is a **real MODERATE
  firing-gain residual above the 1–2pp "mild" band**, refining the §1i "MILD and
  dataset-stable" framing for the hardest greyscale margin. Gap orders monotonically by
  dataset margin: MNIST 0.21 < KMNIST 2.09 < FMNIST 5.14pp. **AC2 MET (near-lossless) on
  MNIST; partially MET (mild) on KMNIST; bounded-lossy on FMNIST** — all far from the
  `deep_mlp` death-cascade (10–16pp).
- **The §1q KMNIST gap is REPRODUCED on an independent cascaded arm.** `plnmargin` cascaded
  KMNIST (0.931, NF↔SCM agreement 1.0000) → sync gap +2.09pp ≈ §1q's +1.79pp (`csr_lenet`
  arm) → the KMNIST mild residual is stable across cascaded vehicles.
- **SVHN cascaded is a deployment-fidelity FAILURE.** All 3 `plnmargin` SVHN cascaded seeds
  `rc=1` in `q/failed/`, crashing the TTFS Cycle Fine-Tuning `_run_nf_scm_parity_gate`
  (`soft_core_mapping_step.py:312` → `nf_scm_parity.py:176` `NfScmParityError`) with cascaded
  decision agreement **0.8906/0.7812/0.8750 < 0.98**. Post-crash deployed floats (~0.69/0.66)
  are gate-fail artifacts, NOT a metric ⇒ `cascaded_to_sync_gap_pp=null`,
  `cascaded_run_finalized=false`. The §1q parallel `plncpair` cascaded SVHN arm ALSO `rc=1`,
  corroborating. The synchronized arm (0.8593, sync→ANN 3.52pp) is the only valid SVHN number.

Run ids: `csr_lenet_{MNIST,FashionMNIST}_DataProvider_cascaded_n1000_s{0,1,2}` paired with
`plnsync_lenet_{MNIST,FashionMNIST}_DataProvider_synchronized_n1000_s{0,1,2}`;
`plnmargin_lenet_{KMNIST,SVHN}_DataProvider_cascaded_n1000_s{0,1,2}` paired with
`plncpair_lenet_{KMNIST,SVHN}_DataProvider_synchronized_n1000_s{0,1,2}`.
**Confounds:** (1) No at-chance confound — every ANN ref ≫ chance (SVHN 0.893–0.897 vs 0.196).
(2) `mss=1000` on all valid arms → pp-gaps and 2–3 sig-fig reads trustworthy; KMNIST cascaded
NF↔SCM = torch↔sim = 1.0000 all seeds. (3) lenet5 depth-axis stress is modest (IR max-latency
~3) → this is the dataset-margin breadth axis, not the deep death-cascade (§1d/§1r). (4) SVHN
cascaded matched gap stays OPEN pending a fidelity fix for the parity-gate crash. **This closes
the §1i/§1q/§3 "no paired n=1000 synchronized lenet5 run" gap for MNIST/FMNIST/KMNIST and flags
SVHN cascaded sync-only.**

---

## 2h. The §2g d8 escalation extends to d10 against a FULL-EVAL trained-ANN reference — the cascaded AC2 deficit WIDENS d8(2.26pp)→d10(4.83pp) and `conversion_policy` is NET-NEGATIVE at BOTH rungs (`item_id=dcnn_deep_n1000_gatefix_d8_d10`, 2026-06-25)

§2g read the n=1000 bigcores-gatefix `deep_cnn` d8 escalation against an *in-log* ANN
(~0.9744). This row re-frames d8 **and the next rung d10** against the **full-eval trained
ANN** (0.9949 / 0.9916) and pairs the `conversion_policy` lever at both depths. `deep_cnn`
(w16), `ttfs_cycle_based`, `ttfs_cycle_schedule=cascaded`, S=4, `max_simulation_samples=1000`,
on-chip-majority VALID. Ledger: `cluster:"WS3"`, `kind:"depth"`,
`item_id:"dcnn_deep_n1000_gatefix_d8_d10"`.

| depth | cpFalse cascaded baseline (3-seed) | trained ANN (AC2 target) | **AC2 casc→ANN gap** | cpTrue rescue | cp lift | rescue n | AC2 verdict |
|:-----:|-----------------------------------:|-------------------------:|---------------------:|--------------:|--------:|:--------:|:------------|
| **d8**  | **0.9723** (.96/.981/.976) | 0.9949 | **2.26pp** | 0.9477 | **−2.46pp** | 3 | near-lossless (mild), cp REFUTED |
| **d10** | **0.9433** (.892/.96/.978) | 0.9916 | **4.83pp** | 0.925 | **−1.83pp** | 2 (s2 `rc=1`) | degraded (widens), cp REFUTED |

- **The cascaded AC2 deficit WIDENS with depth** (d8 2.26pp → d10 4.83pp) against the
  full-eval trained ANN — degraded but **near-lossless, not collapse** (conv inductive bias
  caps severity ~5pp), consistent with the §1s sharp-onset → bounded-plateau ladder.
- **`conversion_policy` rescue REFUTED at BOTH rungs** (d8 −2.46pp, d10 −1.83pp) — a genuine
  net-negative lever, **not a benign no-op**; at d10 it additionally **trips the NF↔SCM parity
  gate** (s2 `rc=1`, `NfScmParityError` agreement 0.9531 < 0.98, a wrong-NF-dynamics incident
  *induced by the lever*). This is the **depth-extension of §2g** (cp net-negative at d8) and
  confirms **no working `conversion_policy` firing-gain rescue at depth on the convnet**;
  synchronized remains the unconditional deep_cnn default.

**Confounds.** (1) **No at-chance confound** — trained ANN ~0.99 at every cell (d8
.9961/.993/.9955, d10 .9888/.9956/.9904 ≫ 0.1135) ⇒ genuine firing-gain regime. (2)
`mss=1000` → adequate resolution. (3) **No synchronized arm in this batch** — the pairing axis
is `conversion_policy`, so `casc→sync gap = null`; the ~0.99 lossless reference is the trained
ANN plus the §1g/§1s synchronized deep_cnn ceiling (0.990–0.994). (4) **d10 cpTrue is n=2**
(s2 excluded per `rc==0`; its 0.9054 `__target_metric` is a pre-crash value not counted). (5)
The companion `ttfs_theta_cotrain` lever is unanalyzed — all cotTrue runs crashed `rc=1`
(`Conv2DPerceptronMapper features_3`, same break as §2d–§2g). (6) The d8 run_ids are also
cited in §2g (`dcnn_deep_controller_escalation`, in-log ANN 0.9744); this row uses the
full-eval ANN 0.9949 and the depth-pairing framing. Run ids: cpFalse
`pdcnnbcn1000fix_d{8,10}_cotFalse_cpFalse_s{0,1,2}`; cpTrue
`pdcnnbcn1000fix_d{8,10}_cotFalse_cpTrue_s{0,1,2}` (d10 s2 `rc=1`). (Detailed analysis:
`docs/research/findings/WS3_depth_firing_gain.md` §4z.)

---

## 2i. AC2 on the in-distribution VALID `mlp_mixer_core` keystone vehicle, OFF-MNIST — the near-lossless MNIST cascade is dataset-stable on FashionMNIST (−3.25pp) but DEGRADES on KMNIST (−9.53pp, ~3× FMNIST) (`item_id=mmix_blendoff_dataset_axis`, 2026-06-25)

The WS7 §0.7 keystone MATCH (blend-OFF raw cascade survives, no escalation needed) was
demonstrated on `mlp_mixer_core` **MNIST**. This row extends that exact blend-OFF / cpTrue
cascade to the **off-MNIST dataset axis** to test whether the keystone vehicle's near-lossless
AC2 is dataset-stable. `mlp_mixer_core`, `ttfs_cycle_based`, `ttfs_cycle_schedule=cascaded`,
`ttfs_blend_fast=false`, `conversion_policy=true`, S=4, `max_simulation_samples=1000`. 6 runs,
3 seeds/dataset, all `rc=0`. Ledger: `cluster:"WS7"`, `kind:"escalation"`,
`item_id:"mmix_blendoff_dataset_axis"`.

| dataset | deployed (cascaded, 3-seed mean) | seeds | ANN ref (AC2 target) | **AC2 deployed→ANN gap** | parity | AC2 verdict |
|:--------|---------------------------------:|:------|---------------------:|-------------------------:|:-------|:------------|
| MNIST (§0.7 ref) | 0.9547 | .954/.956/.954 | 0.9832 | −2.85pp | 1.0 / 1.0 | keystone MATCH (in-distribution) |
| **FashionMNIST** | **0.8547** | .870/.858/.836 | 0.8871 | **−3.25pp** | NF↔SCM 1.0, torch↔sim 1.0 | **MATCH / robust off-MNIST** |
| **KMNIST** | **0.8067** | .815/.798/.807 | 0.9020 | **−9.53pp** (~3× FMNIST) | NF↔SCM 1.0, torch↔sim 1.0 | **DEGRADE off-MNIST** |

- **AC2 keystone MATCH holds on FashionMNIST but NOT on KMNIST.** The blend-OFF cascade
  survives architecturally robust on FashionMNIST (−3.25pp, close to the MNIST keystone's
  −2.85pp), but on the harder KMNIST distribution it opens a real **−9.53pp AC2 gap (~3×
  FMNIST)** — a genuine firing-gain deficit the raw cascade does not survive. This **bounds
  the §0.7 "no escalation needed" verdict to MNIST + FashionMNIST**.

**Confounds.** (1) **NOT a cascaded-vs-synchronized nor cp true-vs-false pairing** — all 6 runs
are cascaded blend-OFF cpTrue (no blend-ON `pm_casc_mmix`, no cpFalse arm on FMNIST/KMNIST), so
`cascaded_to_sync_gap_pp` is **repurposed as the AC2 ANN gap** and `synchronized_*` is
null/empty. (2) **Not a chance/untrained artifact** — ANN healthy (FMNIST 0.8871, KMNIST 0.9020
≫ 0.10) ⇒ genuine firing-gain result. (3) **Deployed metric faithful** — NF↔SCM cascaded
agreement 1.0 and torch↔sim parity 1.0 on all 6 runs. (4) `mss=1000` → ~±1pp granularity; read
the AC2 gaps (FMNIST seed spread .836–.870 ≈ 3.4pp). (5) All 6 `rc=0`, artifact_ok, 3 seeds.
(6) **ESCALATE-vs-MATCH not separable** from logs — the KMNIST −9.53pp gap shows the cascade IS
deficient there but cannot prove whether the absent controller/blend recovery would close it.
Run ids: `pmmixnb_{FashionMNIST,KMNIST}_DataProvider_cpTrue_s{0,1,2}`. (Detailed analysis:
`docs/research/findings/WS7_keystone_automatic.md` §0.8.)

---

## 1y. CONSOLIDATED — the two CONFIRMED `deep_cnn` AC2 death-cascade items with the synchronized ceiling attached: a BOUNDED MNIST depth ladder + a depth-first dataset-breadth re-opening off MNIST (`item_id`s `dcnn_deep_death_cascade_ladder` + `dcnn_dataset_breadth_cascaded`, 2026-06-25)

These two consolidated items fold the clean-`rc=0` `deep_cnn` cascaded corpus into the AC2
form §8/§10 demand — every cell carries its **own ANN reference** and the **synchronized arm**
(the AC2-lossless instrument). VALID on-chip-majority (99.6% on-chip) `deep_cnn` w16 bigcores,
`ttfs_cycle_based`, S=4. AC2 read = cascaded→ANN gap; synchronized→ANN gap reported separately
to show the deployable ceiling. Ledger: `cluster:"WS3"`, `kind:"depth"` (4 rows, MNIST ladder)
+ `kind:"arch_dataset"` (4 rows, dataset breadth).

**(A) MNIST depth ladder — n=1000 lower-noise family.**

| depth | cascaded (AC2 deployed) | synchronized | ANN ref (AC2 target) | **cascaded→ANN (AC2)** | sync→ANN | validity | AC2 verdict |
|:-----:|------------------------:|-------------:|---------------------:|-----------------------:|---------:|:---------|:------------|
| d5 (ref) | 0.9917 | 0.9924 | 0.9913 | ≈0 | ≈0 | VALID | **near-lossless / tied** |
| **d6**  | 0.9563 | 0.9924 | 0.9914 | **−3.51pp** | +0.10pp | VALID | onset |
| **d8**  | 0.9450 | 0.9912 | 0.9921 | **−4.71pp** | −0.09pp | VALID | degraded |
| **d10** | 0.9328 | 0.9912 | 0.9928 | **−6.00pp** | −0.16pp | VALID | degraded |
| **d12** | 0.9353 | 0.9920 | 0.9923 | **−5.69pp** | −0.03pp | VALID | degraded (bounded) |

Synchronized **MEETS AC2 at every depth** (sync→ANN within ±0.16pp). Cascaded AC2 is
**near-lossless through d5, opens at d6, widens to a BOUNDED ~4–6pp plateau** — NOT a monotone
collapse (the n=200 family reads a sharper d12 9.48pp; read the ladder direction).

**(B) Dataset breadth off MNIST — n=200.**

| dataset | depth | cascaded (AC2 deployed) | synchronized | ANN ref | **cascaded→ANN (AC2)** | sync→ANN | AC2 verdict |
|:--------|:-----:|------------------------:|-------------:|--------:|-----------------------:|---------:|:------------|
| FashionMNIST | d5  | 0.8383 | 0.8986 | 0.9283 | **−9.00pp** | −2.97pp | lossy |
| KMNIST       | d5  | 0.9167 | 0.9629 | 0.9696 | **−5.29pp** | −0.67pp | lossy |
| FashionMNIST | d10 | 0.7250 | 0.9041 | 0.9336 | **−20.86pp** | −2.95pp | collapse |
| KMNIST       | d10 | 0.8025 | 0.9623 | 0.9616 | **−15.91pp** | +0.07pp | collapse |

The cascade **re-opens off MNIST**, ordered by **DEPTH first** (d10 ≫ d5), then dataset, while
synchronized stays near-lossless (sync→ANN +0.07 to −2.97pp). The cascaded deficit is **LARGER
on FashionMNIST than on the nominally-harder KMNIST** at both depths — the CNN cascade does NOT
reproduce the `deep_mlp` "harder-dataset = bigger-gap" ordering.

**Headline-gating ruling.** The depth × firing-gain death-cascade is **REAL on the VALID
vehicle but BOUNDED** vs the retired host-majority `deep_mlp` (delayed onset ~d6 vs d4; smaller
MNIST deficit ~4–6pp vs 4.3→9.3pp). **Synchronized is the unconditional deep-model AC2 default.**

**Confounds.** (1) **Two resolution families — read pp gaps, not 3rd decimals.** Both pair a
subsampled-cascaded arm against a full-10000-sample synchronized SCM (means unbiased; gaps
robust >2–4× the per-seed band; sub-0.2pp d5 gaps are noise). (2) **d10 dataset cells are n=2**
(FMNIST cascaded s1 `rc=-9`, KMNIST cascaded s0 `rc=1` — excluded per strict `rc==0`; their
consistent artifacts would not change the verdict direction). (3) **Crash exclusions** —
`pdcnnbc_d12_cascaded` s0/s2 (OOM `rc=-9`) and the `dcnn_d6/d8_*` / `pdcnnladder_d6/d7_*` core-
exhaustion `rc=1` runs are quarantined; the valid evidence is the `pdcnnbcn1000*` (A) and
`pdcnnd{5,10}data_*` (B) families. (4) **No at-chance confound** (every ANN ≫ 0.10/0.1135;
parity gates clean). Run ids — A: `pdcnnbcn1000plat_d{6,8,10}_*_s{0,1,2}` +
`pdcnnbcn1000seed_d{8,10,12}_*_s{3,4,5}`; B: `pdcnnd{5,10}data_{FashionMNIST,KMNIST}_DataProvider_{cascaded,synchronized}_s{0,1,2}`.
(Detailed analysis: `docs/research/findings/WS3_depth_firing_gain.md` §4aa.)

---

## 1z. AC2 on the shallow `deep_cnn` d4 MNIST rung — the depth floor reads ZERO on the VALID vehicle; cascaded HOLDS at the synchronized/ANN ceiling, replacing closeout v2 §10.1's INVALID `deep_mlp` shallow datapoint (`item_id=dcnn_d4_mnist_cascaded_vs_sync_ci`, 2026-06-25)

The §1s ladder placed the death-cascade **onset** at ~d6 and a BOUNDED ~4–7pp plateau above it.
This rung pins the **floor below onset** on the VALID on-chip-majority `deep_cnn` vehicle (w16,
MNIST, `ttfs_cycle_based` S=4; sole config diff `ttfs_cycle_schedule` cascaded vs synchronized),
and is the VALID-vehicle replacement for closeout v2 §10.1's INVALID host-majority `deep_mlp`
shallow-rung death-cascade datapoint.

| model | dataset | depth | regime | deployed (mean ± sd_pp) | n_seeds | AC2 ref (ANN) | AC2 deployed→ANN | casc→sync gap | validity | AC2 verdict |
|:------|:--------|:-----:|:-------|:------------------------|:-------:|--------------:|-----------------:|--------------:|:---------|:------------|
| deep_cnn | MNIST | d4 | cascaded TTFS S=4 | **0.9867** (±1.89) | 3 | 0.9922 | **−0.55pp** | **−0.34pp** | VALID (rc=0) | **near-lossless** (gap within n=50 resolution) |
| deep_cnn | MNIST | d4 | synchronized TTFS S=4 | 0.9901 (±0.12) | 5 | 0.9924 | −0.23pp | — | VALID (rc=0) | **lossless** |

- **AC2 effectively MET; depth floor = ZERO at d4.** Cascaded holds **at/above** synchronized;
  the −0.34pp casc→sync and −0.55pp cascaded→ANN "gaps" are **within the 50-sample 2pp/sample
  resolution** and read as **zero**, so the §6 depth-risk floor is **nonzero=FALSE at d4**.
- **Confound (read gaps, not 3rd decimals):** `max_simulation_samples=50` → cascaded values are
  exactly `{50/50, 48/50, 50/50}`; the 0.96 seed is a single 2-error sample, std 1.89pp is
  small-sample noise. Cascaded seeds **s1/s4 FAILED `rc=1`** (excluded → n=3); synchronized is
  n=5. ANN ≈ 0.992 ≫ chance (genuinely trained, legitimate firing-gain comparison). Run ids:
  `f1_deep_cnn_mnist_ci_MNIST_DataProvider_{cascaded_d4_s{0,2,3},synchronized_d4_s{0,1,2,3,4}}`;
  excluded `...cascaded_d4_s{1,4}` (`rc=1`).

## 2j. NEITHER firing-gain rescue lever recovers the d6 off-MNIST cascaded deficit on the VALID `deep_cnn` vehicle — θ-cotrain CRASHES `rc=1` (unmeasurable) and `conversion_policy=controller` REGRESSES; the best arm never reaches the synchronized ceiling, contradicting closeout §10.2's positive controller-rescue on the INVALID `deep_mlp` d8 (`item_id=dcnn_d6_theta_cotrain_cp_rescue_fmnist_kmnist`, 2026-06-25)

Companion to §2g/§2h (which probed the MNIST d8/d10 rescue, where the cell is already
near-lossless). This batch tests the same two levers at the d6 **onset** rung **off MNIST**,
where there is a real deficit to rescue — a 2×2 over `ttfs_theta_cotrain` ×
`conversion_policy=controller` on the VALID `deep_cnn` d6 (w16, S=4, `ttfs_cycle_based`
cascaded) vehicle, FashionMNIST and KMNIST.

| dataset | arm | deployed (mean ± sd_pp) | n_seeds (rc=0) | AC2 ref (ANN) | →sync ceiling | reaches ceiling? |
|:--------|:----|:------------------------|:--------------:|--------------:|--------------:|:----------------:|
| FashionMNIST | cotFalse_cpFalse (**best**) | **0.8283** (±2.78) | 3 | 0.9312 | **+6.79pp below** | **no** |
| FashionMNIST | cotFalse_cpTrue | 0.8217 (±2.01) | 3 | 0.9307 | +7.45pp below | no |
| FashionMNIST | cotTrue_cp{False,True} | **CRASH `rc=1`** | 0/3 each | — | unmeasurable | — |
| FashionMNIST | synchronized (ceiling) | 0.8962 | 3 | — | — | — |
| KMNIST | cotFalse_cpFalse (**best**) | **0.9167** (±1.03) | 3 | 0.9654 | **+4.53pp below** | **no** |
| KMNIST | cotFalse_cpTrue | 0.8583 (±4.52) | 3 | 0.9711 | +10.36pp below | no |
| KMNIST | cotTrue_cp{False,True} | **CRASH `rc=1`** | 0/3 each | — | unmeasurable | — |
| KMNIST | synchronized (ceiling) | 0.9619 | 3 | — | — | — |

- **Neither lever rescues; AC2 NOT MET on any arm.** `theta_cotrain` is **UNMEASURABLE** (all
  12 `cotTrue` runs crash `rc=1`). `conversion_policy=controller` **REGRESSES** the cascade:
  cpFalse→cpTrue lift **−0.67pp** (FMNIST), **−5.83pp** (KMNIST). The best arm stays **+6.79pp**
  (FMNIST) / **+4.53pp** (KMNIST) **below** the synchronized ceiling.
- **Contradicts closeout §10.2.** The positive controller-auto-rescue lift cited there was
  measured on the **INVALID host-majority** `deep_mlp` d8; on the VALID `deep_cnn` vehicle the
  controller is **net-negative**, consistent with §2g/§2h (controller net-negative at d8/d10 MNIST).
- **Confounds:** (1) All `cotTrue` crash `rc=1` with `Conv2DPerceptronMapper(name=features_3)`
  forward failure (`tensor a (28) vs b (16) at dim 3`) at TTFS-cycle-FT start; on-disk
  `__target_metric` ~0.929–0.932 is a **STALE ANN-stage** value, not deployed. (2) KMNIST
  cpTrue s0 = 0.80 is a **genuine `rc=0` finalized collapse** (99.41% on-chip, agreement/parity
  1.0), i.e. the controller is high-variance — not a crash. (3) ANN ≫ chance. Run ids:
  `pdcnnd6datacotfix_{FashionMNIST,KMNIST}_DataProvider_cot{False,True}_cp{False,True}_s{0,1,2}`
  (cotTrue `rc=1`-excluded), synchronized `pdcnnbcd6data_{FashionMNIST,KMNIST}_DataProvider_synchronized_s{0,1,2}`.

## 2k. At the SHALLOWEST off-MNIST cascaded rung (`deep_cnn` d5, S=4), the `ttfs_staircase_ste` gradient only PARTIALLY closes the AC2 deficit — a clean +1.4pp lift on KMNIST, a wash (+0.67pp, sign-flips by seed) on FashionMNIST; neither reaches the prior-item synchronized ceiling (`item_id=dcnn_d5_ste_onset`, 2026-06-25)

Companion to §2f (the same STE knob at the EASIER MNIST d6 onset, where it HALVES the deficit)
and §2e (STE REGRESSES the d6 dataset axis). This batch (`pdcnnd5stefix_*`) probes the STE lever
one rung **shallower**, at the d5 cascaded onset off MNIST, on the VALID `deep_cnn` vehicle. The
in-batch lever is `ttfs_staircase_ste` (steTrue vs steFalse), **both at
`ttfs_cycle_schedule=cascaded`** — NOT a cascaded-vs-sync contrast. The sync ceilings/cascaded
baselines come from the PRIOR `pdcnnd5data_` item (§1n, `item_id=dcnn_d5_dataset_axis`).

| dataset | arm | deployed mean | n (rc=0) | AC2 ref (ANN) | STE Δ (pp) | →prior sync ceiling |
|:--------|:----|:-------------:|:--------:|--------------:|:----------:|--------------------:|
| FashionMNIST | steFalse (cascaded) | 0.8067 | 3 | 0.9270 | — | +9.19pp below 0.8986 |
| FashionMNIST | steTrue | 0.8133 | 3 | 0.9270 | **+0.67** (sign-flips +5.5/+2.5/−6.0) | +8.53pp below 0.8986 |
| KMNIST | steFalse (cascaded) | 0.8775 | **2** | 0.9689 | — | +8.54pp below 0.9629 |
| KMNIST | steTrue | 0.8917 | 3 | 0.9689 | **+1.42** (+5.5/+1.5 shared seeds) | +7.12pp below 0.9629 |

- **STE PARTIAL on KMNIST, WASH on FashionMNIST; AC2 NOT MET on either dataset.** Clean +1.42pp
  KMNIST lift (best steTrue 0.935 toward 0.9629 sync) vs a +0.67pp FashionMNIST wash (within
  200-sample noise, per-seed sign-flips). Neither reaches the prior-item sync ceiling (residual
  +7.12pp KMNIST / +8.53pp FashionMNIST).
- **Consistent with §2f/§2e.** STE is the dominant-but-not-lossless gate-fix knob (§2f) and is
  dataset-dependent (§2e regression on the d6 dataset axis); at the d5 onset it narrows but does
  not close the gap, motivating a stronger gradient (depth-aware surrogate) as the next CODE bet.
- **Confounds:** (1) lever is STE-on/off, NOT cascaded-vs-sync — both arms cascaded. (2)
  `max_simulation_samples=200` → 0.005 grid, read pp not 3rd decimals. (3) KMNIST steFalse is
  **n=2**: seed s1 (`pdcnnd5stefix_KMNIST_DataProvider_steFalse_s1`) is in `q/failed/` (`rc=-9`
  SIGKILL, log stops at TTFS-cycle-FT before deployment); its on-disk `__target_metric` 0.9559
  is a **STALE pre-deployment ANN** value, excluded. (4) ANN ≫ chance → genuine firing-gain gap.
  (5) FashionMNIST per-seed STE delta unstable (+5.5/+2.5/−6.0) → +0.67pp not significant at n=3.
  Run ids: `pdcnnd5stefix_{FashionMNIST,KMNIST}_DataProvider_ste{False,True}_s{0,1,2}`
  (KMNIST steFalse s1 `rc=-9`-excluded).

## 2l. NEITHER firing-gain rescue lever recovers the cascaded AC2 deficit at the DEEPEST trainable CNN rung (`deep_cnn` d10) — θ-cotrain CRASHES `rc=1` (unmeasurable) and `conversion_policy=controller` is NET-NEGATIVE; the genuine floor orders by dataset margin, extending §2j's d6 verdict one rung deeper (`item_id=dcnn_d10_firing_gain_rescue_levers`, 2026-06-25)

Companion to §2j (the same `ttfs_theta_cotrain` × `conversion_policy=controller` 2×2 at the
**d6 onset**) carried to the **deepest trainable CNN rung** (`deep_cnn` d10, w16, S=4,
`ttfs_cycle_based` cascaded) on MNIST, FashionMNIST and KMNIST — the §1g/§1w death-cascade rung
where the AC2 deficit is largest. **Both axes are cascaded (NO synchronized arm in this batch)**;
the sync ceilings are sibling-`bigcores`-item context (FMNIST 0.9041, KMNIST 0.9623), so every
record's `cascaded_to_sync_gap_pp` is NULL and the AC2 gap below is **cascaded→ANN only**.

| dataset | arm | deployed (mean) | n_seeds (rc=0) | AC2 ref (ANN) | casc→ANN gap | reaches ANN? |
|:--------|:----|:---------------:|:--------------:|--------------:|-------------:|:------------:|
| MNIST | cotFalse_cpTrue (**only valid cell**) | **0.79** (0.865/0.715) | 2 | 0.9899 | **+19.99pp** | **no** |
| MNIST | cotFalse_cpFalse *(corroborative)* | ~0.962 | **0** (all `rc=-9`) | — | — | EXCLUDED |
| FashionMNIST | cotFalse_cpFalse (**floor**) | **0.75** (0.755/0.745) | 2 | 0.9377 | **+18.77pp** | **no** |
| FashionMNIST | cotFalse_cpTrue | — | **0** (all crash) | — | — | unmeasurable |
| KMNIST | cotFalse_cpFalse (**floor**) | **0.8625** (0.885/0.84) | 2 | 0.9615 | **+9.90pp** | **no** |
| KMNIST | cotFalse_cpTrue | 0.865 | 1 | 0.9807 | +11.57pp | no |
| ALL | cotTrue_cp{False,True} | — | **0/18** (`rc=1`) | — | — | unmeasurable |

- **Neither lever rescues; AC2 NOT MET on any arm.** `theta_cotrain` is **UNMEASURABLE** (all 18
  `cotTrue` runs crash `rc=1` on the unlanded `Conv2DPerceptronMapper(features_3)` forward bug).
  `conversion_policy=controller` is **NET-NEGATIVE**: the only directly-paired comparison (KMNIST)
  reads cpFalse 0.8625 vs cpTrue 0.865 = **+0.0025pp FLAT**, and MNIST cpTrue **collapses to 0.79**
  (s1 0.715) vs the `rc=-9`-corroborative cpFalse ~0.962. The genuine `cotFalse_cpFalse` floor
  orders by dataset margin (MNIST > KMNIST +9.9pp > FMNIST +18.77pp).
- **Extends §2j one rung deeper.** Confirms the d6 "neither firing-gain lever rescues / θ-cotrain
  broken / controller net-negative" verdict at d10, the deepest trainable rung — consistent with
  §2g/§2h (controller net-negative at d8/d10 MNIST). **Synchronized stays the unconditional deep ×
  hard default;** the cascaded gate-fix levers do NOT close the deep d10 deficit on the VALID
  `deep_cnn` vehicle.
- **Confounds:** (1) all 12 `cotTrue` crash `rc=1`; their `__target_metric` == ANN bit-for-bit
  (MNIST 0.9947, FMNIST 0.9351, KMNIST 0.9676) = STALE pre-deploy artifact, EXCLUDED. (2) MNIST
  `cotFalse_cpFalse` all 3 seeds `rc=-9` → corroborative-only, EXCLUDED (only valid MNIST cell is
  cpTrue, n=2). (3) NO synchronized arm → `cascaded_to_sync_gap_pp` NULL; AC2 gap is casc→ANN. (4)
  n≤2 on every valid cell; FMNIST cpTrue has ZERO valid seeds. (5) `max_simulation_samples=200` →
  0.005 grid, read gaps. (6) ANN ≫ chance on every cell → genuine firing-gain. (7) cp directly
  paired only on KMNIST. Run ids: `pdcnnd10fix_cot{False,True}_cp{False,True}_s{0,1,2}` (MNIST) +
  `pdcnnd10datafix_{FashionMNIST,KMNIST}_DataProvider_cot{False,True}_cp{False,True}_s{0,1,2}`
  (cotTrue all `rc=1`-excluded; MNIST cpFalse all `rc=-9`-excluded).

## 3. Open AC gaps (what these cells do NOT yet certify)

- ~~**No paired n=1000 synchronized lenet5 run**~~ **CLOSED (§1x, 2026-06-25)** — the
  MNIST/FMNIST/KMNIST cascaded→synchronized AC2 gaps are now read at matched n=1000
  (MNIST +0.21 / KMNIST +2.09 / FMNIST +5.14pp). **Residual open item:** the **SVHN
  cascaded** lenet5 cell is a PARITY-GATE FAILURE (all 3 `plnmargin` seeds `rc=1`,
  cascaded NF↔SCM agreement 0.78–0.89 < 0.98) → SVHN matched cascaded→sync gap stays
  OPEN pending a deployment-fidelity fix (proposed: WS3 `plan_stage:28`).
- **Within-CNN depth ladder now reaches d12 — and the death-cascade APPEARS (§1d)** —
  the d6/d8/d10/d12 rungs closed the §1c "no-collapse" question: cascaded AC2 breaks with
  depth. **The d4–d8 rungs are now CLEAN-FINALIZED `rc=0` (§1f)** on the `pdcnnbc_`/
  `pdcnnladder_`/`dcnn_` vehicle — a sharp d5→d6 onset to a **~5pp plateau** (d6 5.39pp,
  d8 5.00pp), tighter than §1d's `rc=1` d10/d12 (~11–14pp). **The d10/d12 rungs are now
  CLEAN-FINALIZED `rc=0` on the `pdcnnbc_` bigcores vehicle (§1g):** d10 = **−4.00pp**
  (the prior `rc=1` ~13.86pp was crash-inflated), confirming a **BOUNDED ~4–5pp plateau,
  NOT a deepening collapse**; synchronized is **lossless through d12** (+0.30pp vs ANN).
  A genuine n=1000 cross-check (§1g) HARDENS rather than shrinks the gap (8.51/11.14pp).
  **Remaining open cell:** **d12 cascaded is UNMEASURED (n=1** — only s1 `rc=0`; s0/s2
  killed `returncode=-9`); re-run d12 cascaded s0/s2 to finalize the deepest rung
  (proposed: WS3 `plan_stage:20`). The gate-fix at the d6 onset rung is the follow-up
  (proposed: WS3 `plan_stage:19` companion).
- **The deep_cnn dataset axis is now opened (§1e) and FULLY `rc=0`-CLEAN (§1j, §1h)** —
  d4/d8/d10 × FMNIST/KMNIST show the cascade re-opens off MNIST (d4 FMNIST 5.76pp, KMNIST
  8.17pp) and widens with margin. **The d4 cells are clean `rc=0`; the d8 cells are now
  CLEAN-FINALIZED `rc=0` on the `pdcnnd8databc_` bigcores vehicle (§1j, `plan_stage:17`):**
  FMNIST casc→sync **+11.34pp** (confirms the §1e `rc=1` +11.98pp), KMNIST **+7.19pp**;
  the §1e d8 `NON_FINALIZED_rc1` confound is **CLOSED** (only KMNIST cascaded s2 still
  running, n=2). The **d10 collapse rung on the harder datasets is now CLOSED (§1h):** on the
  clean `rc=0` `bigcores` vehicle the deep × hard corner is the **worst case in the table**
  — FMNIST d10 casc→ANN **20.97pp** (casc→sync +17.91), KMNIST d10 casc→ANN **16.38pp**
  (casc→sync +15.98), synchronized AC2 MET (sync→ANN ≤3.06pp). FMNIST widens monotonically
  with depth (5.76 → 15.36 → 20.97pp). **Remaining open:** a 3rd cascaded seed on each d10
  cell (FMNIST s1 `rc=-9` / KMNIST s0 `rc=1` did not finalize) + the firing-gain gate-fix
  on the deep × hard collapse cell — proposed: WS3 `plan_stage:24` (gate-fix) and
  `plan_stage:25` (d6/d8 dataset gate-fix).
- **lenet5 KMNIST/SVHN not yet paired at n=1000** — KMNIST cascaded n=1000 now in the
  table (§1c, mild) but paired against an n=50 synchronized arm; the SVHN cascaded@n1000
  arm crashed rc=1. A paired n=1000 cascaded-vs-synchronized KMNIST/SVHN re-run would
  close the resolution mix and recover SVHN (proposed: WS3 `plan_stage:10`).
- **AC1 absolute targets** for these cells are not frozen in the floor book here; the
  table reports AC2 (deployed→ANN) which is the firing-gain-relevant verdict. Freezing
  per-model near-SOTA AC1 references is WS4 work.
