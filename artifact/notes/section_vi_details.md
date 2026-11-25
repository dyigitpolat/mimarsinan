# Section VI-C Details

These four paragraphs complete Section VI-C of the paper. They stood in its text before we shortened it for length, and we keep them here word for word, with every value as it was printed. After each passage we name the record and the field of every number in it. Record paths are paths inside the records archive of the release.

In these passages, "the evidence bundle" means the folder `artifact/` of this repository together with the records archive of the release.

## Static-Power Estimate of the Built Selections

It followed the sentence on the bit-exact multi-pass selections of the batch-64 and loose batch-1 pairs, built by our extended mapper, in the paragraph on the gains and their cost.

> Their paper-version estimate with static power is in the evidence bundle.

This passage prints no value. The record that holds the estimate with static power is not published (see the last part of this note).

## Selection Threshold on Five-Seed Mean Accuracies

It followed the sentence that no selection changes below the nominal point, in the paragraph on sensitivity.

> On five-seed mean accuracies, the loose batch-1 cell's lowest value changing selection moves from 98.4 µW to 184 µW per tile.

| Printed | Record | Field |
|---|---|---|
| five | `records/06_build/headline_v46_pairs/r3_nominal/selections_r3.json` | cells/R1'/selected/trained_s1_to_s5 (seeds 1 to 5) |
| 1 | `records/06_build/headline_v46_pairs/r3_nominal/selections_r3.json` | cells/R1'/words |
| 98.4 µW | `records/tiers/paper_v6/fix5/study_build/reread/keys.json` | keys, entry named thtileronep, value |
| 98.4 µW | `records/tiers/paper_v6/fix5/study_build/reread/out/thresh/p_tile_up.json` | bisected/cell R1'/threshold |
| 184 µW | `records/tiers/paper_v6/fix5/study_build/reread/out/thresh/p_tile_up.json` | bisected/cell R1' mean/threshold |

## Router Cycle, Tile Placement and Values Carried over the Mesh

It followed the sentence that costs growing with chip size raise energy-budget gains, in the paragraph on sensitivity.

> So with a slower on-chip network under this placement, three of the four gains rise and none falls (router cycles of 1 ns to 8 ns, the evidence bundle). Placing tiles active at once on different routers cuts the batch-1 reference design's energy at 4 ns from 1.37 to 1.14 times its nominal value, but this toolchain version places tiles sequentially. At a router cycle of 4 ns, carrying values between passes over the mesh gives the tight batch-1 cell 5.60 pp, against 4.42 pp when they bypass it.

| Printed | Record | Field |
|---|---|---|
| three of the four | `records/tiers/paper_v6/final/audit_fix/keys/derived_values.json` | study/hopraises/value and study/hopraises/of |
| none | `records/tiers/paper_v6/final/audit_fix2/keys/derived_values.json` | study/hopfalls/value |
| 1 ns | `records/tiers/paper_v6/fix5/study_build/reread/keys.json` | keys, entry named hopnsone, value |
| 8 ns | `records/tiers/paper_v6/fix5/study_build/reread/keys.json` | keys, entry named hopnseight, value |
| 1 | `records/06_build/headline_v46_pairs/r3_nominal/selections_r3.json` | cells/R1'/words |
| 4 ns | `records/tiers/paper_v6/fix5/study_build/reread/keys.json` | keys, entry named hopnsfour, value |
| 1.37 | `records/tiers/paper_v6/fix5/p0/keys/derived_values.json` | placement/placefillfour/value |
| 1.14 | `records/tiers/paper_v6/fix5/p0/keys/derived_values.json` | placement/placeawarefour/value |
| 5.60 pp | `records/tiers/paper_v6/fix5/study_build/reread/keys.json` | keys, entry named carrymesha, value |
| 4.42 pp | `records/tiers/paper_v6/fix5/study_build/reread/keys.json` | keys, entry named carryoncea, value |

## Core Geometry Tests

The paper states their result in one sentence, in the paragraph on co-design. These two paragraphs stood in its place.

> Fixed before 48 further geometries were priced, tests compare, per accuracy target, the cheapest joint design of network and core geometry with the cheapest network on chips sized for ResNet-18 w1 (the evidence bundle). The tests' main analysis, priced by the release version's estimate, charges area as if a tile read rows one after another and time as if together. Under it, those chips cost ResNet-18 w1/4 10.9% to 22.0% more inference time than joint design, confirmed in 6 of 36 tests. Confirmed means at least 5% more time beyond seed noise, after Benjamini-Hochberg correction at 0.10. None stays confirmed under all three declared variants: rows read one after another for both, together for both, or energy charged per cell bit.

| Printed | Record | Field |
|---|---|---|
| 48 | `records/13_geometry_study/confirm_analysis_fix/derived.json` | held_out_geometries |
| ResNet-18 w1 | `records/13_geometry_study/plan_r5.json` | declared/reference_anchor_identity |
| ResNet-18 w1/4 | `records/13_geometry_study/confirm_analysis_fix/derived.json` | confirmed_target_members/0 |
| 10.9% to 22.0% | `records/13_geometry_study/confirm_analysis_fix/derived.json` | ranges/T_pct |
| 6 | `records/13_geometry_study/confirm_analysis_fix/derived.json` | label_counts/CONFIRMED |
| 36 tests | `records/13_geometry_study/confirm_analysis_fix/derived.json` | label_counts (the sum over its four labels) |
| 5% | `records/13_geometry_study/plan_r5.json` | declared/reference_anchor_identity (the size floor of 0.05 it names) |
| 0.10 | not published | declared in the plan of the tests, outside the published part of its record |
| None | `records/tiers/paper_v6/final/audit_fix/keys/derived_values.json` | geo/geoconfnone/words |

## Not Included

The record of the deployed selections' prices, which holds the estimate with static power of the first passage, is not published, as it holds internal working notes or paths of our machines.
The run declarations of the core geometry tests are not published. `records/13_geometry_study/plan_r5.json` is published in part: it holds only the field `declared/reference_anchor_identity`, and its other fields, among them the correction level 0.10, are not published.
The authors can be asked for them.
