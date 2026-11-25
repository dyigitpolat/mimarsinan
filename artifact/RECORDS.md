# Records Zip

The records behind the paper's numbers and figures are not in the source of this repository. They are in one zip
file on a release of the repository.

| Item | Value |
|---|---|
| File | `mimarsinan-tcad-records-v1.zip` |
| Release | the release tagged `tcad-records-v1` of https://github.com/dyigitpolat/mimarsinan |
| Download address | https://github.com/dyigitpolat/mimarsinan/releases/download/tcad-records-v1/mimarsinan-tcad-records-v1.zip |
| Size | 97,298,093 bytes |
| SHA-256 | `0a745ae453e4cfa626917c4f0b38f85d479661608a66ac4914c0480bef6be74a` |
| Unpacked | 675 records, 1.38 GB, with a copy of the repository's MIT license and a short README, all under one folder `records/` |

## Unpacking

The notes in this folder and the index of printed numbers write every record as a path that begins with `records/`.
Unpack the zip inside this folder, `artifact/`, so that those paths resolve from here:

    cd artifact
    curl -LO https://github.com/dyigitpolat/mimarsinan/releases/download/tcad-records-v1/mimarsinan-tcad-records-v1.zip
    sha256sum mimarsinan-tcad-records-v1.zip
    unzip mimarsinan-tcad-records-v1.zip

The digest that `sha256sum` prints should equal the one in the table above. A record path such as
`records/frozen/thru_branch_92d90e2c/e3_grid.json` then opens as `artifact/records/frozen/thru_branch_92d90e2c/e3_grid.json`.

Four records are large: the regime maps of the exploration runs (292 MB, 306 MB and 89.6 MB) and the result of the
matched-accuracy comparison of co-design (531 MB). The three regime maps are JSON on a single line. Loading one of
them whole can take several gigabytes of memory, and a streaming JSON reader avoids that.

## Folders of the Zip

Most records keep the place they had in our working repository. Under `records/frozen/`, each folder holds a copy of
a run's records that we froze at one point of the study, and most of these folder names end in the date and time of
the copy (for example `20261121T001425`). A folder or file name that carries a 12-digit hexadecimal digest, such as
`644b4c3bc5d7`, names the toolchain source a run used. Each experiment's note under `experiments/` says which version
of `VERSIONS.md` that is. Eight records are published in part (`README.md` names them): each begins with a field
`_note` that says so and holds only the fields that printed numbers read.

| Folder | Files | Size | What it holds |
|---|---|---|---|
| `06_build` | 9 | 657 kB | The network pool and the system cost model of the equal-budget study |
| `06_build/codesign` | 2 | 44.5 kB | The cost model's site table and the text output of its preview table |
| `06_build/headline_v46` | 4 | 552 kB | The pool's pre-registered rules, its lock, its measured table and the admissible masks |
| `06_build/headline_v46_pairs` | 2 | 26.1 kB | The selections and the record of the pairs at the nominal point |
| `06_build/screen_v42` | 1 | 34.3 kB | The indices of the CIFAR-100 development split E01 |
| `08_significance/probes` | 1 | 67.3 kB | The census of network weights against published chip capacities (Fig. 1) |
| `10_wave1/g6_exactness_contract` | 1 | 33.1 kB | The model-level runs of a separate simulator, which switches off conditions that the code cannot switch off |
| `13_geometry_study` | 2 | 25.8 kB | The pre-registered plan of the core geometry tests, published in part, and the derived label counts |
| `e2e` | 9 | 18.0 kB | Configurations of the LIF deployment rows of Table VI |
| `e2e/routeb` | 3 | 5.57 kB | Configurations of the LIF rows at three core budgets |
| `e2e/tlif_tuned` | 6 | 12.5 kB | Configurations and training records of the fine-tuned LIF rows |
| `exactness_block/644b4c3bc5d7_r3` | 1 | 52.9 kB | Values derived from the exactness records |
| `excerpts` | 7 | 12.9 kB | Lines quoted word for word from documents that stay out, each block with the line numbers of the original, for the printed values that read them |
| `figures_v2/data` | 1 | 623 B | The data of Table I, published in part |
| `figures_v3/data` | 3 | 428 kB | Earlier figure data files, which the current figure data files read |
| `frozen` | 313 | 1.26 GB | Frozen copies of run records, one folder per freeze |
| `frozen/census_baseline_t1_20261109T230030` | 1 | 127 kB | The summary of the conversion test of 80 torchvision classifiers (Fig. 3(b)) |
| `frozen/cl5_exploration_t1_20261110T065255` | 1 | 292 MB | The regime map of an earlier exploration run, whose regime thresholds the paper prints |
| `frozen/cl5_exploration_t1_20261122T161849` | 7 | 316 MB | The exploration run of record: regime map, capacity counts, member map, resolved deployments, the ResNet-1001 point, summary and run record (Figs. 1, 12 and 13) |
| `frozen/cl6_t1_20261120T183429` | 34 | 155 kB | The converter-width grid on MNIST and CIFAR-10: points, snapshots and summaries (Fig. 15(a)) |
| `frozen/event_census_static_check_t1_20261124T164942` | 76 | 320 kB | One static-check record per deployed program (Section IV-B) |
| `frozen/exactness_block_t1_20261124T165039` | 2 | 135 kB | The report of the randomized tests (Fig. 7(b)) and the summary of the random multi-pass programs (Figs. 6 and 7(a)) |
| `frozen/exactness_block_t1_20261125T031754` | 11 | 1.24 MB | The exactness records of the patched release: random programs, static checks, the multiplier census, the registry of cost-only parameters and the SANA-FE probes |
| `frozen/gate_and_test_read_t1_20261109T230030` | 5 | 479 kB | The first pre-registered read of the test split: lock (published in part), result and the records of the pairs |
| `frozen/gate_and_test_read_t1_20261124T231409` | 3 | 272 kB | The second pre-registered read of the test split: lock (published in part), completion record and result |
| `frozen/iso_accuracy_t1_20261121T155409` | 2 | 531 MB | Co-design against sequential design at matched accuracy: rules (published in part) and result |
| `frozen/latency_diag_t1_20261123T125524` | 9 | 255 kB | Our estimator's runs at the points of the comparison with CIMFlow's simulator (Fig. 9(c)) |
| `frozen/n2_cl7_t1_20261107T093743` | 11 | 2.78 MB | The attention block on the release: counts, prices, the aggregate and the block's data file (Fig. 16, Table VIII) |
| `frozen/n2d_keys_t1_20261120T183656` | 3 | 23.5 kB | The attention block's deployed difference on the patched release |
| `frozen/provisional` | 2 | 89.6 MB | An earlier regime map, whose service life and inference rate the paper uses, and an earlier figure record (published in part) |
| `frozen/routeb_budget_ladder_t1_20261120T183320` | 12 | 34.9 kB | Configurations and table rows of the core budget ladder of Fig. 8 |
| `frozen/routeb_layer_sharing_t1_20261120T183214` | 3 | 19.1 kB | Cost reports of the shared-kernel variant of the co-designed chip (Section V-C) |
| `frozen/routeb_tint_supplement_t1_20261121T001425` | 105 | 22.1 MB | Configurations and results of the integer deployment rows of Table VI, their tables and the area figure record (Figs. 4(b), 8 and 11) |
| `frozen/routeb_tlif_untuned_t1_20261124T164838` | 5 | 1.34 MB | Results and tables of the LIF rows of Table VI |
| `frozen/thru_branch_92d90e2c` | 1 | 596 kB | The throughput grid of the pipelining branch (Section V-D) |
| `frozen/tuned_tlif_t1_20261124T164742` | 3 | 1.16 MB | Results of the fine-tuned LIF rows of Table VI |
| `frozen/w34_adc_sweep_t1_20261120T183547` | 11 | 298 kB | The converter-width sweep of the CIFAR-100 ResNet-18: its ten points and their collection (Fig. 15(a)) |
| `frozen/w38_imagenet_t1_20261120T224345` | 6 | 150 kB | The ImageNet records on NeuRRAM, one per network and weight form (Table VI) |
| `hardware/multicore` | 3 | 104 kB | The readings of the FPGA measurements, the derived keys, and the expectations with whether each held (Figs. 9 and 10) |
| `imagenet/members` | 1 | 11.8 kB | The ImageNet networks and the rule that selected them |
| `other/card_execution` | 1 | 512 B | The FPGA execution record, published in part: the declared axon lines, the decisions as predicted and the checks of each realization |
| `tiers` | 323 | 115 MB | Records of the area, host and FPGA studies, the compiler measurements, the derived values and the re-read of the equal-budget scenarios |
| `tiers/fidelity` | 9 | 1.63 MB | Area estimates (unit facts and figure records), the runs of the host CPU pricing and the census tables of the FPGA cores |
| `tiers/paper_locks` | 10 | 1.19 MB | PIMCOMP-NN measurements, the derived counts of the compiler comparison (published in part), derived latency values, the single-pass chip that would hold each multi-pass selection, and an earlier throughput grid |
| `tiers/paper_t9` | 1 | 24.7 kB | The reading of the largest weight overflow at batch 1 (Section VI-D) |
| `tiers/paper_v6` | 303 | 112 MB | Derived values of printed numbers, the rounding bound of the conversion check and the re-read of the equal-budget scenarios, with one reading per named setting |

## Records Not in the Zip

These are the records a reader of the paper or of our notes might look for and not find, each with its reason.

- **The compiler comparison on the conversion test** (Section V-D, Fig. 9(a) and (b)). The runs of PIMCOMP-NN and
  CIMFlow on the conversion test, their protocol and its tables are held out under a rule of our study. The paper's
  text is our account of them. The PIMCOMP-NN measurements of Table IV are in `records/tiers/paper_locks/`.
- **The regime robustness study** (Section VI-B, key `robregime`). Its run, its lock and its record are held out under
  the same rule. The index gives the printed value with the paper's section.
- **The estimator run of record of the simulator weight-load comparison** (Section V-D). It lies beside held-out
  comparison data, so it stays out. Our estimator's runs at the stated points are in
  `records/frozen/latency_diag_t1_20261123T125524/`.
- **Datasets.** CIFAR-100, CIFAR-10, MNIST and ImageNet are the work of their authors and available from them. They
  and our lists of ImageNet files stay out. A deployment record names the digest of the split it read (for example
  `inputs.provider.split_sha256`), and `records/06_build/screen_v42/val_split_e01.json` gives the indices of the
  CIFAR-100 development split.
- **Trained weights and checkpoints.** They stay out by rule. Each deployment row names its checkpoint by digest
  (`inputs.checkpoint_sha256`).
- **Pages of other authors' papers.** For a value from the literature, the index names the publication and page.
- **Records that held internal notes or names of our machines.** A record that a printed number or a figure reads is
  in the zip with such text replaced, or published in part. A further record that no printed number reads directly
  stays out when it held such text. Among them are records that our notes name: the timing model of the FPGA runs and the predictions fixed
  before measuring, the pricing record of the deployed selections (`w48_pairs.json`), the planned budget ladder of
  Fig. 8 (`planned_ladder.json`), the five-seed pairs on the development split (`pairs_result.json`), the full
  analysis of the re-read of the scenarios (`analysis.json`) and the table of published chip areas
  (`published.json`). The figure data files under `figures/data/` hold every value the figures draw from them.
- **Locks and rules that quote internal decisions.** Where a lock or a set of pre-registered rules quotes internal
  decisions or names our machines, it stays out. `records/excerpts/` quotes the lines that printed values read, word
  for word.
- **Run bulk.** Per-row arrays, packed programs, chip mappings, work folders, logs and console captures that no
  printed number reads.
- **Scripts and programs.** Scripts are in the source, under `artifact/experiments/` and `artifact/figures/`, and not
  in the zip. The toolchain is the rest of this repository.

The records zip is under the repository's MIT license, like the rest of this folder, and carries a copy of it.
