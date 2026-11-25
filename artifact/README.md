# Scripts, Records and Notes behind the Paper

This folder holds the material behind our paper "MimarSinan: Design-Space Exploration of In-Memory and Neuromorphic
Chips by Bit-Exact Mapping of Trained Neural Networks". Section I of the paper points to it: our repository holds the toolchain at every version we
name, the scripts and records behind each figure and table, and details on each experiment. The toolchain is the rest
of this repository. This folder adds the notes, the figure scripts with their data, the index of printed numbers and
the scripts of each experiment. The records themselves are in one zip file on a release of this repository, which
`RECORDS.md` describes.

## Layout

| Path | What it holds |
|---|---|
| `README.md` | This overview |
| `RECORDS.md` | The records zip: where to get it, how to unpack it, what it holds and what it does not hold |
| `VERSIONS.md` | The toolchain versions the paper names, the commit behind each and the other tools we ran |
| `numbers/index.md`, `numbers/index.json` | Every number the paper prints, 356 in all, by its key, with its record and field |
| `figures/` | The sixteen figure scripts, one data file per figure, the style kit, the data builders and the keys files of our build (`figures/README.md`) |
| `experiments/<name>/` | One folder per experiment, 23 in all: a `README.md` and, for 20 of them, the scripts that ran it |
| `notes/section_vi_details.md` | Four passages that complete Section VI-C, with the record and field of every value in them |

## Finding the Record behind a Printed Number

1. Look up the value in `numbers/index.md`. It groups the values by the section where they first print, in reading
   order. Each row gives the key (the name of the value in our paper source), the value as printed, what it is, its
   class as the paper uses the word (measured, estimate, declared or published) and the record with the field it
   reads.
2. `numbers/index.json` holds the same entries for a reader who looks values up by program. Each entry has the fields
   `key`, `section`, `value`, `what`, `class` and `sources`. Each source names a record with the fields it reads, or a
   public stand-in.
3. Download and unpack the records zip as `RECORDS.md` says. A record path in the index, such as
   `records/frozen/...`, is a path inside the zip. A field written `a/b/c` names the key `a`, then `b` inside it, then
   `c`, and a number among them is a position in a list, counted from 0. For a text record the field is a line
   number.
4. Where a record stays out of the public copy, the entry names a stand-in instead: the publication and page of a
   value from the literature, the section of the paper for a setting of the study, or the reason the record is
   withheld. 18 entries have such a stand-in.

## Rebuilding a Figure

`figures/README.md` gives the requirements and the details. From the folder `figures/`:

    python3 make_all.py --no-copy               # all sixteen figures into figures/out/
    python3 make_all.py --no-copy fig_fronts    # one or more figures by name

A figure script reads only its data file in `figures/data/`, the style kit and the label table, so every figure
rebuilds without the records zip. `make_all.py` first compares each data file with `figures/data/MANIFEST.sha256` and
stops if one differs. The table at the end of `figures/README.md` gives, for each figure, the paper's figure number
and the records its data file was built from.

## Reading an Experiment's Folder

Each folder under `experiments/` covers one experiment, or several that one run produced. Its `README.md` has these
parts where they apply.

- **What the paper reports:** the section, figure or table, and the keys of the printed numbers.
- **Setup:** the toolchain version by the paper's name for it (`VERSIONS.md` gives the commit), the networks, chip
  descriptions, data and splits, seeds and budgets, each with the record and field it comes from.
- **How it was run:** the scripts in order, and what each reads and writes.
- **Records:** a table of the records in the zip, what each holds and which printed keys read it.
- **How to read the records:** the fields a reader needs.

The scripts under `scripts/` show how each experiment ran. They read and write files at their places in our working
repository and call the toolchain at the version their note names, so most of them do not run unchanged from this
folder. Their paths begin with neutral folders: `/work/research` is our working repository, `/work/research/tcad_paper`
our paper folder in it, and `/work/worktrees` holds checkouts of the toolchain. A record path that a script writes out
in full reads as the record's place in the zip, `records/<path>`. Where a script joins a path from parts, the parts
keep the folder names of our working repository. A few path parts that named our machines read differently in the
zip.

## Experiments

| Folder | Paper | What it holds |
|---|---|---|
| `area_estimates` | Sections III and V-D, Fig. 11, Table III | Leave-one-out area estimates against published chip areas, the SRAM array estimates and area against scheduled passes |
| `attention_block` | Sections III-A and VI-D, Fig. 16, Table VIII | The ViT-Lite-7/4 attention block on dynamic-weight cores: share of multiply-accumulates on cores, product energy, host transfer and deployed accuracy difference |
| `attention_block_seed_pairs` | Section VI-D, Table VIII | The five-seed pairs of the attention block on the development split and the non-inferiority margin |
| `co_design` | Section VI-C | Co-design against sequential design of chip and network at matched accuracy |
| `compiler_comparison` | Section V-D, Fig. 9(a) and (b), Table IV | PIMCOMP-NN and CIMFlow on networks larger than the chip: dropped products, additions and missing instructions for fully connected layers |
| `conversion_test` | Sections V-A and V-B, Fig. 3(b) | The conversion test of 80 torchvision ImageNet classifiers at random weights in double precision |
| `converter_precision` | Section VI-D, Fig. 15(a) | Accuracy loss of the CIFAR-100 ResNet-18 against converter bits at two core budgets |
| `core_geometry` | Sections VI-C and VIII | The pre-registered tests of joint design of network and core geometry against chips sized for a reference network |
| `equal_budget_scenarios` | Section VI-C, Fig. 14, Table VII | Multi-pass execution at equal area and energy budgets: the network pool, the development-split pairs and the system cost model's scenarios |
| `explored_space` | Sections I, VI-A, VI-B and VI-D, Figs. 1, 12, 13 and 15(b), Table III | One exploration run: explored points, regime map, capacity counts and write endurance |
| `fpga_pass_time` | Sections V-A, V-B and V-D, Fig. 10 | ODIN on the FPGA at two core geometries: checks against the golden model and measured against predicted pass times |
| `host_cpu_pricing` | Section VI-D | Host work priced from a Jetson-class CPU and the energy budgets whose verdict it moves |
| `imagenet_deployment` | Section V-C, Table VI | ImageNet classifiers with their published weights deployed on NeuRRAM descriptions, the ImageNet block of Table VI |
| `multi_pass_fuzz` | Sections V-B and VII, Figs. 6 and 7(a) | Random multi-pass programs compared with their single-pass program under each semantics |
| `per_deployment_check` | Sections IV-B and V-C | The static check of each deployment at every core budget, its time against compilation and the census of rounding multipliers |
| `pipelining_throughput` | Abstract, Sections V-D and VII | Throughput of our pipelining branch on COMPASS's ResNet-18 against COMPASS's reported throughput |
| `printed_value_derivations` | Several sections | The small scripts that derive counts, ranges and ratios from experiment records, each with its derived-value record |
| `random_deployments` | Sections IV-A and V-B, Figs. 4(a) and 7(b), Table IV | Randomized tests of Theorem 1 with random cost-only parameters and with conditions switched off one at a time |
| `regime_robustness` | Section VI-B, Fig. 13 | The regime map's robustness to the cost constants, as the paper reports it (its record is held out) |
| `sanafe_comparison` | Section V-B, Table IV | Synchronized TTFS on the SANA-FE simulator against the golden model on six single-pass programs |
| `simulator_weight_load` | Section V-D, Fig. 9(c) | The weight load in our estimate at the points of the comparison with CIMFlow's simulator |
| `test_read` | Sections V-A, VI-C and VI-D, Tables VI, VII and VIII | The two pre-registered reads of the CIFAR-100 test split |
| `trained_network_deployment` | Sections IV and V-C, Figs. 4(b) and 8, Table VI | Trained CIFAR-100 networks deployed end to end at the core budgets of Table VI and Fig. 8 |

## Preparation of This Copy

We prepared this copy from our working repository so that it holds nothing private and no detail of the computers we
ran on. We changed the files in these ways.

- Python scripts are published without their comments and docstrings.
- A script that named one of our machines, our job scheduler, a resource limit, a set of processor cores, an
  internal decision or approval, or the editing of our drafts is not published, and shell scripts are not published.
  An experiment's note may still name such a script, so that a reader knows what it did.
- In every text file, paths on our machines read as neutral paths (`/work`, `/scratch`, `/home/user`, `/tmp/work`),
  our user name reads `user`, and the names of our servers, compute nodes and clusters read as neutral words. A few
  words that named roles in our working process were replaced by neutral words too. A path of a record in our working
  repository reads as the record's place in the zip.
- In JSON records, a text value that named an internal decision, a cluster, a scheduler tool, a job, a graphics card
  or processor model that the paper does not name, or a process that watched or limited our runs reads
  "(internal note withheld)". Job ids read "(withheld)". A key that described the setup of our machines (their
  environment, processor cores, devices, drivers, queues and limits) or an internal approval reads `internal_note_<n>`,
  with a number for n, and its value reads "(withheld)".
- Eight records are published in part. Each holds only the fields that printed numbers read and begins with a field
  `_note` that says so: `records/13_geometry_study/plan_r5.json`, `records/figures_v2/data/tab_prior.json`,
  `records/frozen/gate_and_test_read_t1_20261109T230030/e2e/test_lock.json`,
  `records/frozen/gate_and_test_read_t1_20261124T231409/e2e/test_read_644b4c3bc5d7/test_lock.json`,
  `records/frozen/iso_accuracy_t1_20261121T155409/e2e/iso_accuracy/rules.json`,
  `records/frozen/provisional/10_wave1/fig1/fig1_instance.json`,
  `records/other/card_execution/hw_execution.json` and
  `records/tiers/paper_locks/cut/cut_exhibit_place/derived/exhibit_values.json`.
- The figure data files are published without a few fields that recorded how they were built or held notes.
  `figures/data/MANIFEST.sha256` gives the digests of the published data files.
- No measured or computed number was changed.
- A digest that a record states for another file was taken on our working copy before these changes, so it can
  differ from the digest of the published file.

What is not included, by kind. The counts come from the lists we kept while preparing the copy. An entry can name a
single file, a whole folder or a class of files such as the checkpoints of every row.

| What is not included | Entries |
|---|---|
| Material of other authors: a page of a published paper, another project's source and a patch of it, the datasets and their file lists, trained weights and checkpoints | 8 |
| Records under a held-out rule of our study | 17 |
| Internal working documents: plans, step reports, reviews, check reports, decision notes and launch logs | 116 |
| Job submission and machine setup: scripts and records that launch, place, limit or watch jobs on our machines | 172 |
| Programs and bulk outputs that no printed number reads: checkpoints, arrays, chip mappings, work folders, logs, console captures, tests and duplicate copies | 88 |
| Files whose private details could not be removed without changing what they say | 12 |
| Scripts that named our machines, our scheduler or our working process, and shell scripts | 64 |
| Further records, beyond those a printed number or a figure reads, that held an internal note or a name of our machines | 109 |
| Further records that a reading marked for a check by hand | 9 |
| Further records, scripts and notes that a reading of this copy named for leaving out | 31 |
| Further files listed as records that are not data: three scripts and one text table | 4 |
| The built figure files: a PDF and a PNG per figure, which `make_all.py` rebuilds | 32 |

`RECORDS.md` names the records a reader might look for and not find, each with its reason.

## License

The scripts, notes and data in this folder are under the repository's MIT license (`LICENSE` at the repository root).
The records zip on the release carries the same license and a copy of it.

## Citation

If you use this material, please cite the paper:

D. Y. Polat and W.-F. Wong, "MimarSinan: Design-Space Exploration of In-Memory and Neuromorphic Chips by Bit-Exact
Mapping of Trained Neural Networks," manuscript, 2026.

```bibtex
@unpublished{polat_mimarsinan,
  author = {Dogukan Yigit Polat and Weng-Fai Wong},
  title  = {{MimarSinan}: Design-Space Exploration of In-Memory and Neuromorphic Chips by Bit-Exact Mapping of
            Trained Neural Networks},
  note   = {Manuscript},
  year   = {2026}
}
```
