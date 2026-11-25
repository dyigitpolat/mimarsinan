# Toolchain Versions

Section V-A of the paper names the versions of our toolchain MimarSinan in words. This file gives the commit behind each name in the repository https://github.com/dyigitpolat/mimarsinan, says which of them a reader can check out today, and gives the versions of the other tools we ran.

The toolchain's paper version is commit `4525d405`, the patched release. This folder `artifact/` sits on branch `tcad` in commits above it. Later commits on the branches that hold the paper's commits change names and documentation, and they reference the cycle-level simulator nevresim as a submodule with the same files.

## Versions and Commits

| Name in the paper | Commit | Where a reader finds it | What the paper reports from it |
|---|---|---|---|
| The patched release (the release with fixes) | `4525d405` | by its hash, on branches `main` and `tcad` | Most numbers of Sections V and VI (Section V-A), among them the deployments of Section V-C and the second evaluation of the test images |
| The release | `e5f6bb37` | by its hash only, as the parent of `4525d405` | Numbers that the fixes leave unchanged and numbers the text names as the release's (Section V-A), among them the first evaluation of the test images and the main analysis of the core geometry tests (Section VI-C) |
| Our pipelining branch | `92d90e2c` | by its hash, on branch `throughput-replication` | Throughput with layer copies and a streamed batch on COMPASS's ResNet-18 (Section V-D) |
| An earlier version, deployment table | `c9131188` | by its hash only, as an ancestor of `4525d405` | The rows marked a and the marks marked b in the deployment table of Section V-C |
| An earlier version, exactness | not published | | The static-check entries, the count of differing deployments that changed a predicted class and the pass replay row of Section V-B |
| A development build, FPGA | not published | | The programs the FPGA ran, byte-identical to the release's (Sections V-A and V-D) |
| A development build, SANA-FE | not published | | The SANA-FE row of the estimate comparison (Section V-D) |

Full hashes of the nevresim commits: `928193fa758d6f2daffe8bba0c043dc82622a05e` (the simulator of the patched release) and `5de880dc4328e0636a7510b4236d5dceb7b1c8aa` (the simulator the release pins).

Full hashes: `4525d4057cf999e9bbae4c96a1821119e9b58dca` (patched release), `e5f6bb37d6285c6ac218d2558682a7e84fcb4c84` (release), `92d90e2c4fb59894cc717c4ef637fff611eb28f1` (pipelining branch) and `c913118818b83b2225562556427a397321fa6aaf` (earlier version of the deployment table).

Some records name the patched release by a digest of its source, `644b4c3bc5d7`, which also appears in folder names inside the records zip. That digest is not a commit. Its commit is `4525d405`.

## Checking Out Each Version

Every public version below comes from one clone:

```
git clone https://github.com/dyigitpolat/mimarsinan.git
cd mimarsinan
```

**The patched release.** `git checkout 4525d405`. The cycle-level simulator nevresim is part of this tree as plain files, so the checkout needs no submodule. Branches `main` and `tcad` hold this commit, and `tcad` adds this folder above it.

**The heads of `main` and `tcad`.** `git checkout main` or `git checkout tcad`, then `git submodule update --init nevresim`. At the branch heads nevresim is a submodule again, pinned at commit `928193f` of https://github.com/dyigitpolat/nevresim. The files of that commit are byte for byte the plain files of `4525d405`, so the simulator is the same.

**The release.** `git checkout e5f6bb37`, then `git submodule update --init nevresim`. No branch or tag names this commit, but every clone holds it as the parent of `4525d405`. At this commit nevresim is a submodule pinned at commit `5de880d`, which is on the main branch of the public nevresim repository.

**Our pipelining branch.** `git checkout 92d90e2c`, a commit of branch `throughput-replication` above the patched release. It adds layer copies into idle arrays and a batch pipeline through each layer group, and it carries nevresim as plain files.

**The earlier version of the deployment table.** `git checkout c9131188`, then `git submodule update --init`, which fetches nevresim from its public repository. No branch names this commit, and every clone holds it as an ancestor of `4525d405`. The rows marked a were checked on the patched release, with the single-pass configuration and outputs on 64 images byte-identical. The marks marked b keep unchanged configurations and were not checked byte for byte.

## Versions without a Public Commit

**The earlier version of the exactness results.** These results come from development trees that were never committed to a public branch.

**The development builds.** The FPGA ran programs from a development build that has no commit of its own. Those programs are byte-identical to the release's, and the release holds the FPGA path. The SANA-FE row comes from one run of a development build, whose record we read at commit `17be1010`, an ancestor of the patched release.

For each number the paper prints from these versions, the index of printed numbers in `artifact/numbers/` names its record or says why the record is not published.

## Other Tools

| Tool | Version | Public home | Use in the paper |
|---|---|---|---|
| PIMCOMP-NN | commit `0cfbfa55cca22fcfebdb8dcfc75420dcbbbbdf77` | https://github.com/sunxt99/PIMCOMP-NN | Compiler comparison in default batch mode (Section V-D) |
| CIMFlow | commit `67f3da07` | Qi et al., DAC 2025 | Compiler comparison (Section V-D) |
| SANA-FE | version 2.1.1, the optional extra `sanafe` in `pyproject.toml` of the patched release | Boyle et al., IEEE TCAD | Simulator backend and the SANA-FE row (Section V-D) |
| ODIN | commit `1781931` of https://github.com/ChFrenkel/ODIN | Frenkel et al. | The published FPGA core, copied into `hw/vendor/odin` under its own license (see `NOTICE`) |
| sciviz | commit `945c459ad2eef830d9f4468ad6eae068cee94ebf` | https://github.com/dyigitpolat/sciviz | The diagram library our figure scripts use |

## Not Included

The development trees of the earlier exactness results and the two development builds are not published as commits.
The authors can be asked for them.
