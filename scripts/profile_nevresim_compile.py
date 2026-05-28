#!/usr/bin/env python3
"""Synthetic nevresim compile profiling sweeps (E1–E5). No real ViT compiles."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mimarsinan.chip_simulation.nevresim.profiling import (
    NevresimCompileProfile,
    build_multi_segment_fanout,
    build_synthetic_mapping,
    correlate_compile_vs_metric,
    profile_mapping_compile,
    write_profile_rows,
)


def _resolve_nevresim_path() -> str:
    candidate = ROOT / "nevresim"
    if not candidate.is_dir():
        raise SystemExit(f"nevresim tree not found at {candidate}")
    return str(candidate)


def run_e1(args, nevresim_path: str) -> list:
    rows = []
    for core_count in args.core_counts:
        hcm, input_size = build_synthetic_mapping(
            "patch_embed_like", core_count, args.axons, args.neurons,
        )
        rows.append(profile_mapping_compile(
            hcm, input_size,
            experiment="E1_core_scaling",
            preset="patch_embed_like",
            nevresim_path=nevresim_path,
            out_dir=args.out_dir / f"E1_cores_{core_count}",
            compile=not args.codegen_only,
            execute=args.execute,
            optimization=args.optimization,
            time_trace=args.time_trace,
            cache_dir=args.cache_dir,
            verbose=args.verbose,
        ))
    return rows


def run_e2(args, nevresim_path: str) -> list:
    rows = []
    for axons in args.axon_counts:
        hcm, input_size = build_synthetic_mapping(
            "contiguous_input", args.fixed_cores, axons, args.neurons,
        )
        rows.append(profile_mapping_compile(
            hcm, input_size,
            experiment="E2_axon_scaling",
            preset="contiguous_input",
            nevresim_path=nevresim_path,
            out_dir=args.out_dir / f"E2_axons_{axons}",
            compile=not args.codegen_only,
            execute=args.execute,
            optimization=args.optimization,
            connectivity_mode=args.connectivity_mode,
            cache_dir=args.cache_dir,
            verbose=args.verbose,
        ))
    return rows


def run_e3(args, nevresim_path: str) -> list:
    rows = []
    for preset in ("contiguous_input", "fragmented_crosscore"):
        hcm, input_size = build_synthetic_mapping(
            preset, args.fixed_cores, args.axons, args.neurons,
        )
        rows.append(profile_mapping_compile(
            hcm, input_size,
            experiment="E3_span_fragmentation",
            preset=preset,
            nevresim_path=nevresim_path,
            out_dir=args.out_dir / f"E3_{preset}",
            compile=not args.codegen_only,
            execute=args.execute,
            optimization=args.optimization,
            connectivity_mode=args.connectivity_mode,
            cache_dir=args.cache_dir,
            verbose=args.verbose,
        ))
    return rows


def run_e4(args, nevresim_path: str) -> list:
    rows = []
    hcm, input_size = build_synthetic_mapping(
        "contiguous_input", args.fixed_cores, args.axons, args.neurons,
    )
    for sim_steps in args.sim_steps:
        rows.append(profile_mapping_compile(
            hcm, input_size,
            experiment="E4_policy_params",
            preset=f"sim_steps_{sim_steps}",
            nevresim_path=nevresim_path,
            out_dir=args.out_dir / f"E4_steps_{sim_steps}",
            simulation_length=sim_steps,
            compile=not args.codegen_only,
            execute=False,
            optimization=args.optimization,
            cache_dir=args.cache_dir,
            verbose=args.verbose,
        ))
    return rows


def run_e5(args, nevresim_path: str) -> list:
    rows = []
    counts = [3] if args.quick else args.segment_counts
    for n in counts:
        segments = build_multi_segment_fanout(
            n,
            cores_per_segment=args.segment_cores,
            axons_per_core=args.axons,
            neurons_per_core=args.neurons,
        )
        seg_rows = []
        for seg_idx, (hcm, input_size) in enumerate(segments):
            row = profile_mapping_compile(
                hcm, input_size,
                experiment="E5_multi_segment",
                preset=f"seg_{seg_idx}_of_{n}",
                nevresim_path=nevresim_path,
                out_dir=args.out_dir / f"E5_n{n}_seg{seg_idx}",
                compile=not args.codegen_only,
                execute=False,
                optimization=args.optimization,
                cache_dir=args.cache_dir,
                verbose=args.verbose,
            )
            seg_rows.append(row)
            rows.append(row)
        total_compile = sum(r.compile_s for r in seg_rows)
        rows.append(NevresimCompileProfile(
            experiment="E5_multi_segment_total",
            preset=f"n_segments_{n}",
            compile_s=total_compile,
            compile_success=all(r.compile_success for r in seg_rows),
            core_count=args.segment_cores,
            axons_per_core=args.axons,
            extra={"segment_count": n},
        ))
    return rows


def run_ab_static_vs_runtime(args, nevresim_path: str) -> list:
    preset = getattr(args, "ab_preset", "mixed_vit_like")
    cores = args.ab_cores if args.ab_cores is not None else args.fixed_cores
    hcm, input_size = build_synthetic_mapping(
        preset, cores, args.axons, args.neurons,
    )
    rows = []
    for mode in ("compile_time", "runtime"):
        rows.append(profile_mapping_compile(
            hcm, input_size,
            experiment="AB_static_vs_runtime",
            preset=f"{preset}_{mode}",
            nevresim_path=nevresim_path,
            out_dir=args.out_dir / f"AB_{preset}_{mode}",
            connectivity_mode=mode,
            compile=not args.codegen_only,
            execute=args.execute,
            optimization=args.optimization,
            cache_dir=args.cache_dir,
            verbose=args.verbose,
        ))
    return rows


def run_e1_fragmented(args, nevresim_path: str) -> list:
    """E1 variant: core scaling on fragmented_crosscore (compile-time hotspot)."""
    rows = []
    for core_count in args.core_counts:
        hcm, input_size = build_synthetic_mapping(
            "fragmented_crosscore", core_count, args.axons, args.neurons,
        )
        rows.append(profile_mapping_compile(
            hcm, input_size,
            experiment="E1_fragmented_scaling",
            preset="fragmented_crosscore",
            nevresim_path=nevresim_path,
            out_dir=args.out_dir / f"E1_frag_cores_{core_count}",
            compile=not args.codegen_only,
            execute=args.execute,
            optimization=args.optimization,
            time_trace=args.time_trace,
            cache_dir=args.cache_dir,
            verbose=args.verbose,
        ))
    return rows


def _apply_large_defaults(args) -> None:
    args.core_counts = [100, 200, 400, 600, 850]
    args.axon_counts = [128, 256]
    args.fixed_cores = 600
    args.axons = 256
    args.neurons = 256
    args.sim_steps = [4, 32, 128]
    args.segment_counts = [5, 10]
    args.segment_cores = 100
    args.ab_preset = "fragmented_crosscore"
    args.ab_cores = 850


def main() -> None:
    parser = argparse.ArgumentParser(description="Nevresim synthetic compile profiler")
    parser.add_argument("--out-dir", type=Path, default=Path("generated/nevresim_profile"))
    parser.add_argument("--experiments", nargs="+", default=["E1", "E2", "E3", "E4", "E5", "AB"])
    parser.add_argument("--core-counts", nargs="+", type=int, default=[10, 50, 100])
    parser.add_argument("--axon-counts", nargs="+", type=int, default=[64, 128, 256])
    parser.add_argument("--fixed-cores", type=int, default=50)
    parser.add_argument("--axons", type=int, default=128)
    parser.add_argument("--neurons", type=int, default=64)
    parser.add_argument("--sim-steps", nargs="+", type=int, default=[4, 16, 64])
    parser.add_argument("--segment-counts", nargs="+", type=int, default=[5, 10])
    parser.add_argument("--segment-cores", type=int, default=20)
    parser.add_argument("--connectivity-mode", choices=["compile_time", "runtime"], default="compile_time")
    parser.add_argument("--optimization", default="-O3")
    parser.add_argument("--codegen-only", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--time-trace", action="store_true")
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--quick", action="store_true", help="Smaller grids for fast smoke runs")
    parser.add_argument(
        "--large",
        action="store_true",
        help="Large-scale grid targeting >10s g++ compile (fragmented_crosscore, up to 850 cores)",
    )
    parser.add_argument("--ab-preset", default="mixed_vit_like")
    parser.add_argument("--ab-cores", type=int, default=None)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if args.quick:
        args.core_counts = [10, 30]
        args.axon_counts = [64, 128]
        args.segment_counts = [3, 5]
    elif args.large:
        _apply_large_defaults(args)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    nevresim_path = _resolve_nevresim_path()

    all_rows = []
    runners = {
        "E1": run_e1,
        "E1F": run_e1_fragmented,
        "E2": run_e2,
        "E3": run_e3,
        "E4": run_e4,
        "E5": run_e5,
        "AB": run_ab_static_vs_runtime,
    }
    for exp in args.experiments:
        if exp not in runners:
            raise SystemExit(f"Unknown experiment {exp!r}")
        all_rows.extend(runners[exp](args, nevresim_path))

    json_path = args.out_dir / "profile_results.json"
    csv_path = args.out_dir / "profile_results.csv"
    write_profile_rows(all_rows, json_path)
    write_profile_rows(all_rows, csv_path)

    summary = {
        "row_count": len(all_rows),
        "correlation_compile_vs_axon_slots": correlate_compile_vs_metric(all_rows, "total_axon_slots"),
        "correlation_compile_vs_header_bytes": correlate_compile_vs_metric(
            all_rows, "header_bytes",
        ),
        "correlation_compile_vs_max_spans": correlate_compile_vs_metric(
            all_rows, "max_spans_per_core",
        ),
        "slowest": max(
            (r for r in all_rows if r.compile_success),
            key=lambda r: r.compile_s,
            default=None,
        ),
        "slowest_total_pipeline": max(
            (r for r in all_rows if r.compile_success),
            key=lambda r: r.export_s + r.codegen_s + r.compile_s,
            default=None,
        ),
    }
    for key in ("slowest", "slowest_total_pipeline"):
        s = summary[key]
        if s is not None:
            summary[key] = {
                "experiment": s.experiment,
                "preset": s.preset,
                "compile_s": s.compile_s,
                "export_s": s.export_s,
                "codegen_s": s.codegen_s,
                "total_pipeline_s": s.export_s + s.codegen_s + s.compile_s,
                "total_axon_slots": s.total_axon_slots,
                "max_spans_per_core": s.max_spans_per_core,
                "header_bytes": s.artifact_sizes.generate_chip_hpp,
            }

    summary_path = args.out_dir / "profile_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    print(f"Wrote {json_path}")
    print(f"Wrote {csv_path}")
    print(f"Wrote {summary_path}")
    if summary.get("correlation_compile_vs_axon_slots") is not None:
        print(f"r(compile, axon_slots) = {summary['correlation_compile_vs_axon_slots']:.3f}")


if __name__ == "__main__":
    main()
