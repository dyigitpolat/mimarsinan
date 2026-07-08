"""Dev-mode monitor review: drive the redesigned pipeline monitor in a real
browser and save the evidence set to generated/_monitor_review/<round>/:
every section against a REAL finished run (replay), the Hardware/NoC playback
at two scrub positions plus the flow inspector, three enriched step pages,
the legacy-backfill run, and live mode via a synthetic in-process feed over
the /ws channel.

Usage (from the repo root, venv active; playwright + chromium required):
    python scripts/monitor_screenshots.py [--out DIR] [--flow all|replay|noc|live]

NOT part of the test suite: it needs a browser and binds a local port.
"""

from __future__ import annotations

import argparse
import os
import socket
import sys
import threading
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

os.environ.setdefault("MIMARSINAN_RUNS_ROOT", str(REPO_ROOT / "generated"))

MODERN_RUN = "lenet5_baseline_20260708_204201_phased_deployment_run_20260708_204218"
NOC_RUN = "fal_t01_04_sync_mmixcore_wq_s16_pruned10_phased_deployment_run"

SECTION_IDS = ("overview", "steps", "analysis", "noc", "artifacts", "config", "console")

STEP_PAGES = (
    ("Pretraining", "pretraining"),
    ("Activation Analysis", "activation_analysis"),
    ("Soft Core Mapping", "soft_core_mapping"),
    ("Hard Core Mapping", "hard_core_mapping"),
    ("SANA-FE Simulation", "sanafe_simulation"),
)


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _start_server(port: int):
    import uvicorn

    from mimarsinan.gui.runtime.collector import DataCollector
    from mimarsinan.gui.server.app import create_app

    collector = DataCollector()
    app = create_app(collector)
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
    server = uvicorn.Server(config)
    threading.Thread(target=server.run, daemon=True).start()
    deadline = time.time() + 30
    while time.time() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.5):
                return collector
        except OSError:
            time.sleep(0.2)
    raise RuntimeError("GUI server did not come up")


def _settle(page, ms: int = 1200) -> None:
    page.wait_for_load_state("networkidle")
    page.wait_for_timeout(ms)


def _goto_section(page, section_id: str) -> None:
    page.click(f'.wb-nav-item[data-section-id="{section_id}"]')
    _settle(page, 900)


def _shot(page, out_dir: Path, name: str, shots: list[str], full: bool = False) -> None:
    path = out_dir / f"{name}.png"
    page.screenshot(path=str(path), full_page=full)
    shots.append(str(path.relative_to(REPO_ROOT)))
    print(f"  shot {path.name}")


def _select_step(page, step_name: str) -> None:
    page.click(f'.step-item[data-step="{step_name}"]')
    _settle(page, 900)


def _click_tab(page, label: str) -> bool:
    tab = page.locator(f'#step-tabs .tab-btn:text-is("{label}")')
    if tab.count() == 0:
        return False
    tab.first.click()
    _settle(page, 800)
    return True


# ── Flow: replay of a real finished run (every section + step pages) ──────
def flow_replay(page, base_url: str, out_dir: Path, shots: list[str]) -> None:
    page.goto(f"{base_url}/monitor?run_id={MODERN_RUN}")
    _settle(page, 2200)
    for i, section_id in enumerate(SECTION_IDS, start=1):
        _goto_section(page, section_id)
        _shot(page, out_dir, f"replay_{i}_{section_id}", shots, full=(section_id != 'console'))

    # Enriched step pages.
    _goto_section(page, "steps")
    for step_name, slug in STEP_PAGES:
        if page.locator(f'.step-item[data-step="{step_name}"]').count() == 0:
            continue
        _select_step(page, step_name)
        _shot(page, out_dir, f"step_{slug}", shots, full=True)

    # Legacy-backfill run (pre-events.jsonl): overview + analysis keep working.
    page.goto(f"{base_url}/monitor?run_id={NOC_RUN}")
    _settle(page, 2200)
    _shot(page, out_dir, "legacy_1_overview", shots, full=True)
    _goto_section(page, "analysis")
    _shot(page, out_dir, "legacy_2_analysis", shots, full=True)


# ── Flow: the Hardware/NoC instruments ─────────────────────────────────────
def flow_noc(page, base_url: str, out_dir: Path, shots: list[str]) -> None:
    page.goto(f"{base_url}/monitor?run_id={NOC_RUN}")
    _settle(page, 2200)
    _goto_section(page, "noc")
    _settle(page, 2000)
    _shot(page, out_dir, "noc_1_initial", shots, full=True)

    # Scrub to two positions over simulated time.
    scrubber = page.locator("#noc-scrubber")
    if scrubber.count() > 0:
        maximum = int(scrubber.get_attribute("max") or "0")
        for tag, frac in (("early", 0.15), ("late", 0.7)):
            pos = max(0, int(maximum * frac))
            scrubber.evaluate(
                "(el, v) => { el.value = String(v); el.dispatchEvent(new Event('input')); }",
                pos,
            )
            _settle(page, 700)
            _shot(page, out_dir, f"noc_2_scrub_{tag}_t{pos}", shots)

    # Playback running.
    page.click("#noc-play-btn")
    page.wait_for_timeout(1200)
    _shot(page, out_dir, "noc_3_playback", shots)
    page.click("#noc-play-btn")

    # Mesh-links overlay.
    page.click('.seg-btn[data-overlay="links"]')
    _settle(page, 700)
    _shot(page, out_dir, "noc_4_links_overlay", shots)

    # Cumulative totals.
    page.click("#noc-cumulative-btn")
    _settle(page, 700)
    _shot(page, out_dir, "noc_5_cumulative", shots, full=True)
    page.click("#noc-cumulative-btn")
    _settle(page, 500)

    # Flow inspector: pick two tiles with traffic.
    page.click('.seg-btn[data-overlay="flows"]')
    _settle(page, 500)
    tiles = page.locator(".noc-tile")
    if tiles.count() >= 2:
        tiles.nth(0).click()
        _settle(page, 400)
        tiles.nth(min(2, tiles.count() - 1)).click()
        _settle(page, 900)
        _shot(page, out_dir, "noc_6_flow_inspector", shots, full=True)


# ── Flow: live mode via a synthetic in-process feed over /ws ──────────────
def _synthetic_feed(collector) -> None:
    import math

    steps = ["Model Configuration", "Model Building", "Pretraining", "LIF Adaptation"]
    collector.set_pipeline_info(steps, {
        "experiment_name": "live_synthetic",
        "pipeline_mode": "phased",
        "spiking_mode": "lif",
        "weight_bits": 5,
    })

    collector.step_started("Model Configuration")
    time.sleep(0.4)
    collector.step_completed("Model Configuration", target_metric=0.0, metric_kind="carried")
    collector.step_started("Model Building")
    time.sleep(0.4)
    collector.step_completed("Model Building", target_metric=0.0, metric_kind="carried")

    collector.step_started("Pretraining")
    for i in range(240):
        collector.record_metric("Training loss", 2.2 * math.exp(-i / 60) + 0.05)
        if i % 12 == 0:
            collector.record_metric("Validation accuracy", min(0.99, 0.3 + i / 300))
            collector.record_metric("LR", 0.001 * (0.98 ** (i // 12)))
        time.sleep(0.02)
    collector.step_completed("Pretraining", target_metric=0.985, metric_kind="measured")

    collector.step_started("LIF Adaptation")
    collector.record_event("mbh_gate", {"action": "entry", "tuner": "LIF Adaptation",
                                        "best_full_acc": 0.985})
    for rung in range(6):
        rate = min(1.0, (rung + 1) / 6)
        acc = 0.985 - 0.02 * math.sin(rung)
        collector.record_metric("LIF Adaptation", rate)
        collector.record_metric("Adaptation target", acc)
        collector.record_event("mbh_gate", {
            "action": "accept" if rung % 3 != 2 else "reject",
            "tuner": "LIF Adaptation", "rung": rung, "rate": rate,
            "full_acc": acc, "best_full_acc": max(0.985, acc),
        })
        time.sleep(0.8)
    # leaves the step RUNNING — live rail should show it in flight


def flow_live(page, base_url: str, out_dir: Path, shots: list[str], collector) -> None:
    feeder = threading.Thread(target=_synthetic_feed, args=(collector,), daemon=True)
    feeder.start()
    page.goto(f"{base_url}/monitor")
    _settle(page, 1500)
    _shot(page, out_dir, "live_1_early", shots, full=True)
    page.wait_for_timeout(4500)
    _shot(page, out_dir, "live_2_pretraining_streaming", shots, full=True)
    _goto_section(page, "steps")
    page.wait_for_timeout(2500)
    _shot(page, out_dir, "live_3_steps_streaming", shots, full=True)
    feeder.join(timeout=30)
    page.wait_for_timeout(2500)
    _shot(page, out_dir, "live_4_adaptation_events", shots, full=True)
    _goto_section(page, "analysis")
    _shot(page, out_dir, "live_5_analysis_staircase", shots, full=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(REPO_ROOT / "generated" / "_monitor_review" / "round1"))
    parser.add_argument("--flow", default="all", choices=("all", "replay", "noc", "live"))
    args = parser.parse_args()
    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    from playwright.sync_api import sync_playwright

    port = _free_port()
    collector = _start_server(port)
    base_url = f"http://127.0.0.1:{port}"
    shots: list[str] = []

    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        page = browser.new_page(viewport={"width": 1600, "height": 900})
        page.on("console", lambda m: m.type == "error" and print(f"  [console.error] {m.text}"))
        if args.flow in ("all", "replay"):
            print("flow: replay")
            flow_replay(page, base_url, out_dir, shots)
        if args.flow in ("all", "noc"):
            print("flow: noc")
            flow_noc(page, base_url, out_dir, shots)
        if args.flow in ("all", "live"):
            print("flow: live")
            flow_live(page, base_url, out_dir, shots, collector)
        browser.close()

    print(f"\n{len(shots)} screenshots -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
