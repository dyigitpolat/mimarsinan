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

MODERN_RUN = os.environ.get(
    "MONITOR_REVIEW_RUN",
    "lenet5_baseline_20260708_204201_phased_deployment_run_20260708_204218",
)
NOC_RUN = "fal_t01_04_sync_mmixcore_wq_s16_pruned10_phased_deployment_run"

SECTION_IDS = ("overview", "steps", "analysis", "noc", "artifacts", "config", "console")

STEP_PAGES = (
    ("Pretraining", "pretraining", None),
    ("Activation Analysis", "activation_analysis", "Activations"),
    ("Weight Quantization", "weight_quantization", "Adaptation"),
    ("LIF Adaptation", "lif_adaptation", "Gate Story"),
    ("Quantization Verification", "quantization_verification", "Quantization"),
    ("Soft Core Mapping", "soft_core_mapping", None),
    ("Hard Core Mapping", "hard_core_mapping", None),
    ("SANA-FE Simulation", "sanafe_simulation", None),
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
    # A native `title` tooltip from whatever the last click left under the
    # cursor renders into the screenshot; park the cursor on the (title-less)
    # logo mark and let the tooltip fade before capturing.
    page.mouse.move(8, 8)
    page.wait_for_timeout(250)
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


# The step navigator moved into the section rail, so the rail must stay
# navigable on a long pipeline: every section link reachable without scrolling,
# arrow keys walking the steps, and the chevron collapsing the sub-list.
def _check_step_navigator(page, out_dir: Path, shots: list[str]) -> None:
    hidden = page.evaluate("""() => {
      const nav = document.querySelector('.wb-nav').getBoundingClientRect();
      return [...document.querySelectorAll('.wb-nav-item')]
        .filter(el => el.getBoundingClientRect().bottom > nav.bottom + 1)
        .map(el => el.dataset.sectionId);
    }""")
    print(f"  section links pushed out of the rail: {hidden or 'none'}")

    walked = page.evaluate("""() => {
      const items = [...document.querySelectorAll('.step-item')];
      items[0].focus();
      const first = document.activeElement.dataset.step;
      for (const key of ['ArrowDown', 'ArrowDown']) {
        document.getElementById('nav-step-list')
          .dispatchEvent(new KeyboardEvent('keydown', { key, bubbles: true }));
      }
      return [first, document.activeElement.dataset.step];
    }""")
    print(f"  arrow-key walk: {walked[0]!r} -> {walked[1]!r}")

    page.click('.wb-nav-item[data-section-id="steps"] .wb-nav-expander')
    _settle(page, 500)
    _shot(page, out_dir, "replay_8_steps_navigator_collapsed", shots)
    page.click('.wb-nav-item[data-section-id="steps"] .wb-nav-expander')
    _settle(page, 400)


# ── Flow: replay of a real finished run (every section + step pages) ──────
def flow_replay(page, base_url: str, out_dir: Path, shots: list[str]) -> None:
    page.goto(f"{base_url}/monitor?run_id={MODERN_RUN}")
    _settle(page, 2200)
    for i, section_id in enumerate(SECTION_IDS, start=1):
        _goto_section(page, section_id)
        _shot(page, out_dir, f"replay_{i}_{section_id}", shots, full=(section_id != 'console'))

    # Enriched step pages.
    _goto_section(page, "steps")
    for step_name, slug, tab in STEP_PAGES:
        if page.locator(f'.step-item[data-step="{step_name}"]').count() == 0:
            continue
        _select_step(page, step_name)
        if tab:
            _click_tab(page, tab)
        _shot(page, out_dir, f"step_{slug}", shots, full=True)

    _check_step_navigator(page, out_dir, shots)

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


# ── Flow: the Configuration tab across WS overview frames ─────────────────
# Reproduces (and, now, proves fixed) the fallback-to-raw-config bug: every step
# lifecycle event pushes a pipeline_overview frame, and the client used to
# rebuild its pipeline state from a hand-listed literal that omitted
# ``config_view`` — downgrading the Configuration tab to the legacy raw table.
def _config_feed(collector) -> None:
    steps = ["Model Configuration", "Model Building", "Pretraining"]
    collector.set_pipeline_info(steps, {
        "experiment_name": "config_ws_repro",
        "pipeline_mode": "phased",
        "spiking_mode": "lif",
        "weight_bits": 5,
    })
    time.sleep(3.0)
    for step in steps:
        collector.step_started(step)
        time.sleep(0.8)
        collector.step_completed(step, target_metric=0.0, metric_kind="carried")
        time.sleep(0.8)


def flow_config(page, base_url: str, out_dir: Path, shots: list[str], collector) -> bool:
    feeder = threading.Thread(target=_config_feed, args=(collector,), daemon=True)
    feeder.start()
    page.goto(f"{base_url}/monitor")
    _settle(page, 1200)
    _goto_section(page, "config")
    structured_before = page.locator("#config-body .config-view").count()
    _shot(page, out_dir, "config_ws_1_before_lifecycle_frames", shots, full=True)
    feeder.join(timeout=30)
    _settle(page, 1200)
    structured_after = page.locator("#config-body .config-view").count()
    _shot(page, out_dir, "config_ws_2_after_lifecycle_frames", shots, full=True)
    ok = bool(structured_before) and bool(structured_after)
    print(f"  structured config view: before={bool(structured_before)} after={bool(structured_after)}"
          + ("" if ok else "  << BUG: Configuration tab downgraded to the raw table"))
    return ok


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
    best = 0.985
    for rung in range(6):
        rate = min(1.0, (rung + 1) / 6)
        acc = 0.985 - 0.02 * math.sin(rung)
        collector.record_metric("LIF Adaptation", rate)
        collector.record_metric("Adaptation target", acc)
        accepted = rung % 3 != 2
        if accepted:
            best = max(best, acc)
        collector.record_event("mbh_gate", {
            "action": "accept" if accepted else "reject",
            "tuner": "LIF Adaptation", "rung": rung, "rate": rate,
            "full_acc": acc, "best_full_acc": best,
        })
        time.sleep(0.8)
    collector.record_event("mbh_endpoint", {
        "tuner": "LIF Adaptation", "entry": 0.985, "exit": best,
        "budget_steps": 800, "steps_used": 420, "engaged": True,
        "reached": True,
    })
    collector.record_event("quantization_report", {
        "bits": 5, "q_max": 15,
        "layers": [
            {"index": i, "name": f"features_{i}", "parameter_scale": 12.0 + i,
             "n_weights": 4000, "q_max": 15, "zero_frac": 0.15 + 0.05 * i,
             "clip_frac": 0.02 * i, "effective_levels": 31 - 2 * i,
             "int_min": -15, "int_max": 15}
            for i in range(4)
        ],
    })
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
    if _click_tab(page, "Gate Story"):
        _shot(page, out_dir, "live_5_gate_story", shots, full=True)
    if _click_tab(page, "Quantization"):
        _shot(page, out_dir, "live_6_quantization_report", shots, full=True)
    _goto_section(page, "analysis")
    _shot(page, out_dir, "live_7_analysis_staircase", shots, full=True)


# ── Flow: incremental WS streaming — curves grow at several positions ─────
# Drives the in-process collector INLINE (same thread as the screenshots) so the
# feed and the captures are deterministically coordinated: emit a burst, let the
# browser's rAF coalesce + repaint, then capture. Proves the live Overview line
# and a live tuner step advance by extendTraces/restyle (not a full re-plot) —
# the growth is visible across positions, honestly (measured points only; the
# carried preamble never plots a point).
def flow_streaming(page, base_url: str, out_dir: Path, shots: list[str], collector) -> None:
    import math

    steps = [
        "Model Configuration", "Model Building", "Pretraining", "Torch Mapping",
        "Activation Analysis", "Clamp Adaptation", "Weight Quantization",
        "LIF Adaptation",
    ]
    collector.set_pipeline_info(steps, {
        "experiment_name": "live_streaming", "pipeline_mode": "phased",
        "spiking_mode": "lif", "weight_bits": 5,
    })
    page.goto(f"{base_url}/monitor")
    _settle(page, 1200)

    # Carried preamble: contributes event lines, never a plotted point.
    for name in ("Model Configuration", "Model Building"):
        collector.step_started(name)
        collector.step_completed(name, target_metric=0.0, metric_kind="carried")

    # ── A live tuner step: a loss/accuracy/LR curve streaming in ──
    collector.step_started("Pretraining")
    _goto_section(page, "steps")  # auto-follow lands on the running step
    page.wait_for_timeout(500)

    def _pretrain_burst(lo: int, hi: int) -> None:
        # A small per-point delay spaces the metric timestamps like a real
        # trainer (one report per optimizer step), so the curve spreads over
        # elapsed time instead of collapsing onto one instant — and it paces
        # the WS frames so the server's metric_categories reach the client and
        # the loss/accuracy/LR curves split onto their own axes.
        for i in range(lo, hi):
            collector.record_metric("Training loss", 2.2 * math.exp(-i / 60) + 0.05)
            if i % 8 == 0:
                collector.record_metric("Validation accuracy", min(0.99, 0.3 + i / 300))
                collector.record_metric("LR", 0.001 * (0.98 ** (i // 8)))
            time.sleep(0.015)

    for lo, hi, tag in ((0, 45, "1_early"), (45, 130, "2_mid"), (130, 240, "3_late")):
        _pretrain_burst(lo, hi)
        _settle(page, 900)
        _shot(page, out_dir, f"streaming_tuner_{tag}", shots, full=True)
    collector.step_completed("Pretraining", target_metric=0.95, metric_kind="measured")

    # ── The live Overview: the measured line grows point by point ──
    _goto_section(page, "overview")
    _settle(page, 700)
    _shot(page, out_dir, "streaming_overview_1_early", shots, full=True)

    # Each entry adds ONE measured point; some carry a gate verdict (a pass/fail
    # event line slides in alongside the growing line).
    measured = [
        ("Torch Mapping", 0.955, None),
        ("Activation Analysis", 0.958, None),
        ("Clamp Adaptation", 0.96, None),
        ("Weight Quantization", 0.965,
         {"status": "pass", "rule": "weights on the integer grid"}),
    ]
    for idx, (name, metric, verdict) in enumerate(measured):
        collector.step_started(name)
        page.wait_for_timeout(350)
        collector.step_completed(name, target_metric=metric, metric_kind="measured",
                                 verdict=verdict)
        _settle(page, 650)
        if idx == 1:
            _shot(page, out_dir, "streaming_overview_2_mid", shots, full=True)
    _shot(page, out_dir, "streaming_overview_3_full", shots, full=True)

    # Leave a final tuner in flight so the running timing bar + rail read live.
    collector.step_started("LIF Adaptation")
    collector.record_event("mbh_gate", {"action": "entry", "tuner": "LIF Adaptation",
                                        "best_full_acc": 0.965})
    _settle(page, 900)
    _shot(page, out_dir, "streaming_overview_4_running", shots, full=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(REPO_ROOT / "generated" / "_monitor_review" / "round1"))
    parser.add_argument("--flow", default="all", choices=("all", "replay", "noc", "live", "config", "streaming"))
    args = parser.parse_args()
    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    from playwright.sync_api import sync_playwright

    port = _free_port()
    collector = _start_server(port)
    base_url = f"http://127.0.0.1:{port}"
    shots: list[str] = []

    config_ok = True
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
        if args.flow in ("all", "config"):
            print("flow: config")
            config_ok = flow_config(page, base_url, out_dir, shots, collector)
        if args.flow in ("all", "live"):
            print("flow: live")
            flow_live(page, base_url, out_dir, shots, collector)
        if args.flow in ("all", "streaming"):
            print("flow: streaming")
            flow_streaming(page, base_url, out_dir, shots, collector)
        browser.close()

    print(f"\n{len(shots)} screenshots -> {out_dir}")
    return 0 if config_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
