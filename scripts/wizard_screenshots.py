"""Dev-mode workbench review: drive the real configurator in a real browser and
save the round-2 evidence set to generated/_wizard_review/round2/:
fresh per-section shots, the co-design interaction sequence (mapping updating
while arch + grid are edited), each of the five mode switches, the template
flow, and the error/remedy flow.

Usage (from the repo root, venv active; playwright + chromium required):
    python scripts/wizard_screenshots.py [--out DIR]

NOT part of the test suite: it needs a browser and binds a local port.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import tempfile
import threading
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

SECTION_IDS = ("workload", "codesign", "semantics", "tuning", "review")
TEMPLATE_TIER_CONFIG = REPO_ROOT / "test_configs" / "tier0" / "t0_02_lif_lenet5_fp_s8_novena_offload_pruned.json"

MODE_SWITCHES = (
    ("lif", None),
    ("ttfs", None),
    ("ttfs_quantized", None),
    ("ttfs_cycle_based", "cascaded"),
    ("ttfs_cycle_based", "synchronized"),
)


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _start_server(port: int) -> None:
    import uvicorn

    from mimarsinan.gui.runtime.collector import DataCollector
    from mimarsinan.gui.server.app import create_app

    app = create_app(DataCollector())
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
    server = uvicorn.Server(config)
    threading.Thread(target=server.run, daemon=True).start()
    deadline = time.time() + 30
    while time.time() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.5):
                return
        except OSError:
            time.sleep(0.2)
    raise RuntimeError("GUI server did not come up")


def _settle(page, ms: int = 1200) -> None:
    page.wait_for_load_state("networkidle")
    page.wait_for_timeout(ms)


def _goto_section(page, section_id: str) -> None:
    page.click(f'.wb-nav-item[data-section-id="{section_id}"]')
    _settle(page, 600)


def _click_segment(page, key: str, option: str) -> None:
    page.click(f'.field[data-key="{key}"] .seg-btn:text-is("{option}")')
    _settle(page, 900)


def _shot(page, out_dir: Path, name: str, shots: list[str], full: bool = False) -> None:
    path = out_dir / f"{name}.png"
    page.screenshot(path=str(path), full_page=full)
    shots.append(str(path.relative_to(REPO_ROOT)))


def _shoot_sections(page, base_url: str, out_dir: Path, flow: str,
                    shots: list[str], query: str = "") -> None:
    page.goto(base_url + "/wizard" + query)
    _settle(page, 2200)
    for i, section_id in enumerate(SECTION_IDS, start=1):
        _goto_section(page, section_id)
        _shot(page, out_dir, f"{flow}_{i}_{section_id}", shots, full=True)


def _shoot_codesign_sequence(page, base_url: str, out_dir: Path, shots: list[str]) -> None:
    """The J2 loop on one screen: edit grid + arch, watch the mapping re-plan."""
    page.goto(base_url + "/wizard")
    _settle(page, 2200)
    _goto_section(page, "codesign")
    _shot(page, out_dir, "codesign_1_initial", shots, full=True)

    # Shrink the first core type's count: the plan reacts (waste/feasibility).
    count_box = page.locator(".cores-editor-row").first.locator("input").nth(2)
    count_box.fill("4")
    count_box.dispatch_event("change")
    _settle(page, 1600)
    _shot(page, out_dir, "codesign_2_grid_edit_mapping_updates", shots, full=True)

    # Suggest hardware for this model: the grid rewrites, mapping re-plans.
    page.click(".suggest-hw")
    _settle(page, 2000)
    _shot(page, out_dir, "codesign_3_suggest_hardware", shots, full=True)

    # Architecture edit on the same screen: another vehicle, live re-plan.
    page.select_option('.field[data-key="model_type"] select', "simple_mlp")
    _settle(page, 1800)
    _shot(page, out_dir, "codesign_4_arch_edit_mapping_updates", shots, full=True)


def _shoot_defect_evidence(page, base_url: str, out_dir: Path, shots: list[str]) -> None:
    """Targeted evidence: search-mode prominence, the pruning slider knob,
    and the empty-hint discipline inside an Advanced drawer."""
    page.goto(base_url + "/wizard")
    _settle(page, 2200)

    # Defect 1: =search promotes the NAS panel; model_type/model_config vanish.
    _goto_section(page, "codesign")
    _click_segment(page, "model_config_mode", "search")
    _shot(page, out_dir, "defect1_search_mode_promotes_nas", shots, full=True)
    _click_segment(page, "model_config_mode", "user")

    # Defect 3: pruning is deployment-side and opens a slider knob.
    _goto_section(page, "semantics")
    page.click('.field[data-key="pruning"] .toggle-row')
    _settle(page, 900)
    _shot(page, out_dir, "defect3_pruning_slider_in_semantics", shots, full=True)
    page.click('.field[data-key="pruning"] .toggle-row')
    _settle(page, 600)

    # Defect 2: empty boxes state what empty means (Advanced drawer).
    _goto_section(page, "tuning")
    page.locator('.wb-section[data-section-id="tuning"] .advanced-toggle').first.click()
    _settle(page, 500)
    _shot(page, out_dir, "defect2_empty_hints_advanced_drawer", shots, full=True)


def _shoot_mode_switches(page, base_url: str, out_dir: Path, shots: list[str]) -> None:
    """Starter + each single mode switch: semantics + rail stay green."""
    for mode, schedule in MODE_SWITCHES:
        page.goto(base_url + "/wizard")
        _settle(page, 2200)
        _goto_section(page, "semantics")
        _click_segment(page, "spiking_mode", mode)
        if schedule is not None:
            _click_segment(page, "ttfs_cycle_schedule", schedule)
        _settle(page, 900)
        name = mode if schedule is None else f"{mode}_{schedule}"
        _shot(page, out_dir, f"mode_{name}", shots, full=True)


def _shoot_error_flow(page, base_url: str, out_dir: Path, template_id: str,
                      shots: list[str]) -> None:
    page.goto(base_url + "/wizard?template_id=" + template_id)
    _settle(page, 2200)
    _goto_section(page, "review")
    _shot(page, out_dir, "error_1_contract_violation", shots, full=True)
    page.click('.error-remedies .btn-sm:text-is("Declare vanilla (float weights)")')
    _settle(page, 1400)
    _shot(page, out_dir, "error_2_after_one_click_remedy", shots, full=True)


def _save_template(base_url: str, name: str, config: dict) -> str:
    import urllib.request

    req = urllib.request.Request(
        base_url + "/api/templates",
        data=json.dumps({"name": name, "config": config}).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req) as res:
        return json.loads(res.read())["id"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(REPO_ROOT / "generated" / "_wizard_review" / "round2"))
    args = parser.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Templates land in a scratch dir so the repo templates/ stays untouched.
    templates_dir = tempfile.mkdtemp(prefix="wizard_review_templates_")
    os.environ["MIMARSINAN_TEMPLATES_DIR"] = templates_dir
    os.environ.setdefault("MIMARSINAN_GUI_NO_BROWSER", "1")
    os.chdir(REPO_ROOT)

    port = _free_port()
    _start_server(port)
    base_url = f"http://127.0.0.1:{port}"

    with open(TEMPLATE_TIER_CONFIG, encoding="utf-8") as f:
        tier_config = json.load(f)
    template_id = _save_template(base_url, tier_config["experiment_name"], tier_config)

    # A deliberate quantization-contract violation for the error/remedy flow.
    broken = json.loads(json.dumps(tier_config))
    broken["experiment_name"] = "wq_contract_demo"
    broken["pipeline_mode"] = "phased"
    broken["deployment_parameters"]["weight_quantization"] = False
    broken["platform_constraints"]["weight_bits"] = 5
    broken_id = _save_template(base_url, "wq_contract_demo", broken)

    from playwright.sync_api import sync_playwright

    shots: list[str] = []
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1600, "height": 900})
        page.on("console", lambda msg: msg.type == "error" and print(f"[console.error] {msg.text}"))
        page.on("pageerror", lambda err: print(f"[pageerror] {err}"))
        _shoot_sections(page, base_url, out_dir, "fresh", shots)
        _shoot_codesign_sequence(page, base_url, out_dir, shots)
        _shoot_defect_evidence(page, base_url, out_dir, shots)
        _shoot_mode_switches(page, base_url, out_dir, shots)
        _shoot_sections(page, base_url, out_dir, "template", shots,
                        query="?template_id=" + template_id)
        _shoot_error_flow(page, base_url, out_dir, broken_id, shots)
        browser.close()

    print("Saved:")
    for shot in shots:
        print(" ", shot)


if __name__ == "__main__":
    main()
