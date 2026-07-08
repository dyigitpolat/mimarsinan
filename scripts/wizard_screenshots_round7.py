"""Round-7 evidence: drive the real configurator and capture the pretrained-
weights panel states to generated/_wizard_review/round7/.

Five states the mandate names:
  - the "Pretrained" switch DISABLED-with-reason on lenet5 (registers nothing);
  - the switch ON + panel expanded on a torchvision builder (single set locked);
  - the multi-set selection (a builder registering two sets);
  - the locked single-set case (the facts revealed, no chooser);
  - the incompatible-weights keyed error (a pinned inapplicable weight set).

NOT part of the test suite: it needs a browser and binds a local port.
"""

from __future__ import annotations

import argparse
import socket
import sys
import threading
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))


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


def _settle(page, ms: int = 1400) -> None:
    page.wait_for_load_state("networkidle")
    page.wait_for_timeout(ms)


def _goto_section(page, section_id: str) -> None:
    page.click(f'.wb-nav-item[data-section-id="{section_id}"]')
    _settle(page, 600)


def _shot(page, out_dir: Path, name: str, shots, full: bool = False) -> None:
    path = out_dir / f"{name}.png"
    page.screenshot(path=str(path), full_page=full)
    shots.append(str(path.relative_to(REPO_ROOT)))


def _select_model(page, base_url: str, model_type: str) -> None:
    _goto_section(page, "codesign")
    page.select_option('.field[data-key="model_type"] select', model_type)
    _settle(page, 2200)


def _turn_on_preload(page) -> None:
    _goto_section(page, "training")
    _settle(page, 600)
    toggle = page.locator('.pretrained-panel .toggle-row').first
    toggle.click()
    _settle(page, 1600)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(REPO_ROOT / "generated" / "_wizard_review" / "round7"))
    args = parser.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    port = _free_port()
    _start_server(port)
    base_url = f"http://127.0.0.1:{port}"

    from playwright.sync_api import sync_playwright

    shots: list[str] = []
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1600, "height": 1000})
        page.on("console", lambda m: m.type == "error" and print(f"[console.error] {m.text}"))
        page.on("pageerror", lambda e: print(f"[pageerror] {e}"))

        # 1 · lenet5 (the starter) registers nothing → switch disabled + reason.
        page.goto(base_url + "/wizard")
        _settle(page, 2400)
        _goto_section(page, "training")
        _shot(page, out_dir, "round7_1_switch_disabled_lenet5", shots, full=True)
        page.locator('.pretrained-panel').first.screenshot(
            path=str(out_dir / "round7_1_switch_disabled_lenet5_panel.png"))
        shots.append("round7_1_switch_disabled_lenet5_panel.png")

        # 2 · a single-set torchvision builder → switch ON, panel expanded,
        #     the set LOCKED with every registered fact revealed.
        _select_model(page, base_url, "torch_squeezenet11")
        _turn_on_preload(page)
        _shot(page, out_dir, "round7_2_single_set_locked_expanded", shots, full=True)
        page.locator('.pretrained-panel').first.screenshot(
            path=str(out_dir / "round7_2_single_set_locked_panel.png"))
        shots.append("round7_2_single_set_locked_panel.png")

        # 3 · a multi-set builder (resnet50 registers V1 + V2) → the selector.
        page.goto(base_url + "/wizard")
        _settle(page, 2200)
        _select_model(page, base_url, "torch_resnet50")
        _turn_on_preload(page)
        _shot(page, out_dir, "round7_3_multi_set_selection", shots, full=True)
        page.locator('.pretrained-panel').first.screenshot(
            path=str(out_dir / "round7_3_multi_set_selection_panel.png"))
        shots.append("round7_3_multi_set_selection_panel.png")
        # pick the second set — its source + facts follow.
        page.locator('.pretrained-selector .seg-btn').nth(1).click()
        _settle(page, 1400)
        page.locator('.pretrained-panel').first.screenshot(
            path=str(out_dir / "round7_3_multi_set_v2_chosen_panel.png"))
        shots.append("round7_3_multi_set_v2_chosen_panel.png")

        # 4 · locked single-set case, isolated (vit at its native geometry).
        page.goto(base_url + "/wizard")
        _settle(page, 2200)
        _select_model(page, base_url, "torch_vit")
        _turn_on_preload(page)
        page.locator('.pretrained-panel').first.screenshot(
            path=str(out_dir / "round7_4_vit_panel.png"))
        shots.append("round7_4_vit_panel.png")

        # 5 · the incompatible-weights keyed error: pin an unregistered set id
        #     via a saved template, load it, and show the keyed error tray.
        import json
        import urllib.request

        bad = {
            "data_provider_name": "CIFAR10_DataProvider",
            "experiment_name": "round7_incompatible_weightset",
            "generated_files_path": "./generated",
            "seed": 0,
            "start_step": None,
            "platform_constraints": {"cores": [{"max_axons": 256, "max_neurons": 256, "count": 1000}]},
            "deployment_parameters": {
                "model_type": "torch_resnet50",
                "model_config": {},
                "spiking_mode": "lif",
                "preload_weights": True,
                "pretrained_weight_set": "imagenet1k_v99",
            },
        }
        req = urllib.request.Request(
            base_url + "/api/templates",
            data=json.dumps({"name": "round7_incompatible_weightset", "config": bad}).encode(),
            headers={"Content-Type": "application/json"}, method="POST",
        )
        urllib.request.urlopen(req).read()
        page.goto(base_url + "/wizard?template_id=round7_incompatible_weightset")
        _settle(page, 2600)
        _goto_section(page, "training")
        _shot(page, out_dir, "round7_5_incompatible_weightset_keyed_error", shots, full=True)
        _goto_section(page, "review")
        _shot(page, out_dir, "round7_5_incompatible_error_review", shots, full=True)

        browser.close()

    print("Saved:")
    for shot in shots:
        print(" ", shot)


if __name__ == "__main__":
    main()
