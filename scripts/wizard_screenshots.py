"""Dev-mode wizard review: drive the real wizard in a real browser and save
per-step screenshots to generated/_wizard_review/ (fresh + template flows).

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

STEP_IDS = ("workload", "model", "deployment", "tuning", "review")
TEMPLATE_TIER_CONFIG = REPO_ROOT / "test_configs" / "tier0" / "t0_02_lif_lenet5_fp_s8_novena_offload_pruned.json"


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


def _shoot_flow(page, base_url: str, out_dir: Path, flow: str, query: str = "") -> list[str]:
    shots: list[str] = []
    page.goto(base_url + "/wizard" + query)
    _settle(page, 2000)
    for i, step_id in enumerate(STEP_IDS, start=1):
        page.click(f'.stepper-chip[data-step="{step_id}"]')
        _settle(page, 700)
        path = out_dir / f"{flow}_{i}_{step_id}.png"
        page.screenshot(path=str(path), full_page=True)
        shots.append(str(path.relative_to(REPO_ROOT)))
    return shots


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(REPO_ROOT / "generated" / "_wizard_review"))
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

    import urllib.request

    with open(TEMPLATE_TIER_CONFIG, encoding="utf-8") as f:
        tier_config = json.load(f)
    req = urllib.request.Request(
        base_url + "/api/templates",
        data=json.dumps({"name": tier_config["experiment_name"], "config": tier_config}).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req) as res:
        template_id = json.loads(res.read())["id"]

    from playwright.sync_api import sync_playwright

    shots: list[str] = []
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1600, "height": 1000})
        page.on("console", lambda msg: msg.type == "error" and print(f"[console.error] {msg.text}"))
        page.on("pageerror", lambda err: print(f"[pageerror] {err}"))
        shots += _shoot_flow(page, base_url, out_dir, "fresh")
        shots += _shoot_flow(
            page, base_url, out_dir, "template",
            query="?template_id=" + template_id,
        )
        browser.close()

    print("Saved:")
    for shot in shots:
        print(" ", shot)


if __name__ == "__main__":
    main()
