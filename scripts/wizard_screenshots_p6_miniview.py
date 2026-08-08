"""P6 evidence: the mapping mini-view in pixels — segment envelopes, named
host ops, end-to-end badge — to generated/_wizard_review/p6_miniview/.

NOT part of the test suite: it needs a browser and binds a local port.
"""

from __future__ import annotations

import socket
import sys
import threading
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

OUT = REPO_ROOT / "generated" / "_wizard_review" / "p6_miniview"


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


def _goto(page, section_id: str) -> None:
    page.click(f'.wb-nav-item[data-section-id="{section_id}"]')
    _settle(page, 600)


def _shot(page, name: str) -> None:
    page.screenshot(path=str(OUT / f"{name}.png"), full_page=True)
    print("  captured", name)


def _set_model_type(page, model_type: str) -> None:
    field = page.locator('.field[data-key="model_type"]')
    select = field.locator("select")
    if select.count():
        select.select_option(model_type)
    else:
        field.locator(f'button:has-text("{model_type}")').first.click()
    _settle(page)


def main() -> None:
    from playwright.sync_api import sync_playwright

    OUT.mkdir(parents=True, exist_ok=True)
    port = _free_port()
    _start_server(port)

    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        page = browser.new_page(viewport={"width": 1600, "height": 950})
        errors: list[str] = []
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.goto(f"http://127.0.0.1:{port}/wizard")
        _settle(page)

        _goto(page, "codesign")
        _shot(page, "p6_1_starter_simplemlp_end_to_end_badge")

        _set_model_type(page, "lenet5")
        _shot(page, "p6_2_lenet5_multispan_named_pools_envelopes")

        _set_model_type(page, "stream_cnn")
        _shot(page, "p6_3_stream_cnn_end_to_end_badge")

        browser.close()
        if errors:
            raise SystemExit(f"console page errors: {errors}")
        print("done — zero page errors")


if __name__ == "__main__":
    main()
