"""P1 evidence: the domain axis in pixels — Core-semantics card, dormant
switch, streamed variant — to generated/_wizard_review/p1_domain/.

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

OUT = REPO_ROOT / "generated" / "_wizard_review" / "p1_domain"


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

        _goto(page, "semantics")
        _shot(page, "p1_1_core_semantics_card_fresh")

        page.click('.field[data-key="spiking_variant"] button:has-text("streamed")')
        _settle(page)
        _shot(page, "p1_2_streamed_variant_with_advisory")
        _goto(page, "codesign")
        _shot(page, "p1_3_streamed_allow_scheduling_locked")

        _goto(page, "semantics")
        page.click('.field[data-key="spiking_variant"] button:has-text("synchronized")')
        _settle(page, 600)
        page.click('.field[data-key="core_semantics"] button:has-text("mvm")')
        _settle(page)
        _shot(page, "p1_4_mvm_switch_zero_errors_dormant_note")
        _goto(page, "codesign")
        _shot(page, "p1_5_mvm_codesign_no_temporal_grid")

        _goto(page, "semantics")
        page.click('.field[data-key="core_semantics"] button:has-text("spiking")')
        _settle(page)
        _shot(page, "p1_6_switch_back_restored")

        browser.close()
        if errors:
            raise SystemExit(f"console page errors: {errors}")
        print("done — zero page errors")


if __name__ == "__main__":
    main()
