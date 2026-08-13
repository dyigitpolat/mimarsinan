"""C5 evidence: the platform-physics panel in pixels, against docs/ux/platform_physics.md.

Captures the three states the spec defines (no profile / truenorth / an override)
plus the objective picker in both, into generated/_wizard_review/c5_physics/.

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

OUT = REPO_ROOT / "generated" / "_wizard_review" / "c5_physics"


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


def _settle(page, ms: int = 900) -> None:
    page.wait_for_load_state("networkidle")
    page.wait_for_timeout(ms)


def _shot(page, name: str, selector: str | None = None) -> None:
    target = page.locator(selector) if selector else page
    target.screenshot(path=str(OUT / f"{name}.png"))
    print("  captured", name)


def _enable_hardware_search(page) -> None:
    """Turn on hw search so the objective picker renders."""
    field = page.locator('.field[data-key="hw_config_mode"]')
    select = field.locator("select")
    if select.count():
        select.select_option("search")
    else:
        field.locator('button:has-text("search")').first.click()
    _settle(page)


def main() -> None:
    from playwright.sync_api import sync_playwright

    OUT.mkdir(parents=True, exist_ok=True)
    port = _free_port()
    _start_server(port)

    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        page = browser.new_page(viewport={"width": 1600, "height": 1100})
        errors: list[str] = []
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)
        page.goto(f"http://127.0.0.1:{port}/wizard")
        _settle(page, 1600)

        page.click('.wb-nav-item[data-section-id="codesign"]')
        _settle(page)

        # State 1 — no profile declared.
        _shot(page, "c5_1_state1_no_profile", "#physicsPanel")
        _shot(page, "c5_1b_state1_codesign_full")

        _enable_hardware_search(page)
        _shot(page, "c5_2_state1_objective_chips_greyed", ".codesign-cosearch")

        # State 2 — truenorth declared.
        page.locator(".physics-profile-select").select_option("truenorth")
        _settle(page, 1200)
        _shot(page, "c5_3_state2_truenorth", "#physicsPanel")
        _shot(page, "c5_4_state2_objective_chips_offered", ".codesign-cosearch")

        # State 3 — an override on a declared constant.
        row = page.locator('.physics-constant[data-constant="t_cycle"]')
        row.locator("input").fill("500")
        row.locator("input").press("Enter")
        page.locator(".physics-panel-title").click()
        _settle(page, 1200)
        _shot(page, "c5_5_state3_override", "#physicsPanel")
        _shot(page, "c5_6_state3_codesign_full")

        browser.close()
        if errors:
            raise SystemExit(f"console/page errors: {errors}")
        print("done — zero page errors")


if __name__ == "__main__":
    main()
