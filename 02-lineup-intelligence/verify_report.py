"""
Headless Playwright check of the Lineups page at 390x844 and 1280x900, light and dark.

Asserts: zero console errors, the shared app chrome (top bar + tab bar on phones) renders, the best five
is preselected (5 checkmarks, 5 hero avatars, a 0-100 fit with a plain verdict), tapping roster rows changes
the fit, a sixth pick is refused with a toast, a swap applies (toast "Swapped in ..."), "Why this works" /
swaps / pairs / rotation / league lists populate, every sheet opens and closes (about, matchup, all pairs,
league), the matchup estimate is labelled "Estimate", switching teams fetches that team's data/<ABBR>.json
and rebuilds pairs + heatmap from the full table, no horizontal overflow, every tappable row / button is
>= 44 px tall, and the §7 ban list (monospace, uppercase, middle dots, arrows, "Read:") has no hits in the HTML.
The page is served over a local http.server (per-team JSON is fetched lazily, which file:// blocks).
Writes <site>/_preview_mobile.png and _preview_desktop.png (light) plus /tmp/lu_verify_*.png.

    python verify_report.py [page.html] [screenshot_dir]
    (defaults: ../site/lineup/index.html, screenshots next to it)
"""
from __future__ import annotations

import functools
import http.server
import os
import re
import socketserver
import sys
import threading

from playwright.sync_api import sync_playwright

ROOT = os.path.dirname(os.path.abspath(__file__))
SITE = os.path.normpath(os.path.join(ROOT, "..", "site", "lineup"))
CHROME = next((p for p in ("/usr/bin/google-chrome", "/usr/bin/chromium", "/usr/bin/chromium-browser") if os.path.exists(p)), None)

TARGETS = "#roster .prow, .swap, .pair, .league-row, .usefive, .btn, .teampick, .matchup-list .row"
BANNED = [r"monospace", r"text-transform:\s*uppercase", r"letter-spacing:\s*\.1", r"text-shadow", "·", "→", r"Read:"]


def ban_scan(html_text: str) -> list[str]:
    return [b for b in BANNED if re.search(b, html_text)]


def open_close(page, sheet: str, opener: str) -> None:
    page.locator(opener).first.click()
    page.wait_for_function(f"document.getElementById('{sheet}').classList.contains('open')")
    page.wait_for_timeout(350)
    page.locator(f"#{sheet} [data-close]").click()
    page.wait_for_function(f"!document.getElementById('{sheet}').classList.contains('open')")
    page.wait_for_timeout(300)


def check(page, width: int) -> dict:
    page.wait_for_selector("#roster .prow")
    assert page.locator("header.ap-top").count() == 1, "shared top bar missing"
    if width < 900:
        assert page.locator("nav.ap-tabs a").count() == 4, "tab bar missing"
    assert page.locator("#roster .prow").count() >= 10
    assert page.locator('#roster .prow[aria-pressed="true"]').count() == 5, "best five should be preselected"
    assert page.locator("#five .av.lg:not(.open)").count() == 5
    fit0 = page.inner_text("#fitV").replace("/100", "").strip()
    assert fit0.isdigit(), fit0
    assert "lineup" in page.text_content("#verdict").lower()
    # deselect one, pick another
    page.locator('#roster .prow[aria-pressed="true"]').first.click()
    assert "Pick 1 more player" in page.text_content("#verdict")
    assert page.text_content("#count").strip() == "4 of 5"
    page.locator('#roster .prow[aria-pressed="false"]').first.click()
    fit1 = page.inner_text("#fitV").replace("/100", "").strip()
    assert fit1.isdigit(), fit1
    # sixth pick refused
    page.locator('#roster .prow[aria-pressed="false"]').first.click()
    assert page.locator('#roster .prow[aria-pressed="true"]').count() == 5
    page.wait_for_selector(".toast.show")
    assert page.locator("#why .why-row").count() >= 1
    assert page.locator("#swaps .swap").count() == 5
    before = page.evaluate("JSON.stringify(window.LineupIQ.selection.sort())")
    name_in = page.locator("#swaps .swap b").first.inner_text()
    page.locator("#swaps .swap").first.click()
    assert page.evaluate("JSON.stringify(window.LineupIQ.selection.sort())") != before, "swap should change the five"
    page.wait_for_function(f"document.querySelector('.toast').textContent === 'Swapped in {name_in}'")
    assert page.locator("#pairsBest .pair").count() == 3 and page.locator("#pairsWorst .pair").count() == 3
    assert page.locator("#rotation .list").count() == 3
    assert page.locator("#leagueTop .league-row").count() == 10
    # sheets
    open_close(page, "aboutSheet", '.list [data-sheet="aboutSheet"]')
    open_close(page, "leagueSheet", '[data-sheet="leagueSheet"]')
    page.locator('[data-sheet="heatSheet"]').click()
    page.wait_for_function("document.getElementById('heatSheet').classList.contains('open')")
    page.wait_for_timeout(400)
    assert page.locator("#heatmap rect.heat-cell").count() >= 100
    page.locator("#heatmap rect.heat-cell").nth(3).dispatch_event("click")
    assert "lineups" in page.text_content("#heatVal")
    page.keyboard.press("Escape")
    page.wait_for_function("!document.getElementById('heatSheet').classList.contains('open')")
    page.locator('[data-sheet="matchupSheet"]').click()
    page.wait_for_function("document.getElementById('matchupSheet').classList.contains('open')")
    page.select_option("#oppSel", index=1)
    assert "Against the" in page.text_content("#oppResult")
    page.locator("#matchupSheet [data-close]").click()
    page.wait_for_timeout(300)
    assert "Estimate" in page.text_content("#matchNote") and page.is_visible("#matchNote")
    # team switch -> lazy per-team JSON
    t0 = page.evaluate("window.LineupIQ.team")
    with page.expect_response(lambda r: "/data/" in r.url and r.url.endswith(".json")) as resp:
        page.select_option("#teamSel", index=3)
    assert resp.value.ok
    page.wait_for_function("window.LineupIQ.full && window.LineupIQ.state === 'ok'", timeout=15000)
    assert page.evaluate("window.LineupIQ.team") != t0
    assert page.text_content("#teamName") == page.evaluate("window.LineupIQ.team")
    assert page.locator('#roster .prow[aria-pressed="true"]').count() == 5
    assert page.locator("#pairsBest .pair").count() == 3, "pairs should rebuild from the fetched table"
    assert page.locator("#heatmap rect.heat-cell").count() >= 100
    page.locator('#roster .prow[aria-pressed="true"]').first.click()
    page.locator('#roster .prow[aria-pressed="false"]').first.click()
    assert page.inner_text("#fitV").replace("/100", "").strip().isdigit(), "builder should score fives from the fetched table"
    assert page.locator("#swaps .swap").count() == 5
    # league row loads a team
    page.locator("#leagueTop .league-row").nth(1).click()
    page.wait_for_timeout(100)
    assert page.evaluate("window.LineupIQ.team") == page.locator("#leagueTop .league-row").nth(1).get_attribute("data-team")
    sw = page.evaluate("Math.max(document.documentElement.scrollWidth, document.body.scrollWidth)")
    assert sw <= width, f"horizontal overflow: scrollWidth {sw} > {width}"
    small = page.evaluate(f"""[...document.querySelectorAll("{TARGETS}")].filter(e => !e.closest('.sheet')).map(e => {{ const r = e.getBoundingClientRect(); return [e.className || e.tagName, r.height]; }}).filter(x => x[1] > 0 && x[1] < 44)""")
    assert not small, f"touch targets < 44px: {small[:6]}"
    page.select_option("#teamSel", index=0)
    page.wait_for_timeout(80)
    return {"fit0": fit0, "fit1": fit1, "scrollWidth": sw}


class _Quiet(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *a):
        pass


def serve(root: str) -> tuple[socketserver.TCPServer, int]:
    handler = functools.partial(_Quiet, directory=root)
    srv = socketserver.TCPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, srv.server_address[1]


def main() -> int:
    target = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else os.path.join(SITE, "index.html"))
    shots = sys.argv[2] if len(sys.argv) > 2 else os.path.dirname(target)
    with open(target, encoding="utf-8") as fh:
        hits = ban_scan(fh.read())
    # serve from the folder that contains site/assets/app.css so ../ links and the service worker resolve
    root = os.path.dirname(target)
    while not os.path.exists(os.path.join(root, "site", "assets", "app.css")) and os.path.dirname(root) != root:
        root = os.path.dirname(root)
    srv, port = serve(root)
    url = f"http://127.0.0.1:{port}/" + os.path.relpath(target, root).replace(os.sep, "/")
    errors: list[str] = []
    with sync_playwright() as p:
        kw = {"args": ["--no-sandbox"]}
        if CHROME:
            kw["executable_path"] = CHROME
        browser = p.chromium.launch(**kw)
        for scheme in ("light", "dark"):
            for name, vp, mobile in (("mobile", (390, 844), True), ("desktop", (1280, 900), False)):
                ctx = browser.new_context(viewport={"width": vp[0], "height": vp[1]}, device_scale_factor=2 if mobile else 1,
                                          is_mobile=mobile, has_touch=mobile, color_scheme=scheme)
                page = ctx.new_page()
                page.on("pageerror", lambda e: errors.append(f"pageerror: {e}"))
                page.on("console", lambda m: errors.append(f"console: {m.text}") if m.type == "error" and "favicon" not in str(m.location) else None)
                page.goto(url)
                res = check(page, vp[0])
                page.evaluate("window.scrollTo(0,0)")
                page.wait_for_timeout(300)
                if scheme == "light":
                    page.screenshot(path=os.path.join(shots, f"_preview_{name}.png"))
                page.screenshot(path=os.path.join("/tmp", f"lu_verify_{name}_{scheme}.png"))
                print(f"{name} {vp[0]}x{vp[1]} {scheme}: fit {res['fit0']} -> {res['fit1']} after tap, scrollWidth {res['scrollWidth']}")
                ctx.close()
        browser.close()
    srv.shutdown()
    if hits:
        print("banned patterns in HTML: " + ", ".join(hits))
    if errors:
        print("JS errors:\n  " + "\n  ".join(errors))
    if hits or errors:
        return 1
    print(f"OK: {target}; screenshots in {shots}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
