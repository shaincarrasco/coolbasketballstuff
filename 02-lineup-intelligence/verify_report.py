"""
Headless Playwright check of the Lineup Intelligence page at 375x812 and 1280x900.

Asserts: zero console errors, the shared top bar renders, tapping tiles changes the fit,
swaps / heatmap / rotation populate, the opponent heuristic is labelled, no horizontal
overflow at 375, and every control / tile / swap row is >= 40 px tall.
Writes <site>/_preview_mobile.png and _preview_desktop.png.

    python verify_report.py [page.html] [screenshot_dir]
    (defaults: ../site/lineup/index.html, screenshots next to it)
"""
from __future__ import annotations

import os
import sys

from playwright.sync_api import sync_playwright

ROOT = os.path.dirname(os.path.abspath(__file__))
SITE = os.path.normpath(os.path.join(ROOT, "..", "site", "lineup"))
CHROME = next((p for p in ("/usr/bin/google-chrome", "/usr/bin/chromium", "/usr/bin/chromium-browser") if os.path.exists(p)), None)

TARGETS = "#tiles .tile, .swap, .btn, select, details.more summary, .board tbody tr, .topbar nav a"


def check(page, width: int, errors: list[str]) -> dict:
    page.wait_for_selector("#tiles .tile")
    assert page.locator("header.topbar .brand").count() == 1, "shared top bar missing"
    assert page.locator("#tiles .tile").count() >= 10
    assert page.locator("#tiles .tile.on").count() == 5, "best five should be preselected"
    fit0, title0 = page.inner_text("#roFit").strip(), page.text_content("#roTitle")
    assert fit0.isdigit(), fit0
    page.locator("#tiles .tile.on").first.click()
    assert "select 1 more" in page.text_content("#roRank")
    page.locator("#tiles .tile:not(.on)").first.click()
    fit1, title1 = page.inner_text("#roFit").strip(), page.text_content("#roTitle")
    assert fit1.isdigit() and fit1 != fit0 and title1 != title0, f"tap should change fit ({fit0} -> {fit1})"
    assert "rank" in page.text_content("#roRank")
    assert page.locator("#swaps .swap").count() >= 5
    assert page.locator("#roContrib .why-row").count() >= 1
    assert page.locator("#court circle[fill='var(--accent)']").count() == 5
    assert "Read:" in page.text_content("#readLine")
    page.locator("#swaps .swap").first.click()
    assert page.text_content("#roTitle") != title1
    page.select_option("#oppSel", index=1)
    assert "heuristic" in page.text_content("#roAdj").lower()
    assert page.locator("#heatmap rect.heat-cell").count() >= 100
    page.locator("#heatmap rect.heat-cell").nth(3).click(force=True)
    assert "fit vs team mean" in page.text_content("#heatVal")
    assert page.locator("#duosBest li").count() == 5 and page.locator("#duosWorst li").count() == 5
    assert page.locator("#rotation .unit").count() == 3
    page.select_option("#teamSel", index=3)
    page.wait_for_timeout(80)
    assert page.locator("#tiles .tile.on").count() == 5
    sw = page.evaluate("Math.max(document.documentElement.scrollWidth, document.body.scrollWidth)")
    assert sw <= width, f"horizontal overflow: scrollWidth {sw} > {width}"
    small = page.evaluate(f"""[...document.querySelectorAll("{TARGETS}")].filter(e => !e.closest('.topbar')).map(e => {{ const r = e.getBoundingClientRect(); return [e.className || e.tagName, r.height]; }}).filter(x => x[1] > 0 && x[1] < 40)""")
    # shell nav icons (36px) belong to site/assets/shell.css and are excluded
    assert not small, f"touch targets < 40px: {small[:6]}"
    page.select_option("#teamSel", index=0)
    page.wait_for_timeout(80)
    return {"fit0": fit0, "fit1": fit1, "scrollWidth": sw}


def main() -> int:
    target = sys.argv[1] if len(sys.argv) > 1 else os.path.join(SITE, "index.html")
    shots = sys.argv[2] if len(sys.argv) > 2 else os.path.dirname(target)
    url = "file://" + os.path.abspath(target)
    errors: list[str] = []
    with sync_playwright() as p:
        kw = {"args": ["--no-sandbox"]}
        if CHROME:
            kw["executable_path"] = CHROME
        browser = p.chromium.launch(**kw)
        for name, vp, mobile in (("mobile", (375, 812), True), ("desktop", (1280, 900), False)):
            ctx = browser.new_context(viewport={"width": vp[0], "height": vp[1]}, device_scale_factor=2 if mobile else 1,
                                      is_mobile=mobile, has_touch=mobile)
            page = ctx.new_page()
            page.on("pageerror", lambda e: errors.append(f"pageerror: {e}"))
            page.on("console", lambda m: errors.append(f"console: {m.text}") if m.type == "error" else None)
            page.goto(url)
            res = check(page, vp[0], errors)
            page.evaluate("window.scrollTo(0,0)")
            page.wait_for_timeout(50)
            page.screenshot(path=os.path.join(shots, f"_preview_{name}.png"))          # viewport, with the shared top bar
            page.screenshot(path=os.path.join("/tmp", f"_full_{name}.png"), full_page=True)
            print(f"{name} {vp[0]}x{vp[1]}: fit {res['fit0']} -> {res['fit1']} after tap, scrollWidth {res['scrollWidth']}")
            ctx.close()
        browser.close()
    if errors:
        print("JS errors:\n  " + "\n  ".join(errors))
        return 1
    print(f"OK — {target}; screenshots in {shots}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
