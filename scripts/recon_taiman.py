"""Reconnaissance script: capture Taiman Party DOM + network traffic.

Goal: figure out the real selectors and (if any) the XHR endpoint that
gets called when the user clicks リロード, so we can rebuild the fetcher
and parser against actual upstream shape instead of the synthetic fixture.

Outputs everything under ``scripts/out/recon/``:
- ``initial.html``        — DOM right after navigation (pre-interaction)
- ``after_reload.html``   — DOM after picking GL + clicking リロード
- ``screenshot.png``      — full-page screenshot post-reload (sanity check)
- ``network.jsonl``       — every request/response (URL, status, size, ct)
- ``xhr_bodies/*``        — bodies of XHR/Fetch responses (truncated)
- ``selectors.json``      — discovered control selectors + candidates
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from playwright.sync_api import Page, Response, sync_playwright

OUT = Path(__file__).resolve().parent / "out" / "recon"
OUT.mkdir(parents=True, exist_ok=True)
XHR_DIR = OUT / "xhr_bodies"
XHR_DIR.mkdir(exist_ok=True)

URL = "https://pokemongo-get.com/taimanparty/"


def dump_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def safe_name(url: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", url)[-120:]


def main() -> None:
    network: list[dict[str, Any]] = []

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        ctx = browser.new_context(
            locale="ja-JP",
            user_agent=(
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_0) "
                "AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15"
            ),
        )
        page: Page = ctx.new_page()

        def on_response(resp: Response) -> None:
            entry = {
                "url": resp.url,
                "status": resp.status,
                "method": resp.request.method,
                "resource_type": resp.request.resource_type,
                "content_type": resp.headers.get("content-type", ""),
            }
            network.append(entry)
            if resp.request.resource_type in {"xhr", "fetch"}:
                try:
                    body = resp.body()
                    # Save full body for our two endpoints of interest; small cap elsewhere.
                    if "pokemongo-get.com" in resp.url:
                        sample = body
                    else:
                        sample = body[:50_000]
                    (XHR_DIR / f"{safe_name(resp.url)}.bin").write_bytes(sample)
                    entry["body_bytes"] = len(body)
                    if entry["method"] in {"POST", "PUT"}:
                        try:
                            entry["request_post_data"] = resp.request.post_data
                        except Exception:  # noqa: BLE001
                            pass
                except Exception as exc:  # noqa: BLE001
                    entry["body_error"] = str(exc)

        page.on("response", on_response)

        print(f"[recon] navigating {URL}")
        page.goto(URL, wait_until="networkidle", timeout=60_000)
        dump_text(OUT / "initial.html", page.content())

        print("[recon] capturing initial DOM controls…")
        # Heuristic catalog of candidate controls — record what we find.
        catalog: dict[str, Any] = {
            "title": page.title(),
            "all_selects": page.eval_on_selector_all(
                "select",
                "els => els.map((e, i) => ({i, name: e.name, id: e.id, "
                "outerHTML: e.outerHTML.slice(0, 400), options: "
                "[...e.options].map(o => ({value: o.value, text: o.textContent.trim()}))}))",
            ),
            "buttons": page.eval_on_selector_all(
                "button, input[type=button], input[type=submit], a.btn",
                "els => els.map((e, i) => ({i, tag: e.tagName, text: "
                "(e.innerText || e.value || '').trim().slice(0,80), "
                "id: e.id, classes: e.className}))",
            ),
            "checkboxes": page.eval_on_selector_all(
                "input[type=checkbox]",
                "els => els.map((e, i) => ({i, name: e.name, value: e.value, "
                "id: e.id, label: (e.closest('label')?.innerText || '').trim().slice(0,40)}))",
            ),
        }
        dump_text(OUT / "selectors_initial.json", json.dumps(catalog, ensure_ascii=False, indent=2))

        # Try to drive the UI: pick リーグ → Great League, click リロード.
        # We try several strategies because we don't yet know the real DOM.
        try:
            print("[recon] attempting league selection (Great League)…")
            # Strategy 1: a <select> with a Great League option.
            picked = False
            for sel in page.locator("select").all():
                opts = sel.locator("option").all_text_contents()
                gl_idx = next(
                    (i for i, t in enumerate(opts) if "スーパー" in t or "Great" in t or "great" in t.lower()),
                    None,
                )
                if gl_idx is not None:
                    sel.select_option(index=gl_idx)
                    picked = True
                    print(f"[recon]   picked select option idx={gl_idx} text={opts[gl_idx]!r}")
                    break
            if not picked:
                print("[recon]   no <select> with Great League option — trying button text")
                btns = page.locator(
                    "button:has-text('スーパー'), button:has-text('Great'), "
                    "a:has-text('スーパー'), a:has-text('Great')"
                )
                if btns.count():
                    btns.first.click()
                    picked = True
            print(f"[recon] league picked? {picked}")
        except Exception as exc:  # noqa: BLE001
            print(f"[recon]   league pick failed: {exc}")

        # Click rank cells to see how brackets are passed.
        # Tables 1 & 2 in selectors_initial show 1-12 and 13-24 — click 20 (upper-ish).
        try:
            print("[recon] clicking rank cell 20 (upper bracket)…")
            rank_cell = page.locator("table.table-party-type td", has_text="20")
            if rank_cell.count():
                rank_cell.first.click()
                print(f"[recon]   clicked rank 20 (matches={rank_cell.count()})")
            else:
                print("[recon]   rank 20 cell not found")
        except Exception as exc:  # noqa: BLE001
            print(f"[recon]   rank click failed: {exc}")

        # Click リロード if present
        try:
            print("[recon] clicking リロード if available…")
            reload_btn = page.get_by_role("button", name=re.compile("リロード|reload", re.I))
            if reload_btn.count():
                reload_btn.first.click()
                print("[recon]   clicked リロード")
            else:
                alt = page.locator("button:has-text('リロード'), a:has-text('リロード')")
                if alt.count():
                    alt.first.click()
                    print("[recon]   clicked リロード via locator")
                else:
                    print("[recon]   no リロード button found")
        except Exception as exc:  # noqa: BLE001
            print(f"[recon]   リロード click failed: {exc}")

        # Give the SPA a moment to fetch + render.
        try:
            page.wait_for_load_state("networkidle", timeout=20_000)
        except Exception as exc:  # noqa: BLE001
            print(f"[recon]   networkidle wait timed out: {exc}")

        page.wait_for_timeout(2000)

        dump_text(OUT / "after_reload.html", page.content())
        page.screenshot(path=str(OUT / "screenshot.png"), full_page=True)

        # Snapshot tables to ease parser design.
        tables = page.eval_on_selector_all(
            "table",
            "els => els.map((e, i) => ({i, id: e.id, classes: e.className, "
            "header: [...e.querySelectorAll('thead th, tr th')].slice(0,8).map(h=>h.innerText.trim()), "
            "rowCount: e.querySelectorAll('tbody tr').length, "
            "sample: e.querySelector('tbody tr')?.outerHTML?.slice(0,500)}))",
        )
        dump_text(OUT / "tables.json", json.dumps(tables, ensure_ascii=False, indent=2))

        dump_text(OUT / "network.jsonl", "\n".join(json.dumps(e, ensure_ascii=False) for e in network))

        print(f"[recon] done. {len(network)} network entries, {len(tables)} tables.")
        browser.close()


if __name__ == "__main__":
    main()
