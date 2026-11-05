"""End-to-end check of a deployed Vercel URL through the real UI (dev tool, not deployed).

  vercel env run -- python -m scripts.e2e_deployed https://<deployment-url> preview|production

Needs VERCEL_OIDC_TOKEN in the environment (vercel env run provides it) to pass Deployment Protection; the token is
sent only as a request header and never printed. Signs up a throwaway user (this writes a real row to Supabase),
waits for the API container to scale down, then scores the distressed preset twice (cold, then warm) and opens
History, which reads the logged prediction back from Supabase through the API.
Writes docs/verification/vercel_<label>.json.
"""
import json
import os
import re
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from playwright.sync_api import sync_playwright

IDLE_SECONDS = 75  # preview containers scale down after 30 s idle, production after 5 min (Vercel docs)


def latency(page) -> tuple[float, str]:
    text = page.get_by_text(re.compile(r"API responded in")).last.inner_text()
    secs = float(re.search(r"API responded in ([\d.]+) s", text).group(1))
    return secs, ("cache" if "served from cache" in text else "model")


def main(url: str, label: str, idle: int) -> None:
    url = url.rstrip("/")
    token = os.environ.get("VERCEL_OIDC_TOKEN")
    headers = {"x-vercel-trusted-oidc-idp-token": token} if token else {}
    out = Path(__file__).resolve().parent.parent / "docs" / "verification"
    result = {"url": url, "target": label, "checked_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}

    with sync_playwright() as p:
        browser = p.chromium.launch()
        ctx = browser.new_context(viewport={"width": 1440, "height": 1700}, extra_http_headers=headers)
        page = ctx.new_page()

        health = ctx.request.get(f"{url}/health")
        result["health"] = health.json()
        meta = ctx.request.get(f"{url}/meta")
        result["api_publicly_reachable"] = "application/json" in meta.headers.get("content-type", "")

        start = time.perf_counter()
        page.goto(url)
        page.get_by_text("No score yet").wait_for(timeout=120000)  # page load also wakes the API via /meta
        result["page_ready_seconds"] = round(time.perf_counter() - start, 2)

        email = f"e2e-{uuid.uuid4().hex[:8]}@example.com"
        sidebar = page.locator('[data-testid="stSidebar"]')
        sidebar.get_by_role("tab", name="Create account").click()
        sidebar.locator('input[aria-label="Email"]').last.fill(email)
        sidebar.locator('input[type="password"]').last.fill(uuid.uuid4().hex)
        sidebar.get_by_role("button", name="Create account").click()
        sidebar.get_by_text("Signed in as").wait_for(timeout=60000)
        result["signup"] = "ok"
        result["test_user_email"] = email

        page.get_by_role("button", name="Distressed firm").click()
        time.sleep(idle)  # let the API container scale down so the next call is a cold start
        result["idle_seconds_before_first_score"] = idle
        page.get_by_role("button", name="Score company").click()
        page.get_by_text("Path to safety").first.wait_for(timeout=180000)
        result["first_score_after_idle_seconds"], result["first_score_source"] = latency(page)
        result["grade"] = page.locator(".cr-badge__text").first.inner_text()

        before = page.get_by_text(re.compile(r"API responded in")).last.inner_text()
        page.get_by_role("button", name="Score company").click()
        page.wait_for_function(  # wait until the caption is replaced by the second call's timing
            "t => [...document.querySelectorAll('p')].some(p => p.innerText.includes('API responded in') "
            "&& p.innerText !== t)", arg=before, timeout=60000)
        result["second_score_seconds"], result["second_score_source"] = latency(page)
        page.screenshot(path=str(out / f"vercel_{label}_score.png"))

        page.locator('[data-testid="stSidebarNav"]').get_by_text("History", exact=True).click()
        page.get_by_text("Companies scored").wait_for(timeout=60000)
        result["history_rows_read_back_from_supabase"] = int(
            page.locator(".cr-card__value").first.inner_text().replace(",", ""))
        page.screenshot(path=str(out / f"vercel_{label}_history.png"))
        browser.close()

    (out / f"vercel_{label}.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else IDLE_SECONDS)
