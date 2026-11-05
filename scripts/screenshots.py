"""Desktop + mobile screenshots of every page into docs/verification/screenshots/ (dev tool, not deployed).

Needs: pip install playwright && python -m playwright install chromium
Run the demo stack first:  python -m scripts.demo_api  and  BACKEND_URL=http://127.0.0.1:8000 streamlit run streamlit_app.py
Then:                      python -m scripts.screenshots
It signs up a throwaway user on the in-memory demo API and clicks through the real UI.
"""
import uuid
from pathlib import Path

from playwright.sync_api import sync_playwright

URL = "http://127.0.0.1:8501"
OUT = Path(__file__).resolve().parent.parent / "docs" / "verification" / "screenshots"
DESKTOP = {"width": 1440, "height": 1700}
MOBILE = {"width": 390, "height": 2600}  # tall viewport: Streamlit scrolls inside the page, not the window


def shoot(page, name: str) -> None:
    page.wait_for_timeout(1200)  # let Plotly finish animating
    page.screenshot(path=str(OUT / f"{name}_desktop.png"))
    page.set_viewport_size(MOBILE)
    page.wait_for_timeout(1500)
    page.screenshot(path=str(OUT / f"{name}_mobile.png"))
    page.set_viewport_size(DESKTOP)
    page.wait_for_timeout(1000)
    expand = page.locator('[data-testid="stExpandSidebarButton"]')  # sidebar may collapse at phone width
    if expand.count() and expand.is_visible():
        expand.click()
        page.wait_for_timeout(500)


def go(page, title: str, wait_for: str) -> None:
    page.locator('[data-testid="stSidebarNav"]').get_by_text(title, exact=True).click()
    page.get_by_text(wait_for).first.wait_for(timeout=60000)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport=DESKTOP, device_scale_factor=1)
        page.goto(URL)
        page.get_by_text("No score yet").wait_for(timeout=60000)

        sidebar = page.locator('[data-testid="stSidebar"]')
        sidebar.get_by_role("tab", name="Create account").click()
        sidebar.locator('input[aria-label="Email"]').last.fill(f"analyst-{uuid.uuid4().hex[:6]}@example.com")  # unique: demo DB keeps users while running
        sidebar.locator('input[type="password"]').last.fill("demo-password-123")
        sidebar.get_by_role("button", name="Create account").click()
        sidebar.get_by_text("Signed in as").wait_for(timeout=30000)

        page.get_by_role("button", name="Distressed firm").click()
        page.get_by_role("button", name="Score company").click()
        page.get_by_text("Path to safety").first.wait_for(timeout=60000)
        shoot(page, "score")

        go(page, "Portfolio upload", "Upload")
        page.get_by_role("button", name="Use the sample portfolio").click()
        page.get_by_text("Top 10 riskiest firms").wait_for(timeout=60000)
        shoot(page, "portfolio")

        go(page, "Stress test", "Run stress test")
        page.get_by_role("button", name="Run stress test").click()
        page.get_by_text("After shocks").wait_for(timeout=60000)
        shoot(page, "stress")

        go(page, "Model card", "Known limitations")
        shoot(page, "model_card")

        go(page, "History", "History")
        page.get_by_text("Companies scored").wait_for(timeout=60000)
        shoot(page, "history")
        browser.close()
    print(f"Saved screenshots to {OUT}")


if __name__ == "__main__":
    main()
