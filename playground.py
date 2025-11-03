from playwright.sync_api import sync_playwright, TimeoutError as PWTimeoutError
import time

OUTPUT_HTML = "notion_home_dom.html"

with sync_playwright() as p:
    browser = p.chromium.launch(headless=False)  # show the browser
    page = browser.new_page()

    print("🌍 Navigating to Notion…")
    page.goto("https://www.notion.so/", wait_until="load", timeout=30000)

    # Let things settle (fonts/animations/cookie banners)
    try:
        page.wait_for_load_state("networkidle", timeout=10000)
    except PWTimeoutError:
        pass
    time.sleep(1.5)

    # Optional: try to accept cookie banner if present (best-effort)
    try:
        # Common cookie buttons on Notion lander
        selectors = [
            "button:has-text('Accept')",
            "button:has-text('I agree')",
            "button:has-text('Got it')",
            "[aria-label='Accept cookies']",
        ]
        for sel in selectors:
            if page.locator(sel).first.is_visible():
                page.click(sel, timeout=2000)
                time.sleep(0.5)
                break
    except Exception:
        pass

    # One more quick settle
    try:
        page.wait_for_load_state("networkidle", timeout=5000)
    except PWTimeoutError:
        pass
    time.sleep(0.8)

    # Grab DOM
    dom = page.content()
    print("✅ Page title:", page.title())
    print(f"🧾 DOM length: {len(dom):,} chars")
    print("🔎 First 15000 chars:\n", dom[:15000])

    # Save full DOM to file
    with open(OUTPUT_HTML, "w", encoding="utf-8") as f:
        f.write(dom)
    print(f"💾 Full DOM saved to {OUTPUT_HTML}")

    # Screenshot after content settles
    page.screenshot(path="notion_home.png")
    print("📸 Screenshot saved: notion_home.png")

    # Keep browser open for inspection if you want
    # browser.close()
