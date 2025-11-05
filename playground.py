"""Simple playground script that reuses the persistent Notion profile."""

from playwright.sync_api import TimeoutError as PWTimeoutError

from bots._profile_launch import launch_persistent, shutdown as shutdown_persistent

NOTION_URL = "https://www.notion.so/"
PROFILE_DIR = "profiles/notion"
DATABASE_LABELS = [
    ("link[Database]", lambda page: page.get_by_role("link", name="Database")),
    ("button[Database]", lambda page: page.get_by_role("button", name="Database")),
    ("link[Databases]", lambda page: page.get_by_role("link", name="Databases")),
    ("text=Database", lambda page: page.get_by_text("Database", exact=True)),
    ("text contains 'New database'", lambda page: page.get_by_text("New database")),
]

EMPTY_TEMPLATE_LABELS = [
    ("button[Empty]", lambda page: page.get_by_role("button", name="Empty")),
    ("link[Empty]", lambda page: page.get_by_role("link", name="Empty")),
    ("text=Empty", lambda page: page.get_by_text("Empty", exact=True)),
    ("text contains 'Empty page'", lambda page: page.get_by_text("Empty page")),
]
SCREENSHOT_PATH = "notion_database_click.png"


def main() -> None:
    playwright = None
    context = None
    page = None
    try:
        playwright, context, page = launch_persistent(NOTION_URL, PROFILE_DIR, headless=False)

        try:
            page.wait_for_load_state("networkidle", timeout=10000)
        except PWTimeoutError:
            pass

        def click_with_fallbacks(options, description: str) -> None:
            for label, builder in options:
                try:
                    locator = builder(page)
                    if locator.count() == 0:
                        print(f"  • No matches for {label}; trying next…")
                        continue
                    target = locator.first
                    target.wait_for(state="visible", timeout=10000)
                    target.click(timeout=10000)
                    print(f"✅ {description} via selector: {label}")
                    return
                except Exception as exc:
                    print(f"  • Selector {label} failed: {exc}")
            raise RuntimeError(f"Could not locate the {description.lower()} with known selectors.")

        print("🔘 Attempting to click the Database control…")
        click_with_fallbacks(DATABASE_LABELS, "Clicked Database control")

        try:
            page.wait_for_timeout(1000)
        except Exception:
            pass

        print("🆕 Attempting to click the Empty template…")
        click_with_fallbacks(EMPTY_TEMPLATE_LABELS, "Clicked Empty template")

        try:
            page.wait_for_load_state("networkidle", timeout=10000)
        except PWTimeoutError:
            pass

        page.screenshot(path=SCREENSHOT_PATH, full_page=True)
        print(f"📸 Screenshot saved: {SCREENSHOT_PATH}")

        input("✅ Done. Inspect the page, then press Enter to close the browser…")
    finally:
        shutdown_persistent(playwright, context)


if __name__ == "__main__":
    main()
