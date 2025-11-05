"""Simple playground script that reuses the persistent Notion profile."""

import json
import textwrap
from typing import Iterable, List

from openai import OpenAI
from playwright.sync_api import TimeoutError as PWTimeoutError

from bots._profile_launch import launch_persistent, shutdown as shutdown_persistent

NOTION_URL = "https://www.notion.so/"
PROFILE_DIR = "profiles/notion"
DEFAULT_DATABASE_LABELS = [
    ("dialog button[Database]", lambda page: page.locator("[role='dialog']").get_by_role("button", name="Database")),
    ("dialog menuitem[Database]", lambda page: page.locator("[role='dialog']").get_by_role("menuitem", name="Database")),
    ("dialog text=Database", lambda page: page.locator("[role='dialog']").get_by_text("Database", exact=True)),
    ("dialog text contains 'Database'", lambda page: page.locator("[role='dialog']").get_by_text("Database", exact=False)),
    ("link[Database]", lambda page: page.get_by_role("link", name="Database")),
    ("button[Database]", lambda page: page.get_by_role("button", name="Database")),
    ("link[Databases]", lambda page: page.get_by_role("link", name="Databases")),
    ("text=Database", lambda page: page.get_by_text("Database", exact=True)),
    ("text contains 'New database'", lambda page: page.get_by_text("New database")),
]

OVERLAY_LOCATORS = [
    ("role[dialog]", lambda page: page.locator("[role='dialog']")),
    ("aria-modal", lambda page: page.locator("[aria-modal='true']")),
    ("role[menu]", lambda page: page.locator("[role='menu']")),
    ("role[listbox]", lambda page: page.locator("[role='listbox']")),
]

DEFAULT_ADD_NEW_LABELS = [
    ("button[Add new]", lambda page: page.get_by_role("button", name="Add new")),
    ("button[Create new]", lambda page: page.get_by_role("button", name="Create new")),
    ("button[New page]", lambda page: page.get_by_role("button", name="New page")),
    ("link[Add new]", lambda page: page.get_by_role("link", name="Add new")),
    ("link[Create new]", lambda page: page.get_by_role("link", name="Create new")),
    ("text=Add new", lambda page: page.get_by_text("Add new", exact=True)),
    ("text=Create new", lambda page: page.get_by_text("Create new", exact=True)),
    ("text contains 'New page'", lambda page: page.get_by_text("New page")),
]

DEFAULT_NEW_PAGE_LABELS = [
    ("button[Page]", lambda page: page.get_by_role("button", name="Page")),
    ("button[New page]", lambda page: page.get_by_role("button", name="New page")),
    ("link[Page]", lambda page: page.get_by_role("link", name="Page")),
    ("link[New page]", lambda page: page.get_by_role("link", name="New page")),
    ("text=Page", lambda page: page.get_by_text("Page", exact=True)),
    ("text=New page", lambda page: page.get_by_text("New page", exact=True)),
]

DEFAULT_EMPTY_DATABASE_LABELS = [
    ("dialog button[Empty database]", lambda page: page.locator("[role='dialog']").get_by_role("button", name="Empty database")),
    ("dialog menuitem[Empty database]", lambda page: page.locator("[role='dialog']").get_by_role("menuitem", name="Empty database")),
    ("dialog text contains 'Empty database'", lambda page: page.locator("[role='dialog']").get_by_text("Empty database", exact=False)),
    ("button[Empty database]", lambda page: page.get_by_role("button", name="Empty database")),
    ("link[Empty database]", lambda page: page.get_by_role("link", name="Empty database")),
    ("text=Empty database", lambda page: page.get_by_text("Empty database", exact=True)),
    ("text contains 'Empty database'", lambda page: page.get_by_text("Empty database", exact=False)),
    ("role=menuitem[Empty database]", lambda page: page.get_by_role("menuitem", name="Empty database")),
    ("button[Empty]", lambda page: page.get_by_role("button", name="Empty")),
    ("link[Empty]", lambda page: page.get_by_role("link", name="Empty")),
    ("text=Empty", lambda page: page.get_by_text("Empty", exact=True)),
    ("text contains 'Empty'", lambda page: page.get_by_text("Empty", exact=False)),
    ("text contains 'Empty page'", lambda page: page.get_by_text("Empty page")),
]
SCREENSHOT_PATH = "notion_database_click.png"
client = OpenAI()


def _capture_prioritized_dom(page, limit: int = 30000) -> str:
    """
    Return HTML prioritizing overlay content (menus, dialogs) before the base page.
    """
    overlay_html: List[str] = []
    for desc, builder in OVERLAY_LOCATORS:
        try:
            locator = builder(page)
            count = min(locator.count(), 3)
            if count == 0:
                continue
            for idx in range(count):
                try:
                    snippet = locator.nth(idx).inner_html(timeout=2000)
                except Exception as exc:
                    print(f"  • Overlay capture failed for {desc}[{idx}]: {exc}")
                    continue
                if snippet:
                    overlay_html.append(snippet)
        except Exception as exc:
            print(f"  • Overlay locator {desc} failed: {exc}")

    try:
        base_html = page.content()
    except Exception as exc:
        print(f"  • Base DOM capture failed: {exc}")
        base_html = ""

    combined = "\n".join(overlay_html + [base_html] if overlay_html else [base_html])
    return combined[:limit]


def _overlay_present(page) -> bool:
    """
    Detect if any known overlay/dialog elements are currently visible.
    """
    for _, builder in OVERLAY_LOCATORS:
        try:
            if builder(page).count() > 0:
                return True
        except Exception:
            continue
    return False


def _looks_like_dashboard(page) -> bool:
    """
    Heuristic to detect when we are still on the Notion home dashboard.
    """
    if _overlay_present(page):
        return False

    try:
        current_url = page.url.rstrip("/")
        if current_url.lower() == NOTION_URL.rstrip("/").lower():
            return True
    except Exception:
        pass

    try:
        sidebar_add_new = page.get_by_role("button", name="Add new")
        if sidebar_add_new.count() > 0:
            return True
    except Exception:
        pass

    return False


def _extract_json_array(text: str) -> str:
    """
    Pull the first JSON array (or fallback to raw text) from a model response.
    """
    text = text.strip()
    start = text.find("[")
    end = text.rfind("]")
    if start != -1 and end != -1 and end > start:
        snippet = text[start : end + 1]
        return snippet.strip()
    return "[]"


def _suggest_label_candidates(dom_html: str, description: str, defaults: Iterable[str]) -> List[str]:
    """
    Ask an LLM to suggest button/link labels based on the live DOM.
    Returns a short, deduplicated list of candidate strings.
    """
    dom_excerpt = dom_html[:30000] if dom_html else ""
    default_list = ", ".join(f'"{d}"' for d in defaults)
    prompt = textwrap.dedent(
        f"""
        You are analyzing a live HTML snippet. Identify up to five distinct button, link,
        or quick-start chip labels that a user would click to {description}.

        Return your answer as a JSON array of strings, ordered from most to least likely.
        Avoid duplicates and keep each label short (ideally 1-3 words).

        Known reliable defaults you may reuse: [{default_list}]

        HTML SNIPPET (truncated):
        {dom_excerpt}
        """
    ).strip()

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You extract actionable UI labels from HTML and respond with JSON arrays only."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
        )
        raw = response.choices[0].message.content or "[]"
        json_snippet = _extract_json_array(raw)
        labels = json.loads(json_snippet)
        candidates: List[str] = []
        seen: set[str] = set()
        for label in labels:
            if not isinstance(label, str):
                continue
            norm = label.strip()
            key = norm.lower()
            if not norm or len(norm) > 64 or key in seen:
                continue
            seen.add(key)
            candidates.append(norm)
        return candidates[:5]
    except Exception as exc:
        print(f"  • LLM label suggestion failed: {exc}")
        return []


def _make_label_locators(label: str):
    """Return a list of locator builders for a given label."""
    label = label.strip()
    if not label:
        return []
    return [
        (f"role=button[{label}]", lambda page, label=label: page.get_by_role("button", name=label, exact=False)),
        (f"role=link[{label}]", lambda page, label=label: page.get_by_role("link", name=label, exact=False)),
        (f"text~='{label}'", lambda page, label=label: page.get_by_text(label, exact=False)),
    ]


def _build_locator_options(suggestions: List[str], defaults: List[tuple]) -> List[tuple]:
    """Merge LLM suggestions and default fallbacks into an ordered locator list."""
    options: List[tuple] = []
    seen_desc: set[str] = set()

    for label in suggestions:
        for desc, builder in _make_label_locators(label):
            if desc in seen_desc:
                continue
            options.append((desc, builder))
            seen_desc.add(desc)

    for desc, builder in defaults:
        if desc in seen_desc:
            continue
        options.append((desc, builder))
        seen_desc.add(desc)

    return options


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

        def attempt_step(description: str, prompt_desc: str, prompt_defaults: List[str], locator_defaults: List[tuple]) -> bool:
            dom_snapshot = _capture_prioritized_dom(page)
            suggestions = _suggest_label_candidates(dom_snapshot, prompt_desc, prompt_defaults)
            if suggestions:
                print(f"🔍 LLM suggestions for {description}: {suggestions}")
            options = _build_locator_options(suggestions, locator_defaults)
            try:
                click_with_fallbacks(options, description)
                try:
                    page.wait_for_timeout(800)
                except Exception:
                    pass
                return True
            except RuntimeError as exc:
                print(f"  • Step '{description}' failed: {exc}")
                return False

        is_dashboard = _looks_like_dashboard(page)
        add_new_clicked = False
        database_clicked = False
        if is_dashboard:
            print("🏠 Detected Notion home dashboard. Following Add new → Empty database flow.")

            add_new_clicked = attempt_step(
                "Opened Add new menu",
                "open the add new menu",
                ["Add new", "Create new"],
                DEFAULT_ADD_NEW_LABELS,
            )

            if not add_new_clicked:
                print("  • Add new menu did not open; falling back to Database flow.")
        else:
            print("📝 Detected page context. Following Database → Empty database flow.")

        empty_clicked = False

        if not is_dashboard or not add_new_clicked:
            database_clicked = attempt_step(
                "Clicked Database control",
                "begin creating a new database",
                ["Database", "New database"],
                DEFAULT_DATABASE_LABELS,
            )
            if not database_clicked and is_dashboard:
                print("  • Database control not found after fallback; retrying Add new flow.")
                add_new_clicked = attempt_step(
                    "Opened Add new menu",
                    "open the add new menu",
                    ["Add new", "Create new"],
                    DEFAULT_ADD_NEW_LABELS,
                )

        empty_clicked = attempt_step(
            "Clicked Empty database template",
            "choose an empty database template for the new page",
            ["Empty database", "Empty", "Empty page"],
            DEFAULT_EMPTY_DATABASE_LABELS,
        )

        if not empty_clicked:
            database_clicked = attempt_step(
                "Clicked Database control",
                "begin creating a new database",
                ["Database", "New database"],
                DEFAULT_DATABASE_LABELS,
            )
            if database_clicked:
                empty_clicked = attempt_step(
                    "Clicked Empty database template",
                    "choose an empty database template for the new page",
                    ["Empty database", "Empty", "Empty page"],
                    DEFAULT_EMPTY_DATABASE_LABELS,
                )

        if not empty_clicked:
            raise RuntimeError("Unable to click the Empty database template after multiple attempts.")

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
