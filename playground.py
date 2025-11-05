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
    ("link[Database]", lambda page: page.get_by_role("link", name="Database")),
    ("button[Database]", lambda page: page.get_by_role("button", name="Database")),
    ("link[Databases]", lambda page: page.get_by_role("link", name="Databases")),
    ("text=Database", lambda page: page.get_by_text("Database", exact=True)),
    ("text contains 'New database'", lambda page: page.get_by_text("New database")),
]

DEFAULT_CREATE_NEW_LABELS = [
    ("button[Create new]", lambda page: page.get_by_role("button", name="Create new")),
    ("button[New page]", lambda page: page.get_by_role("button", name="New page")),
    ("link[Create new]", lambda page: page.get_by_role("link", name="Create new")),
    ("text=Create new", lambda page: page.get_by_text("Create new", exact=True)),
    ("text contains 'New page'", lambda page: page.get_by_text("New page")),
]

DEFAULT_EMPTY_TEMPLATE_LABELS = [
    ("button[Empty]", lambda page: page.get_by_role("button", name="Empty")),
    ("link[Empty]", lambda page: page.get_by_role("link", name="Empty")),
    ("text=Empty", lambda page: page.get_by_text("Empty", exact=True)),
    ("text contains 'Empty page'", lambda page: page.get_by_text("Empty page")),
]
SCREENSHOT_PATH = "notion_database_click.png"
client = OpenAI()


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
    dom_excerpt = dom_html[:15000] if dom_html else ""
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

        dom_snapshot = page.content()
        create_new_suggestions = _suggest_label_candidates(
            dom_snapshot, "open the create new menu", ["Create new", "New page"]
        )
        if create_new_suggestions:
            print(f"🔍 LLM create-new suggestions: {create_new_suggestions}")
        create_new_options = _build_locator_options(create_new_suggestions, DEFAULT_CREATE_NEW_LABELS)

        print("➕ Attempting to open the Create new menu…")
        click_with_fallbacks(create_new_options, "Opened Create new menu")

        try:
            page.wait_for_timeout(1000)
        except Exception:
            pass

        dom_snapshot = page.content()
        database_suggestions = _suggest_label_candidates(
            dom_snapshot, "begin creating a new database", ["Database", "New database"]
        )
        if database_suggestions:
            print(f"🔍 LLM database suggestions: {database_suggestions}")
        database_options = _build_locator_options(database_suggestions, DEFAULT_DATABASE_LABELS)

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
        click_with_fallbacks(database_options, "Clicked Database control")

        try:
            page.wait_for_timeout(1000)
        except Exception:
            pass

        dom_snapshot = page.content()
        empty_suggestions = _suggest_label_candidates(
            dom_snapshot, "choose an empty template for the new database", ["Empty", "Empty page"]
        )
        if empty_suggestions:
            print(f"🔍 LLM empty-template suggestions: {empty_suggestions}")
        empty_options = _build_locator_options(empty_suggestions, DEFAULT_EMPTY_TEMPLATE_LABELS)

        print("🆕 Attempting to click the Empty template…")
        click_with_fallbacks(empty_options, "Clicked Empty template")

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
