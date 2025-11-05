"""Simple playground script that reuses the persistent Notion profile."""

import json
import os
import sys
import textwrap
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Optional

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
DEFAULT_GOAL = "create a database in notion"

LocatorOption = tuple[str, Callable[[object], object]]


@dataclass
class StepConfig:
    name: str
    prompt_desc: str
    prompt_defaults: List[str]
    locator_defaults: List[LocatorOption]
    required: bool = True
    condition: Optional[Callable[[object], bool]] = None
    retries: int = 1
    post_wait_ms: int = 800


@dataclass
class IntegrationConfig:
    provider: str
    action: str
    base_url: str
    profile_dir: str
    workflow: List[StepConfig]
    launch_kwargs: dict = field(default_factory=dict)


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


def _extract_json_object(text: str) -> dict:
    """
    Pull the first JSON object from a model response.
    """
    text = text.strip()
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        snippet = text[start : end + 1]
        try:
            return json.loads(snippet)
        except Exception:
            return {}
    return {}


def _suggest_label_candidates(dom_html: str, description: str, defaults: Iterable[str], goal_hint: Optional[str] = None) -> List[str]:
    """
    Ask an LLM to suggest button/link labels based on the live DOM.
    Returns a short, deduplicated list of candidate strings.
    """
    dom_excerpt = dom_html[:30000] if dom_html else ""
    default_list = ", ".join(f'"{d}"' for d in defaults)
    goal_context = f"User goal/context: {goal_hint}\n\n" if goal_hint else ""
    prompt = textwrap.dedent(
        f"""
        You are analyzing a live HTML snippet. Identify up to five distinct button, link,
        or quick-start chip labels that a user would click to {description}.

        Return your answer as a JSON array of strings, ordered from most to least likely.
        Avoid duplicates and keep each label short (ideally 1-3 words).

        Known reliable defaults you may reuse: [{default_list}]

        {goal_context}HTML SNIPPET (truncated):
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


def click_with_fallbacks(page, options: List[tuple], description: str) -> None:
    """
    Attempt to click using a list of locator builders. Raises RuntimeError when all fail.
    """
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


def _run_step(page, step: StepConfig, goal: str) -> bool:
    """
    Execute a single workflow step, returning True on success. Optional steps
    should be marked with ``required=False`` in the configuration and callers
    can inspect the return value to decide how to proceed.
    """
    if step.condition and not step.condition(page):
        print(f"⏭️  Skipping {step.name} (condition not met).")
        return True

    for attempt in range(1, step.retries + 1):
        dom_snapshot = _capture_prioritized_dom(page)
        suggestions = _suggest_label_candidates(dom_snapshot, step.prompt_desc, step.prompt_defaults, goal)
        if suggestions:
            print(f"🔍 LLM suggestions for {step.name} (attempt {attempt}): {suggestions}")
        else:
            print(f"🔍 No LLM suggestions for {step.name}; falling back to defaults.")
        options = _build_locator_options(suggestions, step.locator_defaults)
        try:
            click_with_fallbacks(page, options, step.name)
            if step.post_wait_ms:
                try:
                    page.wait_for_timeout(step.post_wait_ms)
                except Exception:
                    pass
            return True
        except RuntimeError as exc:
            print(f"  • Step '{step.name}' attempt {attempt} failed: {exc}")
    return False


def _resolve_goal() -> str:
    """
    Resolve the user's desired goal from CLI args, environment, or stdin.
    """
    if len(sys.argv) > 1:
        goal = " ".join(sys.argv[1:]).strip()
        if goal:
            return goal

    env_goal = os.environ.get("PLAYGROUND_GOAL")
    if env_goal:
        env_goal = env_goal.strip()
        if env_goal:
            return env_goal

    try:
        user_goal = input(f"Enter task goal (default '{DEFAULT_GOAL}'): ").strip()
        if user_goal:
            return user_goal
    except EOFError:
        pass

    return DEFAULT_GOAL


INTENT_KEYWORD_MAP = [
    {
        "provider": "notion",
        "action": "create_database",
        "provider_terms": ("notion",),
        "action_terms": ("create database", "create a database", "database", "databases", "db"),
    },
    {
        "provider": "linear",
        "action": "create_project",
        "provider_terms": ("linear",),
        "action_terms": ("project", "issue", "ticket"),
    },
]


def _classify_intent_llm(goal: str) -> Optional[tuple[str, str]]:
    """
    Fallback classifier that asks the LLM to provide provider/action JSON.
    """
    prompt = textwrap.dedent(
        f"""
        Analyze the user's goal: "{goal}".

        Respond with a JSON object using the keys "provider" and "action".
        Keep providers in lowercase (e.g., "notion", "linear").
        Use snake_case verbs for the action (e.g., "create_database").
        If the goal is unclear, respond with {{"provider": null, "action": null}}.
        """
    ).strip()

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Classify the request and respond with JSON only."},
                {"role": "user", "content": prompt},
            ],
            temperature=0,
        )
        raw = response.choices[0].message.content or "{}"
        data = _extract_json_object(raw)
        provider = (data.get("provider") or "").strip().lower()
        action = (data.get("action") or "").strip().lower()
        if provider and action:
            return provider, action
    except Exception as exc:
        print(f"  • Intent classification via LLM failed: {exc}")
    return None


def parse_intent(goal: str) -> tuple[str, str]:
    """
    Map a free-form goal description to (provider, action).
    """
    normalized = goal.lower()
    for entry in INTENT_KEYWORD_MAP:
        if any(term in normalized for term in entry["provider_terms"]) and any(
            term in normalized for term in entry["action_terms"]
        ):
            return entry["provider"], entry["action"]

    llm_guess = _classify_intent_llm(goal)
    if llm_guess:
        return llm_guess

    raise ValueError(f"Unable to determine provider/action for goal: {goal!r}")


NOTION_CREATE_DATABASE_WORKFLOW: List[StepConfig] = [
    StepConfig(
        name="Opened Add new menu",
        prompt_desc="open the add new menu",
        prompt_defaults=["Add new", "Create new"],
        locator_defaults=DEFAULT_ADD_NEW_LABELS,
        required=False,
        condition=_looks_like_dashboard,
        retries=2,
        post_wait_ms=600,
    ),
    StepConfig(
        name="Clicked Database control",
        prompt_desc="begin creating a new database",
        prompt_defaults=["Database", "New database"],
        locator_defaults=DEFAULT_DATABASE_LABELS,
        retries=2,
        post_wait_ms=600,
    ),
    StepConfig(
        name="Clicked Empty database template",
        prompt_desc="choose an empty database template for the new page",
        prompt_defaults=["Empty database", "Empty", "Empty page"],
        locator_defaults=DEFAULT_EMPTY_DATABASE_LABELS,
        retries=3,
        post_wait_ms=800,
    ),
]

INTEGRATIONS: Dict[tuple[str, str], IntegrationConfig] = {
    ("notion", "create_database"): IntegrationConfig(
        provider="notion",
        action="create_database",
        base_url=NOTION_URL,
        profile_dir=PROFILE_DIR,
        workflow=NOTION_CREATE_DATABASE_WORKFLOW,
    ),
}


def main() -> None:
    goal = _resolve_goal()
    print(f"🎯 Goal: {goal}")

    try:
        provider, action = parse_intent(goal)
    except ValueError as exc:
        print(f"❌ {exc}")
        return

    print(f"🤖 Resolved intent → provider='{provider}', action='{action}'")
    config = INTEGRATIONS.get((provider, action))
    if not config:
        print(f"❌ No integration configured for provider '{provider}' and action '{action}'.")
        return

    headless = config.launch_kwargs.get("headless", False)

    playwright = None
    context = None
    page = None
    try:
        playwright, context, page = launch_persistent(config.base_url, config.profile_dir, headless=headless)

        try:
            page.wait_for_load_state("networkidle", timeout=10000)
        except PWTimeoutError:
            pass

        for step in config.workflow:
            success = _run_step(page, step, goal)
            if not success:
                if step.required:
                    raise RuntimeError(f"Required step '{step.name}' failed after {step.retries} attempts.")
                print(f"⚠️ Optional step '{step.name}' failed; continuing.")

        try:
            page.wait_for_load_state("networkidle", timeout=5000)
        except PWTimeoutError:
            pass

        page.screenshot(path=SCREENSHOT_PATH, full_page=True)
        print(f"📸 Screenshot saved: {SCREENSHOT_PATH}")

        input("✅ Done. Inspect the page, then press Enter to close the browser…")
    finally:
        shutdown_persistent(playwright, context)


if __name__ == "__main__":
    main()
