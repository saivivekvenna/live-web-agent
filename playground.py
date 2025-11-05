"""Simple playground script that reuses the persistent Notion profile."""

import json
import os
import sys
import textwrap
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from openai import OpenAI
from playwright.sync_api import TimeoutError as PWTimeoutError

from bots._profile_launch import launch_persistent, shutdown as shutdown_persistent

NOTION_URL = "https://www.notion.so/"
PROFILE_DIR = "profiles/notion"
OVERLAY_LOCATORS = [
    ("role[dialog]", lambda page: page.locator("[role='dialog']")),
    ("aria-modal", lambda page: page.locator("[aria-modal='true']")),
    ("role[menu]", lambda page: page.locator("[role='menu']")),
    ("role[listbox]", lambda page: page.locator("[role='listbox']")),
]

NOTION_CONTROL_HINTS = [
    "Add new",
    "Create new",
    "New page",
    "Database",
    "New database",
    "Empty database",
    "Empty",
    "Empty page",
]
SCREENSHOT_PATH = "notion_database_click.png"
client = OpenAI()
DEFAULT_GOAL = "create a database in notion"
MAX_ACTIONS = 12


@dataclass
class ActionRecord:
    label: str
    success: bool
    message: str


@dataclass
class IntegrationConfig:
    provider: str
    action: str
    base_url: str
    profile_dir: str
    label_hints: List[str] = field(default_factory=list)
    extra_prompt: str = ""
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


def _make_label_locators(label: str, hints: Optional[List[str]] = None) -> List[tuple]:
    """Return a prioritized list of locator builders for a given label."""
    label = label.strip()
    if not label:
        return []

    hints = {hint.lower() for hint in (hints or [])}
    prefers_exact = "exact" in hints or len(label.split()) <= 2
    include_dialog = "dialog" in hints or "overlay" in hints or not hints
    include_menu = "menu" in hints or "menuitem" in hints

    def overlay_button(page, label=label, exact=prefers_exact):
        return page.locator("[role='dialog']").get_by_role("button", name=label, exact=exact)

    def overlay_menuitem(page, label=label):
        return page.locator("[role='dialog']").get_by_role("menuitem", name=label, exact=False)

    def overlay_text(page, label=label, exact=prefers_exact):
        return page.locator("[role='dialog']").get_by_text(label, exact=exact)

    def role_button(page, label=label, exact=prefers_exact):
        return page.get_by_role("button", name=label, exact=exact)

    def role_link(page, label=label, exact=prefers_exact):
        return page.get_by_role("link", name=label, exact=exact)

    def role_menuitem(page, label=label):
        return page.get_by_role("menuitem", name=label, exact=False)

    def text_match(page, label=label, exact=prefers_exact):
        return page.get_by_text(label, exact=exact)

    def locator_contains(page, label=label):
        return page.locator(f"text={label}")

    options: List[tuple] = []
    seen: set[str] = set()

    def add(desc: str, builder: Callable) -> None:
        if desc in seen:
            return
        seen.add(desc)
        options.append((desc, builder))

    if include_dialog:
        add(f"dialog role=button[{label}]", overlay_button)
        add(f"dialog role=menuitem[{label}]", overlay_menuitem)
        add(f"dialog text~='{label}'", overlay_text)

    add(f"role=button[{label}]", role_button)
    add(f"role=link[{label}]", role_link)

    if include_menu:
        add(f"role=menuitem[{label}]", role_menuitem)

    add(f"text~='{label}'", text_match)
    add(f"locator text={label}", locator_contains)

    return options


def _attempt_click(page, label: str, hints: Optional[List[str]] = None) -> tuple[bool, str]:
    """
    Attempt to click a label using generated locator strategies.
    Returns (success, message).
    """
    options = _make_label_locators(label, hints)
    if not options:
        return False, "No locator strategies generated."

    last_error = "No matching elements located."
    for desc, builder in options:
        try:
            locator = builder(page)
            if locator.count() == 0:
                last_error = f"{desc} matched 0 elements"
                continue
            target = locator.first
            target.wait_for(state="visible", timeout=10000)
            target.click(timeout=10000)
            return True, f"Clicked via {desc}"
        except Exception as exc:
            last_error = str(exc)
    return False, f"All strategies failed. Last error: {last_error}"


def _format_history(history: List[ActionRecord]) -> str:
    if not history:
        return "No actions taken yet."
    lines = []
    for idx, record in enumerate(history[-8:], start=max(len(history) - 7, 1)):
        outcome = "success" if record.success else "failure"
        lines.append(f"{idx}. {record.label} → {outcome} ({record.message})")
    return "\n".join(lines)


def _plan_next_action(page, goal: str, config: IntegrationConfig, history: List[ActionRecord]) -> dict:
    dom_snapshot = _capture_prioritized_dom(page)
    history_text = _format_history(history)
    hints_text = ", ".join(config.label_hints) if config.label_hints else "None"
    overlay_state = "visible" if _overlay_present(page) else "not visible"
    dashboard_hint = "likely on dashboard" if _looks_like_dashboard(page) else "not clearly on dashboard"

    prompt = textwrap.dedent(
        f"""
        You are operating a browser automation agent.

        Goal: {goal}
        Provider: {config.provider}
        Known helpful controls: {hints_text}
        Overlay/dialog visibility: {overlay_state}
        Dashboard heuristic: {dashboard_hint}

        Recent action history:
        {history_text}

        Examine the DOM snippet and decide the single next best action to advance toward the goal.
        Respond with a JSON object containing:
          - "finish": boolean indicating whether the goal is already satisfied.
          - "reason": short string explaining your decision.
          - "targets": array (max 3) of action plans. Each object must include:
                • "action": currently only "click" is supported.
                • "label_variants": array of 1-3 plausible UI labels to try, ordered by priority.
                • "locator_hints": optional array of hint strings (e.g., "dialog", "menu", "exact") to guide selectors.
                • "notes": optional string giving context for the click.

        If you believe the task is complete, set "finish": true and provide an empty "targets" array.

        DOM SNIPPET (truncated to 30k chars):
        {dom_snapshot}
        """
    ).strip()

    if config.extra_prompt:
        prompt += f"\n\nAdditional context: {config.extra_prompt}"

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You plan browser actions. Always respond with JSON only."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
        )
        raw = response.choices[0].message.content or "{}"
        plan = _extract_json_object(raw)
        if not isinstance(plan, dict):
            raise ValueError("Planner did not return a JSON object.")
        return plan
    except Exception as exc:
        raise RuntimeError(f"Planning LLM failed: {exc}") from exc


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


INTEGRATIONS: Dict[tuple[str, str], IntegrationConfig] = {
    ("notion", "create_database"): IntegrationConfig(
        provider="notion",
        action="create_database",
        base_url=NOTION_URL,
        profile_dir=PROFILE_DIR,
        label_hints=NOTION_CONTROL_HINTS,
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

        history: List[ActionRecord] = []
        for step_idx in range(1, MAX_ACTIONS + 1):
            try:
                plan = _plan_next_action(page, goal, config, history)
            except RuntimeError as exc:
                print(f"❌ Planner failure: {exc}")
                break

            if plan.get("finish"):
                print(f"🎉 Planner marked goal complete: {plan.get('reason', 'done')}")
                break

            targets = plan.get("targets") or []
            if not targets:
                print(f"❌ Planner returned no targets on iteration {step_idx}.")
                break

            print(f"🧭 Iteration {step_idx}: {plan.get('reason', 'No reason provided.')}")

            executed = False
            for target in targets:
                if target.get("action") != "click":
                    print(f"  • Unsupported action '{target.get('action')}', skipping.")
                    continue

                label_variants_raw = target.get("label_variants") or []
                if isinstance(label_variants_raw, str):
                    label_variants = [label_variants_raw]
                else:
                    label_variants = [str(item) for item in label_variants_raw if isinstance(item, str)]
                if not label_variants:
                    print("  • Planner target missing label variants; skipping.")
                    continue

                locator_hints_raw = target.get("locator_hints") or []
                if isinstance(locator_hints_raw, str):
                    locator_hints = [locator_hints_raw]
                else:
                    locator_hints = [str(item) for item in locator_hints_raw if isinstance(item, str)]
                notes = target.get("notes")
                if notes:
                    print(f"    ↳ Planner note: {notes}")

                for label in label_variants:
                    success, message = _attempt_click(page, label, locator_hints)
                    history.append(ActionRecord(label=label, success=success, message=message))
                    outcome_icon = "✅" if success else "⚠️"
                    print(f"    {outcome_icon} Tried '{label}' → {message}")
                    if success:
                        executed = True
                        try:
                            page.wait_for_timeout(800)
                        except Exception:
                            pass
                        break

                if executed:
                    break

            if not executed:
                print("  • All planner suggestions failed; requesting a new plan…")
                continue

        else:
            print("⚠️ Reached maximum iterations without planner finishing the task.")

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
