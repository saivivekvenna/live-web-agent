# executor.py
import json
import time
import re
from pathlib import Path
from typing import List, Dict, Any
from playwright.sync_api import sync_playwright
from planner import low_level_plan, evaluate_goal_completion

def execute_action(page, action_json):
    """Executes a low-level JSON action on a Playwright page."""
    action = json.loads(action_json)
    a = action.get("action")
    sel = action.get("selector")
    val = action.get("value")
    key = action.get("key")
    timeout = action.get("timeout", 8000)

    print(f"🔹 Executing: {a} -> {sel or val or key}")

    try:
        if a == "navigate":
            page.goto(action["url"], wait_until="load")
        elif a == "click":
            page.click(sel, timeout=timeout)
        elif a == "type":
            page.fill(sel, val or "")
        elif a == "press":
            page.press(sel, key or "")
        elif a == "hover":
            page.hover(sel, timeout=timeout)
        elif a == "check":
            page.check(sel, timeout=timeout)
        elif a == "uncheck":
            page.uncheck(sel, timeout=timeout)
        elif a == "upload":
            page.set_input_files(sel, val)
        elif a == "scroll":
            direction = (val or "down").lower()
            if direction == "up":
                page.evaluate("window.scrollBy(0, -window.innerHeight)")
            else:
                page.evaluate("window.scrollBy(0, window.innerHeight)")
        elif a == "select":
            page.select_option(sel, val)
        elif a == "focus":
            page.focus(sel, timeout=timeout)
        elif a == "clear":
            page.fill(sel, "")
        elif a == "paste":
            page.fill(sel, val or "")
        elif a == "wait":
            wait_for = action.get("wait_for")
            if wait_for:
                page.wait_for_selector(wait_for, timeout=timeout)
            else:
                time.sleep(action.get("duration", timeout / 1000))
        elif a == "assert":
            page.wait_for_selector(action.get("assert_selector"), timeout=timeout)
        elif a == "user_prompt":
            prompt_message = action.get("message") or "Human intervention required. Complete the necessary steps and press Enter."
            print(f"🛑 Human action needed: {prompt_message}")
            input("👉 Press Enter once you've finished the requested action...")
            return True
        elif a == "replan":
            print("⚠️ Replanning required.")
            return False

        if action.get("expect_navigation"):
            page.wait_for_load_state("networkidle")

        return True
    except Exception as e:
        print(f"❌ Error executing {a}: {e}")
        return False


def collect_page_context(page, text_limit: int = 8000, max_clickables: int = 40) -> Dict[str, Any]:
    """
    Gather multiple perspectives of the current page to aid reasoning.
    """
    context: Dict[str, Any] = {
        "url": "",
        "title": "",
        "dom": "",
        "rendered_text": "",
        "actionables": [],
        "accessibility": {},
        "meta": {}
    }

    try:
        context["url"] = page.url
    except Exception:
        pass

    try:
        context["title"] = page.title()
    except Exception:
        pass

    try:
        context["dom"] = page.content()
    except Exception:
        pass

    try:
        rendered_text = page.evaluate(
            "(limit) => (document.body?.innerText || '').slice(0, limit)",
            text_limit,
        )
        context["rendered_text"] = rendered_text or ""
    except Exception:
        pass

    try:
        locator = page.locator("button, [role='button'], a, input, select, textarea")
        count = locator.count()
        items: List[Dict[str, Any]] = []
        for i in range(min(count, max_clickables)):
            el = locator.nth(i)
            try:
                item = {
                    "tag": el.evaluate("el => el.tagName") or "",
                    "role": el.get_attribute("role"),
                    "name": "",
                    "text": "",
                    "aria_label": el.get_attribute("aria-label"),
                    "href": el.get_attribute("href"),
                    "data_testid": el.get_attribute("data-testid"),
                    "placeholder": el.get_attribute("placeholder"),
                    "type": el.get_attribute("type"),
                    "id": el.get_attribute("id"),
                    "enabled": None,
                    "visible": None,
                }
                try:
                    item["text"] = (el.inner_text(timeout=500) or "").strip()
                except Exception:
                    pass
                try:
                    item["name"] = el.get_attribute("name")
                except Exception:
                    pass
                try:
                    item["enabled"] = el.is_enabled()
                except Exception:
                    pass
                try:
                    item["visible"] = el.is_visible()
                except Exception:
                    pass
                items.append(item)
            except Exception:
                continue
        context["actionables"] = items
    except Exception:
        pass

    try:
        ax_snapshot = page.accessibility.snapshot()
        if isinstance(ax_snapshot, dict):
            # remove potentially massive children lists beyond first level
            pruned = dict(ax_snapshot)
            children = pruned.get("children")
            if isinstance(children, list) and len(children) > 10:
                pruned["children"] = children[:10]
                pruned["children_truncated"] = True
            context["accessibility"] = pruned
    except Exception:
        pass

    try:
        context["meta"] = page.evaluate(
            """
            () => ({
                title: document.title,
                metas: Array.from(document.querySelectorAll('meta')).slice(0, 25).map(m => ({
                    name: m.name || null,
                    property: m.getAttribute('property'),
                    content: m.content || null
                }))
            })
            """
        )
    except Exception:
        pass

    return context


def iteration(plan_json: str, max_low_level_rounds: int = 5):
    """Executes the plan with nested high-level and low-level loops."""
    plan = json.loads(plan_json)
    steps: List[Dict[str, Any]] = plan.get("steps", [])
    task = plan.get("task", "")
    overall_goal = plan.get("overall_goal", "")
    starting_url = plan.get("url")

    recall_context: List[Dict[str, Any]] = []
    execution_summary: List[Dict[str, Any]] = []
    MAX_RECALL_ENTRIES = 30
    state_capture_dir = Path("state_captures")
    state_capture_dir.mkdir(exist_ok=True)

    def add_recall(entry: Dict[str, Any]):
        recall_context.append(entry)
        if len(recall_context) > MAX_RECALL_ENTRIES:
            del recall_context[:-MAX_RECALL_ENTRIES]

    def dom_excerpt(text: str, limit: int = 800) -> str:
        if not text:
            return ""
        return text[:limit]

    def slugify(text: str, fallback: str) -> str:
        base = re.sub(r"[^a-zA-Z0-9]+", "_", text).strip("_")
        return base or fallback

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        page = browser.new_page()

        print(f"\n🌐 Starting execution loop for task: {task or 'Unnamed task'}")

        if starting_url:
            print(f"➡️  Navigating to starting URL: {starting_url}")
            page.goto(starting_url, wait_until="load")
            add_recall({
                "type": "navigation",
                "url": starting_url,
                "timestamp": time.time()
            })
            page.screenshot(path="step_0_navigate.png", full_page=True)

        for idx, step in enumerate(steps, start=1):
            step_id = step.get("id", idx)
            step_goal = step.get("goal", "")
            success_criteria = step.get("success_criteria", "")
            step_state_capture = step.get("state_capture", "")

            print(f"\n=== HIGH-LEVEL STEP {idx}/{len(steps)} ===")
            print(f"Intent: {step.get('intent')} | Action: {step.get('action')}")
            print(f"🎯 Goal: {step_goal}")
            add_recall({
                "type": "high_level_step",
                "step_id": step_id,
                "intent": step.get("intent"),
                "goal": step_goal,
                "action": step.get("action"),
                "timestamp": time.time()
            })

            step_completed = False
            for attempt in range(1, max_low_level_rounds + 1):
                print(f"\n  ➿ Low-level attempt {attempt}/{max_low_level_rounds}")
                page_context_before = collect_page_context(page)
                dom_before = page_context_before.get("dom", "")
                rendered_excerpt = (page_context_before.get("rendered_text") or "")[:400]
                add_recall({
                    "type": "observation",
                    "step_id": step_id,
                    "attempt": attempt,
                    "dom_excerpt": dom_excerpt(dom_before),
                    "rendered_text": rendered_excerpt,
                    "timestamp": time.time()
                })

                action_json = low_level_plan(step, page_context_before, recall_context=recall_context)
                print("  🤖 Low-level plan:", action_json)

                try:
                    action_data = json.loads(action_json)
                except json.JSONDecodeError:
                    print("  ❌ Low-level planner returned invalid JSON. Skipping to next attempt.")
                    add_recall({
                        "type": "low_level_error",
                        "step_id": step_id,
                        "attempt": attempt,
                        "raw_output": action_json
                    })
                    continue

                add_recall({
                    "type": "low_level_decision",
                    "step_id": step_id,
                    "attempt": attempt,
                    "target_goal": action_data.get("target_goal"),
                    "action": action_data.get("action"),
                    "selector": action_data.get("selector"),
                    "value": action_data.get("value"),
                    "timestamp": time.time()
                })

                action_success = execute_action(page, action_json)

                screenshot_path = f"step_{step_id}_iter{attempt}_{action_data.get('action', 'unknown')}.png"
                page.screenshot(path=screenshot_path, full_page=True)

                page_context_after = collect_page_context(page)
                dom_after = page_context_after.get("dom", "")
                goal_eval = evaluate_goal_completion(
                    step_goal,
                    success_criteria,
                    dom_after,
                    recall_context=recall_context,
                    page_url=page_context_after.get("url", page.url),
                    page_title=page_context_after.get("title"),
                    rendered_text=page_context_after.get("rendered_text"),
                    actionables=page_context_after.get("actionables"),
                    meta=page_context_after.get("meta")
                )
                add_recall({
                    "type": "goal_evaluation",
                    "step_id": step_id,
                    "attempt": attempt,
                    "status": goal_eval.get("status"),
                    "confidence": goal_eval.get("confidence"),
                    "feedback": goal_eval.get("feedback"),
                    "timestamp": time.time()
                })

                summary_entry = {
                    "step_id": step_id,
                    "attempt": attempt,
                    "low_level_action": action_data,
                    "goal_status": goal_eval.get("status"),
                    "feedback": goal_eval.get("feedback"),
                    "dom_excerpt": dom_excerpt(dom_after, limit=1200),
                    "screenshot": screenshot_path,
                    "success": action_success,
                    "human_prompt": action_data.get("message")
                }

                if goal_eval.get("status") == "achieved":
                    state_slug = slugify(
                        step_state_capture or step_goal or step.get("intent") or f"step_{step_id}",
                        f"step_{step_id}"
                    )
                    html_path = state_capture_dir / f"{state_slug}.html"
                    png_path = state_capture_dir / f"{state_slug}.png"
                    try:
                        html_path.write_text(dom_after, encoding="utf-8")
                        page.screenshot(path=str(png_path), full_page=True)
                        summary_entry["state_capture_html"] = str(html_path)
                        summary_entry["state_capture_screenshot"] = str(png_path)
                        add_recall({
                            "type": "state_capture",
                            "step_id": step_id,
                            "attempt": attempt,
                            "goal": step_goal,
                            "state_hint": step_state_capture,
                            "html_path": str(html_path),
                            "screenshot_path": str(png_path),
                            "timestamp": time.time()
                        })
                    except Exception as capture_err:
                        summary_entry["state_capture_error"] = str(capture_err)
                        print(f"  ⚠️ Failed to persist state capture: {capture_err}")
                    print(f"  ✅ Step goal achieved with confidence {goal_eval.get('confidence')}.")
                    step_completed = True
                    execution_summary.append(summary_entry)
                    break

                if not action_success or action_data.get("action") == "replan":
                    print("  ⚠️ Action failed or requested replan. Proceeding to next high-level attempt.")
                    execution_summary.append(summary_entry)
                    break

                time.sleep(1.0)

                execution_summary.append(summary_entry)

            if not step_completed:
                print(f"❗ High-level step {step_id} did not reach its goal within allotted attempts.")

            # Optional overall goal check after each high-level step
            if overall_goal:
                page_context_now = collect_page_context(page)
                dom_now = page_context_now.get("dom", "")
                overall_eval = evaluate_goal_completion(
                    overall_goal,
                    overall_goal,
                    dom_now,
                    recall_context=recall_context,
                    page_url=page_context_now.get("url", page.url),
                    page_title=page_context_now.get("title"),
                    rendered_text=page_context_now.get("rendered_text"),
                    actionables=page_context_now.get("actionables"),
                    meta=page_context_now.get("meta")
                )
                if overall_eval.get("status") == "achieved":
                    overall_slug = slugify("overall_goal", "overall_goal")
                    html_path = state_capture_dir / f"{overall_slug}.html"
                    png_path = state_capture_dir / f"{overall_slug}.png"
                    try:
                        html_path.write_text(dom_now, encoding="utf-8")
                        page.screenshot(path=str(png_path), full_page=True)
                        add_recall({
                            "type": "state_capture",
                            "step_id": "overall",
                            "attempt": None,
                            "goal": overall_goal,
                            "state_hint": "overall success",
                            "html_path": str(html_path),
                            "screenshot_path": str(png_path),
                            "timestamp": time.time()
                        })
                    except Exception as capture_err:
                        print(f"⚠️ Failed to persist overall state capture: {capture_err}")
                    print(f"\n🏁 Overall goal achieved early with confidence {overall_eval.get('confidence')}. Terminating execution.")
                    break

        print("\n✅ Execution loop finished. Summary of attempts:")
        for entry in execution_summary:
            print(
                f" - Step {entry['step_id']} attempt {entry['attempt']}: "
                f"action={entry['low_level_action'].get('action')} "
                f"status={entry['goal_status']} feedback={entry['feedback']} "
                f"state_capture={entry.get('state_capture_html', 'n/a')} "
                f"human_prompt={entry.get('human_prompt') or 'n/a'}"
            )

        browser.close()
