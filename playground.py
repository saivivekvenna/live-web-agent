
import json
import os
import re
import sys
import textwrap
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
from urllib.parse import urlparse

 
from openai import OpenAI
from playwright.sync_api import TimeoutError as PWTimeoutError


from bots._profile_launch import launch_persistent, shutdown as shutdown_persistent


PROFILES_ROOT = Path("profiles")
GENERATED_CONFIG_PATH = Path("generated_integrations.json")
SCREENSHOT_OUTPUT_DIR = Path("state_captures")
CURRENT_HOME_URL: Optional[str] = None
OVERLAY_LOCATORS = [
   ("role[dialog]", lambda page: page.locator("[role='dialog']")),
   ("aria-modal", lambda page: page.locator("[aria-modal='true']")),
   ("role[menu]", lambda page: page.locator("[role='menu']")),
   ("role[listbox]", lambda page: page.locator("[role='listbox']")),
]


client = OpenAI()
DEFAULT_GOAL = "make a database on notion"
MAX_ACTIONS = 12


@dataclass
class ActionRecord:
   label: str
   success: bool
   message: str
   timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
   state: Dict[str, Any] = field(default_factory=dict)
@dataclass
class IntegrationConfig:
   provider: str
   base_url: str
   profile_dir: str
   label_hints: List[str] = field(default_factory=list)
   extra_prompt: str = ""
   launch_kwargs: dict = field(default_factory=dict)
DEFAULT_INTEGRATIONS: Dict[str, IntegrationConfig] = {}



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




def _looks_like_dashboard(page, home_url: Optional[str] = None) -> bool:
   """
   Heuristic to detect when we are still on the provider's primary landing page.
   """
   if _overlay_present(page):
       return False


   normalized_home = (home_url or CURRENT_HOME_URL or "").rstrip("/").lower()
   if not normalized_home:
       return False

   try:
       current_url = page.url.rstrip("/").lower()
       if current_url.startswith(normalized_home):
           return True
   except Exception:
       pass


   return False




def _summarize_page_state(
   page, dom_limit: int = 30000, excerpt_limit: int = 2000, home_url: Optional[str] = None
) -> Dict[str, Any]:
   """
   Capture a structured snapshot of the current page so the planner can reason over it.
   """
   state: Dict[str, Any] = {
       "url": "",
       "title": "",
       "overlay_visible": False,
       "dashboard_hint": "unknown",
       "timestamp": datetime.now(UTC).isoformat(),
       "dropdown_open": False,
       "menu_open": False,
       "listbox_open": False,
       "dropdown_options": [],
   }

   try:
       state["url"] = page.url
   except Exception:
       state["url"] = ""

   try:
       state["title"] = page.title()
   except Exception:
       state["title"] = ""

   state["overlay_visible"] = _overlay_present(page)
   dashboard_home = home_url or CURRENT_HOME_URL
   state["dashboard_hint"] = (
       "likely_on_dashboard" if _looks_like_dashboard(page, dashboard_home) else "not_clearly_dashboard"
   )

   def _collect_text_list(selector: str, max_items: int = 8) -> List[str]:
       try:
           expr = (
               f"(els) => Array.from(els)"
               f".map(el => (el.innerText || '').trim())"
               f".filter(Boolean)"
               f".slice(0, {max_items})"
           )
           items = page.eval_on_selector_all(selector, expr)
           if isinstance(items, list):
               return [str(item) for item in items if isinstance(item, str)]
       except Exception:
           pass
       return []

   state["visible_headings"] = _collect_text_list("h1, h2, h3")
   state["visible_buttons"] = _collect_text_list("button, [role='button']")
   state["open_dialogs"] = _collect_text_list("[role='dialog'] *", max_items=3)

   menu_items = _collect_text_list(
       "[role='menu'] [role='menuitem'], [role='menu'] [role='option'], [role='menu'] button, [role='menu'] a",
       max_items=10,
   )
   listbox_items = _collect_text_list(
       "[role='listbox'] [role='option'], [role='listbox'] [role='menuitem'], [role='listbox'] button, [role='listbox'] a",
       max_items=10,
   )
   state["menu_options"] = menu_items
   state["listbox_options"] = listbox_items
   dropdown_options = (menu_items + listbox_items)[:10]
   state["dropdown_options"] = dropdown_options
   state["menu_open"] = bool(menu_items)
   state["listbox_open"] = bool(listbox_items)
   state["dropdown_open"] = bool(dropdown_options)

   dom_snapshot = _capture_prioritized_dom(page, limit=dom_limit)
   state["dom_snippet"] = dom_snapshot
   state["dom_excerpt"] = dom_snapshot[:excerpt_limit] if dom_snapshot else ""
   return state


def _failure_history_digest(history: List[ActionRecord], limit: int = 5) -> List[Dict[str, Any]]:
   failures = [record for record in history if not record.success]
   if not failures:
       return []

   digest: List[Dict[str, Any]] = []
   for record in failures[-limit:]:
       entry: Dict[str, Any] = {
           "label": record.label,
           "message": record.message,
           "timestamp": record.timestamp.isoformat(),
       }
       if record.state:
           entry["state_hint"] = {
               "url": record.state.get("url"),
               "title": record.state.get("title"),
               "overlay_visible": record.state.get("overlay_visible"),
               "dashboard_hint": record.state.get("dashboard_hint"),
               "visible_headings": (record.state.get("visible_headings") or [])[:3],
           }
           excerpt = record.state.get("dom_excerpt") or record.state.get("dom_snippet")
           if isinstance(excerpt, str) and excerpt:
               entry["dom_excerpt"] = excerpt[:400]
       digest.append(entry)
   return digest


def _detect_progress(
   before: Optional[Dict[str, Any]], after: Optional[Dict[str, Any]]
) -> tuple[bool, str]:
   """
   Determine whether the page state changed meaningfully after an action.
   """
   if after is None:
       return True, "Updated page state unavailable"
   if not before:
       return True, "No baseline to compare"

   comparisons = [
       ("url", "URL changed"),
       ("dropdown_open", "Dropdown/listbox state changed"),
       ("menu_options", "Menu options changed"),
       ("listbox_options", "Listbox options changed"),
       ("open_dialogs", "Dialog content changed"),
       ("visible_headings", "Visible headings changed"),
   ]

   for key, reason in comparisons:
       if before.get(key) != after.get(key):
           return True, reason

   before_dom = (before.get("dom_excerpt") or "").strip()
   after_dom = (after.get("dom_excerpt") or "").strip()
   if before_dom != after_dom:
       return True, "DOM excerpt changed"

   return False, "DOM excerpt unchanged"


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




def _extract_json_array(text: str) -> List[str]:
   """
   Pull the first JSON array from a model response. Returns [] on failure.
   """
   text = text.strip()
   start = text.find("[")
   end = text.rfind("]")
   if start != -1 and end != -1 and end > start:
       snippet = text[start : end + 1]
       try:
           data = json.loads(snippet)
           if isinstance(data, list):
               return data
       except Exception:
           return []
   return []




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



def _make_input_locators(label: str, hints: Optional[List[str]] = None) -> List[tuple]:
   """Return locator builders that target text entry controls matching the label."""
   label = label.strip()
   if not label:
       return []

   hints_set = {hint.lower() for hint in (hints or [])}
   prefers_exact = "fuzzy" not in hints_set and len(label.split()) <= 4

   def textbox(page, label=label, exact=prefers_exact):
       return page.get_by_role("textbox", name=label, exact=exact)

   def searchbox(page, label=label, exact=prefers_exact):
       return page.get_by_role("searchbox", name=label, exact=exact)

   def combobox(page, label=label, exact=prefers_exact):
       return page.get_by_role("combobox", name=label, exact=exact)

   def labeled(page, label=label, exact=prefers_exact):
       return page.get_by_label(label, exact=exact)

   def placeholder(page, label=label, exact=prefers_exact):
       return page.get_by_placeholder(label, exact=exact)

   def aria_label_contains(page, label=label):
       escaped = label.replace("\\", "\\\\").replace('"', '\\"')
       return page.locator(f"[aria-label*=\"{escaped}\"]")

   def data_testid_contains(page, label=label):
       escaped = label.replace("\\", "\\\\").replace('"', '\\"')
       return page.locator(f"[data-testid*=\"{escaped}\"]")

   options: List[tuple] = []
   seen: set[str] = set()

   def add(desc: str, builder: Callable) -> None:
       if desc in seen:
           return
       seen.add(desc)
       options.append((desc, builder))

   add(f"role=textbox[{label}]", textbox)
   add(f"role=searchbox[{label}]", searchbox)
   add(f"role=combobox[{label}]", combobox)
   add(f"label[{label}]", labeled)
   add(f"placeholder[{label}]", placeholder)
   add(f"aria-label contains '{label}'", aria_label_contains)
   add(f"data-testid contains '{label}'", data_testid_contains)

   return options



def _attempt_dropdown_click(page, label: str, exact: bool = True) -> tuple[bool, str]:
   """
   Try to select a visible option inside an open dropdown/listbox.
   """
   label = label.strip()
   if not label:
       return False, "Dropdown selection skipped: empty label."

   attempts: List[tuple[str, Callable]] = [
       ("menu role=menuitem", lambda: page.locator("[role='menu']").get_by_role("menuitem", name=label, exact=exact)),
       ("menu text", lambda: page.locator("[role='menu']").get_by_text(label, exact=exact)),
       ("listbox role=option", lambda: page.locator("[role='listbox']").get_by_role("option", name=label, exact=exact)),
       ("listbox text", lambda: page.locator("[role='listbox']").get_by_text(label, exact=exact)),
   ]

   last_error = "Dropdown selectors not found."
   for desc, builder in attempts:
       try:
           locator = builder()
       except Exception as exc:
           last_error = str(exc)
           continue
       try:
           if locator.count() == 0:
               last_error = f"{desc} matched 0 elements"
               continue
           target = locator.first
           target.wait_for(state="visible", timeout=5000)
           target.click(timeout=5000)
           return True, f"Selected dropdown option via {desc}"
       except Exception as exc:
           last_error = str(exc)
   return False, last_error



def _attempt_click(
   page,
   label: str,
   hints: Optional[List[str]] = None,
   page_state: Optional[Dict[str, Any]] = None,
) -> tuple[bool, str]:
   """
   Attempt to click a label using generated locator strategies.
   Returns (success, message).
   """
   options = _make_label_locators(label, hints)
   if not options:
       return False, "No locator strategies generated."

   last_error = "No matching elements located."
   if page_state and page_state.get("dropdown_open"):
       hint_set = {hint.lower() for hint in (hints or [])}
       dropdown_exact = "fuzzy" not in hint_set
       success, dropdown_msg = _attempt_dropdown_click(page, label, exact=dropdown_exact)
       if success:
           return True, dropdown_msg
       last_error = dropdown_msg

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




def _attempt_type(
   page,
   label: str,
   text: str,
   hints: Optional[List[str]] = None,
   clear_first: bool = True,
   press_enter: bool = False,
) -> tuple[bool, str]:
   """
   Focus a textbox matching the label and enter the provided text.
   """
   label = (label or "").strip()
   if not label:
       return False, "Type skipped: empty label."
   text_value = str(text)
   options = _make_input_locators(label, hints)
   if not options:
       return False, "No input locator strategies generated."

   last_error = "No matching input elements located."
   for desc, builder in options:
       try:
           locator = builder(page)
           if locator.count() == 0:
               last_error = f"{desc} matched 0 elements"
               continue
           target = locator.first
           target.wait_for(state="visible", timeout=10000)
           try:
               target.click(timeout=5000)
           except Exception:
               pass

           typed_via = "fill"
           try:
               target.fill(text_value, timeout=5000)
           except Exception:
               typed_via = "type"
               if clear_first:
                   try:
                       target.press("Meta+A", timeout=1000)
                   except Exception:
                       try:
                           target.press("Control+A", timeout=1000)
                       except Exception:
                           pass
                   try:
                       target.press("Backspace", timeout=1000)
                   except Exception:
                       pass
               target.type(text_value, delay=40, timeout=10000)

           if press_enter:
               target.press("Enter", timeout=2000)
           return True, f"Typed '{text_value}' via {desc} ({typed_via})"
       except Exception as exc:
           last_error = str(exc)
   return False, f"Typing failed. Last error: {last_error}"



def _format_history(history: List[ActionRecord]) -> str:
   if not history:
       return "No actions taken yet."
   lines = []
   for idx, record in enumerate(history[-8:], start=max(len(history) - 7, 1)):
       outcome = "success" if record.success else "failure"
       lines.append(f"{idx}. {record.label} → {outcome} ({record.message})")
   return "\n".join(lines)




def _plan_next_action(
   page,
   goal: str,
   action: str,
   config: IntegrationConfig,
   history: List[ActionRecord],
   page_state: Optional[Dict[str, Any]] = None,
) -> dict:
   page_state = page_state or _summarize_page_state(page)
   history_text = _format_history(history)
   hints_text = ", ".join(config.label_hints) if config.label_hints else "None"
   overlay_state = "visible" if page_state.get("overlay_visible") else "not visible"
   dashboard_hint = "likely on dashboard" if page_state.get("dashboard_hint") == "likely_on_dashboard" else "not clearly on dashboard"
   dropdown_status = "open" if page_state.get("dropdown_open") else "closed"
   dropdown_options = page_state.get("dropdown_options") or []
   dropdown_preview = ", ".join(dropdown_options[:6]) if dropdown_options else "None"

   state_overview = dict(page_state)
   dom_snapshot = state_overview.pop("dom_snippet", "")
   state_summary = json.dumps(state_overview, indent=2) if state_overview else "{}"
   failure_digest = _failure_history_digest(history)
   failure_summary = json.dumps(failure_digest, indent=2) if failure_digest else "[]"


   prompt = textwrap.dedent(
       f"""
       You are operating a browser automation agent.


       Goal: {goal}
       Action focus: {action}
       Provider: {config.provider}
       Known helpful controls: {hints_text}
       Overlay/dialog visibility: {overlay_state}
       Dashboard heuristic: {dashboard_hint}
       Dropdown/listbox status: {dropdown_status} (options: {dropdown_preview})


      Recent action history:
      {history_text}

      Recent failed attempts (JSON):
      {failure_summary}

      Page state summary:
      {state_summary}


       Examine the DOM snippet and decide the single next best action to advance toward the goal.
       If a dropdown/listbox is open, prioritize selecting the appropriate option before navigating elsewhere.
       When the user intent requires entering text (e.g., search queries, form values), prefer focusing the relevant textbox and typing the value before clicking submit/search controls.
       Respond with a JSON object containing:
         - "finish": boolean indicating whether the goal is already satisfied.
         - "reason": short string explaining your decision.
         - "targets": array (max 3) of action plans. Each object must include:
               • "action": choose between "click" or "type".
               • "label_variants": array of 1-3 plausible UI labels/placeholder strings to try, ordered by priority.
               • "locator_hints": optional array of hint strings (e.g., "dialog", "menu", "exact", "textbox") to guide selectors.
               • "notes": optional string giving context for the interaction.
               • when "action" == "type", also include "text" (what to enter) and optional booleans "press_enter" / "clear_first".


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




def _discover_control_hints(
   page,
   goal: str,
   provider: str,
   action: str,
   limit: int = 10,
   page_state: Optional[Dict[str, Any]] = None,
) -> List[str]:
   dom_snapshot = ""
   if page_state and isinstance(page_state.get("dom_snippet"), str):
       dom_snapshot = page_state["dom_snippet"]
   else:
       dom_snapshot = _capture_prioritized_dom(page)
   prompt = textwrap.dedent(
       f"""
       You analyze live DOM fragments to suggest UI labels that help achieve the goal.


       Goal: {goal}
       Action focus: {action}
       Provider: {provider}


       Return a JSON array (max {limit}) of short, distinct control labels or keywords to look for.
       Prefer concrete button, menu, or link text. Avoid duplicates and overly long strings.


       DOM SNIPPET (truncated):
       {dom_snapshot}
       """
   ).strip()


   try:
       response = client.chat.completions.create(
           model="gpt-4o-mini",
           messages=[
               {"role": "system", "content": "Extract relevant UI control labels. Respond with JSON array only."},
               {"role": "user", "content": prompt},
           ],
           temperature=0.2,
       )
       raw = response.choices[0].message.content or "[]"
       candidates = _extract_json_array(raw)
       hints: List[str] = []
       seen: set[str] = set()
       dom_lower = (dom_snapshot or "").lower()
       for candidate in candidates:
           if not isinstance(candidate, str):
               continue
           label = candidate.strip()
           if not label or len(label) > 40:
               continue
           key = label.lower()
           if key in seen:
               continue
           seen.add(key)
           if dom_lower and label.lower() not in dom_lower:
               continue
           hints.append(label)
           if len(hints) >= limit:
               break
       return hints
   except Exception as exc:
       print(f"  • Control hint discovery failed: {exc}")
       return []




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




def _classify_intent_llm(goal: str, provider_hint: Optional[str] = None) -> Optional[tuple[str, str]]:
   """
   Fallback classifier that asks the LLM to provide provider/action JSON.
   """
   prompt = textwrap.dedent(
       f"""
       Analyze the user's goal: "{goal}".
       {f"Known provider hint: {provider_hint}" if provider_hint else ""}


       Respond with a JSON object using the keys "provider" and "action".
       Keep providers in lowercase (e.g., "workspace", "app").
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
       if provider or action:
           return provider or None, action or None
   except Exception as exc:
       print(f"  • Intent classification via LLM failed: {exc}")
   return None




def _fallback_provider(goal: str) -> Optional[str]:
   normalized = goal.lower()
   domain_match = re.search(r"(?:https?://)?([a-z0-9.-]+)\.[a-z]{2,}", normalized)
   if domain_match:
       host = domain_match.group(1)
       return host.split(".")[0]

   tokens = [tok for tok in re.split(r"\W+", normalized) if tok]
   skip_words = {"create", "make", "build", "find", "get", "open", "add", "new", "a", "an", "the", "to", "for", "in", "on"}
   for token in reversed(tokens):
       if token not in skip_words:
           return token
   return None




def _slugify_goal(goal: str, max_tokens: int = 4) -> str:
   tokens = [tok for tok in re.split(r"\W+", goal.lower()) if tok]
   if not tokens:
       return "generic_task"
   return "_".join(tokens[:max_tokens])




def _build_screenshot_path(goal: str, provider: str) -> str:
   slug = _slugify_goal(f"{provider}_{goal}", max_tokens=6)
   timestamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
   SCREENSHOT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
   filename = f"{slug}_{timestamp}.png"
   return str(SCREENSHOT_OUTPUT_DIR / filename)


def parse_intent(goal: str) -> tuple[str, str]:
   """
   Map a free-form goal description to (provider, action) using the LLM with lightweight fallbacks.
   """
   provider_hint = _fallback_provider(goal)
   llm_guess = _classify_intent_llm(goal, provider_hint)
   provider: Optional[str] = None
   action: Optional[str] = None
   if llm_guess:
       provider, action = llm_guess


   provider = provider or provider_hint
   if not provider:
       raise ValueError(f"Unable to determine provider from goal: {goal!r}")


   action = action or _slugify_goal(goal)
   return provider, action




def _load_generated_integration_cache() -> Dict[str, Any]:
   if not GENERATED_CONFIG_PATH.exists():
       return {}
   try:
       raw = GENERATED_CONFIG_PATH.read_text(encoding="utf-8")
       data = json.loads(raw or "{}")
       if isinstance(data, dict):
           return data
   except Exception as exc:
       print(f"  • Failed to read {GENERATED_CONFIG_PATH}: {exc}")
   return {}



def _save_generated_integration_cache(cache: Dict[str, Any]) -> None:
   try:
       payload = json.dumps(cache, indent=2, sort_keys=True, ensure_ascii=True)
       GENERATED_CONFIG_PATH.write_text(payload, encoding="utf-8")
   except Exception as exc:
       print(f"  • Failed to write {GENERATED_CONFIG_PATH}: {exc}")



def _sanitize_url(value: Optional[str]) -> Optional[str]:
   if not value or not isinstance(value, str):
       return None
   candidate = value.strip()
   if not candidate:
       return None
   parsed = urlparse(candidate)
   if parsed.scheme not in {"http", "https"} or not parsed.netloc:
       return None
   return candidate



def _sanitize_label_hints(raw: Any, limit: int = 8) -> List[str]:
   if not raw:
       return []
   if isinstance(raw, str):
       items = [raw]
   elif isinstance(raw, list):
       items = raw
   else:
       return []
   hints: List[str] = []
   seen: set[str] = set()
   for item in items:
       if not isinstance(item, str):
           continue
       label = item.strip()
       if not label or len(label) > 60:
           continue
       key = label.lower()
       if key in seen:
           continue
       hints.append(label)
       seen.add(key)
       if len(hints) >= limit:
           break
   return hints



def _sanitize_launch_kwargs(raw: Any, fallback: Optional[dict] = None) -> dict:
   allowed_keys = {"headless"}
   sanitized: dict = {}
   if isinstance(raw, dict):
       for key, value in raw.items():
           if key in allowed_keys and isinstance(value, bool):
               sanitized[key] = value
   if sanitized:
       return sanitized
   return dict(fallback) if fallback else {}



def _provider_env_suffix(provider: str) -> str:
   suffix = re.sub(r"[^A-Z0-9]+", "_", provider.upper()).strip("_")
   return suffix or "DEFAULT"



def _resolve_headless_preference(requested: bool) -> bool:
   env_value = os.environ.get("PLAYGROUND_HEADLESS")
   if env_value is not None:
       normalized = env_value.strip().lower()
       if normalized in {"1", "true", "yes", "on"}:
           return True
       if normalized in {"0", "false", "no", "off"}:
           return False
       print(f"  • Unrecognized PLAYGROUND_HEADLESS value '{env_value}', defaulting to visible browser.")
   if requested:
       print(
           "ℹ️ Discovery requested headless mode but UI launch is enforced. "
           "Set PLAYGROUND_HEADLESS=1 if you need headless runs."
       )
   return False



def _slugify_identifier(value: str, fallback: str = "profile") -> str:
   tokens = re.findall(r"[a-z0-9]+", value.lower())
   slug = "_".join(tokens).strip("_")
   return slug[:60] if slug else fallback



def _resolve_profile_dir_path(
   provider: str,
   profile_hint: Optional[str] = None,
   fallback_dir: Optional[str] = None,
) -> str:
   PROFILES_ROOT.mkdir(parents=True, exist_ok=True)
   root_override = os.environ.get("PLAYGROUND_PROFILE_ROOT")
   base_root = Path(root_override).expanduser() if root_override else PROFILES_ROOT
   base_root.mkdir(parents=True, exist_ok=True)

   suffix = _provider_env_suffix(provider)
   env_specific = os.environ.get(f"PLAYGROUND_PROFILE_DIR_{suffix}")
   env_global = os.environ.get("PLAYGROUND_PROFILE_DIR")

   fallback_clean = ""
   if fallback_dir:
       fallback_clean = str(fallback_dir).strip()
   hint_clean = (profile_hint or "").strip()

   candidate = (
       (env_specific or "").strip()
       or (env_global or "").strip()
       or fallback_clean
       or hint_clean
   )
   if not candidate:
       candidate = _slugify_identifier(provider)

   candidate_path = Path(candidate).expanduser()
   if candidate_path.is_absolute():
       target = candidate_path
   elif fallback_dir and candidate == fallback_dir:
       target = Path(fallback_dir).expanduser()
   else:
       target = base_root / candidate_path

   target.mkdir(parents=True, exist_ok=True)
   if env_specific:
       print(f"🔐 Using custom profile directory for {provider}: {target}")
   elif env_global:
       print(f"🔐 Using shared profile directory override: {target}")
   return str(target)



def _baseline_integration_config(provider: str) -> IntegrationConfig:
   fallback_dir = str(PROFILES_ROOT / _slugify_identifier(provider))
   profile_dir = _resolve_profile_dir_path(provider, fallback_dir=fallback_dir)
   return IntegrationConfig(
       provider=provider,
       base_url=_fallback_base_url(provider),
       profile_dir=profile_dir,
   )


def _fallback_base_url(provider: str) -> str:
   slug = re.sub(r"[^a-z0-9]+", "", provider.lower()) or "app"
   return f"https://{slug}.com/"



def _integration_search_agent(goal: str, provider: str) -> Optional[dict]:
   prompt = textwrap.dedent(
       f"""
       You are a research scout for a browser automation system.
       Quickly identify the primary web application entry point, login hints, and UI labels for "{provider}".

       Goal: {goal}

       Respond with JSON containing:
         - "primary_url": canonical https URL for the main product dashboard.
         - "alternate_urls": array of optional backup URLs or docs.
         - "login_hint": short text describing how authentication typically works.
         - "ui_keywords": array of UI terms/buttons that are relevant.
         - "profile_hint": recommended short name for the persistent profile directory.
         - "notes": succinct reminder about quirks or layout.
       """
   ).strip()


   try:
       response = client.chat.completions.create(
           model="gpt-4o-mini",
           messages=[
               {"role": "system", "content": "Act like a web search agent. Respond with JSON only."},
               {"role": "user", "content": prompt},
           ],
           temperature=0.2,
       )
       raw = response.choices[0].message.content or "{}"
       data = _extract_json_object(raw)
       return data or None
   except Exception as exc:
       print(f"  • Integration search agent failed: {exc}")
       return None



def _integration_config_agent(goal: str, provider: str, action: str, search_payload: Optional[dict]) -> Optional[dict]:
   search_summary = json.dumps(search_payload or {}, indent=2, ensure_ascii=True)
   prompt = textwrap.dedent(
       f"""
       Use the research summary below to craft a browser integration configuration.

       Provider: {provider}
       Goal: {goal}
       Action shorthand: {action}

       Research summary:
       {search_summary}

       Respond with a JSON object containing:
         - "base_url": https URL to open after launch.
         - "extra_prompt": optional coaching notes for the planner.
         - "label_hints": array (<=8) of helpful UI labels.
         - "profile_hint": short directory-friendly name (omit path).
         - "launch_kwargs": object containing safe Playwright options (allowed: headless boolean).
       """
   ).strip()


   try:
       response = client.chat.completions.create(
           model="gpt-4o-mini",
           messages=[
               {"role": "system", "content": "Produce integration JSON. Keep it factual."},
               {"role": "user", "content": prompt},
           ],
           temperature=0.2,
       )
       raw = response.choices[0].message.content or "{}"
       data = _extract_json_object(raw)
       return data or None
   except Exception as exc:
       print(f"  • Integration config agent failed: {exc}")
       return None



def _materialize_integration_config(
   provider: str,
   payload: Optional[dict],
   fallback: Optional[IntegrationConfig],
) -> IntegrationConfig:
   fallback_labels = list(fallback.label_hints) if fallback else []
   fallback_launch = dict(fallback.launch_kwargs) if fallback else {}

   base_url = _sanitize_url((payload or {}).get("base_url")) if payload else None
   profile_hint = (payload or {}).get("profile_hint") if payload else None
   extra_prompt = ((payload or {}).get("extra_prompt") or "").strip() if payload else ""
   label_hints = _sanitize_label_hints((payload or {}).get("label_hints")) if payload else []
   launch_kwargs = _sanitize_launch_kwargs((payload or {}).get("launch_kwargs"), fallback_launch if fallback else None)

   resolved_base = base_url or (fallback.base_url if fallback else _fallback_base_url(provider))
   fallback_profile_dir = fallback.profile_dir if fallback else None
   resolved_profile = _resolve_profile_dir_path(provider, profile_hint, fallback_profile_dir)
   resolved_extra_prompt = extra_prompt or (fallback.extra_prompt if fallback else "")
   resolved_hints = label_hints or fallback_labels

   return IntegrationConfig(
       provider=provider,
       base_url=resolved_base,
       profile_dir=resolved_profile,
       label_hints=resolved_hints,
       extra_prompt=resolved_extra_prompt,
       launch_kwargs=launch_kwargs,
   )



def resolve_integration_config(goal: str, provider: str, action: str) -> IntegrationConfig:
   cache = _load_generated_integration_cache()
   entry = cache.get(provider) if isinstance(cache, dict) else None
   entry = entry if isinstance(entry, dict) else {}
   cached_payload = entry.get("config")
   cached_search = entry.get("search")
   used_cache = bool(cached_payload)

   payload = cached_payload
   search_payload = cached_search
   generated = False

   if not payload:
       if not search_payload:
           search_payload = _integration_search_agent(goal, provider)
       payload = _integration_config_agent(goal, provider, action, search_payload)
       if payload:
           generated = True
           cache[provider] = {
               "search": search_payload,
               "config": payload,
              "generated_at": datetime.now(UTC).isoformat(),
               "goal_snapshot": goal,
               "action_snapshot": action,
           }
           _save_generated_integration_cache(cache)

   fallback = DEFAULT_INTEGRATIONS.get(provider)
   fallback_source = "preset" if fallback else "baseline"
   if not fallback:
       fallback = _baseline_integration_config(provider)

   config = _materialize_integration_config(provider, payload, fallback)
   if generated:
       print(f"🧬 Integration config for '{provider}' synthesized via discovery agent.")
   elif used_cache:
       print(f"♻️ Loaded cached discovery config for '{provider}'.")
   elif not payload and fallback_source == "preset":
       print(f"ℹ️ Using built-in preset for '{provider}'.")
   elif not payload:
       print(f"ℹ️ Using heuristic baseline config for '{provider}'.")
   return config




def main() -> None:
   goal = _resolve_goal()
   print(f"🎯 Goal: {goal}")


   try:
       provider, action = parse_intent(goal)
   except ValueError as exc:
       print(f"❌ {exc}")
       return


   print(f"🤖 Resolved intent → provider='{provider}', action='{action}'")
   try:
       config = resolve_integration_config(goal, provider, action)
   except RuntimeError as exc:
       print(f"❌ {exc}")
       return


   headless = _resolve_headless_preference(config.launch_kwargs.get("headless", False))
   global CURRENT_HOME_URL
   CURRENT_HOME_URL = config.base_url


   playwright = None
   context = None
   page = None
   try:
       playwright, context, page = launch_persistent(config.base_url, config.profile_dir, headless=headless)


       try:
           page.wait_for_load_state("networkidle", timeout=10000)
       except PWTimeoutError:
           pass


       runtime_config = replace(config)
       try:
           preloop_state: Optional[Dict[str, Any]] = _summarize_page_state(page)
       except Exception:
           preloop_state = None
       dynamic_hints = _discover_control_hints(
           page, goal, provider, action, page_state=preloop_state
       )
       if dynamic_hints:
           runtime_config.label_hints = dynamic_hints
           print(f"🧠 Discovered control hints: {dynamic_hints}")
       else:
           runtime_config.label_hints = list(config.label_hints)
           if runtime_config.label_hints:
               print(f"🧠 Using fallback control hints: {runtime_config.label_hints}")


       history: List[ActionRecord] = []
       for step_idx in range(1, MAX_ACTIONS + 1):
           try:
               page_state = _summarize_page_state(page)
               plan = _plan_next_action(page, goal, action, runtime_config, history, page_state=page_state)
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
               action_kind_raw = target.get("action") or "click"
               action_kind = action_kind_raw.strip().lower()
               if action_kind not in {"click", "type"}:
                   print(f"  • Unsupported action '{action_kind_raw}', skipping.")
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

               type_text_value: Optional[str] = None
               type_press_enter = False
               type_clear_first = True
               if action_kind == "type":
                   raw_text_value = target.get("text", target.get("value"))
                   if raw_text_value is None:
                       print("  • Type target missing 'text'; skipping.")
                       continue
                   type_text_value = str(raw_text_value)
                   if isinstance(target.get("press_enter"), bool):
                       type_press_enter = bool(target["press_enter"])
                   if isinstance(target.get("clear_first"), bool):
                       type_clear_first = bool(target["clear_first"])

               for label in label_variants:
                   if action_kind == "click":
                       success, message = _attempt_click(
                           page, label, locator_hints, page_state=page_state
                       )
                   else:
                       success, message = _attempt_type(
                           page,
                           label,
                           type_text_value or "",
                           locator_hints,
                           clear_first=type_clear_first,
                           press_enter=type_press_enter,
                       )
                   try:
                       updated_state = _summarize_page_state(
                           page, dom_limit=12000, excerpt_limit=1200
                       )
                   except Exception:
                       updated_state = {}

                   progress, progress_reason = _detect_progress(page_state, updated_state)
                   if success and not progress:
                       success = False
                       message = f"No observable change: {progress_reason}"

                   state_snapshot: Dict[str, Any] = {}
                   if not success:
                       state_snapshot = updated_state or {}

                   history_label = f"{action_kind}:{label}" if action_kind != "click" else label
                   history.append(
                       ActionRecord(
                           label=history_label,
                           success=success,
                           message=message,
                           state=state_snapshot,
                       )
                   )

                   outcome_icon = "✅" if success else "⚠️"
                   print(f"    {outcome_icon} Tried '{label}' → {message}")

                   if success:
                       page_state = updated_state or page_state
                       if label not in runtime_config.label_hints:
                           runtime_config.label_hints.append(label)
                       executed = True
                       try:
                           page.wait_for_timeout(800)
                       except Exception:
                           pass
                       break

                   page_state = updated_state or page_state


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


       screenshot_path = _build_screenshot_path(goal, provider)
       page.screenshot(path=screenshot_path, full_page=True)
       print(f"📸 Screenshot saved: {screenshot_path}")


       input("✅ Done. Inspect the page, then press Enter to close the browser…")
   finally:
       CURRENT_HOME_URL = None
       shutdown_persistent(playwright, context)




if __name__ == "__main__":
   main()
