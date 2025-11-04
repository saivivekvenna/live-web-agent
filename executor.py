# executor.py
import json
import time
import re
from pathlib import Path
from typing import List, Dict, Any, Iterable
from playwright.sync_api import sync_playwright
from planner import low_level_plan, evaluate_goal_completion


def _truncate_text(text: Any, limit: int) -> str:
    """Safely truncate text representations while preserving ASCII output."""
    if text is None:
        return ""
    value = str(text)
    if len(value) <= limit:
        return value
    return value[:limit] + "..."


def _format_inline(text: Any, limit: int = 160) -> str:
    """Format text for inline display by collapsing whitespace and quoting safely."""
    collapsed = " ".join((_truncate_text(text, limit) or "").split())
    return collapsed.replace('"', "'")


def build_dom_filter_snapshot(
    context: Dict[str, Any],
    text_limit: int = 15000,
    interactable_limit: int = 150,
    text_block_limit: int = 200,
) -> str:
    """
    Create a compact but information-dense string representation of the current page.
    Includes URL, title, metadata, interactable inventory, text snippets, and DOM excerpts.
    """

    def _iter_dict_items(items: Iterable[Dict[str, Any]], limit: int) -> Iterable[Dict[str, Any]]:
        count = 0
        for item in items or []:
            if count >= limit:
                break
            yield item or {}
            count += 1

    lines: List[str] = []
    url = context.get("url")
    if url:
        lines.append(f"URL: {url}")

    title = context.get("title")
    if title:
        lines.append(f"TITLE: {title}")

    meta = context.get("meta") or {}
    meta_title = meta.get("title")
    if meta_title and meta_title != title:
        lines.append(f"META_TITLE: {meta_title}")

    metas = meta.get("metas", []) if isinstance(meta, dict) else []
    if metas:
        lines.append("META_TAGS:")
        for meta_idx, entry in enumerate(_iter_dict_items(metas, 20), start=1):
            name = entry.get("name") or entry.get("property") or f"meta-{meta_idx}"
            content = _format_inline(entry.get("content"), limit=240)
            lines.append(f"  - {name}: {content}")

    categories = context.get("actionable_categories") or {}
    if categories:
        lines.append("INTERACTABLE_CATEGORIES:")
        for cat_name, entries in categories.items():
            cat_label = cat_name.upper()
            preview = [
                f"{_format_inline(e.get('text'), limit=120)}"
                for e in _iter_dict_items(entries, 3)
            ]
            preview_line = "; ".join([p for p in preview if p])
            lines.append(f"  - {cat_label}: {preview_line or 'n/a'} (total captured={len(entries)})")

    interactables = context.get("actionables") or []
    if interactables:
        lines.append("INTERACTABLES:")
        for idx, item in enumerate(_iter_dict_items(interactables, interactable_limit), start=1):
            tag = item.get("tag") or "element"
            role = item.get("role")
            type_ = item.get("type")
            cats = ",".join(item.get("categories") or [])
            text = _format_inline(item.get("text"), limit=220)
            href = _format_inline(item.get("href"), limit=200)
            name = _format_inline(item.get("name"), limit=120)
            placeholder = _format_inline(item.get("placeholder"), limit=120)
            selector = item.get("css_path") or ""
            details = []
            if role:
                details.append(f"role={role}")
            if type_:
                details.append(f"type={type_}")
            if name:
                details.append(f"name=\"{name}\"")
            if placeholder:
                details.append(f"placeholder=\"{placeholder}\"")
            if href:
                details.append(f"href=\"{href}\"")
            if item.get("id"):
                details.append(f"id=\"{item['id']}\"")
            if item.get("visible") is not None:
                details.append(f"visible={item['visible']}")
            if item.get("disabled") is not None:
                details.append(f"disabled={item['disabled']}")
            if item.get("checked") is not None:
                details.append(f"checked={item['checked']}")
            if cats:
                details.append(f"cats={cats}")
            if selector:
                details.append(f"selector=\"{selector}\"")
            line = f"  [{idx}] <{tag}>"
            if text:
                line += f" text=\"{text}\""
            if details:
                line += " " + " ".join(details)
            lines.append(line)

    text_blocks = context.get("visible_text_blocks") or []
    if text_blocks:
        lines.append("VISIBLE_TEXT_BLOCKS:")
        for idx, block in enumerate(_iter_dict_items(text_blocks, text_block_limit), start=1):
            tag = block.get("tag") or "node"
            text = _format_inline(block.get("text"), limit=240)
            selector = block.get("css_path") or ""
            line = f"  [{idx}] <{tag}> text=\"{text}\""
            if selector:
                line += f" selector=\"{selector}\""
            lines.append(line)

    rendered = context.get("rendered_text") or ""
    if rendered:
        lines.append("RENDERED_TEXT_EXCERPT:")
        lines.append(_truncate_text(rendered, text_limit))

    dom_html = context.get("dom") or ""
    if dom_html:
        lines.append("DOM_HTML_EXCERPT:")
        lines.append(_truncate_text(dom_html, text_limit))

    return "\n".join(lines)

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
    Returns URL, title, raw DOM HTML, rendered text, interactable inventory,
    accessibility snapshot, page metadata, and a synthesized DOM filter summary.
    """
    context: Dict[str, Any] = {
        "url": "",
        "title": "",
        "dom": "",
        "rendered_text": "",
        "actionables": [],
        "actionable_categories": {},
        "visible_text_blocks": [],
        "dom_filter": "",
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
        interactable_payload = page.evaluate(
            """
            (limit) => {
                const categorySelectors = {
                    button: [
                        "button",
                        "input[type='button']",
                        "input[type='submit']",
                        "input[type='reset']",
                        "[role='button']",
                        "summary"
                    ],
                    link: [
                        "a[href]",
                        "[role='link']"
                    ],
                    input: [
                        "input",
                        "[role='textbox']",
                        "[contenteditable='true']"
                    ],
                    textarea: [
                        "textarea"
                    ],
                    select: [
                        "select",
                        "[role='combobox']",
                        "[role='listbox']"
                    ],
                    checkbox: [
                        "input[type='checkbox']",
                        "[role='checkbox']",
                        "[role='switch']"
                    ],
                    radio: [
                        "input[type='radio']",
                        "[role='radio']"
                    ],
                    tab: [
                        "[role='tab']"
                    ],
                    menuitem: [
                        "[role='menuitem']",
                        "[role='menuitemcheckbox']",
                        "[role='menuitemradio']",
                        "[role='option']"
                    ],
                    focusable: [
                        "[tabindex]",
                        "[role='gridcell']",
                        "[role='treeitem']"
                    ]
                };

                const seen = new WeakMap();
                const elements = [];

                const toPlainDataset = (dataset) => {
                    if (!dataset) return null;
                    const slice = Object.entries(dataset).slice(0, 20).map(([key, value]) => {
                        return [key, value == null ? null : String(value).slice(0, 200)];
                    });
                    if (!slice.length) return null;
                    return Object.fromEntries(slice);
                };

                const computeVisibility = (el) => {
                    if (!el || el.nodeType !== Node.ELEMENT_NODE) return false;
                    const style = window.getComputedStyle(el);
                    if (!style || style.visibility === "hidden" || style.display === "none") return false;
                    if (parseFloat(style.opacity || "1") === 0) return false;
                    const rect = el.getBoundingClientRect();
                    if (rect.width === 0 || rect.height === 0) return false;
                    if (rect.bottom < 0 || rect.right < 0) return false;
                    return true;
                };

                const getCssPath = (el) => {
                    if (!el || el.nodeType !== Node.ELEMENT_NODE) return null;
                    const segments = [];
                    let current = el;
                    let depth = 0;
                    while (current && current.nodeType === Node.ELEMENT_NODE && depth < 8) {
                        let segment = current.nodeName.toLowerCase();
                        if (current.id) {
                            segment += "#" + current.id;
                            segments.unshift(segment);
                            break;
                        }
                        let index = 1;
                        let sibling = current.previousElementSibling;
                        while (sibling) {
                            if (sibling.nodeName === current.nodeName) {
                                index += 1;
                            }
                            sibling = sibling.previousElementSibling;
                        }
                        if (index > 1) {
                            segment += ":nth-of-type(" + index + ")";
                        }
                        segments.unshift(segment);
                        current = current.parentElement;
                        depth += 1;
                    }
                    return segments.join(" > ");
                };

                const toInfo = (el) => {
                    const rect = el.getBoundingClientRect();
                    const style = window.getComputedStyle(el);
                    const rawText = (el.innerText || el.textContent || "").replace(/\s+/g, " ").trim();
                    const info = {
                        tag: el.tagName.toLowerCase(),
                        role: el.getAttribute("role"),
                        type: el.getAttribute("type") || (el.type || null),
                        name: el.getAttribute("name") || null,
                        id: el.id || null,
                        classes: el.className || null,
                        text: rawText ? rawText.slice(0, 900) : "",
                        placeholder: el.getAttribute("placeholder"),
                        title: el.getAttribute("title"),
                        href: el instanceof HTMLAnchorElement && el.href ? el.href : (el.getAttribute("href") || null),
                        value: null,
                        checked: null,
                        disabled: typeof el.disabled === "boolean" ? el.disabled : style.pointerEvents === "none",
                        required: typeof el.required === "boolean" ? el.required : null,
                        readOnly: typeof el.readOnly === "boolean" ? el.readOnly : null,
                        ariaLabel: el.getAttribute("aria-label"),
                        ariaDescribedby: el.getAttribute("aria-describedby"),
                        ariaControls: el.getAttribute("aria-controls"),
                        ariaExpanded: el.getAttribute("aria-expanded"),
                        ariaPressed: el.getAttribute("aria-pressed"),
                        dataset: toPlainDataset(el.dataset),
                        tabIndex: el.getAttribute("tabindex"),
                        visible: computeVisibility(el),
                        boundingBox: {
                            x: rect.x,
                            y: rect.y,
                            width: rect.width,
                            height: rect.height
                        },
                        css_path: getCssPath(el),
                        categories: []
                    };

                    if (el instanceof HTMLInputElement || el instanceof HTMLTextAreaElement) {
                        info.value = el.value ? String(el.value).slice(0, 400) : null;
                        if (el.type === "checkbox" || el.type === "radio") {
                            info.checked = el.checked;
                        }
                    } else if (typeof el.value !== "undefined") {
                        try {
                            info.value = el.value ? String(el.value).slice(0, 400) : null;
                        } catch (err) {
                            info.value = null;
                        }
                    }

                    if ("labels" in el && el.labels) {
                        const labels = Array.from(el.labels)
                            .map((label) => (label.innerText || label.textContent || "").replace(/\s+/g, " ").trim())
                            .filter(Boolean)
                            .slice(0, 5);
                        if (labels.length) {
                            info.labels = labels;
                        }
                    }

                    if (info.href) {
                        info.href = info.href.slice(0, 800);
                    }
                    if (info.classes) {
                        info.classes = info.classes.toString().trim() || null;
                    }
                    if (info.dataset && !Object.keys(info.dataset).length) {
                        delete info.dataset;
                    }

                    return info;
                };

                const addElement = (el, category) => {
                    if (!el || el.nodeType !== Node.ELEMENT_NODE) return;
                    let record = seen.get(el);
                    if (!record) {
                        record = toInfo(el);
                        record.categories = [];
                        seen.set(el, record);
                        elements.push(record);
                    }
                    if (category && !record.categories.includes(category)) {
                        record.categories.push(category);
                    }
                };

                Object.entries(categorySelectors).forEach(([category, queries]) => {
                    queries.forEach((selector) => {
                        try {
                            document.querySelectorAll(selector).forEach((el) => addElement(el, category));
                        } catch (err) {
                            // ignore malformed selectors
                        }
                    });
                });

                const tabbableSelectors = [
                    "a[href]",
                    "button",
                    "input",
                    "select",
                    "textarea",
                    "[tabindex]",
                    "[contenteditable='true']"
                ];
                tabbableSelectors.forEach((selector) => {
                    try {
                        document.querySelectorAll(selector).forEach((el) => addElement(el, "focusable"));
                    } catch (err) {
                        // ignore
                    }
                });

                elements.sort((a, b) => {
                    if (a.visible !== b.visible) return a.visible ? -1 : 1;
                    if (a.boundingBox && b.boundingBox) {
                        if (a.boundingBox.y !== b.boundingBox.y) {
                            return a.boundingBox.y - b.boundingBox.y;
                        }
                        return a.boundingBox.x - b.boundingBox.x;
                    }
                    return 0;
                });

                const trimmed = elements.slice(0, limit).map((item) => {
                    const copy = Object.assign({}, item);
                    if (Array.isArray(copy.categories)) {
                        copy.categories = copy.categories.slice(0, 10);
                    }
                    if (copy.text && copy.text.length > 900) {
                        copy.text = copy.text.slice(0, 900);
                    }
                    if (copy.value && String(copy.value).length > 400) {
                        copy.value = String(copy.value).slice(0, 400);
                    }
                    return copy;
                });

                const categoryIndex = {};
                trimmed.forEach((item) => {
                    (item.categories || []).forEach((category) => {
                        if (!categoryIndex[category]) {
                            categoryIndex[category] = [];
                        }
                        if (categoryIndex[category].length < 120) {
                            categoryIndex[category].push({
                                tag: item.tag,
                                text: item.text,
                                href: item.href,
                                name: item.name,
                                placeholder: item.placeholder,
                                css_path: item.css_path,
                                id: item.id
                            });
                        }
                    });
                });

                return {
                    all: trimmed,
                    categories: categoryIndex
                };
            }
            """,
            max_clickables * 4 if max_clickables else 160,
        )
        if isinstance(interactable_payload, dict):
            context["actionables"] = interactable_payload.get("all") or []
            context["actionable_categories"] = interactable_payload.get("categories") or {}
    except Exception:
        pass

    try:
        context["visible_text_blocks"] = page.evaluate(
            """
            (limit) => {
                const selectors = [
                    "h1",
                    "h2",
                    "h3",
                    "h4",
                    "h5",
                    "h6",
                    "p",
                    "li",
                    "label",
                    "legend",
                    "summary",
                    "span",
                    "dt",
                    "dd",
                    "th",
                    "td"
                ];
                const seen = new Set();
                const nodes = [];

                const getCssPath = (el) => {
                    if (!el || el.nodeType !== Node.ELEMENT_NODE) return null;
                    const segments = [];
                    let current = el;
                    let depth = 0;
                    while (current && current.nodeType === Node.ELEMENT_NODE && depth < 8) {
                        let segment = current.nodeName.toLowerCase();
                        if (current.id) {
                            segment += "#" + current.id;
                            segments.unshift(segment);
                            break;
                        }
                        let index = 1;
                        let sibling = current.previousElementSibling;
                        while (sibling) {
                            if (sibling.nodeName === current.nodeName) {
                                index += 1;
                            }
                            sibling = sibling.previousElementSibling;
                        }
                        if (index > 1) {
                            segment += ":nth-of-type(" + index + ")";
                        }
                        segments.unshift(segment);
                        current = current.parentElement;
                        depth += 1;
                    }
                    return segments.join(" > ");
                };

                selectors.forEach((selector) => {
                    try {
                        document.querySelectorAll(selector).forEach((el) => {
                            const raw = (el.innerText || el.textContent || "").replace(/\\s+/g, " ").trim();
                            if (!raw) return;
                            const text = raw.slice(0, 900);
                            if (seen.has(text)) return;
                            seen.add(text);
                            const rect = el.getBoundingClientRect();
                            nodes.push({
                                tag: el.tagName.toLowerCase(),
                                text,
                                css_path: getCssPath(el),
                                visible: rect.width > 0 && rect.height > 0,
                                boundingBox: {
                                    x: rect.x,
                                    y: rect.y,
                                    width: rect.width,
                                    height: rect.height
                                }
                            });
                        });
                    } catch (err) {
                        // ignore
                    }
                });

                nodes.sort((a, b) => {
                    if (a.visible !== b.visible) return a.visible ? -1 : 1;
                    if (a.boundingBox && b.boundingBox) {
                        if (a.boundingBox.y !== b.boundingBox.y) {
                            return a.boundingBox.y - b.boundingBox.y;
                        }
                        return a.boundingBox.x - b.boundingBox.x;
                    }
                    return 0;
                });

                return nodes.slice(0, limit);
            }
            """,
            max_clickables * 6 if max_clickables else 240,
        ) or []
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

    try:
        context["dom_filter"] = build_dom_filter_snapshot(
            context,
            text_limit=max(text_limit, 4000),
            interactable_limit=max_clickables * 4 if max_clickables else 150,
            text_block_limit=max_clickables * 6 if max_clickables else 200,
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

    profile_root = Path("profiles")
    profile_root.mkdir(exist_ok=True)
    chromium_profile_dir = profile_root / "chromium_user_data"
    chromium_profile_dir.mkdir(exist_ok=True)
    chromium_profile_path = str(chromium_profile_dir.resolve())

    with sync_playwright() as p:
        context = p.chromium.launch_persistent_context(
            user_data_dir=chromium_profile_path,
            headless=False,
        )
        page = context.pages[0] if context.pages else context.new_page()

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

        context.close()
