# executor.py
import base64
import io
import json
import time
import re
from itertools import count
from pathlib import Path
from typing import List, Dict, Any, Iterable
from playwright.sync_api import sync_playwright
from planner import low_level_plan, evaluate_goal_completion

DOM_DEBUG_DIR = Path("dom_dumps")
DOM_DEBUG_DIR.mkdir(exist_ok=True)
_DOM_SNAPSHOT_COUNTER = count(1)
NETWORK_LOG_LIMIT = 200
NETWORK_EXPORT_LIMIT = 120
CANVAS_EXPORT_LIMIT = 5
PLAYWRIGHT_SELECTOR_PREFIXES = (
    "text=",
    "xpath=",
    "css=",
    "id=",
    "role=",
    "label=",
    "name=",
    "placeholder=",
    "alt=",
    "title=",
    "nth=",
    "locator=",
    "has=",
    "near=",
)

try:
    from PIL import Image
    import pytesseract

    _OCR_AVAILABLE = True
except Exception:
    Image = None  # type: ignore
    pytesseract = None  # type: ignore
    _OCR_AVAILABLE = False


def _save_dom_snapshots(context: Dict[str, Any], debug_payload: Dict[str, Any] | None = None) -> None:
    """Persist rich diagnostic artifacts to disk for debugging."""
    debug_payload = debug_payload or {}
    try:
        raw_dom = context.get("dom") or ""
        filtered_dom = context.get("dom_filter") or ""
        rendered_text = context.get("rendered_text") or ""
        accessibility_tree = context.get("accessibility") or {}

        any_primary = raw_dom or filtered_dom or rendered_text or accessibility_tree
        if not any_primary and not debug_payload:
            return

        snapshot_index = next(_DOM_SNAPSHOT_COUNTER)
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        base_name = f"{snapshot_index:04d}_{timestamp}"

        files_saved: List[str] = []

        def _write_text_file(filename: str, content: str) -> None:
            if not content:
                return
            path = DOM_DEBUG_DIR / filename
            path.write_text(content, encoding="utf-8")
            files_saved.append(path.name)

        def _write_json_file(filename: str, data: Any) -> None:
            if not data:
                return
            path = DOM_DEBUG_DIR / filename
            serialized = json.dumps(data, indent=2, ensure_ascii=False)
            path.write_text(serialized, encoding="utf-8")
            files_saved.append(path.name)

        def _write_binary_file(filename: str, payload: bytes) -> Path | None:
            if not payload:
                return None
            path = DOM_DEBUG_DIR / filename
            path.write_bytes(payload)
            files_saved.append(path.name)
            return path

        _write_text_file(f"{base_name}_raw_dom.html", raw_dom)
        _write_text_file(f"{base_name}_filtered_dom.txt", filtered_dom)
        _write_text_file(f"{base_name}_rendered_text.txt", rendered_text)
        _write_json_file(f"{base_name}_accessibility.json", accessibility_tree)

        frames_payload = debug_payload.get("frames")
        if frames_payload:
            _write_json_file(f"{base_name}_frames.json", frames_payload)

        shadow_payload = debug_payload.get("shadow")
        if shadow_payload:
            _write_json_file(f"{base_name}_shadow_dom.json", shadow_payload)

        cdp_payload = debug_payload.get("cdp_dom_snapshot")
        if cdp_payload:
            _write_json_file(f"{base_name}_cdp_snapshot.json", cdp_payload)

        network_payload = debug_payload.get("network_log")
        if network_payload:
            _write_json_file(f"{base_name}_network.json", network_payload)

        canvas_payload = debug_payload.get("canvas")
        if canvas_payload:
            canvas_meta: List[Dict[str, Any]] = []
            for idx, entry in enumerate(canvas_payload[:CANVAS_EXPORT_LIMIT], start=1):
                canvas_info = {
                    "index": entry.get("index"),
                    "width": entry.get("width"),
                    "height": entry.get("height"),
                    "bounding_rect": entry.get("bounding_rect"),
                    "to_data_url_error": entry.get("error"),
                }
                data_url = entry.get("data_url")
                image_path: Path | None = None
                if data_url and isinstance(data_url, str) and data_url.startswith("data:image"):
                    header, _, b64_data = data_url.partition(",")
                    try:
                        binary = base64.b64decode(b64_data)
                        image_path = _write_binary_file(
                            f"{base_name}_canvas{idx}.png", binary
                        )
                        canvas_info["image_file"] = image_path.name if image_path else None
                        if _OCR_AVAILABLE and image_path:
                            try:
                                with Image.open(io.BytesIO(binary)) as img:  # type: ignore[arg-type]
                                    text = pytesseract.image_to_string(img)  # type: ignore[call-arg]
                                ocr_text = text.strip()
                                if ocr_text:
                                    _write_text_file(
                                        f"{base_name}_canvas{idx}_ocr.txt", ocr_text
                                    )
                                    canvas_info["ocr_text_file"] = f"{base_name}_canvas{idx}_ocr.txt"
                                else:
                                    canvas_info["ocr_text_file"] = None
                            except Exception as ocr_exc:
                                canvas_info["ocr_error"] = str(ocr_exc)
                        elif data_url:
                            canvas_info["ocr_text_file"] = None
                            if not _OCR_AVAILABLE:
                                canvas_info["ocr_error"] = "pytesseract not available"
                    except Exception as decode_exc:
                        canvas_info["image_decode_error"] = str(decode_exc)
                canvas_meta.append(canvas_info)
            _write_json_file(f"{base_name}_canvas.json", canvas_meta)

        if files_saved:
            saved_str = ", ".join(files_saved)
            print(f"  🗂️ Saved diagnostics: {saved_str}")
    except Exception as exc:
        print(f"  ⚠️ Failed to save DOM diagnostics: {exc}")


def _gather_frame_data(page, text_limit: int = 1500, html_limit: int = 6000) -> List[Dict[str, Any]]:
    frames_details: List[Dict[str, Any]] = []
    main_frame = None
    try:
        main_frame = page.main_frame
    except Exception:
        pass

    for frame in page.frames:
        try:
            info: Dict[str, Any] = {
                "name": frame.name,
                "url": frame.url,
                "is_main_frame": frame is main_frame,
                "is_detached": frame.is_detached(),
            }
            parent = frame.parent_frame
            if parent:
                info["parent_url"] = parent.url
        except Exception:
            continue

        try:
            text_excerpt = frame.evaluate(
                "(limit) => (document.body?.innerText || '').slice(0, limit)",
                text_limit,
            )
            if text_excerpt:
                info["text_excerpt"] = text_excerpt
        except Exception:
            pass

        if frame is not main_frame:
            try:
                html_excerpt = frame.evaluate(
                    "(limit) => (document.body?.innerHTML || '').slice(0, limit)",
                    html_limit,
                )
                if html_excerpt:
                    info["html_excerpt"] = html_excerpt
            except Exception:
                pass

        frames_details.append(info)

    return frames_details


def _gather_shadow_dom(page, host_limit: int = 12, snippet_limit: int = 1200) -> List[Dict[str, Any]]:
    try:
        return page.evaluate(
            """
            (limit, snippetLimit) => {
                const results = [];

                const getCssPath = (el) => {
                    if (!el || el.nodeType !== Node.ELEMENT_NODE) return null;
                    const segments = [];
                    let current = el;
                    let depth = 0;
                    while (current && current.nodeType === Node.ELEMENT_NODE && depth < 10) {
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

                const walker = document.createTreeWalker(
                    document,
                    NodeFilter.SHOW_ELEMENT
                );

                const seen = new Set();
                let node;
                while (results.length < limit && (node = walker.nextNode())) {
                    if (!node.shadowRoot || seen.has(node)) continue;
                    seen.add(node);

                    const rect = node.getBoundingClientRect();
                    const snippet = Array.from(node.shadowRoot.children || [])
                        .map(child => child.outerHTML || "")
                        .join("\\n")
                        .slice(0, snippetLimit);

                    results.push({
                        host_tag: node.tagName.toLowerCase(),
                        host_id: node.id || null,
                        host_classes: node.className || null,
                        css_path: getCssPath(node),
                        child_count: node.shadowRoot.children ? node.shadowRoot.children.length : 0,
                        bounding_rect: {
                            x: rect.x,
                            y: rect.y,
                            width: rect.width,
                            height: rect.height
                        },
                        markup_excerpt: snippet
                    });
                }
                return results;
            }
            """,
            host_limit,
            snippet_limit,
        ) or []
    except Exception:
        return []


def _capture_cdp_snapshot(page) -> Dict[str, Any] | None:
    session = None
    try:
        session = page.context.new_cdp_session(page)
        snapshot = session.send(
            "DOMSnapshot.captureSnapshot",
            {
                "computedStyles": [],
                "includeDOMRects": True,
                "includePaintOrder": False,
            },
        )
        snapshot["captured_at"] = time.time()
        return snapshot
    except Exception:
        return None
    finally:
        if session:
            try:
                session.detach()
            except Exception:
                pass


def _gather_canvas_data(page, limit: int = CANVAS_EXPORT_LIMIT) -> List[Dict[str, Any]]:
    try:
        return page.evaluate(
            """
            (limit) => {
                const canvases = Array.from(document.querySelectorAll('canvas')).slice(0, limit);
                return canvases.map((canvas, idx) => {
                    const rect = canvas.getBoundingClientRect();
                    let dataUrl = null;
                    let error = null;
                    try {
                        dataUrl = canvas.toDataURL('image/png');
                    } catch (err) {
                        error = String(err);
                    }
                    return {
                        index: idx,
                        width: canvas.width,
                        height: canvas.height,
                        bounding_rect: {
                            x: rect.x,
                            y: rect.y,
                            width: rect.width,
                            height: rect.height
                        },
                        data_url: dataUrl,
                        error
                    };
                });
            }
            """,
            limit,
        ) or []
    except Exception:
        return []

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



def _sanitize_selector(selector: str | None) -> str | None:
    if not selector:
        return selector

    def _repl(match: re.Match[str]) -> str:
        attr = match.group(1).strip()
        value = match.group(3)
        value = value.replace('"', '\\"')
        return f'[{attr}="{value}"]'

    return re.sub(r"\[([^\]=]+)=([\'\"])(.*?)\2\]", _repl, selector)


def _escape_hash_identifiers(segment: str) -> str:
    def _escape_id(match: re.Match[str]) -> str:
        ident = match.group(1)
        escaped_ident = re.sub(r"(?<!\\):", r"\\:", ident)
        return "#" + escaped_ident

    return re.sub(r"#([^\s#\.\>\+\~\[\]]+)", _escape_id, segment or "")


def _escape_playwright_selector(selector: str) -> str:
    if not selector:
        return selector
    stripped_selector = selector.strip()
    if stripped_selector.startswith("//"):
        return selector
    if stripped_selector.startswith("css="):
        prefix, rest = selector.split("=", 1)
        return f"css={_escape_hash_identifiers(rest)}"
    parts = selector.split(">>")
    escaped_parts: List[str] = []
    for part in parts:
        trimmed = part.strip()
        if not trimmed:
            escaped_parts.append(trimmed)
            continue
        if trimmed.startswith(PLAYWRIGHT_SELECTOR_PREFIXES) or trimmed.startswith("//"):
            escaped_parts.append(trimmed)
        else:
            escaped_parts.append(_escape_hash_identifiers(trimmed))
    return " >> ".join(escaped_parts)


def _normalize_selector(selector: str | None) -> str | None:
    if not selector:
        return selector
    sanitized = _sanitize_selector(selector) or selector
    return _escape_playwright_selector(sanitized)


_HAS_TEXT_PATTERN = re.compile(r":has-text\((?:\"|')(.*?)(?:\"|')\)")


def _wait_condition_already_true(selector: str, context_snapshot: Dict[str, Any]) -> bool:
    if not selector:
        return False
    dom_html = context_snapshot.get("dom") or ""
    rendered_text = context_snapshot.get("rendered_text") or ""

    if selector in dom_html:
        return True

    match = _HAS_TEXT_PATTERN.search(selector)
    if match:
        probe = match.group(1)
        if probe and (probe in dom_html or probe in rendered_text):
            return True

    if selector.startswith("text="):
        probe = selector[len("text="):].strip("\"'")
        if probe and (probe in dom_html or probe in rendered_text):
            return True

    simple_match = re.search(r'text\("([^"]+)"\)', selector)
    if simple_match:
        probe = simple_match.group(1)
        if probe and (probe in dom_html or probe in rendered_text):
            return True

    return False


def _sanitize_postconditions(action: Dict[str, Any], context_snapshot: Dict[str, Any]) -> tuple[Dict[str, Any], Dict[str, str]]:
    adjustments: Dict[str, str] = {}
    wait_for = action.get("wait_for")
    if isinstance(wait_for, str) and _wait_condition_already_true(wait_for, context_snapshot):
        adjustments["wait_for"] = wait_for
        action.pop("wait_for", None)

    assert_selector = action.get("assert_selector")
    if isinstance(assert_selector, str) and _wait_condition_already_true(assert_selector, context_snapshot):
        adjustments["assert_selector"] = assert_selector
        action.pop("assert_selector", None)

    return action, adjustments


def _extract_action_keywords(step: Dict[str, Any], action: Dict[str, Any]) -> List[str]:
    sources = [
        step.get("goal"),
        step.get("intent"),
        step.get("description"),
        action.get("target_goal"),
        action.get("reasoning"),
        action.get("selector"),
    ]
    keywords: List[str] = []
    seen: set[str] = set()
    stopwords = {"text", "button", "link", "div", "span", "selector", "input"}

    def add_candidate(text: str) -> None:
        if not text:
            return
        text = text.strip().lower()
        if len(text) < 3:
            return
        if text in stopwords:
            return
        if text in seen:
            return
        seen.add(text)
        keywords.append(text)

    for kw in action.get("keywords") or []:
        if isinstance(kw, str):
            add_candidate(kw)

    for fallback in action.get("fallback_selectors") or []:
        if not isinstance(fallback, str):
            continue
        for match in re.findall(r'"([^"]+)"|\'([^\']+)\'', fallback):
            phrase = next((m for m in match if m), "")
            add_candidate(phrase)
        for token in re.split(r'[^A-Za-z0-9\+]+', fallback):
            add_candidate(token)

    for source in sources:
        if not source:
            continue
        for match in re.findall(r'"([^"]+)"|\'([^\']+)\'', source):
            phrase = next((m for m in match if m), "")
            add_candidate(phrase)
        tokens = re.split(r'[^A-Za-z0-9\+]+', source)
        for token in tokens:
            add_candidate(token)

    return keywords[:8]

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

    fragments = context.get("fragment_summary") or []
    if fragments:
        lines.append("FRAGMENT_CANDIDATES:")
        for idx, frag in enumerate(fragments, start=1):
            text = _format_inline(frag.get("text"), limit=160)
            selector = frag.get("selector") or ""
            source = frag.get("source") or ""
            tag = frag.get("tag") or ""
            line = f"  [{idx}] text=\"{text}\""
            if tag:
                line += f" tag={tag}"
            if selector:
                line += f" selector=\"{selector}\""
            if source:
                line += f" source=\"{source}\""
            lines.append(line)

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


def execute_action(page, action: Dict[str, Any], context_snapshot: Dict[str, Any] | None = None):
    """Executes a low-level action on a Playwright page with locator fallbacks."""
    a = action.get("action")
    sel = action.get("selector")
    val = action.get("value")
    key = action.get("key")
    timeout = action.get("timeout", 8000)

    print(f"🔹 Executing: {a} -> {sel or val or key}")

    def _resolve_locator(selector: str):
        normalized = _normalize_selector(selector) or selector
        locator = page.locator(normalized)
        if "nth" in action and isinstance(action["nth"], int):
            locator = locator.nth(action["nth"])
        elif action.get("last"):
            locator = locator.last
        elif action.get("all"):
            return locator
        else:
            locator = locator.first
        return locator

    def _collect_fragment_locators(keywords: List[str]) -> List[tuple[str, callable]]:
        locators: List[tuple[str, callable]] = []
        if not context_snapshot:
            return locators
        for item in context_snapshot.get("actionables") or []:
            if not isinstance(item, dict):
                continue
            if not item.get("is_fragment"):
                continue
            css_path = item.get("css_path")
            if not css_path:
                continue
            text = (item.get("text") or "").strip()
            if keywords:
                lowered = text.lower()
                if not any(kw in lowered for kw in keywords):
                    continue
            normalized = _normalize_selector(css_path) or css_path
            desc = f"fragment[{text or css_path}]"
            locators.append((desc, lambda norm=normalized: page.locator(norm)))
        return locators

    def _collect_keyword_locators(keywords: List[str]) -> List[tuple[str, callable]]:
        locators: List[tuple[str, callable]] = []
        seen: set[str] = set()
        for kw in keywords:
            kw = kw.strip()
            if not kw:
                continue
            for desc, builder in (
                (f"button(name~='{kw}')", lambda kw=kw: page.get_by_role("button", name=kw, exact=False)),
                (f"link(name~='{kw}')", lambda kw=kw: page.get_by_role("link", name=kw, exact=False)),
                (f"text~='{kw}'", lambda kw=kw: page.get_by_text(kw, exact=False)),
            ):
                if desc in seen:
                    continue
                seen.add(desc)
                locators.append((desc, builder))
        return locators

    def _click_with_fallbacks() -> tuple[bool, str | None]:
        raw_keywords: List[str] = []
        for source in (action.get("keywords"), action.get("_keywords")):
            if not source:
                continue
            raw_keywords.extend([kw for kw in source if isinstance(kw, str)])
        keywords: List[str] = []
        seen_kw: set[str] = set()
        for kw in raw_keywords:
            kw_norm = kw.strip().lower()
            if not kw_norm or kw_norm in seen_kw:
                continue
            seen_kw.add(kw_norm)
            keywords.append(kw)
        candidates: List[tuple[str, callable]] = []
        seen_desc: set[str] = set()

        if sel:
            normalized_selector = _normalize_selector(sel) or sel
            desc = f"planner selector [{sel}]"
            candidates.append((desc, lambda norm=normalized_selector: page.locator(norm)))
            seen_desc.add(desc)

        fallback_selectors = [
            fs for fs in (action.get("fallback_selectors") or []) if isinstance(fs, str)
        ]
        for idx, fallback in enumerate(fallback_selectors, start=1):
            normalized_fallback = _normalize_selector(fallback) or fallback
            desc = f"fallback_selector[{idx}]"
            if desc not in seen_desc:
                candidates.append((desc, lambda norm=normalized_fallback: page.locator(norm)))
                seen_desc.add(desc)

        for desc, builder in _collect_fragment_locators(keywords):
            if desc not in seen_desc:
                candidates.append((desc, builder))
                seen_desc.add(desc)

        for desc, builder in _collect_keyword_locators(keywords):
            if desc not in seen_desc:
                candidates.append((desc, builder))
                seen_desc.add(desc)

        if not candidates and sel:
            normalized_selector = _normalize_selector(sel) or sel
            candidates.append((f"planner selector [{sel}]", lambda norm=normalized_selector: page.locator(norm)))

        ensure_visible = action.get("ensure_visible", True)
        ensure_state = action.get("ensure_state", "visible")
        click_kwargs_base = {"timeout": timeout}
        if "button" in action:
            click_kwargs_base["button"] = action["button"]
        if "click_count" in action:
            click_kwargs_base["click_count"] = action["click_count"]
        if "modifiers" in action:
            click_kwargs_base["modifiers"] = action["modifiers"]
        if action.get("force"):
            click_kwargs_base["force"] = True

        last_error: Exception | None = None

        for desc, builder in candidates:
            try:
                locator = builder()
                count = locator.count()
                if count == 0:
                    print(f"  • {desc}: no matches")
                    continue
                target = locator.first
                if ensure_visible:
                    target.wait_for(state=ensure_state, timeout=timeout)
                click_kwargs = dict(click_kwargs_base)
                try:
                    target.click(**click_kwargs)
                    print(f"✅ Click succeeded via {desc}")
                    return True, None
                except Exception as click_error:
                    msg = str(click_error)
                    if "intercepts pointer events" in msg and not click_kwargs.get("force"):
                        print(f"  🔧 Overlay via {desc}; retrying with force=True")
                        click_kwargs["force"] = True
                        target.click(**click_kwargs)
                        print(f"✅ Click succeeded via {desc} (force)")
                        return True, None
                    print(f"  • {desc} failed: {click_error}")
                    last_error = click_error
            except Exception as exc:
                print(f"  • {desc} failed: {exc}")
                last_error = exc

        error_msg = str(last_error) if last_error else "No viable locator candidates"
        return False, error_msg

    try:
        if a == "navigate":
            page.goto(action["url"], wait_until="load")
        elif a == "click":
            success, error = _click_with_fallbacks()
            if not success:
                return False, error
        elif a == "type":
            if not sel:
                raise ValueError("Type action requires a selector.")
            locator = _resolve_locator(sel)
            if action.get("ensure_visible", True):
                locator.wait_for(state=action.get("ensure_state", "visible"), timeout=timeout)
            if action.get("focus_before", True):
                locator.focus()
            locator.fill(val or "", timeout=timeout)
        elif a == "press":
            if sel:
                locator = _resolve_locator(sel)
                if action.get("ensure_visible", True):
                    locator.wait_for(state=action.get("ensure_state", "visible"), timeout=timeout)
                locator.press(key or "", timeout=timeout)
            else:
                page.keyboard.press(key or "")
        elif a == "hover":
            if not sel:
                raise ValueError("Hover action requires a selector.")
            locator = _resolve_locator(sel)
            if action.get("ensure_visible", True):
                locator.wait_for(state=action.get("ensure_state", "visible"), timeout=timeout)
            locator.hover(timeout=timeout)
        elif a == "check":
            if not sel:
                raise ValueError("Check action requires a selector.")
            locator = _resolve_locator(sel)
            if action.get("ensure_visible", True):
                locator.wait_for(state=action.get("ensure_state", "visible"), timeout=timeout)
            locator.check(timeout=timeout, force=action.get("force", False))
        elif a == "uncheck":
            if not sel:
                raise ValueError("Uncheck action requires a selector.")
            locator = _resolve_locator(sel)
            if action.get("ensure_visible", True):
                locator.wait_for(state=action.get("ensure_state", "visible"), timeout=timeout)
            locator.uncheck(timeout=timeout, force=action.get("force", False))
        elif a == "upload":
            if not sel:
                raise ValueError("Upload action requires a selector.")
            locator = _resolve_locator(sel)
            locator.set_input_files(val)
        elif a == "scroll":
            direction = (val or "down").lower()
            if direction == "up":
                page.evaluate("window.scrollBy(0, -window.innerHeight)")
            else:
                page.evaluate("window.scrollBy(0, window.innerHeight)")
        elif a == "select":
            if not sel:
                raise ValueError("Select action requires a selector.")
            locator = _resolve_locator(sel)
            if action.get("ensure_visible", True):
                locator.wait_for(state=action.get("ensure_state", "visible"), timeout=timeout)
            locator.select_option(val, timeout=timeout)
        elif a == "focus":
            if not sel:
                raise ValueError("Focus action requires a selector.")
            locator = _resolve_locator(sel)
            if action.get("ensure_visible", True):
                locator.wait_for(state=action.get("ensure_state", "visible"), timeout=timeout)
            locator.focus()
        elif a == "clear":
            if not sel:
                raise ValueError("Clear action requires a selector.")
            locator = _resolve_locator(sel)
            if action.get("ensure_visible", True):
                locator.wait_for(state=action.get("ensure_state", "visible"), timeout=timeout)
            locator.fill("", timeout=timeout)
        elif a == "paste":
            if not sel:
                raise ValueError("Paste action requires a selector.")
            locator = _resolve_locator(sel)
            if action.get("ensure_visible", True):
                locator.wait_for(state=action.get("ensure_state", "visible"), timeout=timeout)
            locator.fill(val or "", timeout=timeout)
        elif a == "wait":
            wait_for = action.get("wait_for")
            if wait_for:
                normalized_wait_for = _normalize_selector(wait_for) or wait_for
                page.wait_for_selector(normalized_wait_for, timeout=timeout, state=action.get("wait_state", "visible"))
            else:
                time.sleep(action.get("duration", timeout / 1000))
        elif a == "assert":
            assert_selector = action.get("assert_selector")
            if not assert_selector:
                raise ValueError("Assert action requires 'assert_selector'.")
            normalized_assert = _normalize_selector(assert_selector) or assert_selector
            page.wait_for_selector(normalized_assert, timeout=timeout, state=action.get("wait_state", "visible"))
        elif a == "user_prompt":
            prompt_message = action.get("message") or "Human intervention required. Complete the necessary steps and press Enter."
            print(f"🛑 Human action needed: {prompt_message}")
            input("👉 Press Enter once you've finished the requested action...")
            return True, None
        elif a == "replan":
            print("⚠️ Replanning required.")
            return False, "replan requested"

        if action.get("expect_navigation"):
            page.wait_for_load_state("networkidle")

        wait_for_selector = action.get("wait_for")
        if wait_for_selector:
            wait_state = action.get("wait_state", "visible")
            normalized_wait_for = _normalize_selector(wait_for_selector) or wait_for_selector
            page.wait_for_selector(normalized_wait_for, timeout=timeout, state=wait_state)

        return True, None
    except Exception as e:
        print(f"❌ Error executing {a}: {e}")
        return False, str(e)


def collect_page_context(
    page,
    text_limit: int = 8000,
    max_clickables: int = 80,
    network_events: List[Dict[str, Any]] | None = None,
) -> Dict[str, Any]:
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
    debug_payload: Dict[str, Any] = {}

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
                        "div[role='button']",
                        "div[style*='cursor: pointer']",
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

                    const collectFragments = () => {
                        const fragments = [];
                        const seenFragments = new Set();
                        const candidates = el.querySelectorAll("*");
                        for (const node of candidates) {
                            if (!node || node === el) continue;
                            if (node.children && node.children.length > 0) continue;
                            const raw = (node.innerText || node.textContent || "").replace(/\s+/g, " ").trim();
                            if (!raw) continue;
                            if (raw.length > 80) continue;
                            const cssPath = getCssPath(node);
                            if (!cssPath) continue;
                            const key = raw + "::" + cssPath;
                            if (seenFragments.has(key)) continue;
                            seenFragments.add(key);
                            fragments.push({
                                text: raw.slice(0, 120),
                                css_path: cssPath,
                                tag: node.tagName ? node.tagName.toLowerCase() : null
                            });
                            if (fragments.length >= 10) break;
                        }
                        if (fragments.length) {
                            info.fragments = fragments;
                        }
                    };

                    collectFragments();

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
            base_actionables = interactable_payload.get("all") or []
            categories = interactable_payload.get("categories") or {}
            derived_actionables: List[Dict[str, Any]] = []
            for item in base_actionables:
                fragments = item.get("fragments") or []
                for fragment in fragments:
                    frag_text = (fragment.get("text") or "").strip()
                    frag_css = fragment.get("css_path")
                    if not frag_text or not frag_css:
                        continue
                    derived = dict(item)
                    derived.pop("fragments", None)
                    derived["text"] = frag_text
                    derived["css_path"] = frag_css
                    derived["tag"] = fragment.get("tag") or item.get("tag")
                    base_categories = list(derived.get("categories") or [])
                    if "fragment" not in base_categories:
                        base_categories.append("fragment")
                    derived["categories"] = base_categories
                    derived["derived_from"] = item.get("css_path") or item.get("id") or item.get("tag")
                    derived["is_fragment"] = True
                    derived_actionables.append(derived)
            if len(derived_actionables) > 200:
                derived_actionables = derived_actionables[:200]
            if derived_actionables:
                categories = dict(categories)
                fragment_index = categories.get("fragment", [])
                if not isinstance(fragment_index, list):
                    fragment_index = []
                fragment_index.extend(
                    {
                        "tag": entry.get("tag"),
                        "text": entry.get("text"),
                        "css_path": entry.get("css_path"),
                        "derived_from": entry.get("derived_from"),
                    }
                    for entry in derived_actionables[:150]
                )
                categories["fragment"] = fragment_index
                context["fragment_summary"] = [
                    {
                        "text": item.get("text"),
                        "selector": item.get("css_path"),
                        "tag": item.get("tag"),
                        "source": item.get("derived_from"),
                    }
                    for item in derived_actionables[:20]
                ]
            context["actionables"] = derived_actionables + base_actionables
            context["actionable_categories"] = categories
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

    frames_debug = _gather_frame_data(page)
    if frames_debug:
        debug_payload["frames"] = frames_debug
        context["frames_summary"] = [
            {
                "name": item.get("name"),
                "url": item.get("url"),
                "is_main_frame": item.get("is_main_frame"),
                "is_detached": item.get("is_detached"),
                "text_excerpt": (item.get("text_excerpt") or "")[:200],
            }
            for item in frames_debug
        ]

    shadow_debug = _gather_shadow_dom(page)
    if shadow_debug:
        debug_payload["shadow"] = shadow_debug
        context["shadow_summary"] = [
            {
                "host_tag": item.get("host_tag"),
                "css_path": item.get("css_path"),
                "child_count": item.get("child_count"),
                "markup_excerpt": (item.get("markup_excerpt") or "")[:200],
            }
            for item in shadow_debug
        ]

    cdp_snapshot = _capture_cdp_snapshot(page)
    if cdp_snapshot:
        debug_payload["cdp_dom_snapshot"] = cdp_snapshot
        documents = cdp_snapshot.get("documents") or []
        node_count = 0
        for doc in documents:
            nodes = doc.get("nodes") if isinstance(doc, dict) else {}
            if isinstance(nodes, dict):
                node_count += len(nodes.get("nodeName") or [])
        context["cdp_dom_stats"] = {
            "documents": len(documents),
            "node_count": node_count,
            "strings_count": len(cdp_snapshot.get("strings") or []),
        }

    if network_events:
        trimmed_network = network_events[-NETWORK_EXPORT_LIMIT:]
        debug_payload["network_log"] = trimmed_network
        context["network_summary"] = [
            {
                "type": entry.get("type"),
                "url": (entry.get("url") or "")[:200],
                "method": entry.get("method"),
                "status": entry.get("status"),
                "timestamp": entry.get("timestamp"),
                "resource_type": entry.get("resource_type"),
            }
            for entry in trimmed_network[-10:]
        ]

    canvas_debug = _gather_canvas_data(page)
    if canvas_debug:
        debug_payload["canvas"] = canvas_debug
        context["canvas_summary"] = [
            {
                "index": item.get("index"),
                "width": item.get("width"),
                "height": item.get("height"),
                "bounding_rect": item.get("bounding_rect"),
                "has_data_url": bool(item.get("data_url")),
                "error": item.get("error"),
            }
            for item in canvas_debug
        ]

    _save_dom_snapshots(context, debug_payload=debug_payload)

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
    network_events: List[Dict[str, Any]] = []

    def _trim_network_events() -> None:
        if len(network_events) > NETWORK_LOG_LIMIT:
            del network_events[:-NETWORK_LOG_LIMIT]

    def _record_request(request) -> None:
        try:
            network_events.append(
                {
                    "type": "request",
                    "timestamp": time.time(),
                    "url": request.url,
                    "method": request.method,
                    "resource_type": request.resource_type,
                }
            )
            _trim_network_events()
        except Exception:
            pass

    def _record_response(response) -> None:
        try:
            network_events.append(
                {
                    "type": "response",
                    "timestamp": time.time(),
                    "url": response.url,
                    "status": response.status,
                    "method": response.request.method if response.request else None,
                    "resource_type": response.request.resource_type if response.request else None,
                }
            )
            _trim_network_events()
        except Exception:
            pass

    def _record_request_failed(request) -> None:
        try:
            failure = request.failure
            failure_text = failure.error_text if failure else None
            network_events.append(
                {
                    "type": "requestfailed",
                    "timestamp": time.time(),
                    "url": request.url,
                    "method": request.method,
                    "resource_type": request.resource_type,
                    "failure": failure_text,
                }
            )
            _trim_network_events()
        except Exception:
            pass

    def _maybe_patch_action_with_fragments(
        action: Dict[str, Any], context_snapshot: Dict[str, Any]
    ) -> tuple[Dict[str, Any], Dict[str, Any] | None]:
        """
        If the planner picked a text-based selector, swap in a precomputed fragment selector.
        Returns the potentially modified action plus metadata about the patch.
        """
        try:
            selector = action.get("selector")
        except AttributeError:
            return action, None

        if not selector or not isinstance(selector, str):
            return action, None

        fragments = context_snapshot.get("actionables") or []
        lowered_selector = selector.lower()
        best_fragment = None
        for item in fragments:
            if not isinstance(item, dict):
                continue
            if not item.get("is_fragment"):
                continue
            text = (item.get("text") or "").strip()
            css_path = item.get("css_path")
            if not text or not css_path:
                continue
            if css_path == selector:
                continue
            if text.lower() in lowered_selector:
                best_fragment = item
                break

        if not best_fragment:
            return action, None

        original_selector = selector
        action["selector"] = best_fragment["css_path"]
        if action.get("action") == "click":
            action.setdefault("ensure_visible", True)
            action.setdefault("ensure_state", "visible")

        wait_for = action.get("wait_for")
        if isinstance(wait_for, str) and (best_fragment["text"] or "").strip():
            if best_fragment["text"].strip().lower() in wait_for.lower():
                # Avoid waiting on a string that is already visible before the click.
                action.pop("wait_for", None)

        patch_info = {
            "text": best_fragment.get("text"),
            "selector": best_fragment.get("css_path"),
            "original_selector": original_selector,
        }
        return action, patch_info

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
        context.on("request", _record_request)
        context.on("response", _record_response)
        context.on("requestfailed", _record_request_failed)
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
                page_context_before = collect_page_context(
                    page, network_events=network_events
                )
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

                action_data, fragment_patch = _maybe_patch_action_with_fragments(
                    action_data, page_context_before
                )
                if fragment_patch:
                    print(
                        f"  🔁 Using fragment selector for '{fragment_patch['text']}': {fragment_patch['selector']}"
                    )

                action_data, postcondition_adjustments = _sanitize_postconditions(
                    action_data, page_context_before
                )
                if postcondition_adjustments:
                    removed = ", ".join(
                        f"{key}='{value}'" for key, value in postcondition_adjustments.items()
                    )
                    print(f"  🧹 Removed pre-existing postconditions: {removed}")

                keywords = _extract_action_keywords(step, action_data)
                if keywords:
                    action_data["_keywords"] = keywords

                add_recall({
                    "type": "low_level_decision",
                    "step_id": step_id,
                    "attempt": attempt,
                    "target_goal": action_data.get("target_goal"),
                    "action": action_data.get("action"),
                    "selector": action_data.get("selector"),
                    "value": action_data.get("value"),
                    "patched_selector": fragment_patch,
                    "postcondition_adjustments": postcondition_adjustments,
                    "timestamp": time.time()
                })

                action_success, action_error = execute_action(page, action_data, page_context_before)
                action_data.pop("_keywords", None)

                screenshot_path = f"step_{step_id}_iter{attempt}_{action_data.get('action', 'unknown')}.png"
                page.screenshot(path=screenshot_path, full_page=True)

                page_context_after = collect_page_context(
                    page, network_events=network_events
                )
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
                    "human_prompt": action_data.get("message"),
                    "error": action_error,
                    "patched_selector": fragment_patch,
                    "postcondition_adjustments": postcondition_adjustments
                }

                if not action_success:
                    print("  ⚠️ Action failed; preserving context for replan or human review.")
                    add_recall({
                        "type": "low_level_error",
                        "step_id": step_id,
                        "attempt": attempt,
                        "action": action_data.get("action"),
                        "selector": action_data.get("selector"),
                        "error": action_error,
                        "timestamp": time.time()
                    })
                    execution_summary.append(summary_entry)
                    break

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

                if action_data.get("action") == "replan":
                    print("  ⚠️ Action requested replan. Proceeding to next high-level attempt.")
                    execution_summary.append(summary_entry)
                    break

                time.sleep(1.0)

                execution_summary.append(summary_entry)

            if not step_completed:
                print(f"❗ High-level step {step_id} did not reach its goal within allotted attempts.")

            # Optional overall goal check after each high-level step
            if overall_goal:
                page_context_now = collect_page_context(
                    page, network_events=network_events
                )
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
                f"human_prompt={entry.get('human_prompt') or 'n/a'} "
                f"error={entry.get('error') or 'n/a'}"
            )

        context.close()
