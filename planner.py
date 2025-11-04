# planner.py
from openai import OpenAI
import json, re, textwrap

client = OpenAI()


# ------------------------------------------------------------
# 1. High-level planner: like ChatGPT explaining to a human
# ------------------------------------------------------------
def generate_navigation_plan(task_description: str) -> str:
    """
    Uses GPT to generate a high-level navigation plan for a given web task.
    Example input: "Create a project in Linear"
    """
    prompt_template = """
    You are the NLP Task Planner in a multi-agent automation system.

    Your job: given a natural-language user request, generate a concise step-by-step
    navigation plan for completing that task on the web.

    You do NOT execute the steps — you only describe the intended flow so a Playwright
    executor can later perform and screenshot each action.

    The purpose of the application you are part of is it should automatically navigate the live app and capture screenshots of each UI state in the workflow.
    The user will input simple questions such as : "How do I create a database in Notion" or "How do I create a project in linear". Your response to these type of questions should firstly be a generinic response,. Assume user hasnt opened the application page yet.

    Using your knowledge of how you would normally respond to the given question, you shuold then produce a structured plan that makes the goal explicit at both the task and step level.
    The output must highlight *what success looks like* so that another agent can check progress.

    Rememeber everything you will be outputing should be an "how to guide". Don't be scared to type in/choose fields as examples for the sake of showing how the task is to be done. 

    Each output must include:
    - "task": restate the task in your own words.
    - "solution": solution to the task as a normal ChatGPT response (respond how you would normally respond to these questions)
    - "overall_goal": single sentence describing the desired end state.
    - "url": the best starting webpage for the task.
    - "steps": ordered list where each step contains:
        * "id": integer sequence starting at 1.
        * "intent": short phrase for why this step exists.
        * "action": one of navigate / click / type / select / wait / confirm / search / review / upload.
        * "goal": concrete observable UI or data state that means the step succeeded.
        * "success_criteria": specific DOM/text cues that mean the goal is satisfied.
        * "description": additional guidance (do not mention selectors).
        * "state_capture": describe the UI state to record (e.g., "project modal displayed") even if no unique URL exists.

    Keep the plan high-level but observable. Avoid duplicating the same goal twice in a row.
    Assume an executor can re-plan if progress stalls.

    Return valid JSON only following this structure:

    {
      "task": "string",
      "solution": "string",
      "overall_goal": "string",
      "url": "string",
      "steps": [
        {
          "id": 1,
          "intent": "string",
          "action": "string",
          "goal": "string",
          "success_criteria": "string",
          "description": "string",
          "state_capture": "string"
        }
      ]
    }

    ---

    Example 1
    Input: "Book a hotel room in Paris for next weekend."
    Output:
    {
      "task": "Book a weekend hotel stay in Paris.",
      "solution": "Go to a booking site like Booking.com, set the destination to Paris, choose next weekend dates, adjust guests, then review and confirm a hotel booking that meets your needs.",
      "overall_goal": "Have a Paris hotel selected with dates set for next weekend.",
      "url": "https://www.booking.com",
      "steps": [
        {
          "id": 1,
          "intent": "open booking site",
          "action": "navigate",
          "goal": "Booking.com homepage visible",
          "success_criteria": "Hero search form with destination, dates, and guests inputs is visible",
          "description": "Open the Booking.com homepage in the browser to start the hotel search",
          "state_capture": "Homepage with main search module"
        },
        {
          "id": 2,
          "intent": "configure search",
          "action": "type",
          "goal": "Search form populated with Paris and next weekend dates",
          "success_criteria": "Destination input shows Paris and the date picker highlights the selected range",
          "description": "Enter Paris, set check-in and check-out for next weekend, choose the right number of guests, then start the search",
          "state_capture": "Search form completed before submitting"
        }
      ]
    }

    Now, generate the plan for this user request:
    "<<TASK_DESCRIPTION>>"
    """

    prompt = (
        textwrap.dedent(prompt_template)
        .strip()
        .replace("<<TASK_DESCRIPTION>>", task_description)
    )

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "You are the NLP Task Planner in a multi-agent automation system.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.4,
        )

        plan = response.choices[0].message.content
        return plan.strip()

    except Exception as e:
        return f"Error: {e}"


def _extract_first_json_block(text: str) -> str:
    cleaned = re.sub(r"```(?:json)?|```", "", text, flags=re.IGNORECASE).strip()
    m = re.search(r"\{[\s\S]*\}", cleaned)
    return (m.group(0) if m else cleaned).strip()


def low_level_plan(
    high_level_step: dict, page_context: dict | str, recall_context: list = None
) -> str:
    """
    DOM-aware reasoning planner that *thinks before acting*.
    It reads the DOM, reasons like ChatGPT helping a user, and outputs
    exactly one Playwright-compatible JSON action.
    """
    # Normalize context inputs
    if isinstance(page_context, dict):
        dom_snapshot = page_context.get("dom", "")
        page_url = page_context.get("url", "")
        page_title = page_context.get("title", "")
        rendered_text = page_context.get("rendered_text", "")
        click_map = page_context.get("actionables", [])
        accessibility = page_context.get("accessibility", {})
        meta_summary = page_context.get("meta", {})
    else:
        dom_snapshot = str(page_context)
        page_url = ""
        page_title = ""
        rendered_text = ""
        click_map = []
        accessibility = {}
        meta_summary = {}

    recall_text = ""
    if recall_context:
        try:
            recall_text = "\nRecent context:\n" + json.dumps(
                recall_context[-5:], indent=2
            )
        except Exception:
            recall_text = "\nRecent context: <unserializable>"

    DOM_LIMIT = 35000
    TEXT_LIMIT = 6000
    CLICK_LIMIT = 30
    dom_trunc = dom_snapshot[:DOM_LIMIT]

    rendered_excerpt = (rendered_text or "")[:TEXT_LIMIT]

    def _serialize(obj, limit=None):
        if not obj:
            return ""
        try:
            payload = json.dumps(obj, indent=2)
            if limit and len(payload) > limit:
                return payload[:limit] + "... [truncated]"
            return payload
        except Exception:
            return str(obj)[: (limit or 800)]

    click_summary = ""
    if click_map:
        click_summary = _serialize(click_map[:CLICK_LIMIT], limit=5000)

    accessibility_summary = ""
    if accessibility:
        accessibility_summary = _serialize(accessibility, limit=2000)

    meta_text = ""
    if meta_summary:
        meta_text = _serialize(meta_summary, limit=1200)

    step_goal = high_level_step.get("goal", "")
    step_state_capture = high_level_step.get("state_capture", "")
    success_criteria = high_level_step.get("success_criteria", "")

    prompt = f"""
    You are a DOM Reasoning Agent for an autonomous web system.

    You are looking at a live DOM and must decide **the single next action** to move closer to the user goal.

    Think like a normal ChatGPT assistant helping a human navigate a website:
    - Look at what is actually visible in the DOM.
    - If a required element (like a form field or button) isn’t visible, reason what the *likely prerequisite* is and proceed with the new action.
    (e.g., "You must sign in first" → click “Sign in” or “Sign up” link).
    - Only choose one concrete action at a time.
    - Before deciding, mentally rank up to three actionable candidates whose text or aria-label align with the goal keywords (e.g., New/Create/Add for creation tasks) and choose the highest-confidence match.
    - Prefer the most reliable and obvious next UI step.
    - Ensure the selector you output is something Playwright can resolve (CSS, role, or text) and use `nth` when you need a specific instance.
    - Refer to the "Known actionable elements" and "Visible text blocks" inventories when choosing selectors. Reuse the exact attributes shown there (text, aria-label, role, css_path) instead of inventing new ones.
    - When using attribute selectors, wrap the value in double quotes (e.g., `[aria-label="Workspace"]`) so embedded apostrophes do not break the CSS.
    - If a `css_path` is supplied for the element you want, prefer using that path (optionally with `nth`) because it maps directly to the live DOM.
    - Favor Playwright text selectors such as `text="Database"` or `div:has-text("Database")` when only text is available and no role/aria attribute is present.
    - Only include `wait_for` or `assert_selector` when you can cite a DOM cue (from the current snapshot or stated success criteria) that should appear afterward; otherwise leave them out.
    - Before finalizing the JSON, double-check that every attribute you cite (role, aria-label, etc.) is explicitly present in the DOM excerpt or actionable inventory.
    - When the main canvas already shows a creation surface (e.g., a heading like "New page" with quick-start chips such as Database/Form/Templates), treat that as progress and interact with the relevant quick-start control instead of re-triggering navigation.
    - If a dropdown menu listing creation options (Page, Database, Templates, etc.) is open, select the matching item directly (e.g., `div[role='menu'] >> text='Database'`) or dismiss it with Escape before acting elsewhere.

    Do not output any explanation outside JSON. Output only a valid JSON object matching the schema below.

    If the action is "navigate" or "open", include a "url" field instead of "selector".
    Example:
    {{
    "action": "navigate",
    "url": "https://linear.app",
    "expect_navigation": true,
    "wait_for": "nav, header, main",
    "confidence": 1.0,
    "reasoning": "User needs to access Linear; navigate directly to the homepage."
    }}


    ---

    Step goal: {step_goal or "No goal provided"}
    Success criteria: {success_criteria or "No explicit success criteria"}
    Desired captured state: {step_state_capture or "No specific capture guidance"}
    Current URL: {page_url or "Unknown"}
    Page title: {page_title or "Unknown"}

    Rendered text excerpt:
    {rendered_excerpt or "<empty>"}

    Known actionable elements (subset):
    {click_summary or "<none detected>"}

    Accessibility snapshot (subset):
    {accessibility_summary or "<unavailable>"}

    Page metadata (head/meta tags):
    {meta_text or "<none>"} 

    Schema:
    {{
    "action": "click|type|press|select|hover|check|uncheck|upload|scroll|wait|assert|navigate|focus|clear|paste|user_prompt|replan",
    "selector": "string | CSS or text/role selector (if applicable)",
    "value": "string | optional input or key",
    "key": "string | key to press (for press actions)",
    "wait_for": "string | selector expected to appear after the action",
    "wait_state": "string | optional Playwright wait state (visible|attached|detached|hidden)",
    "assert_selector": "string | selector that confirms success",
    "expect_navigation": true/false,
    "timeout": 8000,
    "force": true/false,
    "nth": 0,
    "button": "left|right|middle",
    "click_count": 1,
    "modifiers": ["Shift","Meta"],
    "ensure_visible": true/false,
    "target_goal": "string | the immediate observable result this action aims to achieve",
    "message": "string | instruction for human intervention when action=user_prompt",
    "confidence": 0.0-1.0,
    "reasoning": "concise human-readable summary of why this is the best action"
    }}

    ---

    Rules:
    1. **Observe first** – read the DOM snippet carefully before deciding.
    2. **Handle prerequisites** – surface login/signup flows when required, and output `user_prompt` with clear instructions for the human to complete authentication.
    3. **Choose stability** – prefer visible text, aria-label, role names, or data-testid attributes. When text is missing (icon buttons), lean on `[role=...]`, `[aria-label=...]`, or `aria-labelledby` values instead of `:has-text`.
    4. **Match intent** – if the goal expects creation or addition, favor elements whose text or aria-label includes words like "New", "Create", "Add", or "Database". Quote the matched attribute in the reasoning.
    5. **Ground selectors** – cite the exact element text or attributes you see (e.g., `text="Database"`). Do not fabricate aria labels or IDs that are absent from the actionable list or DOM excerpt.
    6. **Quick starts** – when the content area exposes quick-start buttons or chips whose labels match the goal (e.g., Database, Form, Template), click the relevant control using a direct text selector or the provided css_path rather than repeating sidebar clicks.
    7. **Handle overlays** – when an overlay or dropdown intercepts pointer events, either choose an option inside it or send `{{"action":"press","key":"Escape"}}` before retrying. Reserve `"force": true` for cases where dismissal is impossible.
    8. **Avoid repetition** – if the previous attempt already executed the same selector without progressing, choose a different element or escalate via `press`/`replan`.
    9. **If you cannot find a good move**, output `{{"action":"replan","reasoning":"No actionable element visible","confidence":0.0}}` and explain what is missing.

    ---

    Example 1:
    High-level step:
    {{"intent":"fill email","action":"type","description":"Enter email to log in"}}
    DOM:
    <form><button>Sign in</button></form>

    Output:
    {{
    "action": "click",
    "selector": "button:has-text('Sign in')",
    "expect_navigation": true,
    "wait_for": "input[type='email']",
    "confidence": 0.9,
    "reasoning": "Email field not visible yet; must first open sign-in form by clicking 'Sign in' button."
    }}

    Example 2:
    High-level step:
    {{"intent":"enter email","action":"type","description":"Fill in email address"}}
    DOM:
    <input type="email" placeholder="Enter your email">

    Output:
    {{
    "action": "type",
    "selector": "input[type='email']",
    "value": "robotwebagenttester@gmail.com",
    "assert_selector": "input[value='robotwebagenttester@gmail.com']",
    "confidence": 1.0,
    "reasoning": "Email input field visible; ready to type email."
    }}

    ---

    Now decide the best action for this situation:

    High-level step:
    {json.dumps(high_level_step, indent=2)}

    {recall_text}

    DOM Snapshot (truncated):
    {dom_trunc}
    """

    for _ in range(2):
        try:
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a DOM reasoning and automation expert. Output pure JSON.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
            )
            raw = resp.choices[0].message.content or "{}"
            text = _extract_first_json_block(raw)
            json.loads(text)  # validate JSON
            return text
        except Exception:
            pass

    return json.dumps(
        {
            "action": "replan",
            "reasoning": "Low-level planner failed twice to produce valid JSON.",
            "confidence": 0.0,
        },
        indent=2,
    )


def evaluate_goal_completion(
    goal: str,
    success_criteria: str,
    dom_snapshot: str,
    recall_context: list = None,
    page_url: str = "",
    page_title: str = "",
    rendered_text: str = "",
    actionables: list = None,
    meta: dict = None,
) -> dict:
    """
    Uses an evaluation agent to determine whether the provided DOM snapshot
    satisfies the given goal and success criteria.
    """
    if not goal:
        return {
            "status": "unknown",
            "confidence": 0.0,
            "feedback": "No goal provided for evaluation.",
            "evidence": "",
        }

    recall_text = ""
    if recall_context:
        try:
            recall_text = "\nRecent context:\n" + json.dumps(
                recall_context[-5:], indent=2
            )
        except Exception:
            recall_text = "\nRecent context: <unserializable>"

    DOM_LIMIT = 15000
    TEXT_LIMIT = 4000
    CLICK_LIMIT = 12
    dom_trunc = dom_snapshot[:DOM_LIMIT]
    rendered_excerpt = (rendered_text or "")[:TEXT_LIMIT]

    def _serialize(obj, limit=None):
        if not obj:
            return ""
        try:
            payload = json.dumps(obj, indent=2)
            if limit and len(payload) > limit:
                return payload[:limit] + "... [truncated]"
            return payload
        except Exception:
            return str(obj)[: (limit or 800)]

    actionables_summary = _serialize((actionables or [])[:CLICK_LIMIT], limit=2000)
    meta_summary = _serialize(meta, limit=1200)

    prompt = f"""
    You are the Goal Validation Agent in an autonomous web navigator.

    Your job:
    - Read the DOM snapshot and determine whether the stated goal has been achieved.
    - Compare against the success criteria whenever provided.
    - Respond strictly in JSON so another agent can parse your output.

    Guidelines:
    - Base your decision strictly on the DOM snippet provided and the current URL.
    - Do not claim the page is still loading solely because scripts or network references exist.
    - If the DOM already contains the UI elements described in the goal or success criteria, mark the status as "achieved" and cite them.
    - Only respond "not_yet" when the DOM is missing the required elements or clearly shows a blocking state (e.g., error, loader, login wall).
    - Many intermediate UI states (modals, dropdowns, inline forms) do not have unique URLs; evaluate them based on visible structure and text.
    - Treat a visible creation surface (e.g., a heading such as "New page" accompanied by quick-start chips like Database/Form/Templates) as satisfying steps that require opening the command menu.
    - If a dropdown listing creation options (Page, Database, Templates, etc.) is visible, consider the menu open and the selection step complete once the desired option appears.
    - Use direct evidence from the DOM excerpt whenever possible.

    Definitions:
    - status: "achieved" if the DOM clearly satisfies the goal, "not_yet" if more actions are required,
      or "failed" if the current state conflicts with the goal.
    - confidence: float 0.0-1.0 indicating how sure you are.
    - feedback: concise explanation (max 2 sentences) summarizing evidence for your status.
    - evidence: direct DOM strings or cues that support your decision (keep short).
    - next_hint: optional suggestion for what is missing (omit if status is achieved).

    Goal: {goal}
    Success criteria: {success_criteria or "Not specified"}
    Current URL: {page_url or "Unknown"}
    Page title: {page_title or "Unknown"}
    {recall_text}

    Rendered text excerpt:
    {rendered_excerpt or "<empty>"}

    Known actionable elements (subset):
    {actionables_summary or "<none detected>"}

    Page metadata (head/meta tags):
    {meta_summary or "<none>"} 

    DOM Snapshot (truncated):
    {dom_trunc}

    Output JSON exactly:
    {{
      "status": "achieved|not_yet|failed",
      "confidence": 0.0,
      "feedback": "string",
      "evidence": "string",
      "next_hint": "string | optional"
    }}
    """

    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "You evaluate goal completion for an autonomous agent. Output strict JSON.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
        )
        raw = resp.choices[0].message.content or "{}"
        text = _extract_first_json_block(raw)
        return json.loads(text)
    except Exception as e:
        return {
            "status": "unknown",
            "confidence": 0.0,
            "feedback": f"Goal evaluator error: {e}",
            "evidence": "",
        }
