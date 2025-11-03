from openai import OpenAI

# Initialize client once
client = OpenAI()

def generate_navigation_plan(task_description: str) -> str:
    """
    Uses GPT to generate a high-level navigation plan for a given web task.
    Example input: "Create a project in Linear"
    """
    prompt = f"""
    You are the NLP Task Planner (Agent B) in a multi-agent automation system.

    Your job: given a natural-language user request, generate a concise step-by-step
    navigation plan for completing that task on the web.

    You do NOT execute the steps — you only describe the intended flow so a Playwright
    executor can later perform and screenshot each action.

    Each output must include:
    - "url" : the best starting webpage for the task
    - "steps" : 3–10 ordered steps, each describing a clear intent and action

    Each step should explain **what** to do and **why** (intent), but not low-level selectors.
    Focus on high-level actions and observable UI states — including modals or forms
    that may not have URLs.

    Keep the plan generalizable. Assume the executor will re-plan if stuck.

    Return valid JSON only, following this structure:

    {{
    "url": "string",
    "steps": [
        {{
        "intent": "string | purpose of this step (e.g., open projects list)",
        "action": "string | high-level action (navigate / click / type / select / wait / confirm)",
        "description": "string | human-readable summary of what happens (what to look for or fill)"
        }}
    ]
    }}

    ---

    Example 1
    Input: "Book a hotel room in Paris for next weekend."
    Output:
    {{
    "url": "https://www.booking.com",
    "steps": [
        {{"intent": "open booking site", "action": "navigate", "description": "Go to Booking.com homepage"}},
        {{"intent": "search hotels", "action": "type", "description": "Enter 'Paris' as destination"}},
        {{"intent": "set dates", "action": "select", "description": "Choose next Friday to Sunday"}},
        {{"intent": "view results", "action": "click", "description": "Submit search and wait for results"}},
        {{"intent": "sort and pick", "action": "click", "description": "Sort by rating and select top hotel"}}
    ]
    }}

    Example 2
    Input: "Create a project in Linear."
    Output:
    {{
    "url": "https://linear.app",
    "steps": [
        {{"intent": "open projects view", "action": "navigate", "description": "Go to the Projects page"}},
        {{"intent": "start creation", "action": "click", "description": "Click 'Create Project' button"}},
        {{"intent": "fill form", "action": "type", "description": "Enter project name and details"}},
        {{"intent": "submit", "action": "click", "description": "Press 'Create' to finalize"}},
        {{"intent": "confirm success", "action": "wait", "description": "Wait for project to appear in list"}}
    ]
    }}

    Now, generate the plan for this user request:
    "{task_description}"
    """

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are an expert web navigation planner."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.4,
        )

        plan = response.choices[0].message.content
        return plan.strip()

    except Exception as e:
        return f"Error: {e}"

# if __name__ == "__main__":
#     print(generate_navigation_plan("Create a database in Notion"))
