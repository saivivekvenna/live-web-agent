import os
import openai
from dotenv import load_dotenv
from playwright.sync_api import sync_playwright


# Load .env variables
load_dotenv()

# Set API key
openai.api_key = os.getenv("OPENAI_API_KEY")

def generate_navigation_plan(user_request: str) -> dict:
    """
    Takes a natural language request and outputs:
      1. A procedural step-by-step plan
      2. A general starting URL to open
    """

    prompt = f"""
    You are an intelligent web navigation planner. Your job is to provide a step by step procedure on how to navigate a given page
    Given a user request, output two fields in JSON:
      - "steps": an ordered list of instructions describing how to achieve the goal on the web.
      - "url": the most relevant general webpage to start from.

    User request: "{user_request}"

    Example:
    Input: "Book a hotel room in Paris for next weekend."
    Output:
    {{
      "steps": [
        "Open https://www.booking.com",
        "Search for hotels in Paris",
        "Set check-in to next Friday and check-out to Sunday",
        "Sort by price or rating",
        "Select the top-rated option and note the booking link"
      ],
      "url": "https://www.booking.com"
    }}
    """

    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.4
    )

    # Parse response safely
    try:
        text = response["choices"][0]["message"]["content"].strip()
        plan = eval(text) if text.startswith("{") else {"steps": [], "url": ""}
    except Exception as e:
        plan = {"steps": [], "url": "", "error": str(e)}

    return plan

print(generate_navigation_plan("Create a project in linear")
)