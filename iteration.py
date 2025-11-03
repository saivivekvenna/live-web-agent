import json

from playwright.sync_api import sync_playwright

# EXAMPLE INPUT 
#
# {
#     "url": "https://linear.app",
#     "steps": [
#         {"intent": "open projects view", "action": "navigate", "description": "Go to the Projects page on Linear."},
#         {"intent": "start creation", "action": "click", "description": "Click the 'Create Project' button to initiate project creation."},
#         {"intent": "fill form", "action": "type", "description": "Enter the project name and any relevant details in the provided fields."},
#         {"intent": "submit", "action": "click", "description": "Press the 'Create' button to finalize the project creation."},
#         {"intent": "confirm success", "action": "wait", "description": "Wait for the new project to appear in the project list, indicating successful creation."}
#     ]
# }

def iteration(plan: str):
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)  # show the browser
        page = browser.new_page()
        page.goto(json.loads(plan).get("url"))

        print("Page title:", page.title())

        page.screenshot(path="example.png")


