# main.py
import json
from planner import generate_navigation_plan
from executor import iteration


if __name__ == "__main__":
    # 👇 You can change this task dynamically to test anything
    task = "create a database in notion"

    print(f"\n🧠 Generating high-level plan for: '{task}' ...\n")
    plan = generate_navigation_plan(task)

    # -------------------------------
    # Validate the plan before executing
    # -------------------------------
    try:
        plan_json = json.loads(plan)
    except json.JSONDecodeError:
        print("❌ Planner did not return valid JSON. Here’s what it output:\n")
        print(plan)
        exit(1)

    print("\n--- 🗺️  HIGH LEVEL PLAN ---\n")
    print(json.dumps(plan_json, indent=2))

    if plan_json.get("overall_goal"):
        print(f"\n🎯 Overall goal: {plan_json['overall_goal']}\n")

    # -------------------------------
    # Execute the plan in Playwright
    # -------------------------------
    print("\n🚀 Beginning execution...\n")
    iteration(json.dumps(plan_json))

    print("\n✅ Task execution finished.\n")