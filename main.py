from planner import generate_navigation_plan



if __name__ == "__main__":
    task = "Create a project in linear"
    plan = generate_navigation_plan(task)
    print(plan)
