from planner import generate_navigation_plan
from iteration import iteration


if __name__ == "__main__":
    task = "Create a project in linear"
    plan = generate_navigation_plan(task)
    print(iteration(plan))
    