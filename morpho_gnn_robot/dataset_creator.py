import json
import random

objects = ["red_box", "blue_sphere", "green_cube"]
obstacles = ["wall", "rock", "barrier"]

def generate_sample():
    target = random.choice(objects)
    avoid = random.sample(obstacles, k=random.randint(0,2))

    input_text = f'Task: Go to the {target}\nScene: {{"objects": {objects}, "obstacles": {obstacles}}}'

    plan = {
        "plan": [
            {
                "skill": "navigate",
                "target": target,
                "constraints": {"avoid": avoid}
            }
        ],
        "status": "success"
    }

    return {
        "instruction": "Convert task and scene into robot action plan",
        "input": input_text,
        "output": json.dumps(plan)
    }

dataset = [generate_sample() for _ in range(3000)]

with open("dataset.json", "w") as f:
    json.dump(dataset, f, indent=2)