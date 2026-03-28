"""
dataset_creator.py - Fixed version
Generates synthetic robot task planning data for fine-tuning Qwen2.5-7B-Instruct.

Fixes from original:
  - Bug: objects/obstacles used outer scope lists inside generate_sample(), so
    every sample's scene was identical. Now each sample gets a randomized scene.
  - Added skill diversity: navigate, inspect, pick_up, wait, report
  - Added more objects, obstacles, and spatial relations for variety
  - Output is now in Qwen2.5 ChatML format (required for instruct fine-tuning)
  - Added multi-step plans (not just single-skill plans)
"""

import json
import random

ALL_OBJECTS = [
    "red_box", "blue_sphere", "green_cube", "yellow_cylinder",
    "orange_cone", "white_barrel", "purple_crate", "silver_panel",
    "charging_station", "target_marker"
]

ALL_OBSTACLES = [
    "wall", "rock", "barrier", "fence",
    "pit", "narrow_gap", "steep_slope", "debris_pile"
]

SKILLS = ["navigate", "inspect", "pick_up", "wait_at", "report_position"]

SYSTEM_PROMPT = (
    "You are a robot task planner. Given a natural language task and a scene "
    "description, produce a structured JSON action plan. The plan must be a list "
    "of skill steps the robot will execute in order. Each step must have: skill "
    "(string), target (string), constraints (dict with 'avoid' list). Always "
    "respond with valid JSON only. No explanation, no markdown."
)


def make_scene():
    """Generate a random but consistent scene dict."""
    n_objects = random.randint(2, 5)
    n_obstacles = random.randint(0, 3)
    scene_objects = random.sample(ALL_OBJECTS, k=n_objects)
    scene_obstacles = random.sample(ALL_OBSTACLES, k=n_obstacles)
    return scene_objects, scene_obstacles


def generate_single_step_sample(scene_objects, scene_obstacles):
    """One task -> one skill step."""
    target = random.choice(scene_objects)
    avoid = random.sample(scene_obstacles, k=min(len(scene_obstacles), random.randint(0, 2)))
    skill = random.choice(SKILLS)

    task_templates = [
        f"Go to the {target}.",
        f"Move towards the {target} and {skill.replace('_', ' ')} it.",
        f"Your goal is the {target}. Avoid {', '.join(avoid) if avoid else 'nothing'}.",
        f"Execute {skill.replace('_', ' ')} on {target}.",
    ]

    task = random.choice(task_templates)
    scene_dict = {"objects": scene_objects, "obstacles": scene_obstacles}

    user_msg = f"Task: {task}\nScene: {json.dumps(scene_dict)}"

    plan = {
        "plan": [
            {
                "skill": skill,
                "target": target,
                "constraints": {"avoid": avoid}
            }
        ],
        "status": "success"
    }

    return user_msg, json.dumps(plan)


def generate_multi_step_sample(scene_objects, scene_obstacles):
    """Compound task -> two or three skill steps."""
    if len(scene_objects) < 2:
        return generate_single_step_sample(scene_objects, scene_obstacles)

    steps_count = random.randint(2, min(3, len(scene_objects)))
    targets = random.sample(scene_objects, k=steps_count)
    skills_chosen = [random.choice(SKILLS) for _ in range(steps_count)]
    avoid_global = random.sample(scene_obstacles, k=min(len(scene_obstacles), 2))

    task = (
        f"First {skills_chosen[0].replace('_', ' ')} the {targets[0]}, "
        f"then {skills_chosen[1].replace('_', ' ')} the {targets[1]}"
        + (f", finally {skills_chosen[2].replace('_', ' ')} the {targets[2]}" if steps_count == 3 else "")
        + f". Avoid {', '.join(avoid_global) if avoid_global else 'nothing'}."
    )

    scene_dict = {"objects": scene_objects, "obstacles": scene_obstacles}
    user_msg = f"Task: {task}\nScene: {json.dumps(scene_dict)}"

    plan = {
        "plan": [
            {
                "skill": skills_chosen[i],
                "target": targets[i],
                "constraints": {"avoid": avoid_global}
            }
            for i in range(steps_count)
        ],
        "status": "success"
    }

    return user_msg, json.dumps(plan)


def generate_sample():
    """
    Returns a dict in Qwen2.5 ChatML format suitable for SFTTrainer
    with a 'text' field containing the full formatted conversation.
    """
    scene_objects, scene_obstacles = make_scene()

    if random.random() < 0.4:
        user_msg, output = generate_multi_step_sample(scene_objects, scene_obstacles)
    else:
        user_msg, output = generate_single_step_sample(scene_objects, scene_obstacles)

    # Qwen2.5 ChatML format
    text = (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{user_msg}<|im_end|>\n"
        f"<|im_start|>assistant\n{output}<|im_end|>"
    )

    return {"text": text}


if __name__ == "__main__":
    random.seed(42)
    n = 10000
    dataset = [generate_sample() for _ in range(n)]

    with open("dataset.json", "w") as f:
        json.dump(dataset, f, indent=2)

    print(f"Generated {n} samples.")
    print("\n--- Example sample (text field) ---")
    print(dataset[0]["text"])
