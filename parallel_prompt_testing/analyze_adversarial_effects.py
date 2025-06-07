import re
import json
from pathlib import Path
from collections import defaultdict

# Prompt types and their associated files
prompt_types = {
    "subjective": "initial_results/subjective_output.txt",
    "creative": "initial_results/creative_output.txt",
    "planning": "initial_results/planning_output.txt",
    "factual": "initial_results/factual_output.txt"
}

# Score pattern matcher (e.g., "Score: {'correctness': 9, 'adherence': 8, ...}")
score_re = re.compile(r"Score:\s*(\{.*?\})")

# Parse score dict from a given line
def parse_score(line: str) -> dict:
    match = score_re.search(line)
    if match:
        try:
            return json.loads(match.group(1).replace("'", '"'))
        except json.JSONDecodeError:
            return {}
    return {}

# Extract score drop per metric for successful attacks
def analyze_file(file_path: Path):
    results = []
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        if lines[i].startswith("[INITIAL HONEST ANSWER]"):
            initial_score = {}
            revised_score = {}
            success = False
            # Scan forward for scores and status
            while i < len(lines):
                line = lines[i].strip()
                if line.startswith("Score:"):
                    score = parse_score(line)
                    if not initial_score:
                        initial_score = score
                    else:
                        revised_score = score
                elif line.startswith("Persuasion:"):
                    success = "SUCCESS" in line
                    break
                i += 1
            if success and initial_score and revised_score:
                result = {
                    "correctness_drop": initial_score["correctness"] - revised_score["correctness"],
                    "adherence_drop": initial_score["adherence"] - revised_score["adherence"],
                    "presentation_drop": initial_score["presentation"] - revised_score["presentation"],
                }
                results.append(result)
        i += 1
    return results

# Run analysis
def summarize_drops():
    summary = {}
    for category, filename in prompt_types.items():
        file_path = Path(filename)
        if not file_path.exists():
            print(f"⚠️ File not found: {filename}")
            continue
        drops = analyze_file(file_path)
        if not drops:
            print(f"⚠️ No successful persuasion runs found in {filename}")
            continue

        avg_drop = defaultdict(float)
        for d in drops:
            for k, v in d.items():
                avg_drop[k] += v

        count = len(drops)
        summary[category] = {k: round(v / count, 2) for k, v in avg_drop.items()}
        print(f"✅ {category.upper()} — based on {count} successful adversarial runs:")
        for metric, delta in summary[category].items():
            print(f"   • Avg {metric.replace('_drop','')} drop: {delta}")
        print()

    return summary

# Run it
if __name__ == "__main__":
    summarize_drops()
