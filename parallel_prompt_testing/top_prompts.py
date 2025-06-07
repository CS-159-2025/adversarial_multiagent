import re
import pandas as pd
from pathlib import Path

# ---- Input files ----
input_files = {
    "subjective": "initial_results/subjective_output.txt",
    "creative": "initial_results/creative_output.txt",
    "planning": "initial_results/planning_output.txt",
    "factual": "initial_results/factual_output.txt"
}

# ---- Regex pattern to extract summary table ----
summary_pattern = r"AVERAGED SUMMARY.*?\n((?:.|\n)*?)\n═"  # Grab text block between summary header and divider

all_summaries = []

# ---- Extract tables from each file ----
for prompt_type, file_path in input_files.items():
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    match = re.search(summary_pattern, content)
    if not match:
        print(f"No summary found in {file_path}")
        continue

    table_text = match.group(1).strip()
    lines = table_text.splitlines()

    # Extract header
    header = re.split(r'\s{2,}', lines[0].strip())
    rows = [re.split(r'\s{2,}', line.strip(), maxsplit=len(header)-1) for line in lines[1:]]

    # Build dataframe
    df = pd.DataFrame(rows, columns=header)
    df["prompt_type"] = prompt_type

    # Convert numeric columns
    for col in df.columns:
        if col not in {"prompt", "prompt_type"}:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Calculate drop
    df["drop"] = df["initial_overall"] - df["revised_overall"]
    all_summaries.append(df)

# ---- Combine all summaries ----
summary_df = pd.concat(all_summaries, ignore_index=True)

# ---- Remove duplicates by averaging over prompt + type ----
grouped_df = (
    summary_df
    .groupby(["prompt_type", "prompt"], as_index=False)
    .mean(numeric_only=True)
)

# ---- Get Top 3 per Category ----
top3_per_category = (
    grouped_df.sort_values(["prompt_type", "drop"], ascending=[True, False])
    .groupby("prompt_type")
    .head(3)
    .reset_index(drop=True)
)

# ---- Get Top 10 Overall ----
top10_overall = grouped_df.sort_values("drop", ascending=False).head(10)

# ---- Save Results ----
top3_per_category.to_csv("top_3_per_type_avg.tsv", sep="\t", index=False)
top10_overall.to_csv("top_10_overall_avg.tsv", sep="\t", index=False)

print("✅ Results saved:")
print(" - top_3_per_type_avg.tsv")
print(" - top_10_overall_avg.tsv")
