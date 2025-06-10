import pandas as pd
import re
import os

def parse_summary_table(file_path):
    if not os.path.exists(file_path):
        return pd.DataFrame()

    with open(file_path, "r") as f:
        content = f.read()

    match = re.search(r"AVERAGED SUMMARY.*?\n(-+\n)?(.*?)(?=\n\S|$)", content, re.DOTALL)
    if not match:
        return pd.DataFrame()

    table_text = match.group(2).strip()
    lines = [line for line in table_text.split("\n") if line.strip()]
    if len(lines) < 2:
        return pd.DataFrame()

    headers = re.split(r"\s{2,}", lines[0].strip())
    rows = []
    for line in lines[1:]:
        if re.match(r"[-=]+", line):
            continue
        parts = re.split(r"\s{2,}", line.strip(), maxsplit=len(headers)-1)
        if len(parts) != len(headers):
            continue
        rows.append(parts)

    df = pd.DataFrame(rows, columns=headers)

    for col in df.columns:
        if col.startswith("initial_") or col.startswith("revised_") or col == "avg_persuasion_success":
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["delta_correctness"] = df["revised_correctness"] - df["initial_correctness"]
    df["delta_adherence"] = df["revised_adherence"] - df["initial_adherence"]
    df["delta_presentation"] = df["revised_presentation"] - df["initial_presentation"]
    df["delta_overall"] = df["revised_overall"] - df["initial_overall"]

    return df

categories = ["factual", "creative", "subjective", "planning"]
sources = {
    "primary": "other_results/",
    "redo": "redo_initial_results/",
    "secondary": "secondary_results/"
}

combined_results = {}
averages = {}

for cat in categories:
    all_dfs = []
    for source_path in sources.values():
        file_path = f"{source_path}{cat}_output.txt"
        df = parse_summary_table(file_path)
        if not df.empty:
            df["category"] = cat
            all_dfs.append(df)

    if all_dfs:
        merged_df = pd.concat(all_dfs, ignore_index=True)
        top5 = merged_df.sort_values("avg_persuasion_success", ascending=False).head(8)
        combined_results[cat] = top5

        averages[cat] = {
            "persuasion_success": top5["avg_persuasion_success"].mean(),
            "delta_correctness": top5["delta_correctness"].mean(),
            "delta_adherence": top5["delta_adherence"].mean(),
            "delta_presentation": top5["delta_presentation"].mean(),
            "delta_overall": top5["delta_overall"].mean()
        }

# Write to file
with open("top_prompts_summary.txt", "w") as f:
    for cat in categories:
        f.write(f"\n===== CATEGORY: {cat.upper()} =====\n")
        if cat in combined_results:
            for _, row in combined_results[cat].iterrows():
                f.write(f"Prompt: {row['prompt']}\n")
                f.write(f"Persuasion Success: {row['avg_persuasion_success']:.2f}\n")
                f.write(f"Δ Correctness: {row['delta_correctness']:.2f}, "
                        f"Δ Adherence: {row['delta_adherence']:.2f}, "
                        f"Δ Presentation: {row['delta_presentation']:.2f}, "
                        f"Δ Overall: {row['delta_overall']:.2f}\n\n")

            avg = averages[cat]
            f.write(f"Average of Top 5 — Success: {avg['persuasion_success']:.2f}, "
                    f"Δ Correctness: {avg['delta_correctness']:.2f}, "
                    f"Δ Adherence: {avg['delta_adherence']:.2f}, "
                    f"Δ Presentation: {avg['delta_presentation']:.2f}, "
                    f"Δ Overall: {avg['delta_overall']:.2f}\n")
        else:
            f.write("No data found.\n")

print("Summary with top 5 prompts (combined sources) saved to top_prompts_summary.txt")
