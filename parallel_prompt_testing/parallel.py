import multiprocessing
import subprocess

def launch_process(prompt_file: str, output_file: str):
    subprocess.run(["python", "run_experiment.py", prompt_file, output_file])

if __name__ == "__main__":
    prompt_files = [
        "subjective.json",
        "creative.json",
        "planning.json",
        "factual.json",
    ]
    output_files = [
        "subjective_output.txt",
        "creative_output.txt",
        "planning_output.txt",
        "factual_output.txt",
    ]

    procs = []
    for pfile, ofile in zip(prompt_files, output_files):
        p = multiprocessing.Process(target=launch_process, args=(pfile, ofile))
        p.start()
        procs.append(p)

    for p in procs:
        p.join()