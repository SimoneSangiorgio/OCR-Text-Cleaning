from pathlib import Path

# -----------

base_path = Path(__file__).resolve().parent

# -----------

dataset_path = base_path / "dataset"
dataset_subset = dataset_path / "the_vampyre_subset.json"

# -----------

cleaned_file = base_path / "clean_judge_files"

cleaning_results = cleaned_file / "cleaning_results.json"
judging_results = cleaned_file / "judging_results.json"



results = base_path / "results"

full_pipeline_results = results / "full_pipeline_results.json"
