# LLM-Powered OCR Post-Correction and Evaluation Pipeline

This project implements a pipeline for cleaning and evaluating text extracted via Optical Character Recognition (OCR), primarily focusing on historical texts like "Le avventure di Pinocchio" and "The Vampyre". It leverages various LLMs (Google Gemini, Groq-hosted models like Mistral) for the cleaning task, employs quantitative metrics for evaluation, and uses LLMs (Gemini, Prometheus) as automated judges of cleaning quality.

## Project Overview

The core goal is to take imperfect OCR output, automatically correct it using LLMs, and then rigorously assess the quality of these corrections. The pipeline is designed to be modular, allowing for different cleaning models, judging mechanisms, and evaluation steps.

## Key Features

*   **Data Handling:** Loads and processes paired OCR/clean text datasets (JSON format).
*   **Pre-cleaning:** Applies rule-based and dictionary-based text normalization before LLM processing.
*   **LLM-based Cleaning:**
    *   Supports multiple LLM providers (Google Gemini, Groq).
    *   Employs a **two-stage cleaning process**:
        1.  Initial character/word-level correction (e.g., fixing `Kon` to `Non`, `Cilirgìa` to `Ciliegia`, joining hyphenated words).
        2.  Secondary paragraph structure, style, and flow correction (e.g., merging lines into paragraphs, correcting chapter numbers, removing noise).
    *   Utilizes detailed and specific prompts tailored for each stage and LLM.
*   **Quantitative Evaluation:** Calculates standard NLP metrics:
    *   Word Error Rate (WER)
    *   Character Error Rate (CER)
    *   ROUGE scores (for n-gram overlap, recall, and precision)
*   **LLM-based Judging:**
    *   Uses LLMs (Gemini, Prometheus) to provide a qualitative score (0-5 or 1-5) on the cleaning quality.
    *   Prometheus judging uses a custom, detailed rubric for OCR quality.
*   **Judge Validation:** Includes a script to calculate Cohen's Kappa score to assess the agreement between human-assigned scores and LLM judge scores, helping to validate the automated judge's reliability.
*   **Configuration:** Manages API keys via `.env` files and file paths (likely through `pathinator.py`).
*   **Detailed Output:** Generates comprehensive JSON files containing original OCR, ground truth, cleaned text from each model, all metrics, judge scores, and detailed word-level differences.

## Project Workflow

The main workflow, orchestrated by `main_V2.py`, is as follows:

1.  **Initialization:**
    *   Load API keys (Google, Groq) from `.env`.
    *   Initialize LLM clients (Gemini, Groq) and the ROUGE metric evaluator.
2.  **Data Loading:**
    *   Load the dataset (e.g., `the_vampyre_subset.json`) specified by `INPUT_PATH` in `main_V2.py`. This subset is created by `dataset_handler.py`.
3.  **Iterate Through Dataset Items:** For each OCR/clean pair:
    *   **Pre-cleaning (`pre_clean.py`):** Apply initial rule-based replacements (e.g., `replacement_rules`) to the OCR text. Spelling correction (`correct_spelling`) can also be part of this stage.
    *   **Iterate Through Configured LLM Cleaners (`MODELS_TO_RUN` in `main_V2.py`):**
        *   **Stage 1 Cleaning (Character/Word Level):** Pass the pre-cleaned OCR text to the first LLM cleaning function (e.g., `clean_with_gemini` or `clean_with_groq` from `cleaner_LLM_V2.py`).
        *   **Stage 2 Cleaning (Paragraph/Style Level):** Pass the output of Stage 1 to the corresponding "corrector" function (e.g., `corrector_gemini` or `corrector_groq` from `cleaner_LLM_V2.py`).
        *   **(Optional) Post-Pre-cleaning:** `replacement_rules` can be applied again.
    *   **Quantitative Evaluation (`main_V2.py`):**
        *   Calculate WER and CER (using `jiwer`) by comparing the LLM's final cleaned text against the ground truth.
        *   Calculate ROUGE scores (using `evaluate.load("rouge")`).
    *   **LLM-based Judging (`judge_LLM.py`):**
        *   The `judge_with_gemini` function (or potentially `judge_with_prometheus`) evaluates the LLM's cleaned text against the ground truth, producing a quality score.
    *   **Difference Analysis (`main_V2.py`):**
        *   `get_detailed_diffs` function provides a word-by-word comparison highlighting differences between the ground truth and the model's cleaned output.
    *   **Aggregate Results:** Store all data (original OCR, ground truth, cleaned text, metrics, judge scores, differences) for the current item and model.
4.  **Save Results:**
    *   Write all aggregated results to a JSON file (e.g., `full_pipeline_results_{START_INDEX}_{END_INDEX}.json`).

## Core Scripts Explained

*   **`dataset_handler.py`:**
    *   Loads raw OCR data and corresponding "clean" (ground truth) data from separate JSON files.
    *   Merges these into a `unified_dict` where each key maps to an "ocr" and "clean" text pair.
    *   Creates a smaller `final_dict` (subset) for focused processing or testing.
    *   Saves this subset to a new JSON file (e.g., `the_vampyre_subset.json`).

*   **`pre_clean.py`:**
    *   Contains functions for rule-based text cleaning and spelling correction.
    *   `replacement_rules()`: Applies a dictionary of find-and-replace rules (e.g., fixing hyphenation, standardizing quotes).
    *   `correct_spelling()`: Uses `pyspellchecker` with a custom Italian dictionary (`index.dic`, loaded via `pathinator.py`) to correct misspelled words, while attempting to preserve proper nouns and original archaic spellings.
    *   `clean_symbols()`, `fix_ocr_specific_errors()`: Other utility functions for pre-cleaning.

*   **`cleaner_LLM_V2.py`:**
    *   Houses the LLM-based text cleaning logic.
    *   `clean_with_gemini(client, ocr_text)`: Sends OCR text to a Gemini model with a detailed prompt focused on character-level corrections, word rejoining, and artifact removal, while strictly preserving line breaks and original style.
    *   `corrector_gemini(client, ocr_text)`: Takes the output from `clean_with_gemini` and uses another Gemini call with a prompt focused on paragraph reconstruction, chapter number correction, and removing residual noise/didascalia, again with strict rules to preserve original orthography.
    *   `clean_with_groq(client, ocr_text, model_id)`: Similar to `clean_with_gemini` but for Groq-hosted models, using a chat-based prompt.
    *   `corrector_groq(client, ocr_text, model_id)`: Similar to `corrector_gemini` but for Groq-hosted models.
    *   The prompts are highly specific, providing examples and "inviolable rules" to guide the LLMs.

*   **`judge_LLM.py`:**
    *   Contains functions for LLMs to act as judges of cleaning quality.
    *   `judge_with_gemini(client, cleaned_text, ground_truth)`: Prompts Gemini to score the `cleaned_text` against the `ground_truth` on a 0-5 scale.
    *   (In one version) `initialize_prometheus_judge()` and `judge_with_prometheus(...)`: Sets up and uses a Prometheus model (e.g., `prometheus-7b-v2.0`) for judging. This uses a detailed `OCR_RUBRIC_DATA` to guide Prometheus on a 1-5 scale.
    *   The `main()` function in this script seems to be for standalone testing of the judging functionality.

*   **`main_V2.py`:**
    *   The main orchestration script for the entire pipeline.
    *   Configures input/output paths, LLM models to run, and processing range (start/end index).
    *   Calls functions from `pre_clean.py`, `cleaner_LLM_V2.py`, and `judge_LLM.py`.
    *   Calculates WER, CER, ROUGE.
    *   `parse_score()`: Extracts the numerical score from the judge's potentially verbose output.
    *   `get_detailed_diffs()`: Uses `difflib` to show specific word-level changes between ground truth and cleaned text.
    *   Saves comprehensive results in JSON format.

*   **`evaluator.py`:**
    *   `analyze_rater_agreement(json_file_path, ...)`:
        *   Reads a JSON results file (presumably one that has been manually annotated with `human_score` alongside the LLM judge's `score`).
        *   Calculates Cohen's Kappa (unweighted and quadratic weighted) to measure inter-rater agreement between human and LLM judges.
        *   Prints a detailed report and saves it to a text file. This is crucial for validating the reliability of the automated LLM judge.

*   **`pathinator.py` (Assumed):**
    *   This file is imported in several scripts but not provided. It's assumed to define various file paths used throughout the project (e.g., dataset paths, results paths, dictionary path) in a centralized way, likely using `pathlib`. For example: `dataset_subset`, `cleaning_results`, `judging_results`, `dictionary`.

## Key Concepts and Techniques

*   **LLM Prompt Engineering:** The quality of LLM output heavily relies on the prompts. This project uses detailed, structured prompts with specific instructions, examples, and constraints (e.g., "NON TOCCARE NULL'ALTRO," "Regole di Preservazione Assoluta").
*   **Two-Stage Cleaning:** Breaking down the complex cleaning task into two simpler, focused stages (character/word correction then paragraph/style correction) likely improves LLM performance and controllability.
*   **Multi-Model Comparison:** The pipeline is set up to run and evaluate multiple LLMs, allowing for comparative analysis.
*   **Hybrid Approach:** Combines rule-based pre-cleaning with advanced LLM capabilities.
*   **Comprehensive Evaluation:** Uses both intrinsic (WER, CER, ROUGE) and extrinsic (LLM-based judging) evaluation methods.
*   **Automated Quality Judging:** Leverages LLMs (Gemini, Prometheus) to automate the otherwise laborious task of quality assessment. The custom rubric for Prometheus is a key aspect here.
*   **Judge Validation:** The use of Cohen's Kappa (`evaluator.py`) provides a quantitative measure of how well the automated LLM judge aligns with human judgment, lending credibility to the automated scores.
*   **Detailed Error Analysis:** The `get_detailed_diffs` function allows for a granular understanding of the types of errors made or corrected by the LLMs.

## Setup and Installation

1.  **Clone the repository (if applicable).**
2.  **Create a Python virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```
3.  **Install dependencies:**
    Create a `requirements.txt` file with the following (and any other specific versions you used):
    ```txt
    google-generativeai
    python-dotenv
    jiwer
    evaluate
    torch # Often a dependency for evaluate or transformers
    transformers # Often a dependency for evaluate
    pandas
    scikit-learn
    groq
    pyspellchecker
    # For Prometheus (if using that version of judge_LLM.py):
    # prometheus-eval
    # vllm # Or other backends like text-generation-inference
    ```
    Then run:
    ```bash
    pip install -r requirements.txt
    ```
4.  **Set up API Keys:**
    Create a `.env` file in the project root with your API keys:
    ```env
    GOOGLE_API_KEY="your_google_api_key"
    GROQ_API_KEY="your_groq_api_key"
    ```
5.  **Prepare `pathinator.py`:**
    Ensure `pathinator.py` is present and correctly defines all necessary file paths. Example structure:
    ```python
    # pathinator.py
    from pathlib import Path

    BASE_DIR = Path(__file__).resolve().parent

    # Dataset paths
    dataset_base = BASE_DIR / "dataset2" / "eng"
    input_ocr_path = dataset_base / "the_vampyre_ocr.json"
    input_clear_path = dataset_base / "the_vampyre_clean.json"
    dataset_subset = dataset_base / "the_vampyre_subset.json" # Output of dataset_handler.py

    # Results paths
    results_dir = BASE_DIR / "results"
    cleaning_results = results_dir / "cleaning_results.json" # Intermediate from cleaner_LLM_V2.py standalone
    judging_results = results_dir / "judging_results.json" # Intermediate from judge_LLM.py standalone
    # The main_V2.py output path is constructed dynamically in the script.

    # Pre-clean paths
    dictionary_dir = BASE_DIR / "dictionary_files" # Assuming a directory for dictionaries
    dictionary = dictionary_dir / "index.dic" # Path to your custom dictionary
    ```
6.  **Prepare `index.dic`:**
    Ensure your custom Italian dictionary `index.dic` (referenced in `pre_clean.py`) is in the location specified by `pathinator.py`.

## Running the Pipeline

1.  **Prepare the dataset:**
    *   Ensure your raw OCR and clean JSON files are in the paths specified in `dataset_handler.py`.
    *   Run `dataset_handler.py` to generate the subset file (e.g., `the_vampyre_subset.json`):
        ```bash
        python dataset_handler.py
        ```
2.  **Run the main cleaning and evaluation pipeline:**
    *   Modify `START_INDEX` and `END_INDEX` in `main_V2.py` to process a specific range of items from the subset.
    *   Execute `main_V2.py`:
        ```bash
        python main_V2.py
        ```
    This will generate a JSON output file in the `results` directory.
3.  **Validate the LLM Judge (Optional):**
    *   Manually add `human_score` to some entries in the output JSON from `main_V2.py`.
    *   Run `evaluator.py`, pointing it to your annotated JSON file:
        ```bash
        python evaluator.py
        ```
        (You might need to adjust the `json_file` variable inside `evaluator.py` or pass it as an argument).

## Output

*   **`dataset_handler.py`:** A `the_vampyre_subset.json` (or similar) file containing a structured subset of OCR/clean text pairs.
*   **`main_V2.py`:** A comprehensive JSON file (e.g., `full_pipeline_results_1_2.json`) in the `results/` directory. Each item in this JSON includes:
    *   `item_id`
    *   `original_ocr`
    *   `ground_truth`
    *   `model_outputs`: A list, where each element corresponds to a model run and contains:
        *   `model_name`
        *   `cleaned_text`
        *   `metrics` (WER, CER, ROUGE scores)
        *   `judgement` (LLM judge's score and raw text)
        *   `differences` (detailed word-level diffs)
*   **`evaluator.py`:** A `kappa_analysis_report.txt` file containing the Cohen's Kappa analysis and interpretation.

## Potential Future Work / Improvements

*   **Expand Model Support:** Integrate more LLMs or fine-tuned models.
*   **Advanced Pre-processing:** Explore more sophisticated pre-processing techniques.
*   **Error Categorization:** Automatically categorize the types of errors corrected or introduced by the LLMs.
*   **Hyperparameter Tuning for Prompts:** Systematically test variations in prompts.
*   **Interactive UI:** A simple web interface for uploading OCR text and seeing the cleaned output.
*   **Batch Processing Optimizations:** For very large datasets, optimize API calls and error handling.
*   **Language Generalization:** Adapt prompts and dictionaries for other languages.

This project provides a solid foundation for research and development in OCR post-correction using modern AI techniques.



# TODO
1) manda di nuovo prometheus sul dataset inglese
2) manda prometheus sul dataset italiano
3) rimanda evaluator su entrambi i dataset
4) finire report
    1) dataset inglese senza data processing 
    
    1) dataset italiano con preprocessing e chains of thought  