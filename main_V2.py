import json
import os
import re
import time
from google import genai
from dotenv import load_dotenv
import jiwer
import evaluate
import difflib
from groq import Groq


from cleaner_LLM_V2 import clean_with_gemini, clean_with_groq, corrector_gemini, corrector_groq
from judge_LLM import judge_with_gemini
from pre_clean import *
from pathinator import *

# --- 1. Configuration ---
load_dotenv()

# Model and File Configuration
INPUT_PATH = dataset_subset

START_INDEX = 1
END_INDEX = 2  # Modifica questo valore o impostalo su None per andare fino alla fine

OUTPUT_PATH = results / f"full_pipeline_results_{START_INDEX}_{END_INDEX}.json"

MODELS_TO_RUN = [{
        "name": "Gemini-1.5-Flash",
        "type": "gemini",
        "function": clean_with_gemini,
        "model_id_or_client": None },

    {
        "name": "Mistral",
        "type": "groq",
        "function": clean_with_groq,
        "model_id_or_client": "mistral-saba-24b"}
        ]

def parse_score(response_text: str) -> int:
    """Extracts the first integer from the judge's response for robustness."""
    numbers = re.findall(r'\d+', response_text)
    return int(numbers[0]) if numbers else -1 # -1 indicates a parsing error

def calculate_metrics(reference: str, hypothesis: str) -> dict:
    """Calculates WER and CER, handling edge cases."""
    if not reference.strip() and not hypothesis.strip():
        return {"wer": 0.0, "cer": 0.0}
    if not reference.strip() or not hypothesis.strip():
        return {"wer": 1.0, "cer": 1.0}
    
    return {
        "wer": jiwer.wer(reference, hypothesis),
        "cer": jiwer.cer(reference, hypothesis)
    }

def get_detailed_diffs(text1: str, text2: str) -> list[dict]:
    """
    Compares two strings word by word and returns a list of dictionaries 
    highlighting the specific differing words.
    """
    # Split texts into lists of words. We use a regex to better handle
    # punctuation, treating it as a separate "word".
    import re
    words1 = re.findall(r'\w+|[^\w\s]', text1)
    words2 = re.findall(r'\w+|[^\w\s]', text2)

    s = difflib.SequenceMatcher(None, words1, words2)
    diffs = []

    for tag, i1, i2, j1, j2 in s.get_opcodes():
        if tag == 'equal':
            continue

        diff_item = {
            "type": tag,
            # Join the words to recreate the original text slice
            "ground_truth_slice": " ".join(words1[i1:i2]),
            "model_cleaned_slice": " ".join(words2[j1:j2])
        }
        diffs.append(diff_item)
        
    return diffs

# --- 3. Main Orchestration Logic ---

def main():
    """
    Main function to run the complete clean, evaluate, and judge pipeline for multiple models.
    """

    # --- Initialize APIs (ONCE) ---
    try:
        google_api_key = os.getenv("GOOGLE_API_KEY")
        if not google_api_key:
            raise ValueError("GOOGLE_API_KEY not found in environment variables.")
        gemini_client = genai.Client(api_key=google_api_key)
        
        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            raise ValueError("GROQ_API_KEY not found in environment variables.")
        groq_client = Groq(api_key=groq_api_key)

        rouge_metric = evaluate.load("rouge")
        print("Successfully initialized Gemini Client, Groq Client, and ROUGE metric evaluator.")
    except Exception as e:
        print(f"Fatal Error during initialization: {e}")
        return

    # --- Load Data ---
    try:
        with open(INPUT_PATH, 'r', encoding='utf-8') as f:
            data_dict = json.load(f)
        print(f"Successfully loaded {len(data_dict)} items from '{INPUT_PATH}'")
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error loading input file: {e}")
        return

    list_of_items = list(data_dict.values())

    # Validate the indices to prevent unexpected behavior
    if END_INDEX is not None and START_INDEX > END_INDEX:
        print(f"Error: START_INDEX ({START_INDEX}) cannot be greater than END_INDEX ({END_INDEX}). Exiting.")
        return

    # Slice the list of items based on the specified range
    subset_to_process = list_of_items[START_INDEX-1:END_INDEX]
    
    all_results = []
    print(f"\nStarting pipeline for {len(subset_to_process)} items (from index {START_INDEX} to {END_INDEX or len(list_of_items)}) across {len(MODELS_TO_RUN)} models...")

    # --- Process Each Item in the Dataset ---
    for i, item in enumerate(subset_to_process):
        print(f"\n{'='*20} Processing item {i+1}/{len(subset_to_process)} {'='*20}")
        
        ocr_text = item.get('ocr', '')
        ocr_text = replacement_rules(ocr_text)

        ground_truth = item.get('clean', '')
        
        # Structure to save all results for this item
        item_result = {
            "item_id": i + 1,
            "original_ocr": ocr_text,
            "ground_truth": ground_truth,
            "model_outputs": []
        }

        # --- Loop through each configured model ---
        for model_config in MODELS_TO_RUN:
            model_name = model_config["name"]
            cleaning_function = model_config["function"]
            print(f"\n--- Running on model: {model_name} ---")

            # Step 1: Clean the text
            print("1. Cleaning text...")
            cleaned_text = "[ERROR: Unknown model type in config]"
            if model_config["type"] == "gemini":
                cleaned_text = cleaning_function(gemini_client, ocr_text)
                cleaned_text = corrector_gemini(gemini_client, cleaned_text)
                #cleaned_text = replacement_rules(cleaned_text)
            elif model_config["type"] == "groq":
                model_id = model_config["model_id_or_client"]
                cleaned_text = cleaning_function(groq_client, ocr_text, model_id)
                cleaned_text = corrector_groq(groq_client, cleaned_text, model_id)
                #cleaned_text = replacement_rules(cleaned_text)

            #ocr_text = correct_spelling(ocr_text)

            if "[ERROR:" in cleaned_text or "[GEMINI_" in cleaned_text or "[HUGGINGFACE_" in cleaned_text or "[GROQ_" in cleaned_text:
                print(f"  -> Skipping further processing for {model_name} due to cleaning error: {cleaned_text}")
                model_run_result = {"model_name": model_name, "cleaned_text": cleaned_text, "metrics": None, "judgement": None, "differences": []}
                item_result["model_outputs"].append(model_run_result)
                time.sleep(1) # Still sleep to avoid hammering a failing API
                continue

            # Step 2: Evaluate with quantitative metrics
            print("2. Calculating WER, CER, and ROUGE metrics...")
            edit_metrics = calculate_metrics(ground_truth, cleaned_text)
            rouge_scores = rouge_metric.compute(predictions=[cleaned_text], references=[ground_truth])
            
            # Step 3: Judge the quality with Gemini (using Gemini as the standard judge for all)
            print("3. Judging quality with Gemini...")
            raw_judgement = judge_with_gemini(gemini_client, cleaned_text, ground_truth)
            parsed_judgement_score = parse_score(raw_judgement)

            # Step 4: Get detailed differences
            detailed_differences = get_detailed_diffs(ground_truth, cleaned_text)

            # Step 5: Aggregate all information for this model run
            model_run_result = {
                "model_name": model_name,
                "cleaned_text": cleaned_text,
                "metrics": {
                    "wer": edit_metrics['wer'],
                    "cer": edit_metrics['cer'],
                    "rouge": rouge_scores
                },
                "judgement": {
                    "score": parsed_judgement_score,
                    "raw_score_text": raw_judgement.strip()
                },
                "differences": detailed_differences
            }
            item_result["model_outputs"].append(model_run_result)
            
            print(f"  -> Metrics: WER={edit_metrics['wer']:.4f}, CER={edit_metrics['cer']:.4f}")
            print(f"  -> Judgement: Score={parsed_judgement_score}")
            
            # Avoid hitting API rate limits
            time.sleep(1) 

        all_results.append(item_result)

    # --- Save Final Combined Results ---
    print("\n--- Pipeline Complete ---")
    output_dir = os.path.dirname(OUTPUT_PATH)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
    try:
        with open(OUTPUT_PATH, "w", encoding="utf-8") as outfile:
            json.dump(all_results, outfile, indent=2, ensure_ascii=False)
        print(f"\nAll processed data and results saved to: {OUTPUT_PATH}")
    except IOError as e:
        print(f"\nError saving final results file: {e}")

if __name__ == "__main__":
    main()
