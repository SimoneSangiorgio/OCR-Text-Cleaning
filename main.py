import json
import os
import re
import time
from google import genai
from dotenv import load_dotenv
import jiwer
import evaluate

from cleaner_LLM import clean_with_gemini
from judge_LLM import judge_with_gemini

# --- 1. Configuration ---
load_dotenv()

# Model and File Configuration
INPUT_PATH = "dataset/eng/the_vampyre_subset.json"
OUTPUT_PATH = "results/full_pipeline_results.json"
NUM_ITEM_TO_PROCESS = 6 # Set to a larger number or `None` to process all

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

# --- 3. Main Orchestration Logic ---

def main():
    """
    Main function to run the complete clean, evaluate, and judge pipeline.
    """
    # --- Initialize APIs (ONCE) ---
    try:
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not found in environment variables.")
        client = genai.Client(api_key=api_key)
        rouge_metric = evaluate.load("rouge")
        print("Successfully initialized Gemini Client and ROUGE metric evaluator.")
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
    subset_to_process = list_of_items[:NUM_ITEM_TO_PROCESS] if NUM_ITEM_TO_PROCESS is not None else list_of_items
    
    all_results = []
    print(f"\nStarting pipeline for {len(subset_to_process)} items...")

    # --- Process Each Item in the Pipeline ---
    for i, item in enumerate(subset_to_process):
        print(f"\n--- Processing item {i+1}/{len(subset_to_process)} ---")
        
        ocr_text = item.get('ocr', '')
        ground_truth = item.get('clean', '')

        # Step 1: Clean the text
        print("1. Cleaning text with Gemini...")
        cleaned_text = clean_with_gemini(client, ocr_text)
        
        if "[GEMINI_" in cleaned_text:
            print("  -> Skipping further processing for this item due to cleaning error.")
            # Still save the failed attempt for review
            result_item = {
                "original_ocr": ocr_text,
                "ground_truth": ground_truth,
                "gemini_cleaned": cleaned_text,
                "metrics": None,
                "judgement": None
            }
            all_results.append(result_item)
            time.sleep(1) # Still sleep to avoid hammering a failing API
            continue

        # Step 2: Evaluate with quantitative metrics
        print("2. Calculating WER, CER, and ROUGE metrics...")
        edit_metrics = calculate_metrics(ground_truth, cleaned_text)
        rouge_scores = rouge_metric.compute(predictions=[cleaned_text], references=[ground_truth])
        
        # Step 3: Judge the quality with an LLM
        print("3. Judging quality with Gemini...")
        raw_judgement = judge_with_gemini(client, cleaned_text, ground_truth)
        parsed_judgement_score = parse_score(raw_judgement)

        # Step 4: Aggregate all information for this item
        result_item = {
            "original_ocr": ocr_text,
            "ground_truth": ground_truth,
            "gemini_cleaned": cleaned_text,
            "metrics": {
                "wer": edit_metrics['wer'],
                "cer": edit_metrics['cer'],
                "rouge": rouge_scores
            },
            "judgement": {
                "score": parsed_judgement_score,
                #"raw_score_text": raw_judgement
            }
        }
        all_results.append(result_item)
        
        print(f"  -> Metrics: WER={edit_metrics['wer']:.4f}, CER={edit_metrics['cer']:.4f}")
        print(f"  -> Judgement: Score={parsed_judgement_score} (Raw: '{raw_judgement}')")
        
        # Avoid hitting API rate limits
        time.sleep(1) 

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