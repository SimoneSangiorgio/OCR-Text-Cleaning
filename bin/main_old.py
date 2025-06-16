

import json
import os
from google import genai
import jiwer # For WER/CER calculation
import time # To handle potential rate limits
import evaluate
from dotenv import load_dotenv


load_dotenv()

#GEMINI_MODEL_NAME = "gemini-2.0-flash" # Or other suitable Gemini model
INPUT_PATH = "results/cleaning_results.json" # Path to your dataset file
OUTPUT_PATH = "results/evaluation_metrics.json" # One clear output file
#NUM_ITEM_TO_PROCESS = 6

#api_key = os.getenv("GOOGLE_API_KEY")


# ----------------------------------------------- LLM Cleaning Functions -----------------------------------------------
# def clean_with_gemini(ocr_text: str) -> str:
#     """Cleans OCR text using Google Gemini (Client API style)."""
#     if not ocr_text.strip():
#         return ""

#     try:
#         # Initialize the client. Passing api_key explicitly is robust.
#         client = genai.Client(api_key=api_key)
#     except Exception as e:
#         print(f"Error initializing Gemini Client (genai.Client): {e}")
#         print("This could be due to an invalid API key, network issues, or problems with the 'google-generativeai' library.")
#         return f"[GEMINI_CLIENT_INIT_ERROR: {e}]"

#     contents = f"""Clean the following OCR text. Correct spelling errors, fix punctuation, remouve also the \n present in the text. Preserve the original
#     meaning and style. Do not add new information or summarize. Return only the cleaned text.

# OCR Text:
# ---
# {ocr_text}
# ---
# Cleaned Text:
# """
#     response = client.models.generate_content(
#         model=GEMINI_MODEL_NAME,
#         contents=contents
#     )
#     #print(response.text)
#     return response.text

# ----------------------------------------------- Evaluation Metrics -----------------------------------------------
def calculate_metrics(reference: str, hypothesis: str) -> dict:
    """Calculates WER and CER."""
    if not reference.strip() and not hypothesis.strip(): # Both empty
        return {"wer": 0.0, "cer": 0.0}
    if not reference.strip(): # Reference is empty, hypothesis is not (bad)
        # jiwer handles this as WER=1.0 (if hypothesis has content) or 0.0 (if hypothesis also empty)
        # For CER, similar logic applies.
        # To be explicit and match common interpretations for empty reference:
        return {"wer": 1.0 if hypothesis.strip() else 0.0, "cer": 1.0 if hypothesis.strip() else 0.0}
    if not hypothesis.strip(): # Hypothesis is empty, reference is not (bad)
         return {"wer": 1.0, "cer": 1.0} # All words in reference are deletions.

    # transformation = jiwer.Compose([
    #     jiwer.ToLowerCase(),
    #     jiwer.RemoveMultipleSpaces(),
    #     jiwer.Strip(),
    #     jiwer.RemovePunctuation(), # Punctuation will not count towards WER
    # ])

    # Apply transformations. jiwer.wer will tokenize the resulting string by spaces.
    wer = jiwer.wer(reference, hypothesis)

    # For CER, jiwer by default applies ToLowerCase and RemoveMultipleSpaces.
    # If you want punctuation removed for CER as well, you can pass the same transformation.
    # For now, let's use jiwer's default CER processing.
    cer = jiwer.cer(reference, hypothesis)

    return {"wer": wer, "cer": cer}

# ----------------------------------------------- Main Processing Logic -----------------------------------------------
def main():
    try:
        with open(INPUT_PATH, 'r', encoding='utf-8') as f:
            # The JSON is a list of items, so data_items is already the list.
            data_items = json.load(f)
    except FileNotFoundError:
        print(f"Error: Dataset file not found at {INPUT_PATH}")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {INPUT_PATH}.")
        return

    print(f"Loaded {len(data_items)} items to evaluate from '{INPUT_PATH}'")

    results_with_metrics = []
    rouge = evaluate.load("rouge")

    # The loop should iterate directly over the list of items
    for i, item in enumerate(data_items):
        print(f"\n--- Evaluating item {i+1}/{len(data_items)} ---")

        gemini_cleaned_text = item.get('gemini_cleaned', '')
        ground_truth_clean_text = item.get('ground_truth', '')

        # 1. Calculate ROUGE
        rouge_scores = rouge.compute(predictions=[gemini_cleaned_text], references=[ground_truth_clean_text])
        print("ROUGE scores:", rouge_scores)

        # 2. Calculate WER/CER
        edit_distance_metrics = calculate_metrics(ground_truth_clean_text, gemini_cleaned_text)
        print(f"Metrics: WER={edit_distance_metrics['wer']:.4f}, CER={edit_distance_metrics['cer']:.4f}")

        # Append all data to a new results list
        item['metrics'] = {
            'wer': edit_distance_metrics['wer'],
            'cer': edit_distance_metrics['cer'],
            'rouge': rouge_scores
        }
        results_with_metrics.append(item)

    print("\n\n--- Overall Average Metrics ---")
    if results_with_metrics:
        # Filter out items that had a cleaning error
        successful_results = [r for r in results_with_metrics if not r['gemini_cleaned'].startswith("[GEMINI_")]
        if successful_results:
            avg_wer = sum(r['metrics']['wer'] for r in successful_results) / len(successful_results)
            avg_cer = sum(r['metrics']['cer'] for r in successful_results) / len(successful_results)
            print(f"Average WER (for {len(successful_results)} valid items): {avg_wer:.4f}")
            print(f"Average CER (for {len(successful_results)} valid items): {avg_cer:.4f}")
        else:
            print("No items were successfully processed to calculate average metrics.")
    else:
        print("No items were processed.")

    # Save Results to a single, clearly named JSON File
    output_dir = os.path.dirname(OUTPUT_PATH)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
    try:
        with open(OUTPUT_PATH, "w", encoding="utf-8") as outfile:
            json.dump(results_with_metrics, outfile, indent=2, ensure_ascii=False)
        print(f"\nDetailed results with metrics saved to {OUTPUT_PATH}")
    except IOError as e:
        print(f"\nError saving results to {OUTPUT_PATH}: {e}")

if __name__ == "__main__":
    main()