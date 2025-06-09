

import json
import os
from google import genai
import jiwer # For WER/CER calculation
import time # To handle potential rate limits
import evaluate
from dotenv import load_dotenv

load_dotenv()

GEMINI_MODEL_NAME = "gemini-2.0-flash" # Or other suitable Gemini model
INPUT_PATH = "dataset2/eng/the_vampyre_subset.json" # Path to your dataset file

api_key = os.getenv("GOOGLE_API_KEY")


# ----------------------------------------------- LLM Cleaning Functions -----------------------------------------------
def clean_with_gemini(ocr_text: str) -> str:
    """Cleans OCR text using Google Gemini (Client API style)."""
    if not ocr_text.strip():
        return ""

    try:
        # Initialize the client. Passing api_key explicitly is robust.
        client = genai.Client(api_key=api_key)
    except Exception as e:
        print(f"Error initializing Gemini Client (genai.Client): {e}")
        print("This could be due to an invalid API key, network issues, or problems with the 'google-generativeai' library.")
        return f"[GEMINI_CLIENT_INIT_ERROR: {e}]"

    contents = f"""Clean the following OCR text. Correct spelling errors, fix punctuation, remouve also the \n present in the text. Preserve the original
    meaning and style. Do not add new information or summarize. Return only the cleaned text.

OCR Text:
---
{ocr_text}
---
Cleaned Text:
"""
    response = client.models.generate_content(
        model=GEMINI_MODEL_NAME,
        contents=contents
    )
    #print(response.text)
    return response.text

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
            data_dict = json.load(f)
    except FileNotFoundError:
        print(f"Error: Dataset file not found at {INPUT_PATH}")
        print("Please ensure the path is correct and the file exists.")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {INPUT_PATH}. Check its format.")
        return

    print(f"Loaded data type: {type(data_dict)}")
    print(f"Total key-value pairs in dictionary: {len(data_dict)}")

    list_of_items = list(data_dict.values())
    print(f"Total items (segments with ocr/clean pairs) to process: {len(list_of_items)}")

    num_items_to_process = 6
    subset_to_process = list_of_items[:num_items_to_process]
    print(f"\nProcessing {len(subset_to_process)} items from the dataset...")

    results = []
    gemini_judged = []

    rouge = evaluate.load("rouge")

    for i, item in enumerate(subset_to_process):
        print(f"\n--- Processing item {i+1}/{len(subset_to_process)} ---")

        ocr_text = item['ocr']
        ground_truth_clean_text = item['clean']

        print(f"OCR Text (first 200 chars):\n{ocr_text[:200]}{'...' if len(ocr_text) > 200 else ''}")

        print("Cleaning with Gemini...")
        gemini_cleaned_text = clean_with_gemini(ocr_text)
        if "[GEMINI_" in gemini_cleaned_text: # Check if an error placeholder was returned
            print(f"Gemini cleaning failed for item {i+1}. Returned: {gemini_cleaned_text}")
        else:
            print(f"Gemini Cleaned Text (first 200 chars):\n{gemini_cleaned_text[:200]}{'...' if len(gemini_cleaned_text) > 200 else ''}")

        result = rouge.compute(predictions=[gemini_cleaned_text], references=[ground_truth_clean_text])
        print(f" ROUGE score:...")
        print(result)
        gemini_metrics = calculate_metrics(ground_truth_clean_text, gemini_cleaned_text)
        print(f"Gemini Metrics: WER={gemini_metrics['wer']:.4f}, CER={gemini_metrics['cer']:.4f}")

        # To avoid hitting rate limits, we can add a sleep here.
        time.sleep(1)

        gemini_judged.append({
            "gemini_cleaned": gemini_cleaned_text,
            "ground_truth": ground_truth_clean_text,
        })
        results.append({
            "original_ocr": ocr_text,
            "ground_truth": ground_truth_clean_text,
            "gemini_cleaned": gemini_cleaned_text,
            "gemini_wer": gemini_metrics['wer'],
            "gemini_cer": gemini_metrics['cer'],
        })

    print("\n\n--- Overall Results ---")
    if results:
        successful_results = [r for r in results if not r['gemini_cleaned'].startswith("[GEMINI_")]
        if successful_results:
            avg_gemini_wer = sum(r['gemini_wer'] for r in successful_results) / len(successful_results)
            avg_gemini_cer = sum(r['gemini_cer'] for r in successful_results) / len(successful_results)
            print(f"Average Gemini WER (for {len(successful_results)} successfully processed items): {avg_gemini_wer:.4f}")
            print(f"Average Gemini CER (for {len(successful_results)} successfully processed items): {avg_gemini_cer:.4f}")
            if len(successful_results) < len(results):
                print(f"Note: {len(results) - len(successful_results)} item(s) encountered errors during Gemini cleaning and were excluded from averages.")
        else:
            print("No items were successfully processed by Gemini to calculate average metrics.")
    else:
        print("No items were processed.")

#-------------------------------------------- Save Results to JSON File --------------------------------------------
    output_filename = "cleaning_results.json"
    try:
        with open(output_filename, "w", encoding="utf-8") as outfile:
            json.dump(results, outfile, indent=2, ensure_ascii=False)
        print(f"\nDetailed results saved to {output_filename}")
    except IOError as e:
        print(f"\nError saving results to {output_filename}: {e}")


#-------------------------------------------- Saving of the file ready to be judicate with judge --------------------------------------------
        output_filename = "judge_file.json"
    try:
        with open(output_filename, "w", encoding="utf-8") as outfile:
            json.dump(results, outfile, indent=2, ensure_ascii=False)
        print(f"\njudge file saved correctly {output_filename}")
    except IOError as e:
        print(f"\nError saving results to {output_filename}: {e}")


#------------------------------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    main()
