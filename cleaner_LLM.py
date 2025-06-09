import json
import os
from google import genai
from dotenv import load_dotenv
import time # To handle potential rate limits

load_dotenv()

GEMINI_MODEL_NAME = "gemini-2.0-flash" # Or other suitable Gemini model
INPUT_PATH = "dataset/eng/the_vampyre_subset.json" # Path to your dataset file
OUTPUT_PATH = "clean_judge_files/cleaning_results.json" 
NUM_ITEM_TO_PROCESS = 6

api_key = os.getenv("GOOGLE_API_KEY")

# ----------------------------------------------- LLM Cleaning Functions -----------------------------------------------
def clean_with_gemini(client: genai.Client, ocr_text: str) -> str:
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

#--------------------------------------------------------------------------------------------------------------------------------



def main():
    """
    main function to create a clean text from ocr text using Gemini
    this is actually no more nedded because the main file take the function from cleaner_LLM.py. but could be usefull if we want to create a separate file witout run the main file
    """
  
    try:
        with open(INPUT_PATH, 'r', encoding='utf-8') as f:
            data_dict = json.load(f)
        print(f"Successfully loaded {len(data_dict)} items from '{INPUT_PATH}'")
    except FileNotFoundError:
        print(f"Error: Input file not found at '{INPUT_PATH}'")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from '{INPUT_PATH}'. Please check its format.")
        return
    

    list_of_items = list(data_dict.values())
    print(f"Total items (segments with ocr/clean pairs) to process: {len(list_of_items)}")

    subset_to_process = list_of_items[:NUM_ITEM_TO_PROCESS]
    print(f"\nProcessing {len(subset_to_process)} items from the dataset...")

    results = []

    # --- 3. Process Data ---
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

    # To avoid hitting rate limits, we can add a sleep here.
        time.sleep(1)

        results.append({
            "original_ocr": ocr_text,
            "ground_truth": ground_truth_clean_text,
            "gemini_cleaned": gemini_cleaned_text,
            
        })

# Save Results to JSON File 
    try:
        with open(OUTPUT_PATH, "w", encoding="utf-8") as outfile:
            json.dump(results, outfile, indent=2, ensure_ascii=False)
        print(f"\nDetailed results saved to {OUTPUT_PATH}")
    except IOError as e:
        print(f"\nError saving results to {OUTPUT_PATH}: {e}")


if __name__ == "__main__":
    main()

