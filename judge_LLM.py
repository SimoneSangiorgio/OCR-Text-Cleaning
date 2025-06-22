import os
import json
# Using 'import google.generativeai as genai' for clarity with older libraries
from google import genai
from dotenv import load_dotenv # Recommended for API key management

# --- Configuration ---
load_dotenv()

# NOTE: "gemini-2.0-flash" is not a valid model name. I have corrected it to a valid one.
# "gemini-pro" is also a very safe and common choice.
GEMINI_MODEL_NAME = "gemini-1.5-flash" 
#INPUT_PATH = "clean_judge_files/cleaning_results.json"
INPUT_PATH = "extracted_data_ita.json"
OUTPUT_PATH = "clean_judge_files/judging_results.json"

api_key = os.getenv("GOOGLE_API_KEY")

# ----------------------------------------------- LLM Judge Functions -----------------------------------------------

def judge_with_gemini(client: genai.Client, gemini_cleaned: str, ground_truth: str) -> str:
    """Judges the quality of Gemini-generated text using a pre-initialized Gemini client."""
    
    # Handle empty input gracefully
    if not gemini_cleaned or not gemini_cleaned.strip():
        return "0"
    
    # --- FIX ---
    # The client is now passed in as an argument.
    # We have REMOVED the inefficient 'client = genai.Client(api_key=api_key)' line from here.

    contents = f"""Evaluate the quality of the "cleaned text" against the "ground truth" reference.
Provide a score from 0 to 5 based on the following scale:
5: Perfect. The cleaned text fully and accurately matches the ground truth.
4: Excellent. Very minor errors (e.g., one or two typos, a single punctuation mistake) that do not affect meaning.
3: Good. Some errors persist (e.g., a few OCR mistakes) but the overall meaning is clear and correct.
2: Fair. Multiple issues make the text difficult to understand or it contains misleading information.
1: Poor. Unacceptable quality; the output is mostly unrelated, unreadable, or nonsensical.
0: Empty/No Output. The cleaned text was empty.

---
[GROUND TRUTH]:
{ground_truth}
---
[CLEANED TEXT]:
{gemini_cleaned}
---

Return ONLY the integer score (0-5) and nothing else.
"""
    response = client.models.generate_content(
        model=GEMINI_MODEL_NAME,
        contents=contents
    )


#--------------------------------------------------------------------------------------------------------------------------------

def main():
    """
    Main function to read data, judge it with Gemini, and save the results.
    """

    # --- 1. Read Input File ---
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

    # --- 2. Initialize Client Once ---
    # --- FIX ---
    # Initialize the client here, ONE time, before the loop starts.
    # This uses the initialization method from your original code.
    try:
        client = genai.Client(api_key=api_key)
        print("Gemini Client initialized successfully.")
    except Exception as e:
        print(f"Fatal Error: Could not initialize genai.Client. Check API Key. Error: {e}")
        print("This might be an issue with your API key or an incompatible 'google-generativeai' library version.")
        return

    # --- 3. Process Data ---
    results = []
    # --- MAJOR FIX ---
    # The original error was because you were looping over a dictionary's keys (strings).
    # We must loop over its VALUES to get the data objects.
    for i, item in enumerate(data_dict.values(), 1):

        print(f"Processing item {i}/{len(data_dict)}...")
        # Your original code failed here because 'item' was a string. It is now a dictionary.
        gemini_cleaned = item.get('gemini_cleaned', '')
        llama_cleaned = item.get('llama_cleaned', '')
        mistral_cleaned = item.get('mistral_cleaned', '')
        ground_truth = item.get('ground_truth', '')

        if not ground_truth:
            print(f"  Skipping item {i} due to empty ground_truth.")
            continue
        
        # --- FIX ---
        # Pass the initialized 'client' object to each function call.
        score_gemini = judge_with_gemini(client, gemini_cleaned, ground_truth)
        score_llama = judge_with_gemini(client, llama_cleaned, ground_truth)
        score_mistral = judge_with_gemini(client, mistral_cleaned, ground_truth)

        results.append({
            'ground_truth': ground_truth,
            'gemini_cleaned': gemini_cleaned,
            'score_gemini': score_gemini,
            'llama_cleaned': llama_cleaned,
            'score_llama': score_llama,
            'mistral_cleaned': mistral_cleaned, 
            'score_mistral': score_mistral
        })

    # --- 4. Save Results ---
    output_dir = os.path.dirname(OUTPUT_PATH)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    
    print(f"\nJudging complete! Results saved to '{OUTPUT_PATH}'")


if __name__ == "__main__":
    main()