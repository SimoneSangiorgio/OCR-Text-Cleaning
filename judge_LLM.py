import os
import json
from google import genai
from dotenv import load_dotenv # Recommended for API key management

# --- Configuration ---
load_dotenv()

GEMINI_MODEL_NAME = "gemini-2.0-flash"
INPUT_PATH = "results/cleaning_results.json"
OUTPUT_PATH = "results/judging_results.json"

api_key = os.getenv("GOOGLE_API_KEY")

# ----------------------------------------------- LLM Judge Functions -----------------------------------------------

def judge_with_gemini(gemini_cleaned: str, ground_truth: str) -> str:
    """Judges the quality of Gemini-generated text using a pre-initialized Gemini client."""
    
    # Handle empty input gracefully
    if not gemini_cleaned or not gemini_cleaned.strip():
        return "0" # Assign a score of 0 if the cleaned text is empty
    
    try:
        # Initialize the client. Passing api_key explicitly is robust.
        client = genai.Client(api_key=api_key)
    except Exception as e:
        print(f"Error initializing Gemini Client (genai.Client): {e}")
        print("This could be due to an invalid API key, network issues, or problems with the 'google-generativeai' library.")
        return f"[GEMINI_CLIENT_INIT_ERROR: {e}]"

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
    #print(response.text)
    return response.text

def main():
    """
    Main function to read data, judge it with Gemini, and save the results.
    """
    # # --- 1. Initialize API Client ---
    # # BEST PRACTICE: Get API key from environment variables.
    # api_key = os.getenv("GOOGLE_API_KEY")
    # if not api_key:
    #     print("Error: GOOGLE_API_KEY not found in environment variables.")
    #     print("Please create a .env file or set the environment variable.")
    #     return # Exit if no API key

    # try:
    #     genai.configure(api_key=api_key)
    #     model = genai.GenerativeModel(GEMINI_MODEL_NAME)
    #     print(f"Successfully initialized Gemini model: {GEMINI_MODEL_NAME}")
    # except Exception as e:
    #     print(f"Error initializing Gemini Client: {e}")
    #     return

    # --- 2. Read Input File ---
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

    # --- 3. Process Data ---
    results = []
    for i, item in enumerate(data_dict, 1):
        print(f"Processing item {i}/{len(data_dict)}...")
        gemini_cleaned = item.get('gemini_cleaned', '') # Use .get() for safety
        ground_truth = item.get('ground_truth', '')

        score = judge_with_gemini(gemini_cleaned, ground_truth)
        
        results.append({
            'gemini_cleaned': gemini_cleaned,
            'ground_truth': ground_truth,
            'score': score
        })

    # --- 4. Save Results ---
    output_dir = os.path.dirname(OUTPUT_PATH)
    if output_dir: # Check if there is a directory part
        os.makedirs(output_dir, exist_ok=True)

    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    
    print(f"\n Judging complete! Results saved to '{OUTPUT_PATH}'")


if __name__ == "__main__":
    main()