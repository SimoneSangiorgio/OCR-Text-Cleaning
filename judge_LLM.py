import os
import json
from google import genai
from dotenv import load_dotenv # Recommended for API key management

from pathinator import *

# --- Prometheus Imports ---
from prometheus_eval.vllm import VLLM
from prometheus_eval import PrometheusEval
from prometheus_eval.prompts import ABSOLUTE_PROMPT, SCORE_RUBRIC_TEMPLATE

# --- Configuration ---
load_dotenv()

GEMINI_MODEL_NAME = "gemini-2.0-flash"
INPUT_PATH = "results\cleaning_results.json"
OUTPUT_PATH = judging_results

api_key = os.getenv("GOOGLE_API_KEY")


# --- Prometheus Configuration ---
# Choose your Prometheus model
PROMETHEUS_MODEL_NAME = "prometheus-eval/prometheus-7b-v2.0" 
# PROMETHEUS_MODEL_NAME = "prometheus-eval/prometheus-bgb-8x7b-v2.0" # More capable, but larger
# PROMETHEUS_MODEL_NAME = "prometheus-eval/m-prometheus-7b" # If you need multilingual and strong English


# This rubric is crucial. Tailor it to OCR quality.
# Prometheus expects a 1-5 scale.
OCR_RUBRIC_DATA = {
  "criteria": "How accurately and completely does the cleaned text represent the ground truth, minimizing OCR errors and maintaining readability?",
  "score1_description": "Poor: The cleaned text is mostly unrelated to the ground truth, unreadable, nonsensical, or omits vast portions of the original content. Contains severe and numerous errors.",
  "score2_description": "Fair: The cleaned text has multiple significant errors (e.g., many misrecognized words, incorrect formatting, missing phrases) that make it difficult to understand or misleading.",
  "score3_description": "Good: The cleaned text is largely correct and understandable but contains some noticeable OCR errors (e.g., a few misrecognized words, minor formatting issues, small omissions/additions) that don't obscure the overall meaning.",
  "score4_description": "Excellent: The cleaned text is highly accurate with only very minor errors (e.g., one or two typos, a single punctuation mistake, slight spacing issues) that do not affect meaning or readability significantly.",
  "score5_description": "Perfect: The cleaned text is an exact or near-exact match to the ground truth. It is perfectly readable and free of OCR errors."
}
SCORE_RUBRIC_FOR_OCR = SCORE_RUBRIC_TEMPLATE.format(**OCR_RUBRIC_DATA)

# ----------------------------------------------- LLM Judge Functions -----------------------------------------------

def judge_with_gemini(client: genai.Client, gemini_cleaned: str, ground_truth: str) -> str:
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


def initialize_prometheus_judge():
    """Initializes and returns the Prometheus VLLM model and judge."""
    print(f"Initializing Prometheus model: {PROMETHEUS_MODEL_NAME}")
    # Ensure you have enough VRAM. For 7B, ~16GB. For 8x7B, much more (~50-60GB).
    # You might need to specify tensor_parallel_size for larger models if you have multiple GPUs
    # model_vllm = VLLM(model=PROMETHEUS_MODEL_NAME, tensor_parallel_size=1, trust_remote_code=True) # trust_remote_code might be needed
    try:
        model_vllm = VLLM(model=PROMETHEUS_MODEL_NAME, trust_remote_code=True)
    except Exception as e:
        print(f"Error initializing VLLM model: {e}")
        print("This might be due to insufficient VRAM, or vLLM/CUDA compatibility issues.")
        print("Try a smaller Prometheus model (e.g., prometheus-7b-v2.0) if using a larger one.")
        raise
        
    judge = PrometheusEval(model=model_vllm, absolute_grade_template=ABSOLUTE_PROMPT)
    print("Prometheus judge initialized.")
    return judge

def judge_with_prometheus(
    prometheus_judge: PrometheusEval,
    cleaned_text: str,
    ground_truth: str,
    original_ocr_text: str = None # Optional: The original OCR text before cleaning
) -> tuple[str, int]:
    """
    Judges the quality of cleaned_text against ground_truth using Prometheus.

    Args:
        prometheus_judge: The initialized PrometheusEval object.
        cleaned_text: The text output from your cleaning LLM.
        ground_truth: The reference correct text.
        original_ocr_text: Optional. The raw OCR text that was input to the cleaning LLM.
                           This can help Prometheus understand the task context.

    Returns:
        A tuple (feedback_text, score_integer).
    """
    if not cleaned_text or not cleaned_text.strip():
        return "Cleaned text was empty.", 1 # Assign lowest score for empty, as Prometheus is 1-5

    # The "instruction" tells Prometheus what task the `cleaned_text` (response) was trying to accomplish.
    if original_ocr_text:
        instruction = f"The following text was extracted via Optical Character Recognition (OCR) and may contain errors: \"{original_ocr_text}\". Please clean this text to improve its accuracy and readability, making it as close as possible to the original document."
    else:
        instruction = "The task was to clean a piece of text obtained from OCR to improve its accuracy and readability."

    # print(f"\nInstruction for Prometheus: {instruction[:200]}...") # For debugging
    # print(f"Response for Prometheus: {cleaned_text[:200]}...")     # For debugging
    # print(f"Reference for Prometheus: {ground_truth[:200]}...")   # For debugging
    # print(f"Rubric: {SCORE_RUBRIC_FOR_OCR[:200]}...")             # For debugging

    try:
        feedback, score = prometheus_judge.single_absolute_grade(
            instruction=instruction,
            response=cleaned_text,
            rubric=SCORE_RUBRIC_FOR_OCR,
            reference_answer=ground_truth
        )
        # Ensure score is an integer
        try:
            score_int = int(score)
        except ValueError:
            print(f"Warning: Prometheus returned a non-integer score: '{score}'. Raw feedback: '{feedback}'. Defaulting to 1.")
            score_int = 1 # Default to lowest score if parsing fails
        return feedback, score_int
    except Exception as e:
        print(f"Error during Prometheus grading: {e}")
        return f"[PROMETHEUS_GRADING_ERROR: {e}]", 1 # Default to lowest score on error





#--------------------------------------------------------------------------------------------------------------------------------



def main():
    """
    Main function to read data, judge it with an LLM, and save the results.
    """
    # --- CHOOSE YOUR JUDGE ---
    # To use Gemini, set this to "gemini"
    # To use Prometheus, set this to "prometheus"
    JUDGE_CHOICE = "prometheus" 

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

    # --- 2. Initialize Judge and Process Data ---
    results = []
    
    if JUDGE_CHOICE == "gemini":
        print("Initializing Gemini judge...")
        gemini_model = genai.GenerativeModel(GEMINI_MODEL_NAME)
        for i, item in enumerate(data_dict, 1):
            print(f"Processing item {i}/{len(data_dict)} with Gemini...")
            gemini_cleaned = item.get('gemini_cleaned', '')
            ground_truth = item.get('ground_truth', '')
            score = judge_with_gemini(gemini_model, gemini_cleaned, ground_truth)
            results.append({**item, 'gemini_judge_score': score})

    elif JUDGE_CHOICE == "prometheus":
        try:
            prometheus_judge = initialize_prometheus_judge()
        except Exception as e:
            print(f"Failed to initialize Prometheus. Exiting. Error: {e}")
            return

        for i, item in enumerate(data_dict, 1):
            print(f"\nProcessing item {i}/{len(data_dict)} (ID: {item.get('id', 'N/A')}) with Prometheus...")
            
            cleaned_text_to_judge = item.get('gemini_cleaned', '') 
            ground_truth = item.get('ground_truth', '')
            original_ocr = item.get('original_ocr', None)

            if not ground_truth:
                print(f"Skipping item {item.get('id', 'N/A')} due to missing ground_truth.")
                results.append({**item, 'prometheus_feedback': "Skipped - No ground truth", 'prometheus_score': None})
                continue

            print(f"  Ground Truth: '{ground_truth[:100]}...'")
            print(f"  Cleaned Text: '{cleaned_text_to_judge[:100]}...'")

            feedback, score = judge_with_prometheus(prometheus_judge, cleaned_text_to_judge, ground_truth, original_ocr)
            
            print(f"  Prometheus Feedback: {feedback}")
            print(f"  Prometheus Score: {score}")
            
            results.append({**item, 'prometheus_feedback': feedback, 'prometheus_score': score})

    # --- 3. Save Results ---
    output_dir = os.path.dirname(OUTPUT_PATH)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    
    print(f"\nJudging complete! Results for {len(results)} items saved to '{OUTPUT_PATH}'")

if __name__ == "__main__":
    main()