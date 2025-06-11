import json
import os
from google import genai
from dotenv import load_dotenv
import time # To handle potential rate limits
import requests
from groq import Groq

from pathinator import *

load_dotenv()

GEMINI_MODEL_NAME = "gemini-2.0-flash" # Or other suitable Gemini model
INPUT_PATH = dataset_subset # Path to your dataset file
OUTPUT_PATH = cleaning_results 
NUM_ITEM_TO_PROCESS = 3

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
        print("This could be due to an invalid API key, network issues, or problems with the 'google-genai' library.")
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

def clean_with_huggingface(model_id: str, ocr_text: str) -> str:
    """
    Cleans OCR text using a model from the Hugging Face Inference API.
    """
    if not ocr_text.strip():
        return ""

    api_key = os.getenv("HUGGINGFACE_API_KEY")
    if not api_key:
        return "[HUGGINGFACE_API_KEY_ERROR: Key not found in .env file]"
        
    api_url = f"https://api-inference.huggingface.co/models/{model_id}"
    headers = {"Authorization": f"Bearer {api_key}"}

    # Il prompt deve essere adattato allo stile del modello.
    # I modelli "-Instruct" funzionano bene con un prompt di sistema e utente.
    # Llama-3 ha un formato specifico di chat.
    if "Llama-3" in model_id:
        # Formato specifico per Llama 3
        prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a helpful assistant that cleans OCR text. Correct spelling errors, fix punctuation, and remove newline characters like \\n. Preserve the original meaning and style. Do not add new information or summarize. Return only the cleaned text, without any introductory phrases.<|eot_id|><|start_header_id|>user<|end_header_id|>
OCR Text:
---
{ocr_text}
---
Cleaned Text:<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""
    else:
        # Un formato più generico per altri modelli instruct
        prompt = f"""Clean the following OCR text. Correct spelling errors, fix punctuation, remouve also the \\n present in the text. Preserve the original
meaning and style. Do not add new information or summarize. Return only the cleaned text.

OCR Text:
---
{ocr_text}
---
Cleaned Text:
"""

    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 1024, # Aumenta se il testo è molto lungo
            "return_full_text": False, # Importante per non ricevere il prompt indietro
        }
    }

    try:
        response = requests.post(api_url, headers=headers, json=payload, timeout=60)
        
        if response.status_code == 200:
            result = response.json()
            # Estrai il testo generato
            cleaned_text = result[0].get('generated_text', '').strip()
            return cleaned_text
        else:
            error_message = response.text
            # I modelli potrebbero essere in caricamento su Hugging Face
            if "is currently loading" in error_message:
                print(f"Model {model_id} is loading, try again in a moment.")
                return f"[HUGGINGFACE_API_LOADING_ERROR: {error_message}]"
            return f"[HUGGINGFACE_API_ERROR: Status {response.status_code} - {error_message}]"
            
    except requests.exceptions.RequestException as e:
        return f"[HUGGINGFACE_REQUEST_ERROR: {e}]"

#--------------------------------------------------------------------------------------------------------------------------------

def clean_with_groq(client: Groq, ocr_text: str, model_id: str) -> str:
    """Cleans OCR text using a model from the Groq API."""
    if not ocr_text.strip():
        return ""

    # This chat-based prompt works well for modern instruct models like Llama3 and Mixtral
    messages = [
        {
            "role": "system",
            "content": "Clean the following OCR text. Correct spelling errors, fix punctuation, remouve also the \\n present in the text. Preserve the original meaning and style. Do not add new information or summarize. Return only the cleaned text.",
        },
        {
            "role": "user",
            "content": f"OCR Text:\n---\n{ocr_text}\n---\nCleaned Text:",
        }
    ]

    try:
        chat_completion = client.chat.completions.create(
            messages=messages,
            model=model_id,
            temperature=0.1,  # Lower temperature for more deterministic output
            max_tokens=2048,
        )
        cleaned_text = chat_completion.choices[0].message.content.strip()
        return cleaned_text
    except Exception as e:
        return f"[GROQ_API_ERROR: {e}]"

#--------------------------------------------------------------------------------------------------------------------------------

def main():
    """
    main function to create a clean text from ocr text using different LLMs
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
    
    # Initialize Gemini client once
    try:
        gemini_client = genai.Client(api_key=api_key)
    except Exception as e:
        print(f"Could not initialize Gemini Client: {e}")
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
        gemini_cleaned_text = clean_with_gemini(gemini_client, ocr_text)
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

