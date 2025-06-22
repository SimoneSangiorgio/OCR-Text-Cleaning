import json

def process_pipeline_results(file_path):
    """
    Processes a JSON file with OCR pipeline results to extract specific data.

    Args:
        file_path (str): The path to the input JSON file.

    Returns:
        dict: A dictionary where each key is an 'item_id' and the value is another
              dictionary containing the OCR, ground truth, and cleaned texts of all the models used gemini, llama, mistral.
              
              IT GENERATE THE TWO FILES EXTRACTED_DATA.JSON AND EXTRACTED_DATA_ITA.JSON
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        return {}
    except json.JSONDecodeError:
        print(f"Error: The file '{file_path}' contains invalid JSON.")
        return {}

    extracted_data = {}

    for item in data:
        item_id = item.get("item_id")
        if item_id is None:
            print(f"Skipping an item because it has no 'item_id'.")
            continue

        original_ocr = item.get("original_ocr", "")
        ground_truth = item.get("ground_truth", "")

        # Initialize variables to store cleaned text from each model
        gemini_cleaned = ""
        llama_cleaned = ""
        mistral_cleaned = ""

        # Loop through the model outputs to find the correct text for each model
        model_outputs = item.get("model_outputs", [])
        for output in model_outputs:
            model_name = output.get("model_name")
            cleaned_text = output.get("cleaned_text", "")

            if model_name == "Gemini-1.5-Flash":
                gemini_cleaned = cleaned_text
            elif model_name == "Llama":
                llama_cleaned = cleaned_text
            elif model_name == "Mistral":
                mistral_cleaned = cleaned_text

        # Assemble the dictionary with named keys, as requested.
        value_dict = {
            "original_ocr": original_ocr,
            "ground_truth": ground_truth,
            "gemini_cleaned": gemini_cleaned,
            "llama_cleaned": llama_cleaned,
            "mistral_cleaned": mistral_cleaned
        }

        # Add the entry to the final dictionary
        extracted_data[item_id] = value_dict

    return extracted_data

def save_dict_to_json(data_dict, file_path):
    """
    Saves a dictionary to a JSON file with pretty-printing.

    Args:
        data_dict (dict): The dictionary to save.
        file_path (str): The path to the output JSON file.
    """
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            # Use json.dump to write the dictionary to the file
            json.dump(data_dict, f, indent=2, ensure_ascii=False)
        print(f"Successfully saved data to '{file_path}'")
    except IOError as e:
        print(f"Error writing to file '{file_path}': {e}")


# --- Main execution block ---
if __name__ == "__main__":
    # Define input and output file names for the eng dataset
    input_file = 'results/full_pipeline_results_24.json'
    output_file = 'extracted_data.json'

    # Define input and output file names for ita dataset
    #input_file = 'results/full_pipeline_results_12_ita.json'
    #output_file = 'extracted_data_ita.json'

    # Process the file to get the dictionary
    result_dict = process_pipeline_results(input_file)

    # Save the resulting dictionary to the new JSON file
    if result_dict:
        save_dict_to_json(result_dict, output_file)
    else:
        print("No data was processed, so no output file was created.")