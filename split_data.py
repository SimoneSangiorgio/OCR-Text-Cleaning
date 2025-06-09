import json

# Path to the dataset file
input_ocr_path = "dataset/eng/the_vampyre_ocr.json"
input_clear_path = "dataset/eng/the_vampyre_clean.json"

# Load the full JSON file
with open(input_ocr_path, 'r', encoding='utf-8') as f:
    ocr_data_dict = json.load(f)

with open(input_clear_path, 'r', encoding='utf-8') as f2:
    clear_data_dict = json.load(f2)

print(f"Loaded data type: {type(ocr_data_dict)}")
print(f"Total key-value pairs in dictionary: {len(ocr_data_dict)}")

print(f"Loaded data type: {type(clear_data_dict)}")
print(f"Total key-value pairs in dictionary: {len(clear_data_dict)}")

# Split and take the first 24 lines
ocr_lines = ocr_data_dict["0"].split('\n')
subset_len = 24
ocr_subset_dict = {str(i): ocr_lines[i] for i in range(subset_len)}

# Print each line
for key, sentence in ocr_subset_dict.items():
    print(f"{key}: {sentence}")

# Save to new JSON file
output_subset_path = "dataset/eng/the_vampyre_subset_24_part2.json"
with open(output_subset_path, 'w', encoding='utf-8') as outfile:
    json.dump(ocr_subset_dict, outfile, indent=2, ensure_ascii=False)

print(f"\nSubset dictionary saved to: {output_subset_path}")