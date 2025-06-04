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


unified_dict = {}

print("\nBuilding final_dict...")
for key in ocr_data_dict:  
    if key in clear_data_dict:
        unified_dict[key] = {
            "ocr": ocr_data_dict[key],
            "clean": clear_data_dict[key]
        }
    else:
        print(f"Warning: Key '{key}' found in OCR data but not in clear data. Skipping this entry.")


for key in clear_data_dict:
    if key not in ocr_data_dict:
        print(f"Warning: Key '{key}' found in clear data but not in OCR data. This entry was not added.")

subset = []
print("\nGenerating subset of final_dict...")
 
final_dict = {}
subset_desired = 24 

for i in range(subset_desired):
    key = str(i)  
    if key in unified_dict:
        final_dict[key] = unified_dict[key]
        subset.append(unified_dict[key])
    else:
        print(f"Warning: Key '{key}' not found in unified_dict. Skipping this entry.")

print(f"\nNumber of items in subset: {len(final_dict)}")


# Display the first item in the subset
# print(f"example", final_dict["0"])

#save this subset
output_subset_path = "dataset/eng/the_vampyre_subset_24.json"
with open(output_subset_path, 'w', encoding='utf-8') as outfile:
    json.dump(final_dict, outfile, indent=2, ensure_ascii=False)
print(f"\nSubset dictionary saved to: {output_subset_path}")