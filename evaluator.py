import json
from sklearn.metrics import cohen_kappa_score
import pandas as pd

def analyze_rater_agreement(json_file_path, output_report_path="kappa_analysis_report.txt"):
    """
    Reads a JSON results file, extracts human and LLM judge scores,
    calculates Cohen's Kappa, prints the analysis, and saves it to a file.
    
    Args:
        json_file_path (str): Path to the input JSON file.
        output_report_path (str): Path to save the final text report.
    """
    # This list will capture all output for saving to a file
    report_lines = []

    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: The file '{json_file_path}' was not found.")
        return
    except json.JSONDecodeError:
        print(f"Error: The file '{json_file_path}' is not a valid JSON file.")
        return

    # These lists will store the scores for comparison
    human_scores = []
    llm_judge_scores = []
    comparison_data = []

    # --- Step 1: Parse the JSON and find all samples with a human score ---
    # (No changes in this part)
    for item in data:
        for model_output in item['model_outputs']:
            if 'human_score' in model_output.get('judgement', {}):
                human_score = model_output['judgement']['human_score']
                llm_score = model_output['judgement']['score']
                
                human_scores.append(human_score)
                llm_judge_scores.append(llm_score)
                
                comparison_data.append({
                    "item_id": item['item_id'],
                    "model_name": model_output['model_name'],
                    "human_score": human_score,
                    "llm_judge_score": llm_score,
                })

    # --- Step 2: Check if we found data and perform calculations ---
    if not human_scores:
        print("\nNo samples with a 'human_score' key were found. Cannot calculate Kappa.")
        return

    # --- Step 3: Build the Report ---
    # From here on, every piece of information is both printed and added to `report_lines`
    
    line = f"Found {len(human_scores)} samples with both human and LLM judge scores."
    print(line)
    report_lines.append(line)
    
    summary_df = pd.DataFrame(comparison_data)
    
    line = "\n--- Data Used for Calculation ---"
    print(line)
    report_lines.append(line)
    
    # Use to_string() to get the full DataFrame as a string for the report
    df_string = summary_df.to_string()
    print(df_string)
    report_lines.append(df_string)

    line = "\n--- Raw Score Lists ---"
    print(line)
    report_lines.append(line)

    line = f"Your Scores (Rater 1):      {human_scores}"
    print(line)
    report_lines.append(line)

    line = f"LLM Judge Scores (Rater 2): {llm_judge_scores}"
    print(line)
    report_lines.append(line)

    # Calculations
    agreements = sum(1 for h, l in zip(human_scores, llm_judge_scores) if h == l)
    simple_agreement = (agreements / len(human_scores)) * 100
    unweighted_kappa = cohen_kappa_score(human_scores, llm_judge_scores)
    weighted_kappa = cohen_kappa_score(human_scores, llm_judge_scores, weights='quadratic')

    line = "\n--- Agreement Analysis ---"
    print(line)
    report_lines.append(line)
    
    line = f"Simple Agreement:                 {simple_agreement:.2f}%"
    print(line)
    report_lines.append(line)

    line = f"Cohen's Kappa (Unweighted):       {unweighted_kappa:.3f}"
    print(line)
    report_lines.append(line)

    line = f"Cohen's Kappa (Quadratic Weighted): {weighted_kappa:.3f}  <-- This is the recommended metric"
    print(line)
    report_lines.append(line)

    # Interpretation
    line = "\n--- Interpretation of Weighted Kappa ---"
    print(line)
    report_lines.append(line)

    if weighted_kappa > 0.8:
        interpretation = "Almost Perfect Agreement"
    elif weighted_kappa > 0.6:
        interpretation = "Substantial Agreement"
    elif weighted_kappa > 0.4:
        interpretation = "Moderate Agreement"
    elif weighted_kappa > 0.2:
        interpretation = "Fair Agreement"
    else:
        interpretation = "Slight or Poor Agreement"
    
    line = f"The score of {weighted_kappa:.3f} indicates: {interpretation}"
    print(line)
    report_lines.append(line)
    
    line = "\n--- Recommendation ---"
    print(line)
    report_lines.append(line)

    if weighted_kappa < 0.6:
        recommendation = "The agreement is not yet substantial. It would be wise to analyze the disagreements in the table above to refine the LLM Judge's prompt or logic."
    else:
        recommendation = "The agreement is substantial. You can have a good degree of confidence in the LLM Judge's scoring on the rest of the dataset."
    print(recommendation)
    report_lines.append(recommendation)
    
    # --- Step 4: Save the report to a file ---
    try:
        with open(output_report_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(report_lines))
        print(f"\n Report successfully saved to '{output_report_path}'")
    except IOError as e:
        print(f"\n Error: Could not save the report file. Reason: {e}")

# --- Run the analysis ---
if __name__ == "__main__":
    json_file = "results/full_pipeline_results_human.json"
    analyze_rater_agreement(json_file)