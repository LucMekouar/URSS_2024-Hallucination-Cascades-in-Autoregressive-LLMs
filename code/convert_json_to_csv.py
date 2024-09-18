# conversrs a JSON file to a CSV file

import pandas as pd
import json

# Input and output file paths
input_json_file = '/Users/lucmacbookpro-profile/Desktop/summer research/URSS 2024/data_results/json/responses_qwen2:0.5b.json'
output_csv_file = '/Users/lucmacbookpro-profile/Desktop/summer research/URSS 2024/data_results/responses_qwen2(0.5b).csv'

#Read the JSON data
with open(input_json_file, 'r') as f:
    json_data = json.load(f)

# Check if JSON data is in a list of dictionaries or a dictionary with consistent structure
if isinstance(json_data, list) or isinstance(json_data, dict):
    # Convert the JSON data to a DataFrame
    df = pd.DataFrame(json_data)
    
    # Write the DataFrame to a CSV file
    df.to_csv(output_csv_file, index=False)
    
    print(f"Data has been successfully converted from {input_json_file} to {output_csv_file}.")
else:
    print("Unsupported JSON format. Please provide a list of dictionaries or a consistent dictionary structure.")
