# sample of 20 rows for human labelling

import os
import glob
import json
import pandas as pd
import numpy as np
import random

# Set a fixed random seed for reproducibility
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# Define data sources and model name mapping
data_sources = [
    ('***', 'llama3.1'),
    ('***', 'gemma2')
]

model_name_map = {
    'qwen2(0.5b)': 'Qwen2(0.5B)',
    'qwen2(7b)': 'Qwen2(7B)',
    'qwen2': 'Qwen2',
    'llama3.1': 'Llama3.1',
    'gemma2': 'Gemma2',
    'mistral-nemo': 'Mistral-NeMo',
}

# Function to load data entries with exactly 50 tokens and labels
def load_data_entries(data_sources):
    data_entries = []
    for folder_path, labelling_model in data_sources:
        # Use glob to match files starting with 'token_labelled_' and ending with '.json'
        pattern = os.path.join(folder_path, f'token_labelled_*_(by_{labelling_model}).json')
        for file_path in glob.glob(pattern):
            # Extract filename
            filename = os.path.basename(file_path)
            # Remove 'token_labelled_' prefix and '.json' suffix
            model_part = filename[len('token_labelled_'):-len('.json')]
            # Extract generating model name
            if f'_(by_{labelling_model})' in model_part:
                model_name_raw = model_part.split(f'_(by_{labelling_model})')[0]
            else:
                model_name_raw = model_part
            # Standardize model names
            model_name_key = model_name_raw.lower()
            model_name = model_name_map.get(model_name_key, model_name_raw)
            # Debugging: Print model being processed
            print(f"Processing model: {model_name} labelled by {labelling_model}")
            with open(file_path, 'r') as file:
                try:
                    data = json.load(file)
                except json.JSONDecodeError:
                    print(f"Error decoding JSON in file: {file_path}. Skipping this file.")
                    continue
                # Add generating model and labelling model info to each entry
                for entry in data:
                    # Ensure '# of labels' is exactly 50
                    if entry.get("# of labels") != 50:
                        continue
                    # Ensure 'labeled_tokens' has exactly 50 tokens
                    labeled_tokens = entry.get("labeled_tokens", "")
                    tokens_list = labeled_tokens.split(',')
                    if len(tokens_list) != 50:
                        continue
                    # Optionally, ensure 'tokenized_response' has exactly 50 tokens
                    tokenized_response = entry.get("tokenized_response", "")
                    token_list = tokenized_response.split('\n')
                    if len(token_list) != 50:
                        continue
                    # All conditions met; add to data_entries
                    entry['generating_model'] = model_name.lower()
                    entry['labelling_model'] = labelling_model.lower()
                    data_entries.append(entry)
    return data_entries

# Load the data entries
data_entries = load_data_entries(data_sources)

# Convert to DataFrame
df = pd.DataFrame(data_entries)

# Check if DataFrame is empty
if df.empty:
    print("No data entries found with exactly 50 tokens and labels.")
    exit()

# Ensure 'generating_model' and 'labelling_model' columns are properly formatted
df['generating_model'] = df['generating_model'].str.lower()
df['labelling_model'] = df['labelling_model'].str.lower()

# Define the combinations of generating modes and labelling models
generating_modes = ['llama3.1', 'gemma2', 'mistral-nemo', 'qwen2', 'qwen2(0.5b)']
labelling_models = ['llama3.1', 'gemma2']

# Initialize a list to hold sampled DataFrames
sampled_dfs = []

# Sample 2 entries per combination
for labelling_model in labelling_models:
    for generating_mode in generating_modes:
        subset = df[
            (df['generating_model'] == generating_mode.lower()) &
            (df['labelling_model'] == labelling_model.lower())
        ]
        # Check if there are enough entries to sample
        if len(subset) >= 2:
            sampled_subset = subset.sample(n=2, random_state=RANDOM_SEED)
            sampled_dfs.append(sampled_subset)
            print(f"Sampled 2 entries for generating model '{generating_mode}' labelled by '{labelling_model}'.")
        else:
            print(f"Not enough data for generating model '{generating_mode}' labelled by '{labelling_model}'. Required: 2, Available: {len(subset)}")

# Combine all sampled DataFrames
if sampled_dfs:
    final_df = pd.concat(sampled_dfs, ignore_index=True)
else:
    print("No sufficient data found for any combination. No CSV file will be generated.")
    exit()

# Output the DataFrame to a CSV file
output_csv_path = '***.csv'
final_df.to_csv(output_csv_path, index=False)

print(f"Sampled data saved to '{output_csv_path}'.")
print(f"Total sampled entries: {len(final_df)}")
