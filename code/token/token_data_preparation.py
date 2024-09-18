# data_preparation.py

import os
import glob
import json
import pandas as pd

def load_data(folder_path):
    data_frames = []
    # Use glob to match files starting with 'token_labelled_' and ending with '.json'
    pattern = os.path.join(folder_path, 'token_labelled_*.json')
    for file_path in glob.glob(pattern):
        # Extract model name from filename
        filename = os.path.basename(file_path)
        model_name = filename.replace('token_labelled_', '').replace('_(by_llama3.1).json', '').replace('.json', '').lower()
        # Standardize model names
        model_name = model_name.replace('qwen2(0.5b)', 'qwen2(0.5b)')
        with open(file_path, 'r') as file:
            data = json.load(file)
            df = parse_data(data, model_name)
            data_frames.append(df)
    # Combine all data into a single DataFrame
    combined_df = pd.concat(data_frames, ignore_index=True)
    return combined_df

def parse_data(data, model_name):
    records = []
    for entry in data:
        question_number = entry['question_number']
        input_temperature = float(entry['input_temperature'])
        # Tokens and labels
        tokens = entry['tokenized_response'].split('\n')
        labels = entry['labeled_tokens'].split(',')
        # Ensure tokens and labels have the same length
        if len(tokens) != len(labels):
            continue  # Skip entries with mismatched tokens and labels
        for i in range(len(tokens)):
            try:
                token_number, token = tokens[i].split(':', 1)
                label = int(labels[i])
                records.append({
                    'model': model_name,
                    'question_number': question_number,
                    'temperature': input_temperature,
                    'token_position': i + 1,
                    'token': token.strip(),
                    'label': label
                })
            except ValueError:
                continue  # Skip malformed tokens
    df = pd.DataFrame(records)
    return df

if __name__ == "__main__":
    folder_path = '/Users/lucmacbookpro-profile/Desktop/summer research/URSS 2024/data_results/token'
    combined_df = load_data(folder_path)
    # Save combined data for further analysis
    combined_df.to_csv('combined_token_data.csv', index=False)
