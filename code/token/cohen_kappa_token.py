import os
import re
import pandas as pd
import numpy as np
import csv
from sklearn.metrics import cohen_kappa_score

def load_labelled_data_human(file_path):
    """
    Loads the labelled_sampled_data_human CSV file.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found at: {file_path}")

    # Read the CSV file, ensuring proper handling of multiline fields
    df = pd.read_csv(file_path, engine='python', quoting=csv.QUOTE_ALL, keep_default_na=False)
    return df

def extract_human_labels(tokenized_response):
    """
    Extracts human labels from the tokenized_response field.
    """
    human_labels = []
    lines = tokenized_response.strip().split('\n')

    for line in lines:
        # Example line: "1: Cut0"
        if ':' in line:
            # Split into token number and token with label
            parts = line.split(':', 1)
            token_with_label = parts[1].strip()

            # Extract the label, which should be the last character
            if token_with_label[-1] in ['0', '1']:
                label = token_with_label[-1]
                human_labels.append(int(label))
            else:
                print(f"Could not extract label from line: '{line}'. Skipping this label.")
        else:
            print(f"Line does not contain ':': '{line}'. Skipping this line.")
    return human_labels

def extract_model_labels(labeled_tokens):
    """
    Extracts and recodes labels from the labeled_tokens field.
    """
    model_labels = []
    tokens = [label.strip() for label in labeled_tokens.split(',')]
    for label in tokens:
        try:
            label_int = int(label)
            # Recode label '2' as '0'
            if label_int == 2:
                label_int = 0
            elif label_int not in [0, 1]:
                print(f"Unexpected label value '{label_int}'. Recoding to '0'.")
                label_int = 0
            model_labels.append(label_int)
        except ValueError:
            print(f"Non-integer label '{label}' found. Recoding to '0'.")
            model_labels.append(0)
    return model_labels

def main():
    # Define file path
    human_labels_file = '***/data/human_labelling/token_human_labelled_data.csv'

    print("Loading labelled_sampled_data_human file...")
    try:
        df = load_labelled_data_human(human_labels_file)
    except Exception as e:
        print(f"Error loading file: {e}")
        return

    # Initialize dictionaries to hold labels per model
    human_labels_per_model = {'gemma2': [], 'llama3.1': []}
    model_labels = {'gemma2': [], 'llama3.1': []}

    # Iterate through each row to extract labels
    for idx, row in df.iterrows():
        labelling_model = row['labelling_model'].strip().lower()
        if labelling_model not in ['gemma2', 'llama3.1']:
            print(f"Row {idx+1}: Unknown labelling_model '{labelling_model}'. Skipping this row.")
            continue

        print(f"\nProcessing row {idx+1}/{len(df)} for labelling model '{labelling_model}'...")

        # Extract model labels
        labeled_tokens_str = row['labeled_tokens']
        model_labels_list = extract_model_labels(labeled_tokens_str)

        if len(model_labels_list) != 50:
            print(f"Row {idx+1}: Expected 50 model labels, found {len(model_labels_list)}. Skipping this row.")
            continue

        # Extract human labels from 'tokenized_response' column
        tokenized_response = row['tokenized_response']
        human_labels_list = extract_human_labels(tokenized_response)

        if len(human_labels_list) != 50:
            print(f"Row {idx+1}: Expected 50 human labels, found {len(human_labels_list)}. Skipping this row.")
            continue

        # Append labels to respective lists
        human_labels_per_model[labelling_model].extend(human_labels_list)
        model_labels[labelling_model].extend(model_labels_list)

        # Print the labels for verification
        print(f"Human labels for row {idx+1}: {human_labels_list}")
        print(f"Model labels for row {idx+1}: {model_labels_list}")

    # Compute Cohen's Kappa for each model
    for model_name in ['gemma2', 'llama3.1']:
        human_labels = human_labels_per_model[model_name]
        model_labels_list = model_labels[model_name]

        human_labels_count = len(human_labels)
        model_labels_count = len(model_labels_list)

        print(f"\nSummary for '{model_name}':")
        print(f"Total Human Labels: {human_labels_count}")
        print(f"Total {model_name.capitalize()} Labels: {model_labels_count}")

        if human_labels_count != model_labels_count or human_labels_count == 0:
            print(f"Mismatch in label counts or no labels for '{model_name}'. Skipping Cohen's Kappa calculation.")
            continue

        # Print the combined label sequences for verification
        print(f"\nCombined Human Labels for '{model_name}':\n{human_labels}")
        print(f"Combined {model_name.capitalize()} Labels:\n{model_labels_list}")

        # Compute Cohen's Kappa using sklearn
        kappa = cohen_kappa_score(human_labels, model_labels_list)
        print(f"\nCohen's Kappa between Human and {model_name.capitalize()}: {kappa:.4f}")

if __name__ == "__main__":
    main()
