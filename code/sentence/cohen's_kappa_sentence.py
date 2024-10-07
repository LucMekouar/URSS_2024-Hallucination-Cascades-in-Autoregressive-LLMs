# use cohen's kappa to compare human and automated (both labelling model separately) labels for sentence analysis

import os
import pandas as pd
import numpy as np
from sklearn.metrics import cohen_kappa_score

def extract_human_labels(response_text):
    """
    Extracts labels from the human response text.
    Args:
        response_text (str): The response text containing labels at the end of each line.
    Returns:
        list: List of integer labels extracted from the response.
    """
    labels = []
    lines = response_text.strip().split('\n')
    for line in lines:
        line = line.strip()
        if not line:
            continue  # Skip empty lines
        # Extract the label from the end of the line
        if line[-1] in ['0', '1', '2', '3']:
            label = int(line[-1])
            # Recode label '2' to '0'
            if label == 2:
                label = 0
            labels.append(label)
        else:
            print(f"Could not extract label from line: '{line}'. Skipping this line.")
    return labels

def extract_model_labels(labeled_response):
    """
    Extracts and recodes labels from the labeled_response field.
    Args:
        labeled_response (str): Comma-separated string of labels from the labelling model.
    Returns:
        list: List of integer labels (0, 1, or 3) after recoding '2' to '0'.
    """
    labels = [label.strip() for label in labeled_response.strip().split(',')]
    recoded_labels = []
    for label in labels:
        try:
            label_int = int(label)
            # Recode label '2' to '0'
            if label_int == 2:
                label_int = 0
            elif label_int not in [0, 1, 3]:
                print(f"Unexpected label value '{label_int}'. Recoding to '0'.")
                label_int = 0
            recoded_labels.append(label_int)
        except ValueError:
            print(f"Non-integer label '{label}' found. Recoding to '0'.")
            recoded_labels.append(0)
    return recoded_labels

def main():
    # Define file paths
    human_labels_file = '***/data/human_labelling/sentence_human_labelled_data.csv'
    automated_labels_file = '***.csv'
    
    print("Loading human labelled data file...")
    try:
        df_human = pd.read_csv(human_labels_file)
    except Exception as e:
        print(f"Error loading human labelled data file: {e}")
        return
    
    print("Loading automated labelled data file...")
    try:
        df_auto = pd.read_csv(automated_labels_file)
    except Exception as e:
        print(f"Error loading automated labelled data file: {e}")
        return
    
    # Rename columns in df_human to match df_auto
    df_human = df_human.rename(columns={
        'question_number_': 'question_number',
        'generating model_': 'generating model',
        'input_temperature_': 'input_temperature'
    })
    
    # Ensure data types match for merging
    df_human['question_number'] = df_human['question_number'].astype(int)
    df_human['input_temperature'] = df_human['input_temperature'].astype(float)
    df_auto['question_number'] = df_auto['question_number'].astype(int)
    df_auto['input_temperature'] = df_auto['input_temperature'].astype(float)
    
    # Initialize dictionaries to hold labels per model
    human_labels_per_model = {'gemma2': [], 'llama3.1': []}
    model_labels = {'gemma2': [], 'llama3.1': []}
    
    # Iterate through each row in df_human
    for idx, row in df_human.iterrows():
        question_number = row['question_number']
        generating_model = row['generating model']
        input_temperature = row['input_temperature']
        response_text = row['response_']
        
        # Extract human labels
        human_labels_list = extract_human_labels(response_text)
        
        # For each labelling model
        for labelling_model in ['gemma2', 'llama3.1']:
            # Find matching row in df_auto
            auto_rows = df_auto[
                (df_auto['question_number'] == question_number) &
                (df_auto['generating model'] == generating_model) &
                (df_auto['input_temperature'] == input_temperature) &
                (df_auto['labelling model'] == labelling_model)
            ]
            if auto_rows.empty:
                print(f"No matching automated data for question_number {question_number}, generating_model {generating_model}, input_temperature {input_temperature}, labelling_model {labelling_model}")
                continue
            # There may be multiple matches, take the first one
            auto_row = auto_rows.iloc[0]
            # Extract model labels
            labeled_response = auto_row['labeled_response']
            model_labels_list = extract_model_labels(labeled_response)
            
            # Now, check if the number of labels matches
            if len(human_labels_list) != len(model_labels_list):
                print(f"Label count mismatch for question_number {question_number}, generating_model {generating_model}, input_temperature {input_temperature}, labelling_model {labelling_model}")
                print(f"Human labels: {len(human_labels_list)}, Model labels: {len(model_labels_list)}")
                continue
            # Append labels to the lists
            human_labels_per_model[labelling_model].extend(human_labels_list)
            model_labels[labelling_model].extend(model_labels_list)
    
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
        
        # Compute Cohen's Kappa using sklearn
        kappa = cohen_kappa_score(human_labels, model_labels_list)
        print(f"Cohen's Kappa between Human and {model_name.capitalize()}: {kappa:.4f}")
    
if __name__ == "__main__":
    main()
