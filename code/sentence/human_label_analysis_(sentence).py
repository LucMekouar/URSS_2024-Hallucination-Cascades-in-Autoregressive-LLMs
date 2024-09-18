import pandas as pd
import re
import ast
import sys

def main():
    # Define file paths
    file_path_human_labels = '/Users/lucmacbookpro-profile/Desktop/summer research/URSS 2024/data_results/sentence/sentence_human_labelled_data2.csv'
    file_path_big_data = '/Users/lucmacbookpro-profile/Desktop/summer research/URSS 2024/data_results/sentence/big_data_frame.csv'
    output_comparison_path = '/Users/lucmacbookpro-profile/Desktop/summer research/URSS 2024/data_results/sentence/comparison_results.csv'

    # Load the human-labeled dataset
    try:
        data_human = pd.read_csv(file_path_human_labels)
        print("Loaded 'sentence_human_labelled_data2.csv' successfully.")
    except FileNotFoundError:
        print(f"Error: The file '{file_path_human_labels}' was not found.")
        sys.exit(1)
    except pd.errors.EmptyDataError:
        print(f"Error: The file '{file_path_human_labels}' is empty.")
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred while loading '{file_path_human_labels}': {e}")
        sys.exit(1)

    # Load the big data frame dataset
    try:
        data_big = pd.read_csv(file_path_big_data)
        print("Loaded 'big_data_frame.csv' successfully.")
    except FileNotFoundError:
        print(f"Error: The file '{file_path_big_data}' was not found.")
        sys.exit(1)
    except pd.errors.EmptyDataError:
        print(f"Error: The file '{file_path_big_data}' is empty.")
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred while loading '{file_path_big_data}': {e}")
        sys.exit(1)

    # Function to extract the last character as label, replace non-digits with 2
    def extract_label(line):
        line = line.strip()
        if line:
            last_char = line[-1]
            if last_char.isdigit():
                return int(last_char)
            else:
                return 2
        else:
            return 2  # Default value for empty lines

    # Function to extract and process labels from the 'response_' column
    def extract_labels_from_response(response):
        lines = response.splitlines()
        labels = [extract_label(line) for line in lines if line.strip()]
        # Replace 2 and 3 with 0
        labels = [0 if label in [2, 3] else label for label in labels]
        return labels

    # Check if 'human_labels' column exists; if not, create it
    if 'human_labels' not in data_human.columns:
        if 'response_' in data_human.columns:
            data_human['human_labels'] = data_human['response_'].apply(extract_labels_from_response)
            print("'human_labels' column created successfully.")
        else:
            print("Error: 'response_' column not found in 'sentence_human_labelled_data2.csv'.")
            sys.exit(1)
    else:
        print("'human_labels' column already exists.")

    # Ensure 'human_labels' are lists of integers
    def parse_human_labels(label):
        if isinstance(label, str):
            try:
                # Safely evaluate the string to a list
                return ast.literal_eval(label)
            except (ValueError, SyntaxError):
                print(f"Warning: Unable to parse 'human_labels' entry: {label}. Assigning empty list.")
                return []
        elif isinstance(label, list):
            return label
        else:
            return []

    data_human['human_labels'] = data_human['human_labels'].apply(parse_human_labels)

    # Function to process 'labeled_response' in data_big
    def convert_labeled_response_to_list(labeled_response):
        if pd.isnull(labeled_response):
            return []
        # Remove any non-digit and non-comma characters
        cleaned = re.sub(r'[^\d,]', '', labeled_response)
        # Split by comma
        parts = cleaned.split(',')
        # Convert to integers, replace 2 and 3 with 0
        labels = []
        for part in parts:
            part = part.strip()
            if part.isdigit():
                num = int(part)
                if num in [2, 3]:
                    labels.append(0)
                else:
                    labels.append(num)
            else:
                labels.append(0)  # Default for unexpected formats
        return labels

    # Apply conversion to 'labeled_response' in data_big
    if 'labeled_response' in data_big.columns:
        data_big['processed_labeled_response'] = data_big['labeled_response'].apply(convert_labeled_response_to_list)
        print("'labeled_response' column processed successfully.")
    else:
        print("Error: 'labeled_response' column not found in 'big_data_frame.csv'.")
        sys.exit(1)

    # Define the columns to merge on
    # In data_human: 'question_number_', 'generating model_', 'input_temperature_'
    # In data_big: 'question_number', 'generating model', 'input_temperature'
    merge_columns_human = ['question_number_', 'generating model_', 'input_temperature_']
    merge_columns_big = ['question_number', 'generating model', 'input_temperature']

    # Check if all merge columns exist in both datasets
    missing_columns_human = [col for col in merge_columns_human if col not in data_human.columns]
    missing_columns_big = [col for col in merge_columns_big if col not in data_big.columns]

    if missing_columns_human:
        print(f"Error: Missing columns in 'sentence_human_labelled_data2.csv': {missing_columns_human}")
        sys.exit(1)

    if missing_columns_big:
        print(f"Error: Missing columns in 'big_data_frame.csv': {missing_columns_big}")
        sys.exit(1)

    # Rename columns in data_human to match those in data_big for merging
    data_human.rename(columns={
        'question_number_': 'question_number',
        'generating model_': 'generating model',
        'input_temperature_': 'input_temperature'
    }, inplace=True)

    # Now, 'question_number', 'generating model', 'input_temperature' exist in both datasets
    merge_columns = ['question_number', 'generating model', 'input_temperature']

    # Merge the datasets
    merged_data = pd.merge(data_human, data_big, on=merge_columns, how='inner')
    print(f"Merged data contains {len(merged_data)} rows.")

    # Function to compare two lists of labels
    def compare_labels(human_labels, labeled_response):
        if len(human_labels) != len(labeled_response):
            return {'length_mismatch': True, 'differences': []}
        differences = []
        for idx, (h, l) in enumerate(zip(human_labels, labeled_response)):
            if h != l:
                differences.append({'index': idx, 'human_label': h, 'labeled_response': l})
        return {'length_mismatch': False, 'differences': differences}

    # Initialize list to store comparison results
    comparison_results = []

    # Initialize counters for total labels and matched labels
    total_labels = 0
    matched_labels = 0

    # Iterate over merged_data rows to compare labels
    for _, row in merged_data.iterrows():
        human_labels = row['human_labels']
        labeled_response = row['processed_labeled_response']
        comparison = compare_labels(human_labels, labeled_response)

        if comparison['length_mismatch']:
            # Compare up to the length of the shorter list
            min_length = min(len(human_labels), len(labeled_response))
            for i in range(min_length):
                total_labels += 1
                if human_labels[i] == labeled_response[i]:
                    matched_labels += 1
            # Optionally, count the extra labels as mismatches or ignore them
            # Here, we'll ignore extra labels beyond the minimum length
            comparison_results.append({
                'question_number': row['question_number'],
                'generating model': row['generating model'],
                'input_temperature': row['input_temperature'],
                'length_mismatch': True,
                'differences': comparison['differences']
            })
        else:
            # Compare each label
            for h, l in zip(human_labels, labeled_response):
                total_labels += 1
                if h == l:
                    matched_labels += 1
            if comparison['differences']:
                comparison_results.append({
                    'question_number': row['question_number'],
                    'generating model': row['generating model'],
                    'input_temperature': row['input_temperature'],
                    'length_mismatch': False,
                    'differences': comparison['differences']
                })

    # Create a DataFrame from comparison_results
    if comparison_results:
        comparison_df = pd.DataFrame(comparison_results)
        print(f"Found {len(comparison_df)} rows with mismatched labels.")
    else:
        comparison_df = pd.DataFrame(columns=['question_number', 'generating model', 'input_temperature', 'length_mismatch', 'differences'])
        print("No mismatches found between 'human_labels' and 'labeled_response'.")

    # Save comparison results to CSV
    try:
        comparison_df.to_csv(output_comparison_path, index=False)
        print(f"Comparison results saved to '{output_comparison_path}'.")
    except Exception as e:
        print(f"An error occurred while saving the comparison results: {e}")
        sys.exit(1)

    # Calculate and print the percentage of matching labels
    if total_labels > 0:
        percentage = (matched_labels / total_labels) * 100
        print(f"Percentage of labels that are the same: {percentage:.2f}%")
    else:
        print("No labels were compared.")

if __name__ == "__main__":
    main()
