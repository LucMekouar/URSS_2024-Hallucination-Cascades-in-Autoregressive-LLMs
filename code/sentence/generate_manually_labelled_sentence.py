# sample of 100 rows for human labelling

import pandas as pd

# File paths
input_file_path = '***.csv'
output_file_path = '***.csv'

# Load the dataset
df = pd.read_csv(input_file_path)

# Filter rows where:
# 1. The total_label_count is >= 10
# 2. The sequences have matching numbers of labels (we assume this is reflected in the total_label_count column being non-zero)
df_filtered = df[df['total_label_count'] >= 10]

# Group by 'generating model' and sample 20 rows per model
df_sampled = df_filtered.groupby('generating model_').apply(lambda x: x.sample(n=20, random_state=42)).reset_index(drop=True)

# Drop the specified columns
columns_to_drop = [
    'labeled_response_gemma2', 'labeled_response_llama3.1', 
    'clean_label_gemma2', 'clean_label_llama3.1', 
    'matching_0_count', 'matching_1_count', 
    'matching_2_count', 'total_label_count'
]
df_sampled = df_sampled.drop(columns=columns_to_drop)

# Move the 'response' column to the rightmost position
response_col = df_sampled.pop('response_')  
df_sampled['response_'] = response_col  

# Output the resulting 100 rows to a new CSV file
df_sampled.to_csv(output_file_path, index=False)

print(f"Sampled data saved to: {output_file_path}")
