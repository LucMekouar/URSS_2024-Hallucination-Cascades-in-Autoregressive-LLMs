# Script to calculate Cohen's Kappa from a recoded CSV file, preserving sequence structure and handling labels

import pandas as pd
import numpy as np

# Load the recoded data
file_path = '/Users/lucmacbookpro-profile/Desktop/summer research/URSS 2024/data_results/sentence/data_frame_recoded.csv'
df = pd.read_csv(file_path, encoding='utf-8')

# Process labels, preserving the sequence structure and underscores
def process_labels(label_sequence):
    label_sequence = label_sequence.strip()  # Remove leading/trailing whitespace
    labels = list(label_sequence)  # Convert to list of characters
    return labels

# Apply label processing
df['clean_label_gemma2_processed'] = df['clean_label_gemma2'].apply(process_labels)
df['clean_label_llama3.1_processed'] = df['clean_label_llama3.1'].apply(process_labels)

# Extract label pairs, skipping underscore placeholders
label_pairs = []
for _, row in df.iterrows():
    labels_gemma2 = row['clean_label_gemma2_processed']
    labels_llama3 = row['clean_label_llama3.1_processed']
    
    if len(labels_gemma2) == len(labels_llama3):
        label_pairs.extend([(g, l) for g, l in zip(labels_gemma2, labels_llama3) if g != '_' and l != '_'])

# Convert the label pairs to a DataFrame
label_pairs_df = pd.DataFrame(label_pairs, columns=['gemma2_label', 'llama3.1_label'])

# Create a confusion matrix (contingency table)
confusion_matrix = pd.crosstab(label_pairs_df['gemma2_label'], label_pairs_df['llama3.1_label'], rownames=['gemma2'], colnames=['llama3.1']).to_numpy()

# Calculate Cohen's Kappa
n = np.sum(confusion_matrix)
observed_agreement = np.trace(confusion_matrix) / n
row_totals = np.sum(confusion_matrix, axis=1)
col_totals = np.sum(confusion_matrix, axis=0)
expected_agreement = np.sum(row_totals * col_totals) / (n ** 2)
kappa = (observed_agreement - expected_agreement) / (1 - expected_agreement)

# Output the result
print(f"Cohen's Kappa: {kappa:.4f}")
