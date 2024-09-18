# Script to tokenize responses using the Gemma-2-9B model.
# This script processes responses by merging them with relevant questions and answers,
# then tokenizes the text while preserving individual tokens.
# The tokenized text is saved to a CSV file.

import pandas as pd
from transformers import AutoTokenizer

# Load the Gemma-2-9B tokenizer
tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-9b")

# Load the CSV files
responses_file = "data_results/responses_gemma2.csv"
truthfulqa_file = "data/TruthfulQA.csv"

df_responses = pd.read_csv(responses_file)
df_truthfulqa = pd.read_csv(truthfulqa_file)

# Create a 'question_number' column in df_truthfulqa, starting from 0
df_truthfulqa['question_number'] = range(len(df_truthfulqa))

# Select only the relevant columns from df_truthfulqa
df_truthfulqa = df_truthfulqa[['question_number', 'Question', 'Correct Answers', 'Incorrect Answers']]

# Merge the two dataframes on the 'question_number' column
df_merged = pd.merge(df_responses, df_truthfulqa, on='question_number', how='left')

# Function to tokenize the text in the 'response' column while preserving individual tokens
def tokenize_text(text):
    tokens = tokenizer.tokenize(text)  # Tokenize the text
    tokens = tokens[:50]  # Keep only the first 50 tokens
    numbered_tokens = [f"{i + 1}: {token}" for i, token in enumerate(tokens)]  # Number each token
    return "\n".join(numbered_tokens)

# Apply the tokenization function to the 'response' column
df_merged['tokenized_response'] = df_merged['response'].apply(tokenize_text)

# Save the modified DataFrame back to a CSV file
output_file = "data_results/token/50.tokenized_gemma2.csv"
df_merged.to_csv(output_file, index=False)

print(f"Tokenized responses with questions saved to {output_file}")
