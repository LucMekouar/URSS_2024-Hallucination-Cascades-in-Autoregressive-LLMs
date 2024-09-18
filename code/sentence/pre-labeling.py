# splits responses into individual sentences, numbering each sentence, and adding to the the output file question, correct answers, and incorrect answers

import pandas as pd
import nltk
import re

# Set the NLTK data path
nltk_data_path = '/Users/lucmacbookpro-profile/Desktop/summer research/URSS 2024/punkt'
nltk.data.path.append(nltk_data_path)

# Ensure punkt is available
from nltk.tokenize import sent_tokenize

def segment_text_in_csv(input_csv, output_csv, column_name='response', original_csv_path=None):
    # Load the CSV file
    df = pd.read_csv(input_csv)

    # Load the original CSV file with "Correct Answers" and "Incorrect Answers" if provided
    if original_csv_path:
        original_df = pd.read_csv(original_csv_path)
        
        # Ensure the question numbers are aligned
        original_df = original_df[['Question', 'Correct Answers', 'Incorrect Answers']].reset_index().rename(columns={'index': 'question_number'})
        
        # Merge the original data with the response data
        df = df.merge(original_df, on='question_number', how='left')

    # Function to clean up the text and number the sentences correctly
    def segment_and_number(text):
        # Remove existing numbering using regex
        text = re.sub(r'^\d+\.\s*', '', text, flags=re.MULTILINE)
        
        # Replace newline characters within quotes with spaces to keep sentences together
        text = re.sub(r'\n+', ' ', text)
        
        sentences = sent_tokenize(text)
        # Truncate to the first 25 sentences
        truncated_sentences = sentences[:25]
        
        numbered_sentences = []
        for i, sentence in enumerate(truncated_sentences):
            numbered_sentences.append(f"{i+1}. {sentence.strip()}")
        
        return "\n".join(numbered_sentences)
    
    # Apply the function to the specified column
    df[column_name] = df[column_name].apply(segment_and_number)
    
    # Save the modified DataFrame to a new CSV file
    df.to_csv(output_csv, index=False)

# Input and output file paths
input_csv = '/Users/lucmacbookpro-profile/Desktop/summer research/URSS 2024/data_results/responses_qwen2(0.5b).csv'
output_csv = '/Users/lucmacbookpro-profile/Desktop/summer research/URSS 2024/data_results/cleaned_files/0000.clean_responses_qwen2(0.5b).csv'
question_csv_path = '/Users/lucmacbookpro-profile/Desktop/summer research/URSS 2024/data/TruthfulQA.csv'  

# Implementation
segment_text_in_csv(input_csv, output_csv, column_name='response', original_csv_path=question_csv_path)
