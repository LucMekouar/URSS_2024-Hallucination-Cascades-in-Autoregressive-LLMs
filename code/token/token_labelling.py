# labelling at a token level: hallucinatory, not hallucinatory or factual information that cannot be verified 
# ordered label pairs are introduced in the prompt simply for processing, after a pilot study found this format to output the most consistent results

import pandas as pd
import aiohttp
import asyncio
import logging
import re
import json
from concurrent.futures import ProcessPoolExecutor

# Setting up logging to show only essential information
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

# Function to load a CSV file
def load_csv(file_path):
    try:
        df = pd.read_csv(file_path, encoding='utf-8')
        return df
    except Exception as e:
        logging.error(f"Failed to load CSV file: {e}")
        raise

# Function to normalize and parse labels from the API response
def parse_labels(labeled_response):
    # Step 1: Normalize various formats into a consistent format
    # Replace different separators (space, underscore, period) with a colon
    normalized_response = re.sub(r'[\s_\.]', ':', labeled_response)

    # Step 2: Remove any leading zeros from the numbering
    normalized_response = re.sub(r'(\D|^)(0)(\d)', r'\1\3', normalized_response)

    # Step 3: Use regex to capture pairs in the form "number:label"
    # The pattern matches numbers followed by a colon and a single-digit label
    parsed_labels = re.findall(r'\d+:\d', normalized_response)

    if not parsed_labels:
        logging.warning(f"Could not parse the response properl. Skipping...")
        return None, None

    # Extract only the labels
    labels = [label.split(':')[1].strip() for label in parsed_labels]
    return labels, ','.join(parsed_labels)

# Async function to generate labels for each token in a response using the API
async def generate_labels_for_tokens(session, model_name, question, full_response, tokenized_response, 
                                     correct_answers, incorrect_answers, temperature, top_p, random_seed, max_tokens, retries=1):
    url = "http://localhost:11434/api/generate"
    headers = {'Content-Type': 'application/json'}
    
    prompt = f"""
Read the following question and full response to have the entire context before annotating:

Question: "{question}"

Full Response: "{full_response}"

Correct Answer(s): "{correct_answers}"

Incorrect Answer(s): "{incorrect_answers}"

Instructions for labeling:

1. Each token in the response below should be labeled individually, taking into account both its meaning and position within the response.
2. Compare each token with the full context provided (Full Response, Question, Correct Answer(s), Incorrect Answer(s)) and decide if the token is Hallucinatory or not.
3. Assign one of the following labels to each token:
   - 0: Not Hallucinatory - The token contributes to a passage that is credible and consistent within the given context. If in doubt, default to 0.
   - 1: Hallucinatory - The token is part of a passage that contradicts the context, or you can confidently recall facts that disprove it.
   - 2: Factual information that I cannot verify - The token is part of a passage that may be factual, but you do not know the relevant facts.
4. If there is any uncertainty or ambiguity, default to labeling the token as 0 (Not Hallucinatory).
5. No explanations are needed. Only return the labels in the specified format.
6. Assume that the tokenization is correct, only focus on the labelling task.

Format for the labels:
- Return an ordered label pair for each token.
- An ordered label pair is the order number of a token as it appears in the response, followed by its label as a number (0, 1, or 2).
- The tokens should be labeled in the same order as they appear in the response.
- Only return ordered label pairs. You should return exactly one ordered label pair per token in each response. 
- As the tokenized response contains exactly 50 tokens, you should return exactly 50 ordered label pairs.
- Do not include any additional information. No text should be returned. 

Example of an ordered label pair for the first token of an example response: 
1:0 (where the number before the colon is the token's position, and the number after the colon is the label).

Response for annotation:
{tokenized_response} 
    """

    payload = {
        "model": model_name,
        "prompt": prompt,
        "temperature": temperature,
        "top_p": top_p,
        "random_seed": random_seed,
        "max_tokens": max_tokens,
        "stream": False
    }

    for attempt in range(retries + 1):
        try:
            # Set a 5-minute timeout for each API call
            async with session.post(url, headers=headers, json=payload) as response:
                response.raise_for_status()
                response_data = await asyncio.wait_for(response.json(), timeout=300)  # 300 seconds = 5 minutes
                labeled_response = response_data.get("response")

                # Debug: Log the entire raw response to inspect it
                logging.debug(f"Raw API response: {labeled_response}")

                if not labeled_response:
                    return None, None

                labeled_tokens, raw_labeled_response = parse_labels(labeled_response)
                return labeled_tokens, raw_labeled_response

        except (asyncio.TimeoutError, aiohttp.ClientError) as e:
            logging.error(f"Attempt {attempt + 1} failed with error: {str(e)}")
            if attempt == retries:
                logging.error(f"All {retries + 1} attempts failed. Disregarding this API call.")
                return None, None

# Function to process each response and add labels to the original data
def process_response(row, model_name, temperature, top_p, random_seed, max_tokens):
    tokenized_response = row['tokenized_response'].strip()
    if not tokenized_response:
        logging.warning(f"Empty tokenized response at index {row.name}. Skipping...")
        return None

    # Running async code inside a sync function using asyncio.run
    async def run_async():
        async with aiohttp.ClientSession() as session:
            labeled_tokens, raw_labeled_response = await generate_labels_for_tokens(
                session, model_name, row['Question'], row['response'], tokenized_response, 
                row['Correct Answers'], row['Incorrect Answers'], temperature, top_p, random_seed, max_tokens
            )
        return labeled_tokens, raw_labeled_response
    
    labeled_tokens, raw_labeled_response = asyncio.run(run_async())

    if labeled_tokens is None:
        return None

    # Check if the number of labels matches the expected number of tokens
    expected_labels = 50
    if len(labeled_tokens) != expected_labels:
        logging.error(f"Mismatch in number of tokens. Expected {expected_labels}, got {len(labeled_tokens)}.")
        # Optionally, fill missing labels with a default value
        while len(labeled_tokens) < expected_labels:
            labeled_tokens.append('0')

    result = {
        "question_number": row['question_number'],
        "input_temperature": row['temperature'],
        "question": row['Question'],
        "full_response": row['response'],
        "tokenized_response": tokenized_response,
        "correct_answers": row['Correct Answers'],
        "incorrect_answers": row['Incorrect Answers'],
        "raw_labeled_response": raw_labeled_response,
        "labeled_tokens": ','.join(labeled_tokens),
        "# of labels": len(labeled_tokens)
    }
    return result

# Main function to handle the processing of all responses
def main():
    logging.info("Process started.")

    csv_file_path = '***.csv'
    data = load_csv(csv_file_path)

    # Process only the first x rows for testing
    # data = data.head(100)

    processed_responses = []

    model_name = "***"  # gemma2 or llama3.1
    temperature = 0
    top_p = 1
    random_seed = 17
    max_tokens = 100  

    with ProcessPoolExecutor(max_workers=18) as executor:
        results = list(executor.map(
            process_response, 
            [row for _, row in data.iterrows()],
            [model_name] * len(data),
            [temperature] * len(data),
            [top_p] * len(data),
            [random_seed] * len(data),
            [max_tokens] * len(data)
        ))

    # Filter out None results and log each response's status
    for i, result in enumerate(results):
        if result is not None:
            processed_responses.append(result)
            logging.info(f"Response {i+1} processed successfully.")
        else:
            logging.error(f"Response {i+1} failed to process.")

    output_file_path = '***.json'
    with open(output_file_path, 'w', encoding='utf-8') as f:
        json.dump(processed_responses, f, indent=2, ensure_ascii=False)

    logging.info(f"Labeled data saved successfully to {output_file_path}")
    logging.info("Process completed.")

if __name__ == '__main__':
    main()
