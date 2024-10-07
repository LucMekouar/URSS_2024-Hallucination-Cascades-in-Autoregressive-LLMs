# Concurrently generates detailed responses to questions from a CSV using a local API (5 LLMs) and saves the responses to a JSON file.

import pandas as pd
import requests
import json
import concurrent.futures
import os
from concurrent.futures import ThreadPoolExecutor

# Computing priority
os.nice(0)

def load_csv(file_path):
    df = pd.read_csv(file_path)
    return df

# Function to generate a response from the API
def generate_response(model_name, prompt, temperature, top_p, max_tokens, random_seed):
    url = "http://localhost:11434/api/generate"
    headers = {'Content-Type': 'application/json'}
    detailed_prompt = f"{prompt}\n\nPlease provide a detailed response to the question consisting of at least 20 sentences. Avoid using bullet points or list formats. Number the sentences accordingly."
    payload = {
        "model": model_name,
        "prompt": detailed_prompt,
        "temperature": temperature,
        "top_p": top_p,
        "random_seed": random_seed,
        "max_tokens": max_tokens,  
        "stream": False  
    }

    response = requests.post(url, headers=headers, data=json.dumps(payload))
    
    if response.status_code == 200:
        response_data = response.json()
        text = response_data.get("response", "")
        return text
    else:
        print(f"Error {response.status_code}: {response.text}")
        return None

# Function to process each question
def process_question(index, question, model_name, temperature, top_p, max_tokens, random_seed):
    prompt = question
    response = generate_response(model_name, prompt, temperature, top_p, max_tokens, random_seed)
    if response:
        return {"question_number": index, "temperature": temperature, "response": response}
    else:
        return None

# Load the CSV file containing the questions
csv_file_path = '.../URSS_public_repo/data/TruthfulQA.csv'
data = load_csv(csv_file_path)


questions = data['Question'].tolist()

responses = []

# Parameters (input model name to generate responses for that model)
model_name = "***"  # llama3.1 (7B,Meta)  gemma2 (9B,Google)  qwen2 (7B,Alibaba)  qwen2:0.5b (0.5B,Alibaba)  mistral-nemo (12B,Mistral)
temperatures = [0, 0.75, 1, 1.25, 2]
top_p = 1
max_tokens = 1000  
random_seed = 1337

# ThreadPoolExecutor to make concurrent API calls
with ThreadPoolExecutor(max_workers=18) as executor:
    future_to_question = {executor.submit(process_question, i, q, model_name, temp, top_p, max_tokens, random_seed): (i, temp) for temp in temperatures for i, q in enumerate(questions)}
    for future in concurrent.futures.as_completed(future_to_question):
        question_index, temp = future_to_question[future]
        try:
            result = future.result()
            if result:
                responses.append(result)
                print(f"Processed question {question_index+1}/{len(questions)} with temperature {temp}")
        except Exception as e:
            print(f"Error processing question {question_index+1} with temperature {temp}: {e}")

# Save responses to a JSON fileq
output_file_path = '***.json'
with open(output_file_path, 'w') as f:
    json.dump(responses, f, indent=2)

print(f"Responses saved to {output_file_path}")
