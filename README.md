# URSS 2024

## Description

This repository contains the code scripts, dataset and human labelled data used for this Undergraduate Research Support Scheme (URSS) 2024 summer research project. The project focuses on generating and analyzing responses from various large language models (LLMs) and involves both descriptive and inferential analysis of the results.

## Code Overview

### `response_generation.py`

- **Description**: Concurrently generates detailed responses to questions from a CSV file using a local API that supports five different LLMs (gemma2, qwen(0.5b), qwen2(7b), llama3.1, mistral-nemo). The responses are saved to a JSON file.
- **Location**: `/code/response_generation.py`

### `convert_json_to_csv.py`

- **Description**: Converts a JSON file containing response data into a CSV file.
- **Location**: `/code/convert_json_to_csv.py`

### `sentence_labelling.py`

- **Description**: Concurrently processes and labels on a sentence level responses using a local API (llama3.1 and gemma2).
- **Location**: `/code/sentence/sentence_labelling.py`

### `sentence_level_analysis.py`

- **Description**: Performs descriptive and inferential statistics on the labeled data, including the generation of autocorrelation graphs.
- **Location**: `/code/sentence/sentence_level_analysis.py`

### `generate_manually_labelled_sentence.py`

- **Description**: Samples 100 responses (each of at least 10 sentences) for human labelling benchmark.
- **Location**: `/code/sentence/generate_manually_labelled_sentence.py`

### `cohen's_kappa_sentence.py`

- **Description**: Calculates the Cohen's Kappa value between each labelling model and the human on the labelled sample.
- **Location**: `/code/sentence/cohen's_kappa_sentence.py`

### `token_labelling.py`

- **Description**: Concurrently processes and labels on a token level responses using a local API (llama3.1 and gemma2).
- **Location**: `/code/token/token_labelling.py`

### `token_level_analysis.py`

- **Description**: Calculates the Cohen's Kappa value between each labelling model and the human on the labelled sample.
- **Location**: `/code/token/token_level_analysis.py`

### `generate_manually_labelled_token.py`

- **Description**: Samples 20 responses (each of exactly 50 tokens) for human labelling benchmark.
- **Location**: `/code/sentence/generate_manually_labelled_token.py`

### `cohen's_kappa_token.py`

- **Description**: Calculates the Cohen's Kappa value between each labelling model and the human on the labelled sample.
- **Location**: `/code/token/cohen's_kappa_token.py`

## Using Ollama and Required Models

This project relies on Ollama to run the following large language models locally:

- **llama3.1 (7B, Meta)**
- **gemma2 (9B, Google)**
- **qwen2 (7B, Alibaba)**
- **qwen2 (0.5B, Alibaba)**
- **mistral-nemo (12B, Mistral)**

### Instructions:

1. **Install Ollama**: Ensure that you have Ollama installed on your system. You can find installation instructions on the [Ollama website](https://ollama.com/).
   
2. **Set Up Models**: Download and configure the above-mentioned models within Ollama.

3. **Run the Local API**: Make sure the local API is running and accessible at `http://localhost:11434` before executing the scripts.

## Contact

For any questions or issues, please contact the repository owner at [luc.mekouar@hotmail.com](mailto:luc.mekouar@hotmail.com).
