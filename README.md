# URSS 2024

## Description

This repository contains the code and data analysis scripts used for the URSS 2024 summer research project. The project focuses on generating and analyzing responses from various large language models (LLMs) and involves both descriptive and inferential analysis of the results.

## Code Overview

### `response_generation.py`

- **Description**: Concurrently generates detailed responses to questions from a CSV file using a local API that supports five different LLMs (e.g., qwen2, llama3.1). The responses are saved to a JSON file.
- **Location**: `/code/response_generation.py`

### `convert_json_to_csv.py`

- **Description**: Converts a JSON file containing response data into a CSV file.
- **Location**: `/code/convert_json_to_csv.py`

### `pre-labeling.py`

- **Description**: Splits responses into individual sentences, numbers each sentence, and adds related information such as the question, correct answers, and incorrect answers to the output file.
- **Location**: `/code/sentence/pre-labeling.py`

### `sentence_labelling.py`

- **Description**: Concurrently processes and labels response data from a CSV file using a local API (e.g., llama3.1), and then saves the labeled results to a JSON file.
- **Location**: `/code/sentence/sentence_labelling.py`

### `descriptive_data_analysis_sentence.py`

- **Description**: Performs descriptive statistics on the labeled data, including calculating the number of disregarded responses, the average number of labels per response, and the totals and averages of each label type.
- **Location**: `/code/sentence/descriptive_data_analysis_sentence.py`

### `inferential_data_analysis_sentence.py`

- **Description**: Conducts inferential data analysis by performing autocorrelation analysis on hallucinatory labels across responses, helping to identify patterns over sequential data.
- **Location**: `/code/sentence/inferential_data_analysis_sentence.py`

## Necessary Libraries

The following Python libraries/modules are required to run the scripts in this repository:

- `pandas`: For data manipulation and CSV file handling.
- `requests`: For making HTTP requests to the local API.
- `json`: For working with JSON data.
- `concurrent.futures`: For concurrent execution of tasks (e.g., API calls).
- `os`: For interacting with the operating system (e.g., file paths, process priority).
- `nltk`: For natural language processing, specifically sentence tokenization.
- `re`: For regular expressions used in text processing.
- `matplotlib`: For creating plots and visualizations.
- `statsmodels`: For performing statistical analyses (i.e. autocorrelation).

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
