# sentence level descriptive and inferential statistics

import os
import glob
import json
import pandas as pd
import numpy as np
from scipy.stats import spearmanr, norm
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import warnings

def load_data(data_sources):
    """
    Load and parse sentence-level data from JSON files.

    Parameters:
    - data_sources: List of tuples (folder_path, labelling_model)

    Returns:
    - pd.DataFrame: Combined DataFrame of all data.
    """
    data_frames = []
    for folder_path, labelling_model in data_sources:
        if labelling_model == 'gemma2':
            # For gemma2, files are named 'B_labelled_{generating_model}(label_by_gemma2).json'
            pattern = os.path.join(folder_path, 'B_labelled_*' + '(label_by_gemma2).json')
        elif labelling_model == 'llama3.1':
            # For llama3.1, files are named 'B_labelled_{generating_model}.json' without '(label_by_*)'
            pattern = os.path.join(folder_path, 'B_labelled_*.json')
        else:
            continue  # Unknown labelling model, skip

        for file_path in glob.glob(pattern):
            filename = os.path.basename(file_path)
            # Remove 'B_labelled_' prefix and '.json' suffix
            model_part = filename[len('B_labelled_'):-len('.json')]

            if labelling_model == 'gemma2':
                # For gemma2, model_part ends with '(label_by_gemma2)'
                if '(label_by_gemma2)' not in model_part:
                    continue  # Skip files not labelled by gemma2
                model_name_raw = model_part.split('(label_by_gemma2)')[0]
            elif labelling_model == 'llama3.1':
                # For llama3.1, exclude files that have '(label_by_' in their names
                if '(label_by_' in model_part:
                    continue  # Skip files labelled by other labelling models
                model_name_raw = model_part
            else:
                continue  # Should not reach here

            # Standardize model names with precise mapping
            model_name_map = {
                'qwen2(0.5b)': 'Qwen2(0.5B)',
                'qwen2': 'Qwen2',
                'llama3.1': 'Llama3.1',
                'gemma2': 'Gemma2',
                'mistral-nemo': 'Mistral-NeMo',
            }
            model_name_key = model_name_raw.lower()
            model_name = model_name_map.get(model_name_key, model_name_raw)
            # Debugging: Print model being processed
            print(f"Processing model: {model_name} labelled by {labelling_model}")
            with open(file_path, 'r') as file:
                data = json.load(file)
                df = parse_data(data, model_name, labelling_model)
                data_frames.append(df)
    # Combine all data into a single DataFrame
    if data_frames:
        combined_df = pd.concat(data_frames, ignore_index=True)
    else:
        combined_df = pd.DataFrame()
    return combined_df

def parse_data(data, generating_model, labelling_model):
    """
    Parse the JSON data and create a DataFrame.

    Parameters:
    - data (list): List of JSON entries.
    - generating_model (str): Name of the generating model.
    - labelling_model (str): Name of the labelling model.

    Returns:
    - pd.DataFrame: Parsed DataFrame.
    """
    records = []
    for entry in data:
        question_number = entry.get('question_number')
        input_temperature = float(entry.get('input_temperature', 0))
        # Split response into sentences
        response_text = entry.get('response', '')
        sentences = response_text.strip().split('\n')
        # Process sentences: remove numbering if present
        processed_sentences = []
        for s in sentences:
            s = s.strip()
            # Remove numbering if present at start
            if s and s[0].isdigit():
                # Split on the first dot to remove numbering
                parts = s.split('.', 1)
                if len(parts) > 1:
                    s = parts[1].strip()
            processed_sentences.append(s)
        labels_text = entry.get('labeled_response', '')
        labels = labels_text.strip().split(',')
        labels = [label.strip() for label in labels if label.strip()]
        # Skip entries where the number of sentences and labels don't match
        if len(processed_sentences) != len(labels):
            warnings.warn(f"Mismatch in number of sentences and labels for question {question_number} in model {generating_model} labelled by {labelling_model}. Skipping.")
            continue
        for i in range(len(processed_sentences)):
            try:
                sentence = processed_sentences[i]
                label = int(labels[i])
                # Recode label '2' as '0'
                if label == 2:
                    label = 0
                records.append({
                    'generating_model': generating_model,
                    'labelling_model': labelling_model,
                    'question_number': question_number,
                    'input_temperature': input_temperature,
                    'sentence_position': i + 1,
                    'sentence': sentence,
                    'label': label
                })
            except ValueError:
                warnings.warn(f"Invalid label '{labels[i]}' for sentence {i+1} in question {question_number} of model {generating_model} labelled by {labelling_model}. Skipping this sentence.")
                continue  # Skip malformed labels
    df = pd.DataFrame(records)
    return df

def calculate_proportions(df):
    """
    Calculate the proportion of sentences labelled as hallucinatory (label == 1) for each response.

    Parameters:
    - df (pd.DataFrame): DataFrame with sentence-level data.

    Returns:
    - pd.DataFrame: DataFrame with an additional 'proportion_hallucinatory' column.
    """
    # Group by response (question_number) and compute proportions
    df_grouped = df.groupby(['generating_model', 'labelling_model', 'question_number', 'input_temperature'])
    proportions = df_grouped['label'].mean().reset_index().rename(columns={'label': 'proportion_hallucinatory'})
    return proportions

def logistic_regression_scipy(X, y):
    """
    Perform logistic regression using SciPy's optimization.

    Parameters:
    - X (np.ndarray): Feature matrix (including intercept).
    - y (np.ndarray): Target vector.

    Returns:
    - dict: Dictionary containing 'params', 'p_values', and 'success' status.
    """
    # Define the sigmoid function
    def sigmoid(z):
        return 1 / (1 + np.exp(-z))

    # Define the negative log-likelihood
    def neg_log_likelihood(params, X, y):
        logits = np.dot(X, params)
        # To prevent log(0), clip logits
        logits = np.clip(logits, -500, 500)
        return -np.sum(y * np.log(sigmoid(logits)) + (1 - y) * np.log(1 - sigmoid(logits)))

    # Define the gradient of the negative log-likelihood
    def grad_neg_log_likelihood(params, X, y):
        logits = np.dot(X, params)
        predictions = sigmoid(logits)
        return -np.dot(X.T, y - predictions)

    # Initial guess
    initial_params = np.zeros(X.shape[1])

    # Optimize
    result = minimize(
        fun=neg_log_likelihood,
        x0=initial_params,
        args=(X, y),
        method='BFGS',
        jac=grad_neg_log_likelihood,
        options={'disp': False, 'maxiter': 1000}
    )

    if not result.success:
        warnings.warn(f"Optimization failed: {result.message}")
        return {'params': None, 'p_values': None, 'success': False}

    params = result.x

    # Calculate standard errors using the Hessian inverse
    try:
        hessian_inv = result.hess_inv
        se = np.sqrt(np.diag(hessian_inv))
    except Exception as e:
        warnings.warn(f"Failed to compute standard errors: {e}")
        se = np.full_like(params, np.nan)

    # Calculate z-scores
    z_scores = params / se

    # Calculate p-values for two-tailed test
    p_values = 2 * (1 - norm.cdf(np.abs(z_scores)))

    return {
        'params': params,
        'p_values': p_values,
        'success': True
    }

def test_h1a(df):
    """
    Test H1a: The proportion of sentences labelled as hallucinatory increases with higher temperature.

    Parameters:
    - df (pd.DataFrame): DataFrame with sentence-level data.

    Returns:
    - pd.DataFrame: H1a results with Spearman correlation coefficients and p-values.
    """
    labelling_models = df['labelling_model'].unique()
    h1a_results = []

    print("Performing H1a Analysis...\n")
    for lab_model in labelling_models:
        df_lab = df[df['labelling_model'] == lab_model]
        # Calculate proportion of hallucinatory sentences at each temperature
        temp_group = df_lab.groupby('input_temperature')['label'].mean().reset_index()
        # Remove NaN values
        temp_group = temp_group.dropna(subset=['label'])
        if len(temp_group) < 2:
            warnings.warn(f"Not enough data to compute Spearman correlation for labelling model '{lab_model}'.")
            continue
        # Use Spearman's rank correlation
        corr, p_value = spearmanr(temp_group['input_temperature'], temp_group['label'])
        h1a_results.append({
            'labelling_model': lab_model,
            'spearman_correlation': corr,
            'p_value': p_value
        })
        print(f"H1a - Labelling Model: {lab_model}")
        print(f"Spearman correlation: {corr:.4f}, p-value: {p_value:.4f}\n")

    h1a_df = pd.DataFrame(h1a_results)
    return h1a_df

def test_h1b(df):
    """
    Test H1b: The proportion of sentences labelled as hallucinatory is greater in responses generated by smaller models.

    Parameters:
    - df (pd.DataFrame): DataFrame with sentence-level data.

    Returns:
    - pd.DataFrame: H1b results with Spearman correlation coefficients and p-values.
    """
    labelling_models = df['labelling_model'].unique()
    # Map generating models to sizes
    model_sizes = {
        'Qwen2(0.5B)': 0.5,
        'Qwen2': 6,
        'Llama3.1': 8,
        'Gemma2': 9,
        'Mistral-NeMo': 12
    }
    # Verify that all generating models are mapped
    unmapped_models = set(df['generating_model'].unique()) - set(model_sizes.keys())
    if unmapped_models:
        warnings.warn(f"Unmapped models found: {', '.join(unmapped_models)}. They will have NaN for model_size.")
    df['model_size'] = df['generating_model'].map(model_sizes)
    h1b_results = []

    print("Performing H1b Analysis...\n")
    for lab_model in labelling_models:
        df_lab = df[df['labelling_model'] == lab_model]
        # Remove entries with NaN model_size
        df_lab = df_lab.dropna(subset=['model_size'])
        # Calculate proportion of hallucinatory sentences for each model size
        size_group = df_lab.groupby('model_size')['label'].mean().reset_index()
        if len(size_group) < 2:
            warnings.warn(f"Not enough data to compute Spearman correlation for labelling model '{lab_model}'.")
            continue
        # Use Spearman's rank correlation
        corr, p_value = spearmanr(size_group['model_size'], size_group['label'])
        h1b_results.append({
            'labelling_model': lab_model,
            'spearman_correlation': corr,
            'p_value': p_value
        })
        print(f"H1b - Labelling Model: {lab_model}")
        print(f"Spearman correlation: {corr:.4f}, p-value: {p_value:.4f}\n")

    h1b_df = pd.DataFrame(h1b_results)
    return h1b_df

def test_h2a(df):
    """
    Test H2a: The occurrence of a hallucinatory sentence influences the probability of the next sentence being hallucinatory.

    Parameters:
    - df (pd.DataFrame): DataFrame with sentence-level data.

    Returns:
    - pd.DataFrame: H2a results with coefficients and p-values.
    """
    labelling_models = df['labelling_model'].unique()
    h2a_results = []

    print("Performing H2a Analysis using SciPy's Logistic Regression...\n")
    for lab_model in labelling_models:
        df_lab = df[df['labelling_model'] == lab_model]
        for gen_model in df_lab['generating_model'].unique():
            df_gen = df_lab[df_lab['generating_model'] == gen_model]
            for temp in df_gen['input_temperature'].unique():
                df_temp = df_gen[df_gen['input_temperature'] == temp]
                # Sort by question_number and sentence_position
                df_temp = df_temp.sort_values(['question_number', 'sentence_position'])
                # Group by question_number
                data_records = []
                for question, group in df_temp.groupby('question_number'):
                    labels = group['label'].values
                    for i in range(1, len(labels)):
                        data_records.append({
                            'current_label': labels[i],
                            'prev_label': labels[i-1]
                        })
                df_seq = pd.DataFrame(data_records)
                if df_seq.empty:
                    warnings.warn(f"No sequential data for labelling model '{lab_model}', generating model '{gen_model}', temperature {temp}.")
                    continue
                # Prepare feature matrix X and target vector y
                X = df_seq['prev_label'].values
                y = df_seq['current_label'].values
                # Add intercept
                X = np.column_stack((np.ones(len(X)), X))
                # Perform logistic regression using SciPy
                result = logistic_regression_scipy(X, y)
                if result['success']:
                    coef = result['params'][1]  # Coefficient for 'prev_label'
                    p_value = result['p_values'][1]
                    h2a_results.append({
                        'labelling_model': lab_model,
                        'generating_model': gen_model,
                        'temperature': temp,
                        'coef_prev_label': coef,
                        'p_value': p_value
                    })
                    print(f"H2a - Labelling Model: {lab_model}, Generating Model: {gen_model}, Temperature: {temp}")
                    print(f"Coefficient for prev_label: {coef:.4f}, p-value: {p_value:.4f}\n")
                else:
                    h2a_results.append({
                        'labelling_model': lab_model,
                        'generating_model': gen_model,
                        'temperature': temp,
                        'coef_prev_label': np.nan,
                        'p_value': np.nan
                    })
                    print(f"H2a - Labelling Model: {lab_model}, Generating Model: {gen_model}, Temperature: {temp}")
                    print(f"Logistic regression failed.\n")

    h2a_df = pd.DataFrame(h2a_results)
    return h2a_df

def autocorrelation(x, lag=1):
    """
    Compute autocorrelation for a given lag.

    Parameters:
    - x (np.ndarray): Input array.
    - lag (int): Lag value.

    Returns:
    - float: Autocorrelation coefficient.
    """
    n = len(x)
    if n <= lag:
        return np.nan
    x_mean = np.mean(x)
    numerator = np.sum((x[:n - lag] - x_mean) * (x[lag:] - x_mean))
    denominator = np.sum((x - x_mean) ** 2)
    if denominator == 0:
        return np.nan
    return numerator / denominator

def test_h2b(df):
    """
    Test H2b: Autocorrelation coefficients at higher lags decrease.

    Parameters:
    - df (pd.DataFrame): DataFrame with sentence-level data.

    Returns:
    - pd.DataFrame: H2b results with Spearman correlation coefficients and p-values.
    """
    labelling_models = df['labelling_model'].unique()
    h2b_results = []

    print("Performing H2b Analysis...\n")
    for lab_model in labelling_models:
        df_lab = df[df['labelling_model'] == lab_model]
        for gen_model in df_lab['generating_model'].unique():
            df_gen = df_lab[df_lab['generating_model'] == gen_model]
            for temp in df_gen['input_temperature'].unique():
                df_temp = df_gen[df_gen['input_temperature'] == temp]
                # Calculate autocorrelations for each sequence and average them
                autocorrs = []
                for question, group in df_temp.groupby('question_number'):
                    labels = group.sort_values('sentence_position')['label'].values
                    if len(labels) > 1:
                        # Compute autocorrelations up to lag 10
                        acfs = [autocorrelation(labels, lag) for lag in range(1, 11)]
                        autocorrs.append(acfs)
                if not autocorrs:
                    warnings.warn(f"No autocorrelation data for labelling model '{lab_model}', generating model '{gen_model}', temperature {temp}.")
                    continue
                # Compute mean autocorrelation across sequences
                mean_autocorr = np.nanmean(autocorrs, axis=0)
                lags = np.arange(1, len(mean_autocorr) + 1)
                # Test if autocorrelation decreases with lag using Spearman's rank correlation
                corr, p_value = spearmanr(lags, mean_autocorr)
                h2b_results.append({
                    'labelling_model': lab_model,
                    'generating_model': gen_model,
                    'temperature': temp,
                    'spearman_correlation': corr,
                    'p_value': p_value
                })

    h2b_df = pd.DataFrame(h2b_results)
    
    # Print the H2b results as a table
    print("H2b Results:")
    print(h2b_df.to_string(index=False))
    
    return h2b_df

def plot_autocorrelation(df, max_lag=10):
    """
    Plot the autocorrelation results, averaging autocorrelations across sequences.

    Parameters:
    - df (pd.DataFrame): DataFrame with sentence-level data.
    - max_lag (int): Maximum lag for autocorrelation.

    Returns:
    - None
    """
    labelling_models = df['labelling_model'].unique()
    generating_models = ['Gemma2', 'Llama3.1', 'Mistral-NeMo', 'Qwen2', 'Qwen2(0.5B)']
    temperatures = [0, 0.75, 1, 1.25, 2]
    temperature_colors = {
        0: 'blue',
        0.75: 'green',
        1: 'red',
        1.25: 'purple',
        2: 'orange'
    }

    for lab_model in labelling_models:
        fig, axes = plt.subplots(1, len(generating_models), figsize=(25, 5), sharey=True)
        fig.suptitle(f'Autocorrelation of Hallucinatory Sentences by Temperature for {lab_model}', fontsize=16)

        for idx, gen_model in enumerate(generating_models):
            ax = axes[idx]
            for temp in temperatures:
                df_temp = df[
                    (df['labelling_model'] == lab_model) &
                    (df['generating_model'] == gen_model) &
                    (df['input_temperature'] == temp)
                ]
                # Calculate autocorrelations for each sequence and average them
                autocorrs = []
                for question, group in df_temp.groupby('question_number'):
                    labels = group.sort_values('sentence_position')['label'].values
                    if len(labels) > 1:
                        # Compute autocorrelations up to max_lag
                        acfs = [autocorrelation(labels, lag) for lag in range(1, max_lag + 1)]
                        autocorrs.append(acfs)
                if not autocorrs:
                    continue
                # Compute mean autocorrelation across sequences
                mean_autocorr = np.nanmean(autocorrs, axis=0)
                ax.plot(range(1, len(mean_autocorr)+1), mean_autocorr, marker='o',
                        label=f'Temp: {temp}', color=temperature_colors.get(temp, 'black'))
            ax.set_title(f'Generating Model: {gen_model}', fontsize=14)
            ax.set_ylim(-0.1, 0.5)
            ax.set_xlabel('Lag', fontsize=12)
            ax.set_xticks(range(1, max_lag + 1))
            if idx == 0:
                ax.set_ylabel('Autocorrelation', fontsize=12)
            ax.legend(fontsize=10)
            ax.grid(False)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        # Save the figure
        figure_filename = f'Autocorrelation_{lab_model}.png'
        plt.savefig(figure_filename, dpi=300)
        print(f"Autocorrelation plot saved to '{figure_filename}'.")
        # Remove plt.show() and plt.close(fig) from inside the loop

    # Display all figures at the end
    plt.show()

def compute_descriptive_statistics(df):
    """
    Compute and display descriptive statistics for the data.

    Parameters:
    - df (pd.DataFrame): DataFrame with sentence-level data.

    Returns:
    - None
    """
    print("\n=== Descriptive Statistics ===\n")
    for labelling_model in df['labelling_model'].unique():
        print(f"Labelling Model: {labelling_model}\n")
        df_labeller = df[df['labelling_model'] == labelling_model]
        # Overall statistics
        total_sentences = len(df_labeller)
        total_hallucinatory = df_labeller['label'].sum()
        proportion_hallucinatory = df_labeller['label'].mean()
        print(f"Total Sentences: {total_sentences}")
        print(f"Total Hallucinatory Sentences (label=1): {total_hallucinatory}")
        print(f"Proportion of Hallucinatory Sentences: {proportion_hallucinatory:.4f}\n")
        
        # Save overall statistics to CSV
        overall_stats = pd.DataFrame({
            'Labelling Model': [labelling_model],
            'Total Sentences': [total_sentences],
            'Total Hallucinatory Sentences': [total_hallucinatory],
            'Proportion Hallucinatory': [proportion_hallucinatory]
        })
        overall_stats.to_csv(f'Descriptive_Stats_Overall_{labelling_model}.csv', index=False)

        # Statistics by Generating Model
        print("Descriptive Statistics by Generating Model:")
        model_stats = df_labeller.groupby('generating_model')['label'].agg(['count', 'sum', 'mean']).rename(columns={
            'count': 'Total Sentences',
            'sum': 'Hallucinatory Sentences',
            'mean': 'Proportion Hallucinatory'
        })
        print(model_stats.to_string())
        print()
        
        # Save statistics by generating model to CSV
        model_stats.to_csv(f'Descriptive_Stats_by_Generating_Model_{labelling_model}.csv')

        # Statistics by Temperature
        print("Descriptive Statistics by Temperature:")
        temp_stats = df_labeller.groupby('input_temperature')['label'].agg(['count', 'sum', 'mean']).rename(columns={
            'count': 'Total Sentences',
            'sum': 'Hallucinatory Sentences',
            'mean': 'Proportion Hallucinatory'
        })
        print(temp_stats.to_string())
        print()
        
        # Save statistics by temperature to CSV
        temp_stats.to_csv(f'Descriptive_Stats_by_Temperature_{labelling_model}.csv')

        # Statistics by Generating Model and Temperature
        print("Descriptive Statistics by Generating Model and Temperature:")
        model_temp_stats = df_labeller.groupby(['generating_model', 'input_temperature'])['label'].agg(['count', 'sum', 'mean']).rename(columns={
            'count': 'Total Sentences',
            'sum': 'Hallucinatory Sentences',
            'mean': 'Proportion Hallucinatory'
        })
        print(model_temp_stats.to_string())
        print()
        
        # Save statistics by generating model and temperature to CSV
        model_temp_stats.to_csv(f'Descriptive_Stats_by_Generating_Model_and_Temperature_{labelling_model}.csv')
        
        # Save all descriptive statistics tables into a single CSV file separated by labeling model
        combined_csv_filename = f'Descriptive_Statistics_All_{labelling_model}.csv'
        with open(combined_csv_filename, 'w') as combined_csv:
            # Write Overall Statistics
            combined_csv.write(f"=== Overall Statistics for {labelling_model} ===\n")
            overall_stats.to_csv(combined_csv, index=False)
            combined_csv.write("\n")
            
            # Write Statistics by Generating Model
            combined_csv.write(f"=== Descriptive Statistics by Generating Model for {labelling_model} ===\n")
            model_stats.to_csv(combined_csv, index=True)
            combined_csv.write("\n")
            
            # Write Statistics by Temperature
            combined_csv.write(f"=== Descriptive Statistics by Temperature for {labelling_model} ===\n")
            temp_stats.to_csv(combined_csv, index=True)
            combined_csv.write("\n")
            
            # Write Statistics by Generating Model and Temperature
            combined_csv.write(f"=== Descriptive Statistics by Generating Model and Temperature for {labelling_model} ===\n")
            model_temp_stats.to_csv(combined_csv, index=True)
            combined_csv.write("\n")
        
        print(f"All descriptive statistics tables saved to '{combined_csv_filename}'.\n")

def main():
    # Define the data sources as a list of tuples (folder_path, labelling_model)
    data_sources = [
        ('***/sentence/json', 'llama3.1'),
        ('***/sentence/json', 'gemma2')
    ]

    # Load the data
    print("Loading data...")
    df = load_data(data_sources)
    if df.empty:
        print("No data loaded. Please check the data files and folder paths.")
        return
    print("Data loaded successfully.\n")

    # Calculate proportions
    print("Calculating proportions of hallucinatory sentences...")
    df_proportions = calculate_proportions(df)
    print("Proportions calculated.\n")

    # Compute and display descriptive statistics
    compute_descriptive_statistics(df)

    # Save combined data for reference
    df.to_csv('combined_sentence_data_with_labeller.csv', index=False)
    print("Combined data saved to 'combined_sentence_data_with_labeller.csv'.\n")

    # Perform statistical tests for H1a and H1b
    h1a_results = test_h1a(df)
    h1a_results.to_csv('H1a_results.csv', index=False)
    print("H1a results saved to 'H1a_results.csv'.\n")

    h1b_results = test_h1b(df)
    h1b_results.to_csv('H1b_results.csv', index=False)
    print("H1b results saved to 'H1b_results.csv'.\n")

    # Compute H2a autocorrelation at lag 1 and p-values
    h2a_results = test_h2a(df)
    h2a_results.to_csv("H2a_Autocorrelation_Lag1_Results.csv", index=False)
    print("H2a results saved to 'H2a_Autocorrelation_Lag1_Results.csv'.\n")

    # Compute H2b Spearman correlations
    h2b_results = test_h2b(df)
    h2b_results.to_csv("H2b_Spearman_Correlations_Results.csv", index=False)
    print("H2b results saved to 'H2b_Spearman_Correlations_Results.csv'.\n")

    # Plot the autocorrelation graphs
    print("Plotting autocorrelation graphs...")
    plot_autocorrelation(df)
    print("Autocorrelation plots have been saved and displayed.\n")

    print("All analyses completed successfully.")

if __name__ == "__main__":
    main()
