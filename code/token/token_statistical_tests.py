# token_level_analysis.py

import os
import glob
import json
import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import acf
from statsmodels.stats.stattools import durbin_watson
import warnings

# Load and parse token-level data
def load_data(folder_path):
    data_frames = []
    # Use glob to match files starting with 'token_labelled_' and ending with '.json'
    pattern = os.path.join(folder_path, 'token_labelled_*.json')
    for file_path in glob.glob(pattern):
        # Extract filename
        filename = os.path.basename(file_path)
        
        # Remove 'token_labelled_' prefix and '.json' suffix
        model_part = filename[len('token_labelled_'):-len('.json')]
        
        # If there is '_(by_...', split and take first part
        if '_(' in model_part:
            model_name_raw = model_part.split('_(')[0]
        else:
            model_name_raw = model_part
        
        # Standardize model names with precise mapping
        model_name_map = {
            'qwen2(0.5b)': 'Qwen2(0.5B)',
            'qwen2(7b)': 'Qwen2(7B)',
            'qwen2': 'Qwen2',
            'llama3.1': 'Llama3.1',
            'gemma2': 'Gemma2',
            'mistral-nemo': 'Mistral-NeMo',
        }
        model_name_key = model_name_raw.lower()
        model_name = model_name_map.get(model_name_key, model_name_raw)
        
        # Debugging: Print model being processed
        print(f"Processing model: {model_name}")
        
        with open(file_path, 'r') as file:
            data = json.load(file)
            df = parse_data(data, model_name)
            data_frames.append(df)

    # Combine all data into a single DataFrame
    if data_frames:
        combined_df = pd.concat(data_frames, ignore_index=True)
    else:
        combined_df = pd.DataFrame()
    
    # **ADDED: Recode label '2' as '0'**
    if not combined_df.empty:
        combined_df['label'] = combined_df['label'].replace(2, 0)
        print("\nLabels recoded: '2' has been changed to '0'.")
    
    return combined_df

def parse_data(data, model_name):
    records = []
    for entry in data:
        question_number = entry['question_number']
        input_temperature = float(entry['input_temperature'])
        # Tokens and labels
        tokens = entry['tokenized_response'].split('\n')
        labels = entry['labeled_tokens'].split(',')
        # Ensure tokens and labels have the same length
        if len(tokens) != len(labels):
            continue  # Skip entries with mismatched tokens and labels
        # Skip entries where the number of labels is not exactly 50
        if len(labels) != 50:
            continue  # Skip entries with labels not equal to 50
        for i in range(len(tokens)):
            try:
                token_number, token = tokens[i].split(':', 1)
                label = int(labels[i])
                
                # **ADDED: Recode label '2' as '0' during parsing (optional)**
                # Uncomment the following lines if you prefer to recode during parsing
                # if label == 2:
                #     label = 0

                records.append({
                    'model': model_name,
                    'question_number': question_number,
                    'temperature': input_temperature,
                    'token_position': i + 1,
                    'token': token.strip(),
                    'label': label
                })
            except ValueError:
                continue  # Skip malformed tokens
    df = pd.DataFrame(records)
    return df

# Extract hallucinatory sequences for each generating model and temperature
def extract_hallucinatory_sequences(df):
    sequences = {}
    for model in df['model'].unique():
        sequences[model] = {}
        for temp in df['temperature'].unique():
            temp_df = df[(df['model'] == model) & (df['temperature'] == temp)]
            # Get the labels as a list
            labels = temp_df.sort_values(['question_number', 'token_position'])['label'].tolist()
            sequences[model][temp] = labels
    return sequences

# Perform autocorrelation analysis for the hallucinatory sequences
def perform_autocorrelation_analysis(sequences, max_lag=10):
    autocorrelations = {}
    for model, temp_sequences in sequences.items():
        autocorrelations[model] = {}
        for temp, sequence in temp_sequences.items():
            if len(sequence) > 1:
                autocorr = acf(sequence, nlags=max_lag, fft=False, missing='conservative')[1:]  # Exclude lag 0
                autocorrelations[model][temp] = autocorr
            else:
                autocorrelations[model][temp] = [np.nan] * max_lag
    return autocorrelations

# Compute autocorrelation at lag 1 and its p-value for H2a
def compute_h2a_autocorrelation_pvalues(sequences):
    h2a_results = []
    for model, temp_sequences in sequences.items():
        for temp, sequence in temp_sequences.items():
            if len(sequence) > 1:
                autocorr_result = acf(sequence, nlags=1, fft=False, missing='conservative')
                autocorr_lag1 = autocorr_result[1]
                N = len(sequence)
                # Standard error for autocorrelation at lag 1
                se = 1 / np.sqrt(N)
                z = autocorr_lag1 / se
                p_value = 1 - stats.norm.cdf(z)  # One-tailed test (H0: rho <= 0, H1: rho > 0)
                h2a_results.append({
                    'model': model,
                    'temperature': temp,
                    'autocorr_lag1': autocorr_lag1,
                    'p_value': p_value
                })
            else:
                h2a_results.append({
                    'model': model,
                    'temperature': temp,
                    'autocorr_lag1': np.nan,
                    'p_value': np.nan
                })
    h2a_df = pd.DataFrame(h2a_results)
    return h2a_df

# Compute Durbin-Watson statistic for H2b
def compute_h2b_durbin_watson(sequences):
    h2b_results = []
    for model, temp_sequences in sequences.items():
        for temp, sequence in temp_sequences.items():
            if len(sequence) > 1:
                dw_stat = durbin_watson(sequence)
                h2b_results.append({
                    'model': model,
                    'temperature': temp,
                    'durbin_watson': dw_stat
                })
            else:
                h2b_results.append({
                    'model': model,
                    'temperature': temp,
                    'durbin_watson': np.nan
                })
    h2b_df = pd.DataFrame(h2b_results)
    return h2b_df

# Generate H2b table with Spearman correlations and Durbin-Watson statistics
def generate_h2b_table(autocorr_df, h2b_dw_df):
    temperature_values = [0.0, 0.75, 1.0, 1.25, 2.0]
    temp_labels = [f"Temperature = {temp:g}" for temp in temperature_values]
    
    # Define all generating models
    all_generating_models = ['Llama3.1', 'Gemma2', 'Qwen2', 'Qwen2(0.5B)', 'Mistral-NeMo']
    
    # Initialize a DataFrame for the table with single-level columns
    table = pd.DataFrame(
        "N/A",
        index=temp_labels,
        columns=all_generating_models
    )
    
    # Loop through autocorr_df, compute Spearman correlations, and fill in the table
    for (model, temp), group in autocorr_df.groupby(['model', 'temperature']):
        mean_autocorr_by_lag = group.groupby('lag')['autocorrelation'].mean().reset_index()
        autocorr_values = mean_autocorr_by_lag['autocorrelation'].values
        
        if len(autocorr_values) > 1:
            lags = mean_autocorr_by_lag['lag'].values
            spearman_coef, spearman_p_value = stats.spearmanr(lags, autocorr_values)
            
            temp_label = f"Temperature = {temp:g}"
            # Check if temperature label is in table index
            if temp_label in table.index:
                # Check that the model exists in the columns
                if model in table.columns:
                    # Retrieve Durbin-Watson statistic
                    dw_value = h2b_dw_df[
                        (h2b_dw_df['model'] == model) & (h2b_dw_df['temperature'] == temp)
                    ]['durbin_watson'].values
                    if len(dw_value) > 0:
                        dw_stat = dw_value[0]
                    else:
                        dw_stat = np.nan
                    # Populate the table with formatted values
                    table.loc[temp_label, model] = (
                        f"ρ = {spearman_coef:.3f}, p = {spearman_p_value:.3f}\n"
                        f"DW = {dw_stat:.3f}"
                    )
                else:
                    warnings.warn(f"Model '{model}' not found in H2b table columns.")
            else:
                warnings.warn(f"Temperature '{temp}' not found in H2b table index.")
    
    # Identify models with no data (shouldn't occur since all models have data)
    existing_models = autocorr_df['model'].unique()
    missing_models = set(all_generating_models) - set(existing_models)
    
    if missing_models:
        warnings.warn(f"No data available for the following models: {', '.join(missing_models)}. They are marked as 'N/A' in the H2b table.")
    
    return table

# Plot the autocorrelation results
def plot_combined_autocorrelation_panel(autocorrelations, max_lag=10, save_path=None):
    colors = {0.0: 'blue', 0.75: 'green', 1.0: 'red', 1.25: 'purple', 2.0: 'orange'}
    
    num_models = len(autocorrelations)
    fig_width = 5 * num_models  # Adjust width based on number of models
    fig, axes = plt.subplots(1, num_models, figsize=(fig_width, 5), sharey=True)
    fig.suptitle('Autocorrelation of Hallucinatory Tokens by Temperature for Each Generating Model', fontsize=16)
    
    # If only one model, axes is not a list
    if num_models == 1:
        axes = [axes]
    
    for idx, (model, temp_autocorr) in enumerate(autocorrelations.items()):
        ax = axes[idx]
        for temp, autocorr in temp_autocorr.items():
            if len(autocorr) >= max_lag:
                autocorr_to_plot = autocorr[:max_lag]
            else:
                autocorr_to_plot = autocorr
            ax.plot(range(1, len(autocorr_to_plot) + 1), autocorr_to_plot, marker='o', label=f'Temp: {temp}', color=colors.get(temp, 'black'))
        ax.set_title(f'Model: {model}', fontsize=14)
        ax.set_ylim(-0.1, 1.0)
        ax.set_xticks(range(1, max_lag + 1))
        ax.set_xlabel('Lag', fontsize=12)
        if idx == 0:
            ax.set_ylabel('Autocorrelation', fontsize=12)
        ax.legend(fontsize=10)
        # Remove grid lines
        ax.grid(False)
        # Adjust tick parameters
        ax.tick_params(axis='both', which='major', labelsize=10)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    if save_path:
        fig.savefig(save_path, dpi=300)
    plt.show()

# **ADDED: Function to compute and display descriptive statistics**
def compute_descriptive_statistics(df):
    print("\n=== Descriptive Statistics ===\n")
    
    # Overall statistics
    total_tokens = len(df)
    total_hallucinatory = df['label'].sum()
    proportion_hallucinatory = df['label'].mean()
    print(f"Total Tokens: {total_tokens}")
    print(f"Total Hallucinatory Tokens (label=1): {total_hallucinatory}")
    print(f"Proportion of Hallucinatory Tokens: {proportion_hallucinatory:.4f}\n")
    
    # Statistics by Model
    print("Descriptive Statistics by Model:")
    model_stats = df.groupby('model')['label'].agg(['count', 'sum', 'mean']).rename(columns={
        'count': 'Total Tokens',
        'sum': 'Hallucinatory Tokens',
        'mean': 'Proportion Hallucinatory'
    })
    print(model_stats.to_string())
    print()
    
    # Statistics by Temperature
    print("Descriptive Statistics by Temperature:")
    temp_stats = df.groupby('temperature')['label'].agg(['count', 'sum', 'mean']).rename(columns={
        'count': 'Total Tokens',
        'sum': 'Hallucinatory Tokens',
        'mean': 'Proportion Hallucinatory'
    })
    print(temp_stats.to_string())
    print()
    
    # Statistics by Model and Temperature
    print("Descriptive Statistics by Model and Temperature:")
    model_temp_stats = df.groupby(['model', 'temperature'])['label'].agg(['count', 'sum', 'mean']).rename(columns={
        'count': 'Total Tokens',
        'sum': 'Hallucinatory Tokens',
        'mean': 'Proportion Hallucinatory'
    })
    print(model_temp_stats.to_string())
    print()

# Testing H1a
def test_h1a(df):
    """
    Tests Hypothesis H1a: The proportion of hallucinatory tokens increases with higher temperature.
    """
    # Spearman's rank correlation
    temp_group = df.groupby('temperature')['label'].mean().reset_index()
    corr, p_value = stats.spearmanr(temp_group['temperature'], temp_group['label'])
    print(f"\nSpearman correlation between temperature and proportion of hallucinatory tokens:")
    print(f"Correlation coefficient: {corr:.4f}, p-value: {p_value:.4f}")
    if p_value < 0.05:
        print("Result: Reject H0, accept H1a (proportion increases with temperature).")
    else:
        print("Result: Fail to reject H0 (no significant correlation between temperature and proportion).")

# Testing H1b
def test_h1b(df):
    """
    Tests Hypothesis H1b: The proportion of hallucinatory tokens is greater in smaller models.
    """
    # Map generating models to parameter sizes
    model_sizes = {
        'Qwen2(0.5B)': 0.5,
        'Qwen2(7B)': 7,
        'Qwen2': 6,          # Assign appropriate size
        'Llama3.1': 8,
        'Gemma2': 9,
        'Mistral-NeMo': 12
    }
    # Verify that all generating models are mapped
    unmapped_models = set(df['model'].unique()) - set(model_sizes.keys())
    if unmapped_models:
        warnings.warn(f"Unmapped models found: {', '.join(unmapped_models)}. They will have NaN for model_size.")
    df['model_size'] = df['model'].map(model_sizes)
    
    # Testing H1b: Proportion increases with smaller models
    model_group = df.groupby('model_size')['label'].mean().reset_index()
    # Remove entries with NaN model_size
    model_group = model_group.dropna(subset=['model_size'])
    if model_group.empty:
        print("\nNo valid model sizes available for H1b testing.")
        return
    corr, p_value = stats.spearmanr(model_group['model_size'], model_group['label'])
    print(f"\nSpearman correlation between model size and proportion of hallucinatory tokens:")
    print(f"Correlation coefficient: {corr:.4f}, p-value: {p_value:.4f}")
    if p_value < 0.05:
        print("Result: Reject H0, accept H1b (proportion increases with smaller models).")
    else:
        print("Result: Fail to reject H0 (no significant correlation between model size and proportion).")

def main():
    # Define the folder path containing the data
    folder_path = '/Users/lucmacbookpro-profile/Desktop/summer research/URSS 2024/data_results/token'
    # Load and prepare data
    print("Loading data...")
    df = load_data(folder_path)
    if df.empty:
        print("No data loaded. Please check the data files and folder path.")
        return
    print("Data loaded successfully.")
    
    # **ADDED: Compute and display descriptive statistics**
    compute_descriptive_statistics(df)
    
    # Save combined data for reference
    df.to_csv('combined_token_data.csv', index=False)
    
    # Perform statistical tests for H1a and H1b
    test_h1a(df)
    test_h1b(df)
    
    # Extract hallucinatory sequences
    print("\nProcessing hallucinatory sequences...")
    hallucinatory_sequences = extract_hallucinatory_sequences(df)
    
    # Perform autocorrelation analysis
    print("Performing autocorrelation analysis...")
    autocorrelations = perform_autocorrelation_analysis(hallucinatory_sequences, max_lag=10)
    
    # Prepare autocorr_df for statistical tests
    autocorr_df_list = []
    for model, temp_autocorr in autocorrelations.items():
        for temp, autocorr_values in temp_autocorr.items():
            for lag, autocorr in enumerate(autocorr_values, start=1):
                autocorr_df_list.append({
                    'model': model,
                    'temperature': temp,
                    'lag': lag,
                    'autocorrelation': autocorr
                })
    autocorr_df = pd.DataFrame(autocorr_df_list)
    
    # Compute H2a autocorrelation at lag 1 and p-values
    print("\nComputing H2a autocorrelation at lag 1 and p-values...")
    h2a_results = compute_h2a_autocorrelation_pvalues(hallucinatory_sequences)
    print(h2a_results.to_string(index=False))
    h2a_results.to_csv("H2a_Autocorrelation_Lag1_Results.csv", index=False)
    print("\nH2a autocorrelation lag 1 results have been saved to 'H2a_Autocorrelation_Lag1_Results.csv'.")
    
    # Compute H2b Durbin-Watson statistics
    print("\nComputing H2b Durbin-Watson statistics...")
    h2b_dw_results = compute_h2b_durbin_watson(hallucinatory_sequences)
    print(h2b_dw_results.to_string(index=False))
    h2b_dw_results.to_csv("H2b_Durbin_Watson_Results.csv", index=False)
    print("\nH2b Durbin-Watson results have been saved to 'H2b_Durbin_Watson_Results.csv'.")
    
    # Generate H2b table
    print("\nGenerating H2b table with Spearman correlations and Durbin-Watson statistics...")
    h2b_table = generate_h2b_table(autocorr_df, h2b_dw_results)
    print("\nH2b Table:")
    print(h2b_table.to_string())
    h2b_table.to_csv("H2b_Autocorrelation_and_DW_Table.csv", index=True)
    print("\nH2b table has been saved to 'H2b_Autocorrelation_and_DW_Table.csv'.")
    
    # Plot the autocorrelation graphs
    print("\nPlotting autocorrelation graphs...")
    plot_combined_autocorrelation_panel(
        autocorrelations,
        max_lag=10,
        save_path="Token_Level_Autocorrelation_Panel.png"
    )
    print("\nAutocorrelation plots have been saved to 'Token_Level_Autocorrelation_Panel.png'.")
    
if __name__ == "__main__":
    main()