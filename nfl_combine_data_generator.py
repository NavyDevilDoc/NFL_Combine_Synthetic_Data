import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from scipy import stats

import warnings
warnings.filterwarnings("ignore")

'''
Units:
    Ht        - in
    Wt        - lb
    Forty     - s
    Vertical  - in
    BenchReps - reps
    BroadJump - in
    Cone      - s
    Shuttle   - s
'''


# Function to generate synthetic data for a given position
def generate_synthetic_data(stats, num_samples):
    synthetic_data = {}
    for metric in stats['mean'].index:
        mean = stats['mean'][metric]
        std = stats['std'][metric]
        synthetic_data[metric] = np.random.normal(mean, std, num_samples)
    return pd.DataFrame(synthetic_data)

def load_and_clean_data(file_path):
    """Load and clean the NFL combine data."""
    df = pd.read_csv(file_path)
    columns_to_drop = ['Player', 'Year', 'Pfr_ID', 'AV', 'Team', 'Round', 'Pick']
    df_cleaned = df.drop(columns=columns_to_drop)
    df_filtered = df_cleaned[~df_cleaned['Pos'].isin(['QB', 'K', 'P', 'G', 'EDGE', 'LS', 'DB'])]
    return df_filtered

def map_positions(df):
    """Map similar positions to unified labels."""
    position_mapping = {
        'OLB': 'LB', 'ILB': 'LB', 'CB': 'CB', 'FS': 'S', 'SS': 'S', 'WR': 'WR',
        'RB': 'RB', 'TE': 'TE', 'OT': 'OL', 'OG': 'OL', 'C': 'OL', 'DT': 'DL',
        'DE': 'DL', 'NT': 'DL'
    }
    df.loc[:, 'Pos'] = df['Pos'].map(position_mapping).fillna(df['Pos'])
    return df

def calculate_stats(df_positions):
    """Calculate mean and standard deviation for each metric in each position."""
    stats = {}
    for pos, df_pos in df_positions.items():
        numeric_df_pos = df_pos.select_dtypes(include=[float, int])
        stats[pos] = {
            'mean': numeric_df_pos.mean(),
            'std': numeric_df_pos.std()
        }
    return stats

def calculate_nan_percentage(df):
    nan_percentage = df.isna().mean() * 100
    return nan_percentage

def plot_histograms(df_positions):
    """Plot histograms for each position and metric combination."""
    metrics = df_positions[next(iter(df_positions))].select_dtypes(include=[float, int]).columns
    for pos, df_pos in df_positions.items():
        for metric in metrics:
            plt.figure(figsize=(10, 6))
            df_pos[metric] = df_pos[metric].replace([np.inf, -np.inf], np.nan)
            sns.histplot(df_pos[metric].dropna(), bins=20, kde=True)
            plt.title(f'Histogram of {metric} for {pos}')
            plt.xlabel(metric)
            plt.ylabel('Frequency')
            plt.show()

def plot_qq_plots(df_positions):
    """Plot Q-Q plots for each position and metric combination."""
    metrics = df_positions[next(iter(df_positions))].select_dtypes(include=[float, int]).columns
    for pos, df_pos in df_positions.items():
        for metric in metrics:
            # Replace inf values with NaN
            df_pos[metric] = df_pos[metric].replace([np.inf, -np.inf], np.nan)
            
            plt.figure(figsize=(10, 6))
            stats.probplot(df_pos[metric].dropna(), dist="norm", plot=plt)
            plt.title(f'Q-Q plot of {metric} for {pos}')
            plt.xlabel('Theoretical Quantiles')
            plt.ylabel('Sample Quantiles')
            plt.show()

def shapiro_wilk_test(df_positions):
    """Perform Shapiro-Wilk test for each position and metric combination."""
    metrics = df_positions[next(iter(df_positions))].select_dtypes(include=[float, int]).columns
    results = {}
    for pos, df_pos in df_positions.items():
        results[pos] = {}
        for metric in metrics:
            # Replace inf values with NaN
            df_pos[metric] = df_pos[metric].replace([np.inf, -np.inf], np.nan)
            
            # Perform Shapiro-Wilk test
            stat, p_value = stats.shapiro(df_pos[metric].dropna())
            results[pos][metric] = (stat, p_value)
    return results

def main():
    # Load and clean data
    file_path = r"C:\Users\docsp\Desktop\AI_ML_Folder\Python_Practice_Folder\Datasets\NFL_Combine_Data\combine_data_since_2000_PROCESSED_2018-04-26.csv"
    df = load_and_clean_data(file_path)
    df = map_positions(df)

    # Calculate and display the percentage of NaN values
    nan_percentage = calculate_nan_percentage(df)
    print("Percentage of NaN values in each column:")
    print(nan_percentage)

    # Display unique positions
    print("Unique Positions After Mapping:")
    print(df['Pos'].unique())

    # Create separate dataframes for each combined position
    positions = df['Pos'].unique()
    df_positions = {pos: df[df['Pos'] == pos] for pos in positions}

    # Display the first few rows of each position dataframe
    for pos, df_pos in df_positions.items():
        print(f"DataFrame for {pos}:")
        print(df_pos.head())

    # Calculate stats
    stats = calculate_stats(df_positions)

    # Display the calculated statistics
    for pos, stat in stats.items():
        print(f"Position: {pos}")
        print("Mean values:")
        print(stat['mean'])
        print("Standard Deviation values:")
        print(stat['std'])
        print()

    # Total number of samples to generate
    total_samples = 15000

    # Define positions
    position_labels = ['WR', 'RB', 'OL', 'CB', 'S', 'TE', 'FB', 'LB', 'DL']
    
    # Set a minimum proportion for each position
    min_proportion = 0.01  # Ensure each position has at least 1% representation

    # Generate random proportions
    remaining_proportion = 1.0 - len(position_labels) * min_proportion
    random_proportions = np.random.rand(len(position_labels))
    
    # Normalize the proportions so they add up to 1
    proportions = random_proportions / random_proportions.sum() * remaining_proportion
    
    # Add the minimum proportion to each position
    proportions += min_proportion
    
    # Create a dictionary for proportions
    position_proportions = dict(zip(position_labels, proportions))
    
    # Display the generated proportions
    print("Generated Proportions for Each Position:")
    for position, proportion in position_proportions.items():
        print(f"{position}: {proportion:.2f}")

    # Calculate the number of samples for each position
    num_samples_per_position = {pos: int(total_samples * proportion) for pos, proportion in position_proportions.items()}

    # Adjust the number of samples for the last position to ensure the total is exactly 15000
    total_assigned_samples = sum(num_samples_per_position.values())
    last_position = position_labels[-1]
    num_samples_per_position[last_position] += (total_samples - total_assigned_samples)

    # Generate synthetic data for each position
    synthetic_dfs = {}
    for pos in stats.keys():
        if pos in num_samples_per_position:
            synthetic_dfs[pos] = generate_synthetic_data(stats[pos], num_samples_per_position[pos])
            synthetic_dfs[pos]['Pos'] = pos  # Add position label

    # Combine all synthetic dataframes into one
    synthetic_dataset = pd.concat(synthetic_dfs.values(), ignore_index=True)

    # Plot histograms and Q-Q Plot for each position/metric combination 
    #plot_histograms(df_positions)
    #plot_qq_plots(df_positions)
    
    shapiro_results = shapiro_wilk_test(df_positions)
    for pos, metrics in shapiro_results.items():
        print(f"Shapiro-Wilk Test Results for {pos}:")
        for metric, (stat, p_value) in metrics.items():
            print(f"  {metric}: statistic={stat}, p-value={p_value}")
            if p_value > 0.05:
                print(f"    {metric} for {pos} likely follows a normal distribution (p > 0.05).")
            else:
                print(f"    {metric} for {pos} does not follow a normal distribution (p <= 0.05).")

    # Randomize the row order of the synthetic dataset
    synthetic_dataset_shuffled = synthetic_dataset.sample(frac=1, random_state=42).reset_index(drop=True)

    # Encode the position column
    label_encoder = LabelEncoder()
    synthetic_dataset_shuffled['Pos'] = label_encoder.fit_transform(synthetic_dataset_shuffled['Pos'])

    # Retrieve and print the mapping
    position_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
    print("Position to Label Mapping:")
    print(position_mapping)

    return df, df_positions, stats, synthetic_dataset, synthetic_dataset_shuffled, position_mapping, shapiro_results

if __name__ == "__main__":
    df, df_positions, stats, synthetic_dataset, synthetic_dataset_shuffled, position_mapping, shapiro_results = main()
