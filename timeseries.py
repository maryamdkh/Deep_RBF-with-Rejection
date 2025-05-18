#!/usr/bin/env python3
import os
import argparse
import pandas as pd
import pickle
from timeseries_analysis.FeatureExrtaction import TimeSeriesFeatureProcessor  

def read_csv_safe(file_path):
    """
    Safely read a CSV file. If the file is empty, return an empty DataFrame.

    Args:
        file_path (str): Path to the CSV file.

    Returns:
        pd.DataFrame: DataFrame containing the data from the CSV file, or an empty DataFrame if the file is empty.
    """
    try:
        df = pd.read_csv(file_path)
        return df
    except pd.errors.EmptyDataError:
        print(f"Warning: The file '{file_path}' is empty. Returning an empty DataFrame.")
        return pd.DataFrame()  # Return an empty DataFrame
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' does not exist.")
        return pd.DataFrame()  # Return an empty DataFrame
    except Exception as e:
        print(f"An unexpected error occurred while reading '{file_path}': {e}")
        return pd.DataFrame()  # Return an empty DataFrame
    
def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Process time series features from a DataFrame')
    parser.add_argument('--input', type=str, required=True, 
                       help='Path to input DataFrame pickle file')
    parser.add_argument('--output_df_dir', type=str, required=True,
                       help='Directory to save processed DataFrame with feature paths')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Directory to save feature files')
    parser.add_argument('--method', type=str, default='rocket',
                       choices=['rocket', 'handcrafted', 'tsfresh', 'catch22', 'raw'],
                       help='Feature extraction method')
    parser.add_argument('--rocket_kernels', type=int, default=10000,
                       help='Number of kernels for ROCKET method')
    parser.add_argument('--n_jobs', type=int, default=-1,
                       help='Number of parallel jobs (-1 for all cores)')
    
    args = parser.parse_args()
    os.makedirs(args.output_df_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    # Load input DataFrame
    print(f"Loading DataFrame from {args.input}...")
    df = read_csv_safe(args.input)
    
    # Initialize feature processor
    processor = TimeSeriesFeatureProcessor(
        method=args.method,
        output_dir=args.output_dir,
        rocket_n_kernels=args.rocket_kernels,
        n_jobs=args.n_jobs
    )
    
    # Process DataFrame
    print(f"Processing {len(df)} samples with {args.method} method...")
    features, df_with_paths = processor.process_dataframe(df)
    df_with_paths.to_csv(os.path.join(args.output_df_dir,'sample_df.csv'))
    


if __name__ == '__main__':
    main()