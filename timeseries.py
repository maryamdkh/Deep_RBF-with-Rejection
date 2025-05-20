#!/usr/bin/env python3
import os
import argparse
import pandas as pd
import pickle
from timeseries_analysis.FeatureExrtaction import TimeSeriesFeatureProcessor  
from data.Dataset import ParkinsonTSDataset
from data.utils import plot_group_distribution
from model.utils import set_seed
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

from loss.MLLoss import MLLoss
from loss.SoftMLLoss import SoftMLLoss
from trainer.Trainer import Trainer
from model.Model import DeepRBFNetwork


def train_fold(fold_id, train_df, val_df, args, device):
    """
    Train a model for a single fold.

    Args:
        fold_id (int): ID of the current fold.
        train_df (pd.DataFrame): DataFrame for the training set.
        val_df (pd.DataFrame): DataFrame for the validation set.
        args (argparse.Namespace): Command-line arguments.
        device (torch.device): Device to use for training.
    """
    print(f"Training fold {fold_id + 1}/{args.num_folds}")

    # Initialize feature processor
    processor = TimeSeriesFeatureProcessor(
        method=args.method,
        output_dir=args.output_dir,
        rocket_n_kernels=args.rocket_kernels,
        n_jobs=args.n_jobs
    )
    
    # Process DataFrame
    print(f"Processing {len(train_df)} samples with {args.method} method...")
    features, df_with_paths = processor.process_dataframe(train_df)
    df_with_paths.to_csv(os.path.join(args.output_df_dir,f'train_df_fold_{fold_id + 1}.csv'))
    # Load training dataset
    train_dataset = ParkinsonTSDataset(dataframe=df_with_paths,is_train=True)
    plot_group_distribution(train_dataset,
                            file_path = f"{args.modeling_results_dir}/distributions/train_{fold_id+1}.png", 
                            title=f"Group Distribution Train Fold {fold_id+1}")
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)

    args.input_dim = train_dataset.__getitem__(2)[0].shape[0]
    # Define model, loss, optimizer, and data loaders
    model = DeepRBFNetwork(args=args) 
    criterion = MLLoss(lambda_margin=args.lambda_margin,lambda_min=args.lambda_min) if args.loss_type == "mlloss" else SoftMLLoss(lambda_margin=args.lambda_margin)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.num_epochs, eta_min=args.lr * 0.01)

    model = model.to(device)

    # Initialize validation dataset and DataLoader only if val_df is not empty
    val_loader = None
    if not val_df.empty:
        val_dataset = ParkinsonTSDataset(dataframe=val_df, is_train=False)
        plot_group_distribution(val_dataset,
                            file_path = f"{args.modeling_results_dir}/distributions/valid_{fold_id+1}.png", 
                            title=f"Group Distribution Validation Fold {fold_id+1}")
        val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, num_workers=2)

    # Initialize trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,  # Pass None if val_df is empty
        fold=fold_id + 1,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        save_dir=os.path.join(args.save_dir),
        save_results=os.path.join(args.save_results,"loss")
    )

    # Train the model
    trainer.train(num_epochs=args.num_epochs)
  
  
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
    parser.add_argument('--dfs_dir', type=str, required=True,
                       help='Directory to save processed DataFrame with feature paths')
    parser.add_argument('--features_dir', type=str, required=True,
                       help='Directory to save feature files')
    parser.add_argument('--checkpoints_dir', type=str, required=True,
                       help='Directory to save model checkpoints')
    parser.add_argument('--modeling_results_dir', type=str, required=True,
                       help='Directory to save feature modeling results.')
    parser.add_argument("--folds_root_dir_train", type=str, required=True,
                        help="Root directory containing fold-specific train and validation CSV files")
    parser.add_argument("--folds_root_dir_val", type=str, required=True,
                        help="Root directory containing fold-specific validation CSV files")
    parser.add_argument("--num_folds", type=int, default=25,
                        help="Number of folds (default: 25)")
    parser.add_argument('--method', type=str, default='rocket',
                       choices=['rocket', 'handcrafted', 'tsfresh', 'catch22', 'raw'],
                       help='Feature extraction method')
    parser.add_argument('--rocket_kernels', type=int, default=10000,
                       help='Number of kernels for ROCKET method')
    parser.add_argument('--n_jobs', type=int, default=-1,
                       help='Number of parallel jobs (-1 for all cores)')
    parser.add_argument("--num_classes", type=int, required=True,
                        help="Number of classes in the dataset")
    parser.add_argument("--feature_dim", type=int, required=True,
                        help="Dimension of the feature vector used to calculate the distance.")
    parser.add_argument("--input_dim", type=int, required=True,
                        help="Dimension of the feature vector input to the DeepRBF network")
    parser.add_argument("--lambda_margin", type=float, default=1.0,
                        help="Margin for the hinge loss (default: 1.0)")
    parser.add_argument("--lambda_min", type=float, default=100,
                        help="Minimum intra-distance (default: 100)")
    parser.add_argument("--confidence_threshold", type=float, default=50,
                        help="The minimum distance difference between classes, used during inference.")
    parser.add_argument("--distance_metric", type=str, default='l2',
                        help="The distance metric used to calculate the distance in loss.")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="Learning rate for the optimizer (default: 0.001)")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for training and validation (default: 32)")
    parser.add_argument("--num_epochs", type=int, default=20,
                        help="Number of epochs to train (default: 20)")
    
    args = parser.parse_args()
    os.makedirs(args.dfs_dir, exist_ok=True)
    os.makedirs(args.features_dir, exist_ok=True)
    os.makedirs(args.checkpoints_dir, exist_ok=True)
    os.makedirs(args.modeling_results_dir, exist_ok=True)
    os.makedirs(os.path.join(args.modeling_results_dir,"distributions"), exist_ok=True)
    os.makedirs(os.path.join(args.modeling_results_dir,"loss"), exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(args.random_state)


    for fold_id in range(15):
        # Load train and validation CSV files for the current fold
        train_csv_path = os.path.join(args.folds_root_dir_train, f"train_df_fold_{fold_id + 1}.csv")
        val_csv_path = os.path.join(args.folds_root_dir_val, f"val_df_fold_{fold_id + 1}.csv")

        train_df = read_csv_safe(train_csv_path)
        val_df = read_csv_safe(val_csv_path)

        # Train the model for the current fold
        train_fold(fold_id, train_df, val_df, args, device)
    

    print("Training completed for all folds.")


    


if __name__ == '__main__':
    main()