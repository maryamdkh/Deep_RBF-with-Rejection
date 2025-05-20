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
from trainer.utils import plot_confusion_matrix
from sklearn.preprocessing import StandardScaler



def validate_fold(fold_id, df,train_df, args, device):
    """
    Validate a model for a single fold.

    Args:
        fold_id (int): ID of the current fold.
        df (pd.DataFrame): DataFrame for the dataset.
        args (argparse.Namespace): Command-line arguments.
        device (torch.device): Device to use for training.
    """
    print(f"validating fold {fold_id + 1}/{args.num_folds}")

    # Initialize feature processor
    processor = TimeSeriesFeatureProcessor(
        method=args.method,
        output_dir=args.features_dir,
        rocket_n_kernels=args.rocket_kernels,
        n_jobs=args.n_jobs
    )

    scaler = StandardScaler()
    
    # Process DataFrame
    print(f"Processing {len(train_df)} samples with {args.method} method...")
    features, df_with_paths = processor.process_dataframe(train_df)
    train_dataset = ParkinsonTSDataset(dataframe=df_with_paths,scaler=scaler,is_train=True)

    # Process DataFrame
    print(f"Processing {len(df)} samples with {args.method} method...")
    features, df_with_paths = processor.process_dataframe(df)
    df_with_paths.to_csv(os.path.join(args.dfs_dir,f'val_df_fold_{fold_id + 1}.csv'))
    # Load training dataset
    val_dataset = ParkinsonTSDataset(dataframe=df_with_paths,scaler=scaler,is_train=False)
    plot_group_distribution(val_dataset,
                            file_path = f"{args.modeling_results_dir}/distributions/valid_{fold_id+1}.png", 
                            title=f"Group Distribution Validation Fold {fold_id+1}")
    
    data_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)

    # Define model, loss, optimizer, and data loaders
    args.input_dim = train_dataset.__getitem__(2)[0].shape[0]
    model = DeepRBFNetwork(args=args) 
    model = model.to(device)
  
    # Initialize trainer
    trainer = Trainer(
        model=model,
        fold = fold_id+1,
        device=device,
        save_results=os.path.join(args.modeling_results_dir)
    )
    trainer.best_model_weights = os.path.join(args.pre_model_dir,f"best_model_{fold_id + 1}.pt")
    trainer.load_best_model()

    inference_method = "min_distance" if args.loss_type == "mlloss" else "softml"
    # Perform inference on training data and generate classification report
    return trainer.predict(dataloader=data_loader, threshold=args.rejection_thresh,\
                    inference_method=inference_method, lambda_eval=args.lambda_eval,confidence_threshold=args.confidence_threshold)

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
        output_dir=args.features_dir,
        rocket_n_kernels=args.rocket_kernels,
        n_jobs=args.n_jobs
    )
    scaler = StandardScaler()
    
    # Process DataFrame
    print(f"Processing {len(train_df)} samples with {args.method} method...")
    features, df_with_paths = processor.process_dataframe(train_df)
    df_with_paths.to_csv(os.path.join(args.dfs_dir,f'train_df_fold_{fold_id + 1}.csv'))
    # Load training dataset
    train_dataset = ParkinsonTSDataset(dataframe=df_with_paths,scaler=scaler,is_train=True)
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
        # Process DataFrame
        print(f"Processing {len(val_df)} samples with {args.method} method...")
        features, df_with_paths = processor.process_dataframe(val_df)
        df_with_paths.to_csv(os.path.join(args.dfs_dir,f'valid_df_fold_{fold_id + 1}.csv'))
        val_dataset = ParkinsonTSDataset(dataframe=df_with_paths,scaler=scaler, is_train=False)
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
        save_dir=os.path.join(args.checkpoints_dir),
        save_results=os.path.join(args.modeling_results_dir,"loss")
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
    
def init_config(args):
    os.makedirs(args.dfs_dir, exist_ok=True)
    os.makedirs(args.features_dir, exist_ok=True)
    os.makedirs(args.checkpoints_dir, exist_ok=True)
    os.makedirs(args.modeling_results_dir, exist_ok=True)
    os.makedirs(os.path.join(args.modeling_results_dir,"distributions"), exist_ok=True)
    os.makedirs(os.path.join(args.modeling_results_dir,"loss"), exist_ok=True)
    os.makedirs(os.path.join(args.modeling_results_dir,"scalers"), exist_ok=True)
    
    set_seed(args.random_state)

def init_parser():
        # Set up argument parser
    parser = argparse.ArgumentParser(description='Process time series features from a DataFrame')
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
    # parser.add_argument("--input_dim", type=int, required=True,
    #                     help="Dimension of the feature vector input to the DeepRBF network")
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
    parser.add_argument("--pre_model_dir", type=str, default="",
                        help="Directory to the pretrained models.")
    parser.add_argument("--rejection_thresh", type=float, default=16,
                        help="Threshold used to reject the sample.")
    parser.add_argument("--loss_type", type=str, default="mlloss",
                        help="Type of the loss function to use for training the model.")
    parser.add_argument("--lambda_eval", type=float, default=100, required=False,
                        help="Lambda parameter for inference when using soft mlloss.")
    parser.add_argument("--random_state", type=int, default=42, required=False,
                        help="Random state.")

    return parser
 
def main():
    args = init_parser().parse_args()
    init_config(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # for fold_id in range(15):
    #     # Load train and validation CSV files for the current fold
    #     train_csv_path = os.path.join(args.folds_root_dir_train, f"train_df_fold_{fold_id + 1}.csv")
    #     val_csv_path = os.path.join(args.folds_root_dir_val, f"val_df_fold_{fold_id + 1}.csv")

    #     train_df = read_csv_safe(train_csv_path)
    #     val_df = read_csv_safe(val_csv_path)

    #     # Train the model for the current fold
    #     train_fold(fold_id, train_df, val_df, args, device)
    

    # print("Training completed for all folds.")

    #Initialize lists to collect all predicted and true labels across all folds
    all_folds_predicted_labels = []
    all_folds_doctor_labels = []    
    all_folds_real_labels = []    
    all_folds_subject_ids = []    
    all_folds_distances = []

    # Validate a model for each fold
    for fold_id in range(args.num_folds):
        # Load validation CSV file for the current fold
        val_csv_path = os.path.join(args.folds_root_dir_val, f"val_df_fold_{fold_id + 1}.csv")
        train_csv_path = os.path.join(args.folds_root_dir_train, f"train_df_fold_{fold_id + 1}.csv")
        val_df = read_csv_safe(val_csv_path)
        if not len(val_df):
          continue
        train_df = read_csv_safe(train_csv_path)
        # Validate the model for the current fold
        all_distances,all_predicted_labels, all_doctor_labels = validate_fold(fold_id,train_df=train_df, df=val_df, args=args, device=device)
        
        # Collect the predicted and true labels for this fold
        all_folds_predicted_labels.extend(all_predicted_labels)
        all_folds_doctor_labels.extend(all_doctor_labels)
        all_folds_real_labels.extend(val_df.loc[:,"real_labels"].to_list() )
        all_folds_subject_ids.extend([item.split("/")[-1].split(".")[0] for item in val_df.loc[:,"path"].to_list()])
        all_folds_distances.extend(all_distances)


    print("Validating completed for all folds.")

    val_df_data = {"id":all_folds_subject_ids, "doctor_label":all_folds_doctor_labels,"real_label":all_folds_real_labels,
                  "predicted_label":all_folds_predicted_labels,"distance":all_folds_distances}
    pd.DataFrame(data =val_df_data).to_csv("valiadtion_results.csv")

    # Plot a confusion matrix for all folds combined
    target_names = ["control", "parkinson", "rejected"]  # Adjust based on your labels
    plot_confusion_matrix(all_folds_doctor_labels, all_folds_predicted_labels, target_names,args.save_results)
    args_dict = vars(args)
    args_df = pd.DataFrame.from_dict(args_dict, orient='index', columns=['Value'])
    args_df.to_csv(os.path.join(args.save_results, "config_arguments.csv"))



    


if __name__ == '__main__':
    main()