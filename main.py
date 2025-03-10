import argparse
import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
import pandas as pd

from model.Model import DeepRBFNetwork
from model.utils import load_feature_extractor
from loss.MLLoss import MLLoss
from trainer.Trainer import Trainer
from data.Dataset import ParkinsonDataset
from data.Dataloader import balanced_collate_fn

def validate_fold(fold_id, df, args, device):
    """
    Validate a model for a single fold.

    Args:
        fold_id (int): ID of the current fold.
        df (pd.DataFrame): DataFrame for the dataset.
        args (argparse.Namespace): Command-line arguments.
        device (torch.device): Device to use for training.
    """
    print(f"validating fold {fold_id + 1}/{args.num_folds}")

    # Load the feature extractor model
    feature_extractor = load_feature_extractor( device= device)
    feature_extractor.eval() 

    # Define model, loss, optimizer, and data loaders
    model = DeepRBFNetwork(feature_extractor, args.num_classes, args.feature_dim)
    model = model.to(device)

    dataset = ParkinsonDataset(dataframe= df, data_dir=args.data_dir, is_train=False)
    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, collate_fn=balanced_collate_fn)

    # Initialize trainer
    trainer = Trainer(
        model=model,
        fold = fold_id+1,
        device=device,
        save_results=os.path.join(args.save_results, f"fold_{fold_id + 1}")
    )
    # Perform inference on training data and generate classification report
    trainer.predict(data_loader, threshold=16)



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

    # Load the feature extractor model
    if args.feature_extractor:
      feature_extractor = load_feature_extractor(os.path.join(args.feature_extractor, f"best_model_fold_{fold_id+1}.pth"), device)
    else:
      feature_extractor = load_feature_extractor(device=device)

    feature_extractor.eval()  # Set to evaluation mode (no training)

    # Define model, loss, optimizer, and data loaders
    model = DeepRBFNetwork(feature_extractor, args.num_classes, args.feature_dim)
    criterion = MLLoss(lambda_margin=args.lambda_margin)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.num_epochs, eta_min=args.lr * 0.01)

    model = model.to(device)

    # Load training dataset
    train_dataset = ParkinsonDataset(dataframe=train_df, data_dir=args.data_dir, is_train=True)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2, collate_fn=balanced_collate_fn)

    # Initialize validation dataset and DataLoader only if val_df is not empty
    val_loader = None
    if not val_df.empty:
        val_dataset = ParkinsonDataset(dataframe=val_df, data_dir=args.data_dir, is_train=False)
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
        save_dir=os.path.join(args.save_dir, f"fold_{fold_id + 1}"),
        save_results=os.path.join(args.save_results, f"fold_{fold_id + 1}")
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
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Train Deep-RBF Network with Rejection")
    parser.add_argument("--feature_extractor", type=str, required=False, default=None,
                        help="Root to the PyTorch model dir to use as the feature extractor")
    parser.add_argument("--num_classes", type=int, required=True,
                        help="Number of classes in the dataset")
    parser.add_argument("--feature_dim", type=int, required=True,
                        help="Dimension of the feature vector output by the feature extractor")
    parser.add_argument("--lambda_margin", type=float, default=1.0,
                        help="Margin for the hinge loss (default: 1.0)")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="Learning rate for the optimizer (default: 0.001)")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for training and validation (default: 32)")
    parser.add_argument("--num_epochs", type=int, default=20,
                        help="Number of epochs to train (default: 20)")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Directory containing the dataset")
    parser.add_argument("--folds_root_dir_train", type=str, required=True,
                        help="Root directory containing fold-specific train and validation CSV files")
    parser.add_argument("--folds_root_dir_val", type=str, required=True,
                        help="Root directory containing fold-specific validation CSV files")
    parser.add_argument("--num_folds", type=int, default=25,
                        help="Number of folds (default: 25)")
    parser.add_argument("--save_dir", type=str, default="checkpoints",
                        help="Directory to save model checkpoints (default: checkpoints)")
    parser.add_argument("--save_results", type=str, default="figs",
                        help="Directory to save model results (default: figs)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create directories for saving checkpoints and results
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.save_results, exist_ok=True)

    # Train a model for each fold
    best_model_paths = []
    for fold_id in range(5,11):
        # Load train and validation CSV files for the current fold
        train_csv_path = os.path.join(args.folds_root_dir_train, f"train_df_fold_{fold_id + 1}.csv")
        val_csv_path = os.path.join(args.folds_root_dir_val, f"val_df_fold_{fold_id + 1}.csv")

        train_df = read_csv_safe(train_csv_path)
        val_df = read_csv_safe(val_csv_path)
        # print(val_df)
        # break

        # Train the model for the current fold
        train_fold(fold_id, train_df, val_df, args, device)
    

    print("Training completed for all folds.")


if __name__ == "__main__":
    main()