import argparse
import torch
import torch.optim as optim
from model import DeepRBFNetwork
from loss import CustomLoss
from trainer import Trainer
from dataset import CustomDataset
from dataloader import DataLoader, balanced_collate_fn

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Train Deep-RBF Network with Rejection")
    parser.add_argument("--feature_extractor", type=str, required=True,
                        help="Path to the PyTorch model to use as the feature extractor")
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
    parser.add_argument("--save_dir", type=str, default="checkpoints",
                        help="Directory to save model checkpoints (default: checkpoints)")
    args = parser.parse_args()

    # Load the feature extractor model
    feature_extractor = torch.load(args.feature_extractor)
    feature_extractor.eval()  # Set to evaluation mode (no training)

    # Define model, loss, optimizer, and data loaders
    model = DeepRBFNetwork(feature_extractor, args.num_classes, args.feature_dim)
    criterion = CustomLoss(lambda_margin=args.lambda_margin)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Load dataset
    train_dataset = CustomDataset(dataframe=pd.read_csv("train_labels.csv"), data_dir=args.data_dir, is_train=True)
    val_dataset = CustomDataset(dataframe=pd.read_csv("val_labels.csv"), data_dir=args.data_dir, is_train=False)

    # Create DataLoader
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, collate_fn=balanced_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, collate_fn=balanced_collate_fn)

    # Initialize trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        save_dir=args.save_dir
    )

    # Train the model
    trainer.train(num_epochs=args.num_epochs)

if __name__ == "__main__":
    main()