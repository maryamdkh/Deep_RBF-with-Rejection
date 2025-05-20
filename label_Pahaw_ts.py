import argparse
import torch
import pandas as pd
from data.Dataset import PaHaWTsDataset
from torch.utils.data import DataLoader
from trainer.utils import plot_confusion_matrix

from model.utils import load_all_models

from semi_supervised.PaHaW import label_pahaw_images
from timeseries_analysis.FeatureExrtaction import TimeSeriesFeatureProcessor  
from sklearn.preprocessing import StandardScaler


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Label images using an ensemble of DeepRBFNetwork models')
    parser.add_argument('--dataset', type=str, required=True, help='Path to CSV file containing data paths')
    parser.add_argument('--model_paths', nargs='+', required=True, 
                   help='List of model paths')
    parser.add_argument('--output', type=str, required=True, help='Output CSV file for results')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for inference')
    parser.add_argument('--rejection_threshold', type=float, default=1.0, 
                       help='Threshold for rejection based on distance')
    parser.add_argument('--confidence_threshold', type=float, default=0.5, 
                       help='Threshold for rejection based on confidence')
    parser.add_argument("--input_dim", type=int, required=True,
                        help="Dimension of the feature vector input to the DeepRBF network")
    parser.add_argument('--voting_method', type=str, default='majority', 
                       choices=['majority', 'distance_weighted'], help='Voting method to use')
    parser.add_argument('--num_classes', type=int, required=True, help='Number of classes in the model')
    parser.add_argument('--feature_dim', type=int, default=64, help='Dimension of feature space')
    parser.add_argument('--no_cuda', action='store_true', help='Disable CUDA even if available')
    parser.add_argument("--distance_metric", type=str, default='l2',
                        help="The distance metric used to calculate the distance in loss.")
    parser.add_argument("--lambda_margin", type=float, default=1.0,
                        help="Margin for the hinge loss (default: 1.0)")
    parser.add_argument('--method', type=str, default='rocket',
                       choices=['rocket', 'handcrafted', 'tsfresh', 'catch22', 'raw'],
                       help='Feature extraction method')
    parser.add_argument('--rocket_kernels', type=int, default=10000,
                       help='Number of kernels for ROCKET method')
    parser.add_argument('--n_jobs', type=int, default=-1,
                       help='Number of parallel jobs (-1 for all cores)')
    parser.add_argument('--features_dir', type=str, required=True,
                       help='Directory to save feature files')
    
    args = parser.parse_args()
    
    # Set up device
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"Using device: {device}")
    
    # Load dataset
    df = pd.read_csv(args.dataset)
    print(f"Loaded dataset with {len(df)} samples")

    processor = TimeSeriesFeatureProcessor(
        method=args.method,
        output_dir=args.features_dir,
        rocket_n_kernels=args.rocket_kernels,
        n_jobs=args.n_jobs,
        dataset_name = "pahaw"
    )
    print(f"Processing {len(df)} samples with {args.method} method...")
    features, df_with_paths = processor.process_dataframe(df)
    # Create dataset and data loader
    scaler = StandardScaler()
    dataset = PaHaWTsDataset(df_with_paths, scaler, has_labels=True)
    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    

    # Create template args for model initialization
    class ArgsTemplate:
        pass
    args_template = ArgsTemplate()
    args_template.num_classes = args.num_classes
    args_template.feature_dim = args.feature_dim
    args_template.distance_metric = args.distance_metric
    args_template.lambda_margin = args.lambda_margin
    args_template.input_dim = dataset.__getitem__(2)[0].shape[0]
    
    model_paths = args.model_paths
    # Load all models
    models = load_all_models(model_paths, args_template, device,feature_extractor=False)
    print(f"Loaded {len(models)} models")
    
    # Run inference
    results = label_pahaw_images(
        models=models,
        data_loader=data_loader,
        device=device,
        rejection_threshold=args.rejection_threshold,
        confidence_threshold=args.confidence_threshold,
        voting_method=args.voting_method
    )
    
    # Prepare results for output
    output_data = []
    for path, pred in results.items():
        output_data.append({
            'image_path': path,
            'predicted_label': pred['pred_label'],
            'true_label': pred['true_label'],
            'confidence': pred['confidence']
        })
    
    # Save results to CSV
    target_names = ["control", "parkinson", "rejected"]  # Adjust based on your labels
    output_df = pd.DataFrame(output_data)
    output_df['predicted_label'] = output_df['predicted_label'].apply(lambda x: 2 if x == -1 else x)
    plot_confusion_matrix(output_df['true_label'], output_df['predicted_label'], target_names,'/content/')
    output_df.to_csv(args.output, index=False)
    print(f"Saved results to {args.output}")

if __name__ == '__main__':
    main()