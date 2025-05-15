import argparse
import torch
import pandas as pd
from data.Dataset import PaHaWDataset
from torch.utils.data import DataLoader
from trainer.utils import plot_confusion_matrix

from model.utils import load_feature_extractor, load_all_models

from semi_supervised.PaHaW import label_pahaw_images

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Label images using an ensemble of DeepRBFNetwork models')
    parser.add_argument('--dataset', type=str, required=True, help='Path to CSV file containing image paths')
    parser.add_argument('--image_root', type=str, default=None, help='Root directory for images (optional)')
    parser.add_argument('--model_paths', nargs='+', required=True, 
                   help='List of model paths')
    parser.add_argument('--output', type=str, required=True, help='Output CSV file for results')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for inference')
    parser.add_argument('--rejection_threshold', type=float, default=1.0, 
                       help='Threshold for rejection based on distance')
    parser.add_argument('--confidence_threshold', type=float, default=0.5, 
                       help='Threshold for rejection based on confidence')
    parser.add_argument('--voting_method', type=str, default='majority', 
                       choices=['majority', 'distance_weighted'], help='Voting method to use')
    parser.add_argument('--num_classes', type=int, required=True, help='Number of classes in the model')
    parser.add_argument('--feature_dim', type=int, default=64, help='Dimension of feature space')
    parser.add_argument('--no_cuda', action='store_true', help='Disable CUDA even if available')
    parser.add_argument("--distance_metric", type=str, default='l2',
                        help="The distance metric used to calculate the distance in loss.")
    parser.add_argument("--lambda_margin", type=float, default=1.0,
                        help="Margin for the hinge loss (default: 1.0)")
    
    args = parser.parse_args()
    
    # Set up device
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"Using device: {device}")
    
    # Load dataset
    df = pd.read_csv(args.dataset)
    print(f"Loaded dataset with {len(df)} images")
    
    # Create dataset and data loader
    dataset = PaHaWDataset(df, image_root_dir=args.image_root, has_labels=True)
    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    
    # Initialize feature extractor (replace with your actual feature extractor)
    feature_extractor  = load_feature_extractor(device=device)
    
    # Create template args for model initialization
    class ArgsTemplate:
        pass
    args_template = ArgsTemplate()
    args_template.num_classes = args.num_classes
    args_template.feature_dim = args.feature_dim
    args_template.distance_metric = args.distance_metric
    args_template.lambda_margin = args.lambda_margin

    # print(args.model_paths)
    
    model_paths = args.model_paths
    # Load all models
    models = load_all_models(model_paths, feature_extractor, args_template, device)
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
    plot_confusion_matrix(output_data['true_label'], output_data['predicted_label'], target_names,'/content/')
    output_df = pd.DataFrame(output_data)
    output_df.to_csv(args.output, index=False)
    print(f"Saved results to {args.output}")

if __name__ == '__main__':
    main()