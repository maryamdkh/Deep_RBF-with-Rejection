import torch

def label_pahaw_images(models, data_loader, device, rejection_threshold=1.0, 
                      confidence_threshold=0.5, voting_method='majority'):
    """
    Label images using an ensemble of DeepRBFNetwork models with different voting mechanisms.
    
    Args:
        models (list): List of trained DeepRBFNetwork models
        data_loader (DataLoader): DataLoader for the PaHaW dataset
        device (torch.device): Device to run inference on
        rejection_threshold (float): Threshold for rejection based on distance
        confidence_threshold (float): Threshold for rejection based on confidence
        voting_method (str): Voting method to use - 'majority' or 'distance_weighted'
        
    Returns:
        dict: Dictionary mapping image paths to their predicted labels (2 for rejected),
              true labels, and optionally confidence scores if using distance_weighted method
    """
    results = {}
    
    with torch.no_grad():
        for images, true_labels, paths in data_loader:
            images = images.to(device)
            true_labels = true_labels.cpu().numpy()  # Convert true labels to numpy array
            
            batch_size = images.size(0)
            
            # Initialize storage for this batch
            batch_votes = {path: {'labels': [], 'distances': [], 'true_label': true_label} 
                          for path, true_label in zip(paths, true_labels)}
            
            for model in models:
                model.eval()
                distances, pred_labels, is_rejected = model.inference(
                    images, 
                    rejection_threshold=rejection_threshold,
                    confidence_threshold=confidence_threshold
                )
                
                # Collect votes and distances from each model
                pred_labels = pred_labels.cpu().numpy()
                distances = distances.cpu().numpy()
                
                for i, path in enumerate(paths):
                    batch_votes[path]['labels'].append(pred_labels[i])
                    batch_votes[path]['distances'].append(distances[i])
            
            # Determine final label for each image in this batch
            for path, vote_data in batch_votes.items():
                labels = vote_data['labels']
                distances = vote_data['distances']
                true_label = vote_data['true_label']
                
                if len(labels) == 0:
                    # All models rejected this sample
                    results[path] = {
                        'pred_label': 2, 
                        'confidence': 0.0,
                        'true_label': true_label
                    }
                    continue
                
                if voting_method == 'majority':
                    # Option 1: Simple majority voting
                    counts = {}
                    for label in labels:
                        counts[label] = counts.get(label, 0) + 1
                    # Get label with max votes, with random tie-break
                    final_label = max(counts.items(), key=lambda x: (x[1], x[0]))[0]
                    confidence = counts[final_label] / len(labels)  # Proportion of votes
                    
                elif voting_method == 'distance_weighted':
                    # Option 2: Distance-weighted voting
                    vote_weights = {}
                    total_weight = 0.0
                    
                    for label, dist in zip(labels, distances):
                        # Convert distance to weight (smaller distance = higher weight)
                        weight = 1.0 / (1.0 + dist[label])  # Using only the distance to predicted class
                        
                        vote_weights[label] = vote_weights.get(label, 0.0) + weight
                        total_weight += weight
                    
                    if total_weight > 0:
                        # Normalize weights and select highest weighted label
                        for label in vote_weights:
                            vote_weights[label] /= total_weight
                        final_label = max(vote_weights.items(), key=lambda x: x[1])[0]
                        confidence = vote_weights[final_label]
                    else:
                        final_label = 2
                        confidence = 0.0
                
                else:
                    raise ValueError(f"Unknown voting method: {voting_method}")
                
                results[path] = {
                    'pred_label': final_label, 
                    'confidence': confidence,
                    'true_label': true_label
                }
    
    return results