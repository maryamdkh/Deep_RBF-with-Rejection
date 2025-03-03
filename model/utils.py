import os
import torch
import torch.nn as nn

def load_feature_extractor(data_path, device):
    
    """
    Load a pretrained WideResNet-50-2 model and modify it to use layers before the last two layers (avgpool and fc) as the feature extractor.

    Args:
        data_path (str): Path to the saved model checkpoint.
        device (torch.device): Device to load the model onto (e.g., 'cuda' or 'cpu').

    Returns:
        model (nn.Module): Modified model with the last two layers removed.
    """
    # Check if the model path exists
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Path {data_path} does not exist.")

    # Load the pretrained WideResNet-50-2 model
    model = torch.hub.load('pytorch/vision:v0.10.0', 'wide_resnet50_2', pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, 3)


    # Load the model's state dict (if provided)
    if data_path:
        model.load_state_dict(torch.load(data_path))

    # Remove the last two layers (avgpool and fc)
    model = torch.nn.Sequential(*list(model.children())[:-2])

    # Move the model to the specified device
    model = model.to(device)

    return model