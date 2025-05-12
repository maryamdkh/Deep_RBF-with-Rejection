import os
import torch
import torch.nn as nn
import random
import numpy as np

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_feature_extractor(device, data_path=None):
    """
    Load a pretrained WideResNet-50-2 model and modify it to use layers before the last two layers (avgpool and fc) as the feature extractor.
    Then, append additional layers to reduce dimensions from 2048 → 1024 → 512.

    Args:
        data_path (str): Path to the saved model checkpoint.
        device (torch.device): Device to load the model onto (e.g., 'cuda' or 'cpu').

    Returns:
        model (nn.Module): Modified model with the last two layers removed and new layers appended.
    """
    
    # Load the pretrained WideResNet-50-2 model
    model = torch.hub.load('pytorch/vision:v0.10.0', 'wide_resnet50_2', pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, 3)  # Original fc layer (kept for compatibility if loading weights)

    # Load the model's state dict (if provided)
    if data_path:
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Path {data_path} does not exist.")
        model.load_state_dict(torch.load(data_path))

    # Remove the last two layers (avgpool and fc)
    model = torch.nn.Sequential(*list(model.children())[:-1])
  

    return model