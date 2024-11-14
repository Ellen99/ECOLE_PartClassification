''' Utility function for image features '''
from typing import Tuple, List
import torch

def get_relevant_features(
    image_features: torch.Tensor,
    mask: torch.Tensor,
    patch_size: int,
    device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, List[Tuple[int, int, int, int]]]:
    """Extract relevant features based on mask."""
    h, w = mask.shape
    dh, dw = h // patch_size, w // patch_size

    assert image_features.shape[:2] == (dh, dw), "Invalid features shape"

    patched_mask = mask.view(dh, patch_size, dw, patch_size).max(dim=1)[0].max(dim=2)[0]
    relevant_indices = torch.nonzero(patched_mask).t()
    relevant_feats = image_features[relevant_indices[0], relevant_indices[1]].to(device)

    mask_ranges = [
        (
            y * patch_size,
            y * patch_size + patch_size,
            x * patch_size,
            x * patch_size + patch_size
        )
        for y, x in relevant_indices.t()
    ]
    return relevant_feats, relevant_indices, mask_ranges