import json
import torch
from typing import Tuple
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from pycocotools import mask as mask_utils

class MaskHandler:
    """Handle loading and standardization of RLE-encoded masks"""
    def __init__(self, target_size: Tuple[int, int] = (224, 224)):
        self.target_size = target_size
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(target_size, antialias=True)
        ])

    def get_image_path(self, mask_path: str) -> str:
        '''Get the image path from the mask json file'''
        try:
            with open(mask_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return data["image_path"]
        except (FileNotFoundError, json.JSONDecodeError, KeyError) as e:
            print(f"Error occurred when loading mask {mask_path}: {e}")
            return ""

    def load_image(self, image_path: str) -> torch.Tensor:
        '''Load image from path and apply transformations'''
        return self.transform(Image.open(image_path).convert("RGB"))

    def get_mask(self, mask_path: str) -> torch.Tensor:
        '''Get the mask tensor from the mask json file'''
        try:
            with open(mask_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            rle = {"size": data["size"], "counts": data["counts"]}
            mask = mask_utils.decode(rle)
            mask_tensor = torch.from_numpy(mask).float().unsqueeze(0)
            mask_tensor = F.interpolate(
                mask_tensor.unsqueeze(0),
                size=self.target_size,
                mode='nearest'
            ).squeeze()
            return mask_tensor
        
        except (FileNotFoundError, json.JSONDecodeError, KeyError) as e:
            print(f"Error occurred when loading mask {mask_path}: {e}")
            return torch.zeros(self.target_size)
        
    def validate_mask(self, mask: torch.Tensor) -> bool:
        """Validate mask format and values"""
        if not isinstance(mask, torch.Tensor):
            return False
        if mask.shape != self.target_size:
            return False
        if not torch.all((mask == 0) | (mask == 1)):
            return False
        return True
    