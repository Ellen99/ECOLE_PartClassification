import os
import re
from typing import Dict, Tuple
from PIL import Image
import numpy as np
import torch
from tqdm import tqdm
from torchvision import transforms
from models.dino_encoder import DinoEncoder
from src.utils.mask_handler import MaskHandler
from src.utils.config import TrainingConfig

class DataLoader:
    """Handles loading and preprocessing of training data."""
    def __init__(self, config: TrainingConfig):
        self.mask_dir = config.mask_dir
        self.feat_dir = config.feat_dir
        self.patch_size = config.patch_size
        self.device = config.device
        self.dh = config.dh
        self.dw = config.dw

        self.mask_handler = MaskHandler()
        self.dino_encoder = DinoEncoder(self.device, config.dino_model)

        self.transform = transforms.Compose([
            transforms.Resize((config.image_height, config.image_width)),
            transforms.ToTensor()
        ])

    def load_image(self, image_path: str) -> torch.Tensor:
        """Load and preprocess image."""
        try:
            image = self.transform(Image.open(image_path).convert("RGB"))
            return image
        except (FileNotFoundError, OSError, IOError) as e:
            print(f"Error loading image {image_path}: {e}")
            return None

    def get_image_data_from_path(self) -> Tuple[Dict, Dict]:
        """Load all training data and concept hierarchy."""
        image_cache = {}
        concept_parts = {}

        for concept_folder in tqdm(os.listdir(self.mask_dir), desc="Loading image data"):
            match = re.search(r'--part:([^/]+)', concept_folder)
            if not match:
                # skip non-part annotations
                continue

            part_name = match.group(1)
            concept_name = concept_folder.split('--part:')[0]

            if concept_name not in concept_parts:
                concept_parts[concept_name] = [part_name]
            elif part_name not in concept_parts[concept_name]:
                concept_parts[concept_name].append(part_name)

            self._process_concept_folder(concept_folder,
                                         part_name,
                                         concept_name,
                                         image_cache)

        return image_cache, concept_parts

    def _process_concept_folder(self,
                                concept_folder: str,
                                part_name: str,
                                concept_name: str,
                                image_cache: Dict):
        concept_path = os.path.join(self.mask_dir, concept_folder)
        for mask_file in os.listdir(concept_path):
            mask_path = os.path.join(concept_path, mask_file)

            # handle the case when there are sub-folders for parts
            if os.path.isdir(mask_path):
                sub_part_name = mask_file
                for sub_mask_file in os.listdir(mask_path):
                    sub_mask_path = os.path.join(mask_path, sub_mask_file)
                    if sub_mask_file.endswith('.json'):
                        part_name = f"{part_name} {sub_part_name}"
                        self._process_mask_file(sub_mask_path,
                                                part_name,
                                                concept_name,
                                                image_cache)
                    else:
                        print(f"Skipping non-json file: {sub_mask_file}")
                        continue
            elif mask_file.endswith('.json'):
                self._process_mask_file(mask_path,
                                        part_name,
                                        concept_name,
                                        image_cache)

    def _process_mask_file(self,
                           mask_path: str,
                           part_name: str,
                           concept_name: str,
                           image_cache: Dict):

        image_path = self.mask_handler.get_image_path(mask_path)
        image_name = os.path.basename(image_path)

        feature_dir = self.feat_dir
        feature_path = os.path.join(feature_dir, f"{image_name}_features.npy")

        if len(feature_path) > 255:
            truncated_name = image_name[:len(image_name) // 10]
            new_filename = f"{truncated_name}_{part_name}_{concept_name}"
            image_name = new_filename
            feature_path = os.path.join(feature_dir, f"{image_name}_features.npy")

        if not os.path.exists(feature_dir):
            os.makedirs(feature_dir)

        mask_tensor = self.mask_handler.get_mask(mask_path)

        if image_name in image_cache:
            image_cache[image_name]["parts"][part_name] = mask_tensor
        else:
            if os.path.exists(feature_path):
                image_intermediate_feats = torch.from_numpy(np.load(feature_path))
            else:
                image_tensor = self.load_image(image_path).to(self.device)
                if image_tensor is None:
                    return
                image_tensor = image_tensor.unsqueeze(0)

                if image_tensor.shape != torch.Size([1, 3, 224, 224]):
                    print(f"Skipping image {image_name} with shape {image_tensor.shape}")
                    return

                image_intermediate_feats = self.dino_encoder.get_intermediate_features(image_tensor)
                image_intermediate_feats = image_intermediate_feats.view(self.dh, self.dw, -1)
                # shape: (dh, dw, FEAT_DIM) -> torch.Size([16, 16, 1024]) for large dino model
                try:
                    np.save(feature_path, image_intermediate_feats.cpu().numpy())
                except (IOError, OSError) as e:
                    print(f"Couldn't save the file for some reason: {feature_path} : {e}")
                    return

            image_cache[image_name] = {
                "features": image_intermediate_feats.squeeze(0),
                "parts": {part_name: mask_tensor},
                "concept": concept_name,
                "image_path": image_path
            }