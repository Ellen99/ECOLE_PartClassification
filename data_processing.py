import torch
import os
import numpy as np
import joblib
from PIL import Image
from torchvision import transforms
import pickle
from rembg import remove
from config import IMAGE_SIZE, PATCH_SIZE, CHECKPOINT_DIR

@torch.no_grad
def load_and_preprocess_images(image_dir, dino_encoder):
    '''
    Load images from a directory, preprocess them, remove background, and extract features using a DINO encoder.
    Parameters: 
        image_dir: str - directory containing images
        dino_encoder: torch.nn.Module - DINO encoder model
    Returns:
        image_data: dict - dictionary containing image features, concepts, with format
            {
                "image_name": {
                    "features": torch.Tensor - image features,
                    "concept": str - concept label,
                    "image_path": str - image path
                }
            }
    '''
    image_data = {}
    preprocess = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor()
    ])

    device = next(dino_encoder.parameters()).device  # Get the device of the model

    h = w = IMAGE_SIZE
    dh = h // PATCH_SIZE
    dw = w // PATCH_SIZE
    for image_name in os.listdir(image_dir):
        if image_name.endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(image_dir, image_name)
            image = Image.open(image_path).convert("RGB")
            image_tensor = preprocess(image).to(device)
            
            bg_removed_image = remove(image)
            bg_removed_tensor = preprocess(bg_removed_image).to(device)
            mask = (bg_removed_tensor.sum(dim=0) > 0).float()
            mask_resized = torch.nn.functional.interpolate(mask.unsqueeze(0).unsqueeze(0), size=(dh, dw), mode='nearest').squeeze(0).squeeze(0)

            features = dino_encoder.get_intermediate_layers(image_tensor.unsqueeze(0))[0].view(dh, dw, -1).squeeze(0)
            features = features * mask_resized.unsqueeze(-1)  # Set background features to zero
            
            concept = '-'.join(image_name.split('-')[:-1])
            image_data[image_name] = {
                "features": features,
                "concept": concept,
                "image_path": image_path
            }
    return image_data

def load_concept_hierarchy(base_dir = CHECKPOINT_DIR):
    '''
        Load the concept hierarchy from a file
        Parameters:
            base_dir (str): The directory where the concept hierarchy file(concept_hierarchy.pkl) is stored.

        Returns:
            dict: The concept hierarchy dictionary
    '''
    try:
        filename = 'concept_hierarchy.pkl'
        filepath = os.path.join(base_dir, filename)
        with open(filepath, 'rb') as f:
            concept_hierarchy = pickle.load(f)
    except Exception as e:
        print(f"Error loading concept hierarchy: {e}")
        concept_hierarchy = None
    return concept_hierarchy


def load_part_classifiers(concepts, penalty = 'l1', base_dir = CHECKPOINT_DIR):
    '''
        Load part classifiers for the given concepts
        Parameters:
            concepts (list): A list of root concepts for which the part classifiers need to be loaded.
            penalty (str): The penalty used for training the classifiers. Default is 'l1'.
            base_dir (str): The directory where the part classifiers are
        Returns:
            dict: A dictionary where the keys are root concepts and the values are dictionaries of part classifiers.
            {
                "root_concept": {
                    "part_name": classifier
                }
            }
    '''
    try:
        concept_classifiers = {}
        for root_concept in concepts:
            checkpoint_path = f'{base_dir}/part_classifiers_{penalty}/{root_concept}_classifiers.pkl'
            if os.path.exists(checkpoint_path):
                part_classifier = joblib.load(checkpoint_path)
                concept_classifiers[root_concept] = part_classifier
                # print("loading from checkpoint")
            else:
                print(f"Checkpoint not found for {root_concept} concept at {checkpoint_path}")
    except Exception as e:
        print(f"Error loading part classifiers for {root_concept}: {e}")

    return concept_classifiers

# Example usage
# concept1 = 'airplanes--agricultural'
# concept2 = 'airplanes--attack'

# loaded_classifiers = load_part_classifiers([concept1, concept2])
# print(loaded_classifiers)