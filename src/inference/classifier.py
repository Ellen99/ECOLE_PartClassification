import os
import joblib    
import torch
import numpy as np
import cv2
from tqdm import tqdm
from src.utils.config import InferenceConfig

class PartClassifier:
    ''' Class to classify parts in an image based on the given part classifiers.'''
    def __init__(self, config: InferenceConfig):
        self.penalty = config.penalty
        self.checkpoint_dir = config.checkpoint_dir

        self.img_height = config.image_height
        self.img_width = config.image_width

        self.patch_size = config.patch_size
        self.dh = config.dh
        self.dw = config.dw
        self.threshold = config.threshold
        self.return_binary_mask = config.return_binary_mask
        self.return_upsampled_mask = config.return_upsampled_mask
        self.classifers = {}

        # self.config = config
        # self.feature_extractor = FeatureExtractor(config)
        # os.makedirs(config.checkpoint_dir, exist_ok=True)

    # def classify_parts(self, image_data, concept_parts, part_classifiers, probability_threshold=0.5, upsample=True):

    def classify_parts(self, image_data, concept_parts, thresh=None):
        '''
        Classify image patches based on the given part classifiers.
        
        Parameters:
            image_data (dict): Dictionary containing image features, concepts, with format
                    "image_name": {
                        "features": torch.Tensor - image features,
                        "concept": str - concept label,
                        "image_path": str - image path - optional, not used in context of this function
                    }
            concept_parts (list): List of part names for the concept.
        '''
        if thresh:
            self.threshold = thresh
        concept_name = image_data["concept"]
        image_features = image_data["features"]
        aggregated_part_features = {part: [] for part in concept_parts}
        part_masks = {part: np.zeros((self.dh, self.dw)) for part in concept_parts}

        for y in range(self.dh):
            for x in range(self.dw):
                feat_x_y = image_features[y,x]
                # skip the background features
                if torch.all(feat_x_y == 0):
                    continue
                max_prob, best_part = -1, None
                try:
                    for part, classifier in self.classifers[concept_name].items():
                        prob_val = classifier.predict_proba(feat_x_y.cpu().unsqueeze(0).numpy())[:, 1]
                        if prob_val > max_prob:
                            max_prob = prob_val
                            best_part = part

                        # Only add features with significant probability
                        if max_prob >= self.threshold:
                            aggregated_part_features[best_part].append((max_prob, (y, x)))
                except KeyError as e:
                    print(f"Key error classifying parts for concept {concept_name}: {e}")
                except ValueError as e:
                    print(f"Value error classifying parts for concept {concept_name}: {e}")
                except RuntimeError as e:
                    print(f"Runtime error classifying parts for concept {concept_name}: {e}")
        for part, data in aggregated_part_features.items():
            if not data:
                continue
            for prob, coord in data:
                if self.return_binary_mask:
                    part_masks[part][coord] = 1
                else:
                    part_masks[part][coord] = prob

        if self.return_upsampled_mask:
            for part in part_masks:
                part_masks[part] = cv2.resize(part_masks[part], (self.img_width, self.img_height), interpolation=cv2.INTER_NEAREST)
        return part_masks

    def load_part_classifiers(self, concepts):
        '''
            Load part classifiers for the given concepts
            Parameters:
                concepts (list): A list of root concepts for which the part classifiers need to be loaded.
                penalty (str): The penalty used for training the classifiers. Default is 'l1'.
                base_dir (str): The directory where the part classifiers are
            Returns:
                dict: A dictionary where the keys are root concepts and the values are dictionaries of part classifiers.
                {
                    "concept_name": {
                        "part_name": classifier
                    }
                }
        '''
        try:
            for concept_name in tqdm(concepts, desc="Loading part classifiers", position=0, leave=True):
                checkpoint_path = f'{self.checkpoint_dir}/part_classifiers_{self.penalty}/{concept_name}_classifiers.pkl'
                
                # checkpoint_path = f'{base_dir}/part_classifiers_{penalty}/{concept_name}_classifiers.pkl'
                if os.path.exists(checkpoint_path):
                    part_classifier = joblib.load(checkpoint_path)
                    self.classifers[concept_name] = part_classifier
                    # print("loading from checkpoint")
                else:
                    print(f"Checkpoint not found for {concept_name} concept at {checkpoint_path}")
            return self.classifers
        except FileNotFoundError as e:
            print(f"File not found error loading part classifiers for {concept_name}: {e}")
            return None
        except Exception as e:
            print(f"Unexpected error loading part classifiers for {concept_name}: {e}")
            return None