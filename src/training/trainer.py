import os
import warnings
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import numpy as np
import torch
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import ConvergenceWarning
from src.utils.config import TrainingConfig
from src.utils.feature_utils import get_relevant_features

class PartClassifierTrainer:
    ''' Trainer for part classifiers'''
    def __init__(self, config: TrainingConfig):
        self.config = config
        # self.feature_extractor = FeatureExtractor(config)
        os.makedirs(config.checkpoint_dir, exist_ok=True)

    def _collect_features_for_part(
        self,
        part: str,
        image_data: Dict,
        features: torch.Tensor,
        part_mask: torch.Tensor
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """Collect positive and negative features for a part."""
        positive_features, negative_features = [], []

        relevant_feats, _, mask_ranges = get_relevant_features(features,
                                                               part_mask,
                                                               self.config.patch_size,
                                                               self.config.device)
        positive_features.append(relevant_feats)

        # Collect negative examples from other parts
        for other_part, other_mask in image_data["parts"].items():
            if other_part != part:
                other_feats, _, other_ranges = get_relevant_features(features,
                                                                     other_mask,
                                                                     self.config.patch_size,
                                                                     self.config.device)
                # Filter overlapping features
                filtered_feats = [
                    other_feats[idx]
                    for idx, other_range in enumerate(other_ranges)
                    if not any(other_range == relevant_range for relevant_range in mask_ranges)
                ]

                if filtered_feats:
                    negative_features.append(torch.stack(filtered_feats))

        return positive_features, negative_features

    def _train_single_part(self,
                           part: str,
                           images_data: Dict) -> Optional[Tuple[str, LogisticRegression]]:
        """Train classifier for a single part."""
        try:
            positive_features, negative_features = [], []

            for image_data in images_data.values():
                if part not in image_data["parts"]:
                    # Skip images without the part annotation
                    continue

                pos_feats, neg_feats = self._collect_features_for_part(part,
                                                                       image_data,
                                                                       image_data["features"],
                                                                       image_data["parts"][part])
                positive_features.extend(pos_feats)
                negative_features.extend(neg_feats)

            if not positive_features or not negative_features:
                return None

            # Prepare training data
            X = np.vstack([
                torch.vstack(positive_features).cpu().numpy(),
                torch.vstack(negative_features).cpu().numpy()
            ])
            y = np.hstack([
                np.ones(len(torch.vstack(positive_features))),
                np.zeros(len(torch.vstack(negative_features)))
            ])

            # Training
            # Standardize features
            scaler = StandardScaler()
            X = scaler.fit_transform(X)

            # Suppress convergence warnings during training
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', category=ConvergenceWarning)
                classifier = LogisticRegression(
                    penalty=self.config.penalty,
                    solver=self.config.solver,
                    max_iter=self.config.iterations,
                    random_state=42,
                    n_jobs=-1  # Use all available CPU cores
                )
                classifier.fit(X, y)

            return part, classifier

        except (ValueError, TypeError, RuntimeError) as e:
            print(f"Error training classifier for part {part}: {e}")
            return None

    def train_part_classifiers(
        self,
        images_data: Dict,
        concept_parts: Dict[str, List[str]]
    ) -> Dict[str, Dict[str, LogisticRegression]]:
        """Train classifiers for all parts of all concepts."""
        concept_classifiers = {}
        # pbar = tqdm(concept_parts.items(), desc="Training concepts", position=0, leave=True)
        for concept_name, parts in tqdm(concept_parts.items(), desc="Training part classifiers for concepts", position=0, leave=True):
            checkpoint_path = os.path.join(
                self.config.checkpoint_dir,
                f'part_classifiers_{self.config.penalty}/{concept_name}_classifiers.pkl'
            )

            # Load existing classifiers if available
            if os.path.exists(checkpoint_path):
                concept_classifiers[concept_name] = joblib.load(checkpoint_path)
                if self.config.verbose:
                    print(f"Loaded checkpoint for {concept_name}")
                continue

            # Train classifiers in parallel
            with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
                future_to_part = {
                    executor.submit(self._train_single_part,
                                    part,
                                    images_data
                                    # ,concept_name
                                    ): part for part in parts
                }

                part_classifiers = {}
                # tqdm(future_to_part, desc=f"Training parts for {concept_name}"):
                for future in future_to_part:
                    result = future.result()
                    if result:
                        part, classifier = result
                        part_classifiers[part] = classifier

            concept_classifiers[concept_name] = part_classifiers

            # Save checkpoint
            joblib.dump(part_classifiers, checkpoint_path)
            if self.config.verbose:
                print(f"Saved classifiers for {concept_name}")

        return concept_classifiers

def train_classifiers(images_data: Dict,
                      concept_parts: Dict[str, List[str]],
                      config: Optional[TrainingConfig] = None
                      )-> Dict[str, Dict[str, LogisticRegression]]:
    """Main training function."""
    if config is None:
        config = TrainingConfig()

    trainer = PartClassifierTrainer(config)
    return trainer.train_part_classifiers(images_data, concept_parts)
