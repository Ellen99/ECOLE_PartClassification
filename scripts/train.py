import argparse
import yaml
import sys
import os

project_main_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# Add the project's main directory to sys.path
if project_main_dir not in sys.path:
    sys.path.append(project_main_dir)
os.chdir(project_main_dir)

# Description: This script is used to train the part classifiers for each concept.
from src.utils.data_utils import save_concept_hierarchy , load_concept_hierarchy
from src.utils.config import TrainingConfig
from src.training.data_loader import DataLoader
from src.training.trainer import train_classifiers

def parse_args():
    parser = argparse.ArgumentParser(description='Train part classifiers')
    parser.add_argument('--config', type=str, default='configs/default_training_config.yaml',
                      help='Path to configuration file')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Load configuration
    with open(args.config, 'r', encoding='utf-8') as f:
        config_dict = yaml.safe_load(f)
    config = TrainingConfig.from_dict(config_dict)

    print (f"Training configuration: {config}")
    print("Initializing training pipeline...")

    print("Loading image data...")
    data_loader = DataLoader(config)
    image_cache, concept_parts = data_loader.get_image_data_from_path()

    # Save concept hierarchy
    save_concept_hierarchy(concept_parts, f"{config.checkpoint_dir}/concept_hierarchy.pkl")

    print(f"Loaded data: for {len(image_cache.keys())} images")
    print(f"Concept parts: for {len(concept_parts.keys())} concepts")


    print("Starting classifier training...")
    classifiers = train_classifiers(image_cache, concept_parts, config)
    print(f"Training completed : trained classifiers for {len(classifiers.keys())} concepts")

    # Load concept hierarchy
    loaded_concept_parts = load_concept_hierarchy(f"{config.checkpoint_dir}/concept_hierarchy.pkl")
    print(f"Loaded Concept parts: for {len(loaded_concept_parts.keys())} concepts")

if __name__ == '__main__':
    main()
