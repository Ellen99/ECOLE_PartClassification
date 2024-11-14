import argparse
import yaml
import sys
import os

project_main_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# Add the project's main directory to sys.path
if project_main_dir not in sys.path:
    sys.path.append(project_main_dir)
os.chdir(project_main_dir)
print(f"Changed working directory to {project_main_dir}")

from src.utils.config import InferenceConfig
from src.utils.data_utils import load_concept_hierarchy
from src.inference.image_loader import ImageLoader
from src.inference.classifier import PartClassifier
from src.inference.visualization import MaskVisualizer

def parse_args():
    ''' Parse command line arguments '''
    parser = argparse.ArgumentParser(description='Predict parts in an image')
    parser.add_argument('--config', type=str, default = 'configs/default_pred_config.yaml',
                      help='Path to configuration file')
    args, _ = parser.parse_known_args()
    return args
    # return parser.parse_args()

def main():
    args = parse_args()

    # Load configuration
    with open(args.config, 'r', encoding='utf-8') as f:
        config_dict = yaml.safe_load(f)
    config = InferenceConfig.from_dict(config_dict)
    # config = InferenceConfig()

    part_classifier = PartClassifier(config)
    visualizer = MaskVisualizer(config)

    image_loader = ImageLoader(config)
    images_data = image_loader.load_images()

    concept_hierarchy = load_concept_hierarchy(f"{config.checkpoint_dir}/concept_hierarchy.pkl")
    # format of concept_hierarchy: {concept_name: ["part1", "part2", ...]}

    if concept_hierarchy is None:
        print("No concept hierarchy found")
        return
    part_classifier.load_part_classifiers(list(concept_hierarchy.keys()))

    for img_name, image_data in images_data.items():
        concept_name = image_data["concept"]
        image = image_data["image"]
        concept_parts = concept_hierarchy[concept_name]
        predicted_masks = part_classifier.classify_parts(image_data, concept_parts)

        visualizer.visualize_part_masks(image, img_name, predicted_masks)

if __name__ == "__main__":
    main()
