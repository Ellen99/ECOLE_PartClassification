import argparse
import yaml
import sys
import os
from tqdm import tqdm
import copy
from openai import OpenAI


project_main_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# Add the project's main directory to sys.path
if project_main_dir not in sys.path:
    sys.path.append(project_main_dir)
os.chdir(project_main_dir)
print(f"Changed working directory to {project_main_dir}")

from secret import KEY
from src.utils.config import InferenceConfig
from src.utils.data_utils import load_concept_hierarchy
from src.utils.gpt_utils import filter_relevant_parts_through_api
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
    client = OpenAI(api_key=KEY)

    # Load configuration
    with open(args.config, 'r', encoding='utf-8') as f:
        config_dict = yaml.safe_load(f)
    config = InferenceConfig.from_dict(config_dict)

    config.only_save_visualization = True
    config.checkpoint_dir = "/shared/nas/data/m1/elenc2/PartClassification/checkpoints_vitl14_last"
    config.penalty = "l2"
    part_classifier = PartClassifier(config)
    visualizer = MaskVisualizer(config)

    image_loader = ImageLoader(config)
    images_data = image_loader.load_images()


    root_concepts = load_concept_hierarchy(f"{config.checkpoint_dir}/root_concept_hierarchy.pkl")
    concept_parts = load_concept_hierarchy(f"{config.checkpoint_dir}/concept_hierarchy.pkl")
    # format of concept_hierarchy: {concept_name: ["part1", "part2", ...]}

    all_concepts_lst = list(root_concepts.keys()) + list(concept_parts.keys())
    print("root concepts length: ", len(list(root_concepts.keys())))
    print("core concepts length: ", len(list(concept_parts.keys())))
    print("All concepts length: ", len(all_concepts_lst))

    if concept_parts is None or root_concepts is None:
        print("No concept hierarchy found")
        return

    # Load part classifiers for l1 penalty
    part_classifier.load_part_classifiers(all_concepts_lst)
    print(f"{len(part_classifier.classifers.keys())} Part classifiers loaded successfully!")


    if part_classifier.classifers is None:
        print("No classifiers found")
        return
    images_data_copy = copy.deepcopy(images_data)

    for img_name, image_data in images_data_copy.items():
        print("__________________________________________________________")
        print(f"Processing image: {img_name}")
        image = image_data["image"]

        concept_name = image_data["concept"]
        print("Concept name: ", concept_name)
        root_concept = concept_name.split('--')[0]
        core_concept = concept_name.split('--')[1]

        # FOR TESTING PURPOSES
        # if root_concept not in classifier.classifers.keys(): # and concept_name not in part_classifier.classifers.keys():
        #     print(f"Skipping: {root_concept} : {concept_name} as the concept classifier is not trained yet")
        #     continue
        # print(f"Root concept: {root_concept}, core concept: {core_concept}")

        part_classifier.threshold = 0.9
        # CASE WHEN THE CONCEPT IS NEW
        if concept_name not in concept_parts:
            # we should use root concept classifiers - we'll get part names through api call
            image_data["concept"] = root_concept
            predicted_masks = part_classifier.classify_parts(image_data, root_concepts[root_concept])
            non_empty_masks = {}
            for part_msk in predicted_masks.keys():
                if predicted_masks[part_msk].max() > 0:
                    non_empty_masks[part_msk] = predicted_masks[part_msk]

            parts_to_consider = non_empty_masks.keys()
            # check if there are non empty masks for the parts
            if len(parts_to_consider) == 0:
                print(f"No parts found for new concept - {root_concept} : {core_concept}")
                continue

            parts_to_consider = ', '.join([s.replace("'", "") for s in parts_to_consider])
            # API CALL
            try:
                result = filter_relevant_parts_through_api(parts_to_consider, core_concept, client)
            except Exception as e:
                print(f"Error in API call: {e}")
                continue
            # the result is comma separated list of parts make it a list
            if result is str:
                result_lst = [part.strip() for part in result.split(',')]
            else:
                result_lst = result

            filtered_masks = {part_name: part_mask for part_name, part_mask in non_empty_masks.items() if part_name in result_lst}
            # make sure there are less than 8 masks, if there are more, get rid of those that have the lowest sum in the mask array
            print("Filtered masks: ", filtered_masks.keys())

            if len(filtered_masks) == 0:
                print(f"No parts found for new concept - {root_concept} : {core_concept}")
                continue
            if len(filtered_masks) > 8:
                sorted_masks = sorted(filtered_masks.items(), key=lambda x: x[1].sum())
                filtered_masks = {part_name: part_mask for part_name, part_mask in sorted_masks[:8]}
            try:
                visualizer.visualize_part_masks(image, img_name, filtered_masks, concept_name)
            except Exception as e:
                print(f"Error in visualizing masks: {e}")
        else:
            # CASE WHEN THE CONCEPT IS NOT NEW
            print(f"Concept {root_concept} : {core_concept} is in the concept parts")
            predicted_masks = part_classifier.classify_parts(image_data, concept_parts[concept_name])
            visualizer.visualize_part_masks(image, img_name, predicted_masks, concept_name)
    print("Done")
if __name__ == "__main__":
    main()
