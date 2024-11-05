import torch
from config import DEFAULT_DINO_MODEL
from data_processing import load_and_preprocess_images, load_concept_hierarchy, load_part_classifiers
from classification import classify_parts
from visualization import visualize_masks_on_image

def build_dino(model_name: str = DEFAULT_DINO_MODEL):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.hub.load("facebookresearch/dinov2", model_name).to(device)

def main():
    
    dino_encoder = build_dino()
    dino_encoder.eval()
    test_image_dir = "/shared/nas/data/m1/elenc2/PartClassification/test_images"

    images_data = load_and_preprocess_images(test_image_dir, dino_encoder)
    # format of images_data: {   "image_name": {
                            #         "features": torch.Tensor - image features,
                            #         "concept": str - concept label,
                            #         "image_path": str - image path
                            #     }

    concept_hierarchy = load_concept_hierarchy()
    # format of concept_hierarchy: {root_concept: ["part1", "part2", ...]}

    if concept_hierarchy is not None:
        all_root_concepts = list(concept_hierarchy.keys())

    # defalut penalty is l1
    part_classifiers_l1 = load_part_classifiers(all_root_concepts)
    print(f"Loaded {len(part_classifiers_l1)} part classifiers with l1 penalty")

    part_classifiers_l2 = load_part_classifiers(all_root_concepts, penalty='l2')    
    print(f"Loaded {len(part_classifiers_l2)} part classifiers with l2 penalty")
    
    thresh = 0.95
    for img_name, image_data in images_data.items():
        root_concept = image_data["concept"]
        image_path = image_data["image_path"]
        concept_parts = concept_hierarchy[root_concept]

        predicted_masks_l2 = classify_parts(image_data, concept_parts, part_classifiers_l2, probability_threshold=thresh, upsample=True)
        path_to_save = f"output/masks_{root_concept}-{img_name}.png"
        visualize_masks_on_image(image_path, predicted_masks_l2, f"generated masks for {root_concept}", only_save=False, save_path=path_to_save)

if __name__ == "__main__":
    main()
