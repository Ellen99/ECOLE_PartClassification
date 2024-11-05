from config import IMAGE_SIZE, PATCH_SIZE
import torch
import numpy as np
import cv2

def classify_parts(image_data, concept_parts, part_classifiers, probability_threshold=0.9, return_binary_mask = True, upsample = True):
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
        part_classifiers (dict): Dictionary of part classifiers for the concept.
        probability_threshold (float): Minimum probability threshold for a part to be considered.
        return_binary_mask (bool): Whether to return binary masks or probability masks.
        upsample (bool): Whether to upsample the masks to the original image size.
    '''
    h = w = IMAGE_SIZE
    dh = h // PATCH_SIZE
    dw = w // PATCH_SIZE

    root_concept = image_data["concept"]
    image_features = image_data["features"]
    aggregated_part_features = {part: [] for part in concept_parts}
    part_masks = {part: np.zeros((dh, dw)) for part in concept_parts}

    for y in range(dh):
        for x in range(dw):
            feat_x_y = image_features[y,x]
            # skip the background features
            if torch.all(feat_x_y == 0):
                continue
            max_prob, best_part = -1, None
            for part, classifier in part_classifiers[root_concept].items():
                prob_val = classifier.predict_proba(feat_x_y.cpu().unsqueeze(0).numpy())[:, 1]
                if prob_val > max_prob:
                    max_prob = prob_val
                    best_part = part

                # Only add features with significant probability
                if max_prob >= probability_threshold:
                    aggregated_part_features[best_part].append((max_prob, (y, x)))

    for part, data in aggregated_part_features.items():
        if not data:
            continue
        for prob, coord in data:
            if return_binary_mask:
                part_masks[part][coord] = 1
            else:
                part_masks[part][coord] = prob

    if upsample:
        for part in part_masks:
            part_masks[part] = cv2.resize(part_masks[part], (w, h), interpolation=cv2.INTER_NEAREST)
    return part_masks
