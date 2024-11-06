import numpy as np
from scipy.ndimage import label
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import torch
import os
from config import OUTPUT_PATH


def visualize_ellipses(image: Image, mask_connected_components, title="Image with Mask Overlay", only_save = False, save_path=OUTPUT_PATH):
    """
    Display multiple mask overlays on the original image in a single row.
    
    Parameters:
        image_path (str): Path to the original image.
        masks (dict): Dictionary of masks where keys are part names and values are 2D mask arrays.
        title (str): Title for the figure.
    """
    try:
        # Set up the figure with a row layout based on the number of masks
        num_masks = len(mask_connected_components)
        fig, axes = plt.subplots(1, num_masks, figsize=(5 * num_masks, 5))
        fig.suptitle(title, fontsize=16)
        
        # Ensure `axes` is iterable in case there's only one mask
        if num_masks == 1:
            axes = [axes]

        # Plot each mask overlayed on the original image
        for ax, part_name in zip(axes, mask_connected_components.keys()):
            ax.imshow(image)
            for mask_component in mask_connected_components[part_name]:
                # for component_i in mask_component:
                #     print(component_i)
                ellipse = generate_ellipse(mask_component, facecolor='red', opacity=0.4, edgecolor='lime', n_std=3.5)
                ax.add_patch(ellipse)
            # ax.imshow(mask_resized, alpha=0.5, cmap='jet')
            ax.set_title(part_name)
            ax.axis("off")
            ax.set_aspect('equal')

        if not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        # print(f"Image saved to {save_path}")

        if not only_save:
            plt.show()
        else:
            plt.close()
    except Exception as e:
        print(f"An error occurred during visualization: {e}")


def separate_connected_components(binary_mask):
    labeled_mask, num_features = label(binary_mask)
    components = []

    for i in range(1, num_features + 1):
        component = (labeled_mask == i).astype(np.uint8)
        components.append(component)

    return components

def generate_ellipse(data_i, n_std: float = 3, edgecolor='none', facecolor='red', opacity=0.3):

    # Perform PCA to get principal components for each dataset
    pca = PCA(n_components=2)
    data_i_swapped = data_i[:, ::-1]
    # pca.fit(data_i)
    pca.fit(data_i_swapped)

    # Get the principal components, eigenvalues, and mean
    principal_components = pca.components_
    eigenvalues = pca.explained_variance_
    mean_data = np.mean(data_i_swapped, axis=0)

    # Calculate the angle of rotation for the ellipse
    angle = np.degrees(np.arctan2(principal_components[0, 1], principal_components[0, 0]))

    # Define the width and height of the ellipse based on eigenvalues
    # width, height = 3 * n_std * np.sqrt(eigenvalues)
    width, height = n_std * np.sqrt(eigenvalues)

    ellipse = patches.Ellipse(
        mean_data# (mean_data[1], mean_data[0])
        , width, height, angle=angle, edgecolor=edgecolor, facecolor=facecolor #colors[i]
       , alpha=opacity,
        label=f'{n_std} Std Ellipse)'
    )
    return ellipse

def get_part_components(predicted_masks_l2, image_size):
    masks = {}
    for part in predicted_masks_l2:
        mask = predicted_masks_l2[part]

        mask_tensor = torch.from_numpy(mask).to(torch.float32)
        mask_resized = torch.nn.functional.interpolate(
            mask_tensor.unsqueeze(0).unsqueeze(0),
            size=(image_size[1], image_size[0]),  # Resize to original image size
            mode='nearest'
        ).squeeze().cpu().numpy()

        mask_components_ids = []
        components = separate_connected_components(mask_resized)

        for i, component in enumerate(components):
            # indices = np.where(mask == 1)
            # print("component", len(coordinates))
            coordinates = np.column_stack(np.where(component == 1))
            mask_components_ids.append(coordinates)

        masks[part] = mask_components_ids
    return masks