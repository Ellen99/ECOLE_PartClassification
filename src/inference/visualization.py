import os
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from PIL import Image
from sklearn.decomposition import PCA
from scipy.ndimage import label
from src.utils.config import InferenceConfig

class MaskVisualizer:
    ''' Class to visualize masks on an image. '''
    def __init__(self, config: InferenceConfig):
        self.output_dir = config.output_dir
        self.only_save = config.only_save_visualization

        self.visualization_option = config.visualization_option # ellipse or patch

        self.resize = lambda mask, h, w : torch.nn.functional.interpolate(
            mask.unsqueeze(0).unsqueeze(0),
            size=(h, w),
            mode='nearest').squeeze().cpu().numpy()

    def visualize_part_masks(self,
                             image: Image,
                             image_name: str,
                             masks: dict,
                             concept_name: str):
        '''Visualize the predicted masks on the image.'''

        if self.visualization_option == 'ellipse':
            self.visualize_ellipses(image, image_name, masks, "Image with Ellipse Overlay: " + concept_name)

        elif self.visualization_option == 'patch':
            self.visualize_patches(image, image_name, masks, "Image with Patch Overlay: " + concept_name)

    def visualize_patches(self,
                          image: Image,
                          image_name: str,
                          masks: dict,
                          title: str):
        """
        Display patch-mask overlays for all parts on the image in a single row.

        Parameters:
            image (str): Image to visualize.
            image_name (str): Name of the image.
            masks (dict): Dictionary of masks where keys are part names and values are 2D mask arrays.
            title (str): Title for the figure.
        """
        try:
            # Set up the figure with a row layout based on the number of masks
            num_masks = len(masks)
            fig, axes = plt.subplots(1, num_masks, figsize=(5 * num_masks, 5))
            fig.suptitle(title, fontsize=16)
            
            # Ensure `axes` is iterable in case there's only one mask
            if num_masks == 1:
                axes = [axes]

            # Plot each mask overlayed on the original image
            for ax, (part_name, mask) in zip(axes, masks.items()):
                # Convert and resize the mask to match the original image size
                mask_tensor = torch.from_numpy(mask).to(torch.float32)
                mask_resized = self.resize(mask_tensor, image.size[1], image.size[0])
                
                # Display the original image with the mask overlay
                ax.imshow(image)
                ax.imshow(mask_resized, alpha=0.5, cmap='jet')
                ax.set_title(part_name)
                ax.axis("off")

            save_path = f"{self.output_dir}/patch_masks_{image_name}"
            self.save_plot(save_path)
        except (OSError, ValueError, RuntimeError) as e:
            print(f"An error occurred during visualization: {e}")

    def visualize_ellipses(self,
                           image: Image,
                           image_name: str,
                           masks: dict,
                           title: str):
        """
        Display multiple mask overlays on the original image in a single row.
        
        Parameters:
            image (str): Image to visualize.
            image_name (str): Name of the image.
            masks (dict): Dictionary of masks where keys are part names and values are 2D mask arrays.
            title (str): Title for the figure.
        """
        try:
            # Set up the figure with a row layout based on the number of masks
            part_component_masks = self.get_part_components(masks, image.size)

            num_masks = len(part_component_masks)
            fig, axes = plt.subplots(1, num_masks, figsize=(5 * num_masks, 5))
            fig.suptitle(title, fontsize=16)
            
            # Ensure `axes` is iterable in case there's only one mask
            if num_masks == 1:
                axes = [axes]

            # Plot each part's mask overlay on image
            for ax, part_name in zip(axes, part_component_masks.keys()):
                ax.imshow(image)
                for mask_component in part_component_masks[part_name]:
                    ellipse = self.generate_ellipse(mask_component,
                                                    facecolor='red',
                                                    opacity=0.4,
                                                    edgecolor='lime',
                                                    n_std=3.5)
                    ax.add_patch(ellipse)
                ax.set_title(part_name)
                ax.axis("off")
                ax.set_aspect('equal')
            save_path = f"{self.output_dir}/masks_{image_name}"

            self.save_plot(save_path)
        except (OSError, ValueError, RuntimeError) as e:
            print(f"An error occurred during visualization: {e}")

    def save_plot(self, save_path: str):
        '''Save the plot to a file.'''
        try:
            if not os.path.exists(os.path.dirname(save_path)):
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)

            if not self.only_save:
                plt.show()
            else:
                plt.close()
        except (OSError, ValueError, RuntimeError) as e:
            print(f"An error occurred during saving the image: {e}")

    def generate_ellipse(self,
                         data_i: np.ndarray,
                         n_std: float = 3,
                         edgecolor='none',
                         facecolor='red',
                         opacity=0.3):
        '''Generate an ellipse based on the data.'''
        # Perform PCA to get principal components for each dataset
        pca = PCA(n_components=2)
        data_i_swapped = data_i[:, ::-1]  # Swap x and y coordinates
        pca.fit(data_i_swapped)

        # Get the principal components, eigenvalues, and mean
        principal_components = pca.components_
        eigenvalues = pca.explained_variance_
        mean_data = np.mean(data_i_swapped, axis=0)

        angle = np.degrees(np.arctan2(principal_components[0, 1], principal_components[0, 0]))
        width, height = n_std * np.sqrt(eigenvalues)

        return patches.Ellipse(mean_data,
                                  width, height,
                                  angle=angle,
                                  edgecolor=edgecolor,
                                  facecolor=facecolor,
                                  alpha=opacity,
                                  label=f'{n_std} Std Ellipse)')

    def get_part_components(self,
                            input_masks,
                            image_size):
        '''Get the connected components for each part mask.'''
        masks = {}
        for part in input_masks:
            mask = input_masks[part]

            mask_tensor = torch.from_numpy(mask).to(torch.float32)
            mask_resized = self.resize(mask_tensor, image_size[1], image_size[0])

            mask_components_ids = []
            components = self.separate_connected_components(mask_resized)

            for component in components:
                coordinates = np.column_stack(np.where(component == 1))
                mask_components_ids.append(coordinates)

            masks[part] = mask_components_ids
        return masks
    
    def separate_connected_components(self, binary_mask):
        '''Separate connected components in a binary mask.'''
        labeled_mask, num_features = label(binary_mask)
        components = []

        for i in range(1, num_features + 1):
            component = (labeled_mask == i).astype(np.uint8)
            components.append(component)

        return components