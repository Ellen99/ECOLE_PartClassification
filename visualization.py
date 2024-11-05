import os
import torch
import matplotlib.pyplot as plt
from PIL import Image
from config import OUTPUT_PATH

def visualize_masks_on_image(image_path, masks, title="Image with Mask Overlay", only_save = False, save_path=OUTPUT_PATH):
    """
    Display multiple mask overlays on the original image in a single row.
    
    Parameters:
        image_path (str): Path to the original image.
        masks (dict): Dictionary of masks where keys are part names and values are 2D mask arrays.
        title (str): Title for the figure.
    """
    try:
        # Load the original image
        original_image = Image.open(image_path).convert("RGB")

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
            mask_resized = torch.nn.functional.interpolate(
                mask_tensor.unsqueeze(0).unsqueeze(0),
                size=(original_image.size[1], original_image.size[0]),  # Resize to original image size
                mode='nearest'
            ).squeeze().cpu().numpy()
            
            # Display the original image with the mask overlay
            ax.imshow(original_image)
            ax.imshow(mask_resized, alpha=0.5, cmap='jet')
            ax.set_title(part_name)
            ax.axis("off")

        if not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"Image saved to {save_path}")

        if not only_save:
            plt.show()
        else:
            plt.close()
    except Exception as e:
        print(f"An error occurred during visualization: {e}")