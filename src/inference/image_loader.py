from PIL import Image
import torch
import os
from torchvision import transforms
from rembg import remove
from src.utils.config import InferenceConfig
from models.dino_encoder import DinoEncoder

class ImageLoader:
    '''Class to load images, preprocess them, remove background, and extract features using a DINO encoder.'''
    def __init__(self, config: InferenceConfig):
        # self.config = config
        self.img_height = config.image_height
        self.img_width = config.image_width
        self.patch_size = config.patch_size
        self.feat_dim = config.feat_dim

        self.dh = config.dh
        self.dw = config.dw
        self.device = config.device

        self.image_dir = config.input_dir

        self.dino_encoder = DinoEncoder(self.device, config.dino_model)

        self.resize = lambda mask: torch.nn.functional.interpolate(mask.unsqueeze(0).unsqueeze(0),
                                       size=(self.dh, self.dw),
                                       mode='nearest').squeeze(0).squeeze(0)

        # self.preprocess = transforms.Compose([
        #         transforms.Resize((self.img_height,self.img_width)),
        #         transforms.ToTensor()])
        
        self.transform = transforms.Compose([
            transforms.Resize((config.image_height, config.image_width)),
            transforms.ToTensor()
        ])

    @torch.no_grad
    def load_images(self):
        '''
        Load images from a directory, preprocess them, 
        remove background, and extract features using a DINO encoder.

        Parameters:
            image_dir: str - directory containing images
            dino_encoder: torch.nn.Module - DINO encoder model
        Returns:
            image_data: dict - dictionary containing image features, concepts, with format
                {
                    "image_name": {
                        "features": torch.Tensor - image features,
                        "concept": str - concept label,
                        "image_path": str - image path
                    }
                }
        '''
        image_data = {}

        for image_name in os.listdir(self.image_dir):
            if image_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(self.image_dir, image_name)
                image = self.load_image(image_path)
                if image is None:
                    print(f"Skipping the image : {image_path}")
                    continue
                image_tensor = self.transform(image).to(self.device)

                bg_removed_image = remove(image)
                bg_removed_tensor = self.transform(bg_removed_image).to(self.device)

                mask = (bg_removed_tensor.sum(dim=0) > 0).float()
                mask_resized = self.resize(mask)

            # features = dino_encoder.get_intermediate_layers(image_tensor.unsqueeze(0))[0].view(dh, dw, -1).squeeze(0)
                features = self.dino_encoder.get_intermediate_features(image_tensor.unsqueeze(0))
                features = features.view(self.dh, self.dw, -1).squeeze(0)
                features = features * mask_resized.unsqueeze(-1)  # Set background features to zero

                concept = '-'.join(image_name.split('-')[:-1])
                image_data[image_name] = {
                    "features": features,
                    "concept": concept,
                    # "image_path": image_path,
                    "image": image
                }
        return image_data

    def load_image(self, image_path: str) -> Image:
        """Load image."""
        try:
            return Image.open(image_path).convert("RGB")
        except (FileNotFoundError, OSError, IOError) as e:
            print(f"Error loading image {image_path}: {e}")
            return None
