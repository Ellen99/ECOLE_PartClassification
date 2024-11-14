import torch

class DinoEncoder:
    """Wrapper class for the DINO model."""
    def __init__(self, device_type: str, model_name: str):
        self.device = device_type
        self.model = self._load_model(model_name)
        self.model.eval()  #evaluation mode (no training)

    def _load_model(self, model_name: str):
        '''Load the DINO model via torch hub'''
        model = torch.hub.load("facebookresearch/dinov2", model_name).to(self.device)
        return model

    def get_intermediate_features(self, image_tensor):
        '''Get intermediate features from the DINO model'''
        with torch.no_grad():
            image_tensor = image_tensor.to(self.device)
            intermediate_feats = self.model.get_intermediate_layers(image_tensor)[0]
            return intermediate_feats
