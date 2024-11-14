from dataclasses import dataclass
import torch

@dataclass
class BaseConfig:
    ''' Base configuration class. '''

    dino_model: str = 'dinov2_vitl14'
    penalty: str = 'l1'
    patch_size: int = 14
    feat_dim: int = 1024 # patch size for large dino model
    image_height: int = 224
    image_width: int = 224

    max_workers: int = 6
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    verbose: bool = False

    # TODO move checkpoints to Hugging Face hub
    checkpoint_dir: str = 'checkpoints_vitl14/'

    feat_dir: str = '/shared/nas/data/m1/elenc2/PartClassifiers_training/dino_features-dinov2_vitl14/'

    dh: int = 0
    dw: int = 0

    def __post_init__(self):
        """ Calculate dh and dw based on image size and patch size."""
        self.dh = self.image_height // self.patch_size
        self.dw = self.image_width // self.patch_size

    @classmethod
    def from_dict(cls, config_dict: dict) -> 'BaseConfig':
        """Create config from dictionary."""
        config = cls(**config_dict)
        config.__post_init__()
        return config


@dataclass
class TrainingConfig(BaseConfig):
    ''' Configuration for training parameters and settings. '''

    iterations: int = 2000
    solver: str = 'saga'
    mask_dir: str = '/shared/nas/data/m1/elenc2/PartClassifiers_training/annotations/masks/'

    @classmethod
    def from_dict(cls, config_dict: dict) -> 'TrainingConfig':
        """Create config from dictionary."""
        config = cls(**config_dict)
        super(TrainingConfig, config).__post_init__()

        return config

@dataclass
class InferenceConfig(BaseConfig):
    ''' Configuration for inference parameters and settings. '''

    input_dir: str = 'data/test_images/'
    output_dir: str = 'output/'

    threshold: float = 0.95

    return_binary_mask: bool = True
    return_upsampled_mask: bool = True
    only_save_visualization: bool = False
    visualization_option: str = 'ellipse' # 'ellipse' or 'patch'

    @classmethod
    def from_dict(cls, config_dict: dict) -> 'InferenceConfig':
        """Create config from dictionary."""
        config = cls(**config_dict)
        super(InferenceConfig, config).__post_init__()
        return config