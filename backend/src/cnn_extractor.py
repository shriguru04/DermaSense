# backend/src/cnn_extractor.py
"""
CNN Feature Extractor — ResNet18 (pretrained on ImageNet)
---------------------------------------------------------
Takes a skin lesion image → outputs 512-dimensional feature vector.
These 512 features are merged with the 34 manual clinical/histopath features
to create a 546-dim input vector for XGBoost.

ResNet18 is chosen over ViT because:
- Works well with small datasets (366 samples + augmentation)
- 512-dim output (not too many CNN features overwhelming 34 manual features)
- Runs on CPU without GPU requirement
- Transfer learning from ImageNet captures texture patterns relevant to skin

Architecture:
  Image (224×224 RGB)
    → ResNet18 backbone (all layers frozen except last block)
    → Global Average Pooling
    → 512-dim feature vector
    → Merged with 34 manual features
    → 546-dim total → XGBoost
"""

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import io
import os

# ── IMAGE PREPROCESSING PIPELINE ────────────────────────────────
# ImageNet mean/std normalization — required for pretrained ResNet18
IMAGE_TRANSFORMS = transforms.Compose([
    transforms.Resize((224, 224)),          # ResNet18 input size
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],         # ImageNet mean
        std=[0.229, 0.224, 0.225]           # ImageNet std
    )
])

# Augmentation transforms for training time (generates more samples from few images)
AUGMENT_TRANSFORMS = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05),
    transforms.RandomRotation(degrees=15),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


class CNNFeatureExtractor:
    """
    Extracts 512-dim feature vectors from skin images using pretrained ResNet18.
    The last fully-connected layer is removed — we use the penultimate layer output.
    """

    def __init__(self, device=None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model  = self._build_model()
        self.model.eval()
        print(f"CNN Extractor ready | Device: {self.device} | Output: 512 features/image")

    def _build_model(self):
        """
        ResNet18 with final FC layer removed.
        Outputs: (batch, 512) feature tensor after global average pooling.
        """
        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

        # Freeze all layers — we use ResNet purely as a feature extractor
        # (not fine-tuning because we may have few labeled images)
        for param in resnet.parameters():
            param.requires_grad = False

        # Remove the final classification layer (1000-class ImageNet head)
        # Keep everything up to and including avgpool → outputs 512-dim
        model = nn.Sequential(*list(resnet.children())[:-1])
        return model.to(self.device)

    def extract(self, image_input):
        """
        Extract 512-dim feature vector from one image.

        Args:
            image_input: PIL.Image, file path (str), or raw bytes

        Returns:
            np.ndarray of shape (512,)
        """
        # ── Load image ──────────────────────────────────────────
        if isinstance(image_input, bytes):
            img = Image.open(io.BytesIO(image_input)).convert('RGB')
        elif isinstance(image_input, str):
            img = Image.open(image_input).convert('RGB')
        elif isinstance(image_input, Image.Image):
            img = image_input.convert('RGB')
        else:
            raise ValueError(f"Unsupported image type: {type(image_input)}")

        # ── Preprocess ──────────────────────────────────────────
        tensor = IMAGE_TRANSFORMS(img)              # (3, 224, 224)
        tensor = tensor.unsqueeze(0).to(self.device) # (1, 3, 224, 224)

        # ── Extract features ─────────────────────────────────────
        with torch.no_grad():
            features = self.model(tensor)           # (1, 512, 1, 1)
            features = features.squeeze()           # (512,)

        return features.cpu().numpy().astype(np.float32)

    def extract_batch(self, image_list):
        """
        Extract features from a list of images.
        Returns np.ndarray of shape (n_images, 512).
        """
        return np.stack([self.extract(img) for img in image_list])


def merge_features(manual_features: np.ndarray, cnn_features: np.ndarray) -> np.ndarray:
    """
    Merge 34 manual features + 512 CNN features → 546-dim combined vector.

    Strategy: direct concatenation.
    The scaler applied before XGBoost handles the different value ranges.

    Args:
        manual_features: np.ndarray shape (34,) or (n, 34)
        cnn_features:    np.ndarray shape (512,) or (n, 512)
            Use zeros(512) if no image is provided.

    Returns:
        np.ndarray shape (546,) or (n, 546)
    """
    if manual_features.ndim == 1:
        return np.concatenate([manual_features, cnn_features])
    return np.hstack([manual_features, cnn_features])


def get_zero_cnn_features():
    """Return zero-vector (512,) when no image is provided — safe fallback."""
    return np.zeros(512, dtype=np.float32)


# ── STANDALONE TEST ──────────────────────────────────────────────
if __name__ == '__main__':
    extractor = CNNFeatureExtractor()
    # Test with a random noise image
    dummy = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
    feats = extractor.extract(dummy)
    print(f"Test OK — feature shape: {feats.shape}, dtype: {feats.dtype}")
    print(f"Feature stats: min={feats.min():.3f} max={feats.max():.3f} mean={feats.mean():.3f}")
