"""
Cat vs Dog Patch-Level Heatmap using DINOv3 Prototype Learning

This script:
1. Loads DINOv3 ViT-S/16 (from Meta's facebookresearch/dinov3) as feature extractor
2. Selects 100 cat and 100 dog prototype images
3. Extracts patch-level features for all prototypes
4. Computes prototype centers (mean patch features) for cats and dogs
5. For test images, computes per-patch cosine similarity to cat/dog centers
6. Generates heatmaps: Blue=Cat, Red=Dog, White=Background/Neutral

Usage:
    # With pretrained weights (download first):
    python cat_dog_heatmap.py --weights /path/to/dinov3_vits16_pretrain_lvd1689m-08c60483.pth

    # Without pretrained weights (random init, for testing pipeline):
    python cat_dog_heatmap.py
"""

import argparse
import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path
import random
import warnings

warnings.filterwarnings("ignore")

# Add DINOv3 repo to path
SCRIPT_DIR = Path(__file__).parent
DINOV3_DIR = SCRIPT_DIR / "dinov3-main"
sys.path.insert(0, str(DINOV3_DIR))

# ============================================================
# Configuration
# ============================================================

IMG_SIZE = 224  # DINOv3 default input size
PATCH_SIZE = 16  # DINOv3 ViT-S/16
NUM_PATCHES_PER_SIDE = IMG_SIZE // PATCH_SIZE  # 14

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# ============================================================
# DINOv3 Model Loading
# ============================================================

def load_dinov3_model(weights_path=None, device="cpu"):
    """Load DINOv3 ViT-S/16 model.

    Uses the official DINOv3 repo code from facebookresearch/dinov3.
    DINOv3 uses RoPE (Rotary Position Embedding) instead of learned positional embeddings,
    LayerScale, and storage tokens (registers).

    Args:
        weights_path: Path to .pth weights file. If None, uses random init.
        device: Device to load model on.
    Returns:
        model: DINOv3 DinoVisionTransformer
    """
    from dinov3.hub.backbones import dinov3_vits16

    if weights_path and os.path.exists(weights_path):
        # Load with pretrained weights from local file
        model = dinov3_vits16(pretrained=True, weights=weights_path)
        print(f"  Loaded DINOv3 ViT-S/16 pretrained weights from: {weights_path}")
    else:
        # Create model without pretrained weights
        model = dinov3_vits16(pretrained=False)
        if weights_path:
            print(f"  WARNING: Weights not found at {weights_path}")
        print("  Using DINOv3 ViT-S/16 without pretrained weights (random init)")
        print("  For best results, download weights from Meta:")
        print("    https://dinov3.llamameta.net/dinov3_vits16/dinov3_vits16_pretrain_lvd1689m-08c60483.pth")

    model = model.to(device)
    model.eval()
    return model


def extract_patch_features_dinov3(model, x):
    """Extract patch token features from DINOv3 model.

    DINOv3 returns a dict with:
    - x_norm_clstoken: CLS token features
    - x_storage_tokens: Storage/register token features
    - x_norm_patchtokens: Patch token features (what we need)

    Args:
        model: DINOv3 DinoVisionTransformer
        x: Input tensor (B, 3, H, W)
    Returns:
        patch_features: (B, num_patches, embed_dim)
    """
    with torch.no_grad():
        out = model.forward_features(x)
        return out["x_norm_patchtokens"]


# ============================================================
# Image Processing
# ============================================================

def load_image(path):
    """Load and preprocess an image."""
    img = Image.open(path).convert("RGB")
    return transform(img)


# ============================================================
# Prototype Learning
# ============================================================

@torch.no_grad()
def extract_all_patch_features(model, image_paths, batch_size=16, device="cpu"):
    """Extract patch-level features for a list of images.
    Returns: tensor of shape (N, num_patches, embed_dim)
    """
    model.eval()
    all_features = []

    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i : i + batch_size]
        imgs = []
        for p in batch_paths:
            try:
                imgs.append(load_image(p))
            except Exception as e:
                print(f"  Skipping {p}: {e}")
                continue
        if not imgs:
            continue
        batch = torch.stack(imgs).to(device)
        features = extract_patch_features_dinov3(model, batch)
        all_features.append(features.cpu())

    return torch.cat(all_features, dim=0)


def compute_prototype_center(patch_features):
    """Compute the prototype center from patch features.

    Takes the mean of ALL patch features across ALL prototype images.
    This gives us a single vector representing the "average patch appearance"
    of the class.

    Args:
        patch_features: (N, num_patches, embed_dim)
    Returns:
        center: (embed_dim,) L2-normalized
    """
    all_patches = patch_features.reshape(-1, patch_features.shape[-1])
    center = all_patches.mean(dim=0)
    center = F.normalize(center, dim=0)
    return center


# ============================================================
# Heatmap Generation
# ============================================================

def generate_heatmap(model, image_path, cat_center, dog_center, device="cpu"):
    """Generate a cat-vs-dog heatmap for a single image.

    For each patch:
    - Compute cosine similarity to cat center and dog center
    - Map difference to color: Blue (cat) <-> White (neutral) <-> Red (dog)

    Returns:
        heatmap: (H_patches, W_patches) numpy array
        original_img: PIL Image
    """
    model.eval()
    img = Image.open(image_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        patch_features = extract_patch_features_dinov3(model, img_tensor)
        patch_features = patch_features.squeeze(0)  # (num_patches, embed_dim)
        patch_features = F.normalize(patch_features, dim=1)

        cat_sim = (patch_features @ cat_center.to(device)).cpu()
        dog_sim = (patch_features @ dog_center.to(device)).cpu()

    # Positive = dog, negative = cat
    heatmap_values = dog_sim - cat_sim

    h = w = NUM_PATCHES_PER_SIDE
    heatmap = heatmap_values.reshape(h, w).numpy()
    return heatmap, img


def visualize_heatmap(image_path, heatmap, original_img, output_path, title=None):
    """Visualize the heatmap overlaid on the original image.

    Blue = Cat, Red = Dog, White = Neutral/Background
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # 1. Original image
    axes[0].imshow(original_img.resize((IMG_SIZE, IMG_SIZE)))
    axes[0].set_title("Original Image", fontsize=14)
    axes[0].axis("off")

    # 2. Heatmap only (Blue -> White -> Red)
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "cat_dog", ["#0066FF", "#FFFFFF", "#FF3300"]
    )

    vmax = max(abs(heatmap.min()), abs(heatmap.max()))
    if vmax < 1e-6:
        vmax = 1.0

    im = axes[1].imshow(
        heatmap, cmap=cmap, vmin=-vmax, vmax=vmax, interpolation="nearest", aspect="equal"
    )
    axes[1].set_title("Patch Heatmap\n(Blue=Cat, Red=Dog, White=Neutral)", fontsize=12)
    axes[1].axis("off")
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    # 3. Overlay
    resized_img = original_img.resize((IMG_SIZE, IMG_SIZE))
    axes[2].imshow(resized_img)
    heatmap_upsampled = np.kron(heatmap, np.ones((PATCH_SIZE, PATCH_SIZE)))
    axes[2].imshow(heatmap_upsampled, cmap=cmap, alpha=0.5, vmin=-vmax, vmax=vmax)
    axes[2].set_title("Overlay", fontsize=14)
    axes[2].axis("off")

    if title:
        fig.suptitle(title, fontsize=16, fontweight="bold")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved heatmap: {output_path}")


# ============================================================
# Main Pipeline
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="DINOv3 Cat vs Dog Patch Heatmap")
    parser.add_argument(
        "--weights",
        type=str,
        default=None,
        help="Path to DINOv3 ViT-S/16 weights (.pth file)",
    )
    parser.add_argument("--num-prototypes", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    DATA_DIR = SCRIPT_DIR / "data"
    CAT_DIR = DATA_DIR / "cats"
    DOG_DIR = DATA_DIR / "dogs"
    TEST_DIR = DATA_DIR / "test"
    OUTPUT_DIR = SCRIPT_DIR / "results"
    OUTPUT_DIR.mkdir(exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Step 1: Load DINOv3 Model ---
    print("\n[1/5] Loading DINOv3 ViT-S/16 model...")
    model = load_dinov3_model(args.weights, device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  Model: DINOv3 ViT-S/16 (patch_size={PATCH_SIZE}, embed_dim={model.embed_dim})")
    print(f"  Parameters: {n_params:.1f}M")
    print(f"  Features: RoPE positional encoding, {model.n_storage_tokens} storage tokens")

    # --- Step 2: Select Prototypes ---
    print("\n[2/5] Selecting prototype images...")
    cat_images = sorted(
        [str(CAT_DIR / f) for f in os.listdir(CAT_DIR) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    )
    dog_images = sorted(
        [str(DOG_DIR / f) for f in os.listdir(DOG_DIR) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    )

    random.seed(42)
    cat_prototypes = random.sample(cat_images, min(args.num_prototypes, len(cat_images)))
    dog_prototypes = random.sample(dog_images, min(args.num_prototypes, len(dog_images)))
    print(f"  Cat prototypes: {len(cat_prototypes)} (from {len(cat_images)} total)")
    print(f"  Dog prototypes: {len(dog_prototypes)} (from {len(dog_images)} total)")

    # --- Step 3: Extract Features ---
    print("\n[3/5] Extracting patch features for prototypes...")
    print("  Processing cat prototypes...")
    cat_features = extract_all_patch_features(model, cat_prototypes, args.batch_size, device)
    print(f"    Cat features shape: {cat_features.shape}")

    print("  Processing dog prototypes...")
    dog_features = extract_all_patch_features(model, dog_prototypes, args.batch_size, device)
    print(f"    Dog features shape: {dog_features.shape}")

    # --- Step 4: Compute Prototype Centers ---
    print("\n[4/5] Computing prototype centers...")
    cat_center = compute_prototype_center(cat_features)
    dog_center = compute_prototype_center(dog_features)

    cosine_sim = F.cosine_similarity(cat_center.unsqueeze(0), dog_center.unsqueeze(0)).item()
    print(f"  Cat center norm: {cat_center.norm().item():.4f}")
    print(f"  Dog center norm: {dog_center.norm().item():.4f}")
    print(f"  Cosine similarity between cat/dog centers: {cosine_sim:.4f}")
    print(f"  (Lower = better separation, 1.0 = identical)")

    # --- Step 5: Generate Heatmaps ---
    print("\n[5/5] Generating heatmaps for test images...")
    test_images = sorted(
        [str(TEST_DIR / f) for f in os.listdir(TEST_DIR) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    )

    if not test_images:
        print("  No test images found!")
        return

    for img_path in test_images:
        img_name = os.path.basename(img_path)
        output_name = f"heatmap_{os.path.splitext(img_name)[0]}.png"
        output_path = str(OUTPUT_DIR / output_name)

        try:
            heatmap, original_img = generate_heatmap(model, img_path, cat_center, dog_center, device)

            if "cat" in img_name.lower():
                label = "Cat"
            elif "dog" in img_name.lower():
                label = "Dog"
            else:
                label = "Unknown"

            title = f"DINOv3 Cat vs Dog Heatmap — {img_name} (Label: {label})"
            visualize_heatmap(img_path, heatmap, original_img, output_path, title)
        except Exception as e:
            print(f"  Error processing {img_name}: {e}")

    print(f"\nDone! Results saved to: {OUTPUT_DIR}")
    print(f"Total heatmaps generated: {len(list(OUTPUT_DIR.glob('heatmap_*.png')))}")


if __name__ == "__main__":
    main()
