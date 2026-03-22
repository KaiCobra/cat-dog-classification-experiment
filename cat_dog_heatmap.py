"""
Cat vs Dog Patch-Level Heatmap using ViT (DINOv2-style) Prototype Learning

This script:
1. Loads a pretrained ViT-S/16 as feature extractor
2. Selects 100 cat and 100 dog prototype images
3. Extracts patch-level features for all prototypes
4. Computes prototype centers (mean patch features) for cats and dogs
5. For test images, computes per-patch similarity to cat/dog centers
6. Generates heatmaps: Blue=Cat, Red=Dog, White=Background/Neutral
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path
import random
import warnings
warnings.filterwarnings("ignore")

# ============================================================
# 1. ViT-S/16 Model (compatible with Google's pretrained weights)
# ============================================================

class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=384):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x)  # (B, embed_dim, H/P, W/P)
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=6, qkv_bias=True):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, in_features)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads=num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, int(dim * mlp_ratio))

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class VisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=384,
                 depth=12, num_heads=6, mlp_ratio=4.0):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + num_patches, embed_dim))
        self.blocks = nn.ModuleList([Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)

    def interpolate_pos_encoding(self, x, h, w):
        """Interpolate position encoding for arbitrary input sizes."""
        num_patches = x.shape[1] - 1
        N = self.pos_embed.shape[1] - 1

        if num_patches == N and h == w:
            return self.pos_embed

        cls_pos = self.pos_embed[:, 0:1]
        patch_pos = self.pos_embed[:, 1:]

        dim = x.shape[-1]
        h0 = h // self.patch_size
        w0 = w // self.patch_size

        sqrt_N = int(N ** 0.5)
        patch_pos = patch_pos.reshape(1, sqrt_N, sqrt_N, dim).permute(0, 3, 1, 2)
        patch_pos = F.interpolate(patch_pos, size=(h0, w0), mode='bicubic', align_corners=False)
        patch_pos = patch_pos.permute(0, 2, 3, 1).reshape(1, -1, dim)

        return torch.cat([cls_pos, patch_pos], dim=1)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)

        pos_embed = self.interpolate_pos_encoding(x, H, W)
        x = x + pos_embed

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        return x  # (B, 1 + num_patches, embed_dim)

    def get_patch_features(self, x):
        """Extract only patch token features (no CLS token)."""
        out = self.forward(x)
        return out[:, 1:]  # Remove CLS token


def load_google_vit_weights(model, npz_path):
    """Convert Google's ViT JAX weights to PyTorch format."""
    data = np.load(npz_path)

    # Patch embedding
    kernel = data["embedding/kernel"]  # (16, 16, 3, 384) -> (384, 3, 16, 16)
    model.patch_embed.proj.weight.data = torch.from_numpy(kernel).permute(3, 2, 0, 1).float()
    model.patch_embed.proj.bias.data = torch.from_numpy(data["embedding/bias"]).float()

    # CLS token
    model.cls_token.data = torch.from_numpy(data["cls"]).float()

    # Position embedding
    model.pos_embed.data = torch.from_numpy(data["Transformer/posembed_input/pos_embedding"]).float()

    # Encoder norm
    model.norm.weight.data = torch.from_numpy(data["Transformer/encoder_norm/scale"]).float()
    model.norm.bias.data = torch.from_numpy(data["Transformer/encoder_norm/bias"]).float()

    # Transformer blocks
    for i in range(12):
        prefix = f"Transformer/encoderblock_{i}"

        # LayerNorm 1
        model.blocks[i].norm1.weight.data = torch.from_numpy(data[f"{prefix}/LayerNorm_0/scale"]).float()
        model.blocks[i].norm1.bias.data = torch.from_numpy(data[f"{prefix}/LayerNorm_0/bias"]).float()

        # Attention QKV
        q_w = data[f"{prefix}/MultiHeadDotProductAttention_1/query/kernel"]  # (384, 6, 64)
        k_w = data[f"{prefix}/MultiHeadDotProductAttention_1/key/kernel"]
        v_w = data[f"{prefix}/MultiHeadDotProductAttention_1/value/kernel"]
        q_b = data[f"{prefix}/MultiHeadDotProductAttention_1/query/bias"]  # (6, 64)
        k_b = data[f"{prefix}/MultiHeadDotProductAttention_1/key/bias"]
        v_b = data[f"{prefix}/MultiHeadDotProductAttention_1/value/bias"]

        qkv_w = np.stack([q_w.reshape(384, 384), k_w.reshape(384, 384), v_w.reshape(384, 384)])
        qkv_b = np.stack([q_b.reshape(384), k_b.reshape(384), v_b.reshape(384)])
        model.blocks[i].attn.qkv.weight.data = torch.from_numpy(qkv_w.reshape(3*384, 384)).float()
        model.blocks[i].attn.qkv.bias.data = torch.from_numpy(qkv_b.reshape(3*384)).float()

        # Attention projection
        out_w = data[f"{prefix}/MultiHeadDotProductAttention_1/out/kernel"]  # (6, 64, 384)
        out_b = data[f"{prefix}/MultiHeadDotProductAttention_1/out/bias"]
        model.blocks[i].attn.proj.weight.data = torch.from_numpy(out_w.reshape(384, 384).T).float()
        model.blocks[i].attn.proj.bias.data = torch.from_numpy(out_b).float()

        # LayerNorm 2
        model.blocks[i].norm2.weight.data = torch.from_numpy(data[f"{prefix}/LayerNorm_2/scale"]).float()
        model.blocks[i].norm2.bias.data = torch.from_numpy(data[f"{prefix}/LayerNorm_2/bias"]).float()

        # MLP
        model.blocks[i].mlp.fc1.weight.data = torch.from_numpy(data[f"{prefix}/MlpBlock_3/Dense_0/kernel"].T).float()
        model.blocks[i].mlp.fc1.bias.data = torch.from_numpy(data[f"{prefix}/MlpBlock_3/Dense_0/bias"]).float()
        model.blocks[i].mlp.fc2.weight.data = torch.from_numpy(data[f"{prefix}/MlpBlock_3/Dense_1/kernel"].T).float()
        model.blocks[i].mlp.fc2.bias.data = torch.from_numpy(data[f"{prefix}/MlpBlock_3/Dense_1/bias"]).float()

    print(f"Loaded Google ViT-S/16 weights from {npz_path}")
    return model


# ============================================================
# 2. Image Processing
# ============================================================

IMG_SIZE = 224  # Standard ViT input size
PATCH_SIZE = 16
NUM_PATCHES_PER_SIDE = IMG_SIZE // PATCH_SIZE  # 14

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def load_image(path):
    """Load and preprocess an image."""
    img = Image.open(path).convert("RGB")
    return transform(img)


def load_image_batch(paths, batch_size=16):
    """Load images in batches."""
    all_features = []
    for i in range(0, len(paths), batch_size):
        batch_paths = paths[i:i+batch_size]
        batch = torch.stack([load_image(p) for p in batch_paths])
        yield batch, batch_paths


# ============================================================
# 3. Prototype Learning
# ============================================================

@torch.no_grad()
def extract_patch_features(model, image_paths, batch_size=16, device='cpu'):
    """Extract patch-level features for a list of images.
    Returns: tensor of shape (N, num_patches, embed_dim)
    """
    model.eval()
    all_features = []

    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i+batch_size]
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
        features = model.get_patch_features(batch)  # (B, num_patches, embed_dim)
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
        center: (embed_dim,)
    """
    # Flatten to (N * num_patches, embed_dim) and take mean
    all_patches = patch_features.reshape(-1, patch_features.shape[-1])
    center = all_patches.mean(dim=0)
    # L2 normalize the center
    center = F.normalize(center, dim=0)
    return center


# ============================================================
# 4. Heatmap Generation
# ============================================================

def generate_heatmap(model, image_path, cat_center, dog_center, device='cpu'):
    """Generate a cat-vs-dog heatmap for a single image.

    For each patch:
    - Compute cosine similarity to cat center and dog center
    - Map the difference to a color: Blue (cat) <-> White (neutral) <-> Red (dog)

    Returns:
        heatmap: (H_patches, W_patches) array with values in [-1, 1]
                 -1 = strongly cat (blue), +1 = strongly dog (red), 0 = neutral (white)
        original_img: PIL Image
    """
    model.eval()

    # Load and process image
    img = Image.open(image_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        patch_features = model.get_patch_features(img_tensor)  # (1, num_patches, embed_dim)
        patch_features = patch_features.squeeze(0)  # (num_patches, embed_dim)

        # L2 normalize patch features
        patch_features = F.normalize(patch_features, dim=1)

        # Compute cosine similarity to each center
        cat_sim = (patch_features @ cat_center.to(device)).cpu()  # (num_patches,)
        dog_sim = (patch_features @ dog_center.to(device)).cpu()

    # Compute heatmap values: positive = dog, negative = cat
    # Use softmax-like scoring for better contrast
    heatmap_values = dog_sim - cat_sim  # (num_patches,)

    # Reshape to 2D grid
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

    # 2. Heatmap only
    # Create custom colormap: Blue -> White -> Red
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "cat_dog", ["#0066FF", "#FFFFFF", "#FF3300"]
    )

    # Normalize heatmap for better visualization
    vmax = max(abs(heatmap.min()), abs(heatmap.max()))
    if vmax < 1e-6:
        vmax = 1.0

    im = axes[1].imshow(heatmap, cmap=cmap, vmin=-vmax, vmax=vmax,
                        interpolation='nearest', aspect='equal')
    axes[1].set_title("Patch Heatmap\n(Blue=Cat, Red=Dog, White=Neutral)", fontsize=12)
    axes[1].axis("off")
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    # 3. Overlay: heatmap on original image
    resized_img = original_img.resize((IMG_SIZE, IMG_SIZE))
    axes[2].imshow(resized_img)

    # Upsample heatmap to image size for overlay
    heatmap_upsampled = np.kron(heatmap, np.ones((PATCH_SIZE, PATCH_SIZE)))
    axes[2].imshow(heatmap_upsampled, cmap=cmap, alpha=0.5, vmin=-vmax, vmax=vmax)
    axes[2].set_title("Overlay", fontsize=14)
    axes[2].axis("off")

    if title:
        fig.suptitle(title, fontsize=16, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved heatmap: {output_path}")


# ============================================================
# 5. Main Pipeline
# ============================================================

def main():
    # Configuration
    SCRIPT_DIR = Path(__file__).parent
    DATA_DIR = SCRIPT_DIR / "data"
    CAT_DIR = DATA_DIR / "cats"
    DOG_DIR = DATA_DIR / "dogs"
    TEST_DIR = DATA_DIR / "test"
    OUTPUT_DIR = SCRIPT_DIR / "results"
    WEIGHTS_PATH = "/tmp/vit_s16.npz"
    NUM_PROTOTYPES = 100  # Number of prototype images per class
    BATCH_SIZE = 16

    OUTPUT_DIR.mkdir(exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Step 1: Load Model ---
    print("\n[1/5] Loading ViT-S/16 model...")
    model = VisionTransformer(
        img_size=IMG_SIZE, patch_size=PATCH_SIZE, in_chans=3,
        embed_dim=384, depth=12, num_heads=6, mlp_ratio=4.0
    )

    if os.path.exists(WEIGHTS_PATH):
        model = load_google_vit_weights(model, WEIGHTS_PATH)
    else:
        print(f"WARNING: Weights not found at {WEIGHTS_PATH}. Using random initialization.")
        print("  Download weights: https://storage.googleapis.com/vit_models/augreg/S_16-i21k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0.npz")

    model = model.to(device)
    model.eval()
    print(f"  Model parameters: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")

    # --- Step 2: Select Prototypes ---
    print("\n[2/5] Selecting prototype images...")
    cat_images = sorted([str(CAT_DIR / f) for f in os.listdir(CAT_DIR)
                         if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    dog_images = sorted([str(DOG_DIR / f) for f in os.listdir(DOG_DIR)
                         if f.lower().endswith(('.jpg', '.jpeg', '.png'))])

    random.seed(42)
    cat_prototypes = random.sample(cat_images, min(NUM_PROTOTYPES, len(cat_images)))
    dog_prototypes = random.sample(dog_images, min(NUM_PROTOTYPES, len(dog_images)))

    print(f"  Cat prototypes: {len(cat_prototypes)} (from {len(cat_images)} total)")
    print(f"  Dog prototypes: {len(dog_prototypes)} (from {len(dog_images)} total)")

    # --- Step 3: Extract Features & Compute Centers ---
    print("\n[3/5] Extracting patch features for prototypes...")
    print("  Processing cat prototypes...")
    cat_features = extract_patch_features(model, cat_prototypes, BATCH_SIZE, device)
    print(f"    Cat features shape: {cat_features.shape}")

    print("  Processing dog prototypes...")
    dog_features = extract_patch_features(model, dog_prototypes, BATCH_SIZE, device)
    print(f"    Dog features shape: {dog_features.shape}")

    # --- Step 4: Compute Prototype Centers ---
    print("\n[4/5] Computing prototype centers...")
    cat_center = compute_prototype_center(cat_features)
    dog_center = compute_prototype_center(dog_features)

    cosine_sim = F.cosine_similarity(cat_center.unsqueeze(0), dog_center.unsqueeze(0)).item()
    print(f"  Cat center norm: {cat_center.norm().item():.4f}")
    print(f"  Dog center norm: {dog_center.norm().item():.4f}")
    print(f"  Cosine similarity between centers: {cosine_sim:.4f}")

    # --- Step 5: Generate Heatmaps for Test Images ---
    print("\n[5/5] Generating heatmaps for test images...")
    test_images = sorted([str(TEST_DIR / f) for f in os.listdir(TEST_DIR)
                          if f.lower().endswith(('.jpg', '.jpeg', '.png'))])

    if not test_images:
        print("  No test images found!")
        return

    for img_path in test_images:
        img_name = os.path.basename(img_path)
        output_name = f"heatmap_{os.path.splitext(img_name)[0]}.png"
        output_path = str(OUTPUT_DIR / output_name)

        try:
            heatmap, original_img = generate_heatmap(model, img_path, cat_center, dog_center, device)

            # Determine title based on filename
            if "cat" in img_name.lower():
                label = "Cat"
            elif "dog" in img_name.lower():
                label = "Dog"
            else:
                label = "Unknown"

            title = f"Cat vs Dog Heatmap — {img_name} (Label: {label})"
            visualize_heatmap(img_path, heatmap, original_img, output_path, title)

        except Exception as e:
            print(f"  Error processing {img_name}: {e}")

    print(f"\nDone! Results saved to: {OUTPUT_DIR}")
    print(f"Total heatmaps generated: {len(list(OUTPUT_DIR.glob('heatmap_*.png')))}")


if __name__ == "__main__":
    main()
