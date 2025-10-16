"""
Test script for CLIP Style Extraction using CLIPStyleOperator.
Tests style vector extraction from both PIL images and tensors,
and computes cosine similarity between style vectors.
"""

import sys
import os
from pathlib import Path

# Add the parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as T

# Import the CLIPStyleOperator
from guided_diffusion.measurements import CLIPStyleOperator


def load_image_as_pil(image_path):
    """Load an image as PIL Image."""
    return Image.open(image_path).convert('RGB')


def pil_to_tensor(pil_image, device):
    """Convert PIL image to tensor in [-1, 1] range."""
    transform = T.Compose([
        T.ToTensor(),  # Converts to [0, 1]
    ])
    tensor = transform(pil_image)
    # Convert from [0, 1] to [-1, 1]
    tensor = tensor * 2.0 - 1.0
    return tensor.to(device)


def compute_cosine_similarity(vec1, vec2):
    """Compute cosine similarity between two vectors."""
    # Ensure vectors are 2D [B, D]
    if vec1.dim() == 1:
        vec1 = vec1.unsqueeze(0)
    if vec2.dim() == 1:
        vec2 = vec2.unsqueeze(0)
    
    # Compute cosine similarity
    similarity = F.cosine_similarity(vec1, vec2, dim=1)
    return similarity.item()


def test_clip_style_extraction():
    """Main test function for CLIP style extraction."""
    print("=" * 80)
    print("CLIP Style Extraction Test")
    print("=" * 80)
    
    # Initialize operator with MPS device (macOS)
    device = torch.device("mps")
    print(f"\nUsing device: {device}")
    
    try:
        operator = CLIPStyleOperator(device=device)
        print("✓ CLIPStyleOperator initialized successfully")
    except Exception as e:
        print(f"✗ Failed to initialize CLIPStyleOperator: {e}")
        return
    
    # Define image paths (relative to project root)
    project_root = Path(__file__).parent.parent.parent
    image1_path = project_root / "pics" / "starry_night_full.jpg"
    image2_path = project_root / "pics" / "rgb_toystory2.jpg"
    
    # Check if images exist
    if not image1_path.exists():
        print(f"✗ Image not found: {image1_path}")
        return
    if not image2_path.exists():
        print(f"✗ Image not found: {image2_path}")
        return
    
    print(f"\nLoading images:")
    print(f"  Image 1: {image1_path.name}")
    print(f"  Image 2: {image2_path.name}")
    
    # Load images as PIL
    pil_img1 = load_image_as_pil(image1_path)
    pil_img2 = load_image_as_pil(image2_path)
    print(f"✓ Images loaded successfully")
    print(f"  Image 1 size: {pil_img1.size}")
    print(f"  Image 2 size: {pil_img2.size}")
    
    # -------------------------------------------------------------------------
    # Test Case 1: Tensor to Tensor
    # -------------------------------------------------------------------------
    print("\n" + "-" * 80)
    print("Test Case 1: Tensor to Tensor")
    print("-" * 80)
    
    try:
        # Convert PIL to tensors
        tensor1 = pil_to_tensor(pil_img1, device)
        tensor2 = pil_to_tensor(pil_img2, device)
        
        print(f"Tensor 1 shape: {tensor1.shape}, range: [{tensor1.min():.3f}, {tensor1.max():.3f}]")
        print(f"Tensor 2 shape: {tensor2.shape}, range: [{tensor2.min():.3f}, {tensor2.max():.3f}]")
        
        # Convert tensors to [0, 1] range for style_vec
        tensor1_01 = (tensor1 + 1.0) / 2.0
        tensor2_01 = (tensor2 + 1.0) / 2.0
        
        # Extract style vectors
        style_vec1 = operator.style_vec(tensor1_01)
        style_vec2 = operator.style_vec(tensor2_01)
        
        print(f"Style vector 1 shape: {style_vec1.shape}")
        print(f"Style vector 2 shape: {style_vec2.shape}")
        
        # Compute cosine similarity
        similarity = compute_cosine_similarity(style_vec1, style_vec2)
        print(f"\n✓ Cosine Similarity (Tensor to Tensor): {similarity:.6f}")
        
    except Exception as e:
        print(f"✗ Test Case 1 failed: {e}")
        import traceback
        traceback.print_exc()
    
    # -------------------------------------------------------------------------
    # Test Case 2: Same Image (Tensor) - Should have high similarity
    # -------------------------------------------------------------------------
    print("\n" + "-" * 80)
    print("Test Case 2: Same Image Test (Tensor)")
    print("-" * 80)
    
    try:
        # Extract style vector from same image twice
        style_vec1_a = operator.style_vec(tensor1_01)
        style_vec1_b = operator.style_vec(tensor1_01)
        
        # Compute cosine similarity
        similarity_same = compute_cosine_similarity(style_vec1_a, style_vec1_b)
        print(f"✓ Cosine Similarity (Same Image): {similarity_same:.6f}")
        print(f"  (Expected: ~1.0 for identical images)")
        
    except Exception as e:
        print(f"✗ Test Case 2 failed: {e}")
        import traceback
        traceback.print_exc()
    
    # -------------------------------------------------------------------------
    # Test Case 3: Different crops of same image - Should have moderate/high similarity
    # -------------------------------------------------------------------------
    print("\n" + "-" * 80)
    print("Test Case 3: Tensor Variants Test")
    print("-" * 80)
    
    try:
        # Create a slightly modified version (add small noise)
        tensor1_variant = tensor1_01 + torch.randn_like(tensor1_01) * 0.1
        tensor1_variant = torch.clamp(tensor1_variant, 0.0, 1.0)
        
        style_vec1_orig = operator.style_vec(tensor1_01)
        style_vec1_var = operator.style_vec(tensor1_variant)
        
        similarity_variant = compute_cosine_similarity(style_vec1_orig, style_vec1_var)
        print(f"✓ Cosine Similarity (Original vs Noisy Variant): {similarity_variant:.6f}")
        print(f"  (Expected: High similarity ~0.95+ for small noise)")
        
    except Exception as e:
        print(f"✗ Test Case 3 failed: {e}")
        import traceback
        traceback.print_exc()
    
    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("Test Summary")
    print("=" * 80)
    print(f"Different images (Starry Night vs Toy Story): {similarity:.6f}")
    print(f"Same image (self-similarity):                  {similarity_same:.6f}")
    print(f"Noisy variant (robustness):                    {similarity_variant:.6f}")
    print("\nInterpretation:")
    print("  - Different styles should have lower similarity (< 0.8)")
    print("  - Same image should have similarity ≈ 1.0")
    print("  - Noisy variants should have high similarity (> 0.9)")
    print("=" * 80)


if __name__ == "__main__":
    test_clip_style_extraction()

