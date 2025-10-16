#!/usr/bin/env python3

import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
import sys
import os

# Add the diffusion-posterior-sampling path
sys.path.append('../diffusion-posterior-sampling')

from guided_diffusion.measurements import get_operator

def debug_style_extraction():
    """
    Debug style extraction on different images to see what's happening.
    """
    
    # Initialize the style operator
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    operator = get_operator('style_retrieval', device=device)
    
    # Test images
    test_images = [
        '../../pics/im1.jpg',  # Van Gogh
        '../../pics/im2.jpg',  # Natural image
        '../../pics/im3.jpg'   # Another image
    ]
    
    print("=== Style Extraction Debug ===")
    
    for img_path in test_images:
        if not os.path.exists(img_path):
            print(f"⚠️  Image not found: {img_path}")
            continue
            
        print(f"\n📸 Testing: {img_path}")
        
        # Load image
        pil_img = Image.open(img_path).convert('RGB')
        print(f"   Image size: {pil_img.size}")
        print(f"   Image mode: {pil_img.mode}")
        
        # Extract style features
        try:
            style_features = operator.forward(pil_img)
            print(f"   Style features shape: {style_features.shape}")
            print(f"   Style features norm: {torch.norm(style_features).item():.4f}")
            print(f"   Style features mean: {style_features.mean().item():.4f}")
            print(f"   Style features std: {style_features.std().item():.4f}")
            print(f"   Style features min: {style_features.min().item():.4f}")
            print(f"   Style features max: {style_features.max().item():.4f}")
            
            # Check if features are all zeros or very similar
            if torch.norm(style_features) < 1e-6:
                print("   ⚠️  WARNING: Style features are nearly zero!")
            elif torch.std(style_features) < 1e-6:
                print("   ⚠️  WARNING: Style features have very low variance!")
                
        except Exception as e:
            print(f"   ❌ Error extracting style features: {e}")
    
    # Test tensor input
    print(f"\n🧪 Testing tensor input...")
    try:
        # Create a simple test tensor
        test_tensor = torch.randn(3, 512, 512).to(device)
        print(f"   Test tensor shape: {test_tensor.shape}")
        print(f"   Test tensor device: {test_tensor.device}")
        
        # Extract style from tensor
        tensor_style = operator.forward(test_tensor)
        print(f"   Tensor style features shape: {tensor_style.shape}")
        print(f"   Tensor style features norm: {torch.norm(tensor_style).item():.4f}")
        
    except Exception as e:
        print(f"   ❌ Error with tensor input: {e}")
    
    # Test cosine similarity between different images
    print(f"\n🔍 Testing style similarity between images...")
    style_features_list = []
    image_names = []
    
    for img_path in test_images:
        if os.path.exists(img_path):
            try:
                pil_img = Image.open(img_path).convert('RGB')
                style_features = operator.forward(pil_img)
                style_features_list.append(style_features)
                image_names.append(os.path.basename(img_path))
            except:
                continue
    
    if len(style_features_list) >= 2:
        for i in range(len(style_features_list)):
            for j in range(i+1, len(style_features_list)):
                feat1 = F.normalize(style_features_list[i], dim=-1)
                feat2 = F.normalize(style_features_list[j], dim=-1)
                similarity = F.cosine_similarity(feat1, feat2, dim=-1).item()
                print(f"   Similarity between {image_names[i]} and {image_names[j]}: {similarity:.4f}")
                
                if similarity > 0.95:
                    print(f"   ⚠️  WARNING: Images are very similar in style features!")
                elif similarity < 0.1:
                    print(f"   ✅ Images have very different style features")

def test_style_loss_computation():
    """
    Test the style loss computation to see if it's working properly.
    """
    print(f"\n=== Style Loss Computation Test ===")
    
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    operator = get_operator('style_retrieval', device=device)
    
    # Create two different style features
    style1 = torch.randn(1, 1024).to(device)  # Random style 1
    style2 = torch.randn(1, 1024).to(device)  # Random style 2
    
    # Test cosine similarity loss
    pred_norm = F.normalize(style1, dim=-1)
    target_norm = F.normalize(style2, dim=-1)
    
    cosine_sim = F.cosine_similarity(pred_norm, target_norm, dim=-1)
    style_loss = (1.0 - cosine_sim).mean() * 10.0
    
    print(f"   Style 1 norm: {torch.norm(style1).item():.4f}")
    print(f"   Style 2 norm: {torch.norm(style2).item():.4f}")
    print(f"   Cosine similarity: {cosine_sim.item():.4f}")
    print(f"   Style loss: {style_loss.item():.4f}")
    
    # Test with identical features
    style_loss_identical = (1.0 - F.cosine_similarity(pred_norm, pred_norm, dim=-1)).mean() * 10.0
    print(f"   Style loss (identical): {style_loss_identical.item():.4f}")

if __name__ == "__main__":
    debug_style_extraction()
    test_style_loss_computation()
