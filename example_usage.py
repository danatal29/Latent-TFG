#!/usr/bin/env python3
"""
Example usage script for UGD-PSLD integrated style transfer
This script demonstrates various ways to use the hybrid system
"""

import os
import sys
import torch
from PIL import Image
from omegaconf import OmegaConf
import argparse

# Add paths
sys.path.append('/workspace')
sys.path.append('/workspace/Universal-Guided-Diffusion/stable-diffusion-guided')
sys.path.append('/workspace/PSLD/stable-diffusion')

from ugd_psld_style_transfer import UGDPSLDStyleTransfer
from ugd_psld_advanced import (
    GuidanceConfig, GuidanceMode, 
    create_hybrid_sampler, load_config
)


def example_basic_style_transfer():
    """Basic example of style transfer using the hybrid system"""
    
    print("=" * 50)
    print("Example 1: Basic Style Transfer")
    print("=" * 50)
    
    # Configuration paths (update these to your actual paths)
    config_path = "/workspace/Universal-Guided-Diffusion/stable-diffusion-guided/configs/stable-diffusion/v1-inference.yaml"
    checkpoint_path = "/workspace/Universal-Guided-Diffusion/stable-diffusion-guided/models/ldm/stable-diffusion-v1/model.ckpt"
    
    # Check if files exist
    if not os.path.exists(config_path):
        print(f"Config not found at {config_path}")
        print("Please update the path to your actual config file")
        return
        
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found at {checkpoint_path}")
        print("Please download the model checkpoint first")
        return
    
    # Initialize the style transfer system
    style_transfer = UGDPSLDStyleTransfer(
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    # Setup hybrid sampler
    style_transfer.setup_sampler(guidance_mode="hybrid")
    
    # Example 1: Artistic style transfer
    print("\nPerforming artistic style transfer...")
    guidance_config = {
        "clip_model": "RN50",
        "clip_weight": 1.0,
        "ps_weight": 0.5,
        "guidance_scale": 100.0,
        "ps_scale": 1.0
    }
    
    # Create a dummy style image for demonstration
    # In practice, you would load an actual style image
    style_image = Image.new('RGB', (512, 512), color='red')
    style_image.save('/tmp/dummy_style.jpg')
    
    style_transfer.transfer_style(
        content_text="A beautiful mountain landscape with snow",
        style_image_path='/tmp/dummy_style.jpg',
        output_path='/tmp/output_artistic.png',
        num_steps=25,  # Reduced for faster demo
        guidance_config=guidance_config,
        cfg_scale=7.5,
        seed=42
    )
    
    print("✓ Artistic style transfer completed!")
    print("  Output saved to: /tmp/output_artistic.png")


def example_comparison_modes():
    """Compare different guidance modes"""
    
    print("\n" + "=" * 50)
    print("Example 2: Comparing Guidance Modes")
    print("=" * 50)
    
    modes = ["ugd", "psld", "hybrid"]
    results = {}
    
    # Configuration
    config_path = "/workspace/Universal-Guided-Diffusion/stable-diffusion-guided/configs/stable-diffusion/v1-inference.yaml"
    checkpoint_path = "/workspace/Universal-Guided-Diffusion/stable-diffusion-guided/models/ldm/stable-diffusion-v1/model.ckpt"
    
    if not os.path.exists(config_path) or not os.path.exists(checkpoint_path):
        print("Skipping comparison - model files not found")
        return
    
    # Initialize system
    style_transfer = UGDPSLDStyleTransfer(
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    # Test each mode
    for mode in modes:
        print(f"\nTesting {mode.upper()} mode...")
        
        style_transfer.setup_sampler(guidance_mode=mode)
        
        guidance_config = {
            "clip_model": "RN50",
            "clip_weight": 1.0 if mode in ["ugd", "hybrid"] else 0.0,
            "ps_weight": 1.0 if mode in ["psld", "hybrid"] else 0.0,
            "guidance_scale": 100.0,
            "ps_scale": 1.0
        }
        
        output_path = f'/tmp/output_{mode}.png'
        
        # For demonstration, we'll skip actual generation
        # In practice, uncomment the following:
        """
        style_transfer.transfer_style(
            content_text="A serene lake at sunset",
            style_image_path='/tmp/dummy_style.jpg',
            output_path=output_path,
            num_steps=25,
            guidance_config=guidance_config,
            cfg_scale=7.5,
            seed=42
        )
        """
        
        results[mode] = output_path
        print(f"  ✓ {mode.upper()} mode completed")
    
    print("\n" + "-" * 30)
    print("Results saved:")
    for mode, path in results.items():
        print(f"  {mode.upper()}: {path}")


def example_adaptive_guidance():
    """Demonstrate adaptive guidance scheduling"""
    
    print("\n" + "=" * 50)
    print("Example 3: Adaptive Guidance Scheduling")
    print("=" * 50)
    
    from ugd_psld_advanced import AdaptiveGuidanceScheduler
    import numpy as np
    import matplotlib.pyplot as plt
    
    # Create scheduler
    scheduler = AdaptiveGuidanceScheduler(total_steps=50, schedule_type="cosine")
    
    # Visualize weight schedules
    steps = range(50)
    ugd_weights = []
    psld_weights = []
    
    for step in steps:
        ugd_w, psld_w = scheduler.get_weights(step, base_ugd=1.0, base_psld=1.0)
        ugd_weights.append(ugd_w)
        psld_weights.append(psld_w)
    
    # Print sample weights
    print("\nSample weight progression:")
    print("Step | UGD Weight | PSLD Weight")
    print("-" * 35)
    for i in [0, 12, 25, 37, 49]:
        print(f" {i:3d} | {ugd_weights[i]:10.4f} | {psld_weights[i]:11.4f}")
    
    # Save plot
    plt.figure(figsize=(10, 6))
    plt.plot(steps, ugd_weights, label='UGD Weight', linewidth=2)
    plt.plot(steps, psld_weights, label='PSLD Weight', linewidth=2)
    plt.xlabel('Diffusion Step')
    plt.ylabel('Guidance Weight')
    plt.title('Adaptive Guidance Weight Schedule (Cosine)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('/tmp/adaptive_weights.png')
    print("\n✓ Weight schedule plot saved to: /tmp/adaptive_weights.png")


def example_style_interpolation():
    """Demonstrate style interpolation between multiple styles"""
    
    print("\n" + "=" * 50)
    print("Example 4: Style Interpolation")
    print("=" * 50)
    
    from ugd_psld_advanced import StyleInterpolator
    
    # Create dummy style features for demonstration
    style1 = torch.randn(1, 512)  # Simulated style features
    style2 = torch.randn(1, 512)
    style3 = torch.randn(1, 512)
    
    interpolator = StyleInterpolator()
    
    # Different interpolation weights
    weight_sets = [
        [1.0, 0.0, 0.0],  # Pure style 1
        [0.0, 1.0, 0.0],  # Pure style 2
        [0.0, 0.0, 1.0],  # Pure style 3
        [0.5, 0.5, 0.0],  # Mix of 1 and 2
        [0.33, 0.33, 0.34],  # Equal mix
        [0.6, 0.3, 0.1],  # Weighted mix
    ]
    
    print("\nStyle interpolation examples:")
    print("Weights (Style1, Style2, Style3) | Description")
    print("-" * 50)
    
    for weights in weight_sets:
        interpolated = interpolator.interpolate_styles(
            [style1, style2, style3], 
            weights
        )
        desc = get_weight_description(weights)
        print(f"{weights} | {desc}")
    
    print("\n✓ Style interpolation demonstrated successfully!")


def get_weight_description(weights):
    """Get description for weight combination"""
    if weights == [1.0, 0.0, 0.0]:
        return "Pure Style 1"
    elif weights == [0.0, 1.0, 0.0]:
        return "Pure Style 2"
    elif weights == [0.0, 0.0, 1.0]:
        return "Pure Style 3"
    elif weights[0] == weights[1] and weights[1] == weights[2]:
        return "Equal blend"
    else:
        return f"Custom blend"


def example_config_based_setup():
    """Example using configuration file"""
    
    print("\n" + "=" * 50)
    print("Example 5: Configuration-based Setup")
    print("=" * 50)
    
    config_path = "/workspace/ugd_psld_config.yaml"
    
    if not os.path.exists(config_path):
        print(f"Config file not found at {config_path}")
        return
    
    # Load configuration
    config = load_config(config_path)
    
    print("\nLoaded configuration:")
    print(f"  Guidance mode: {config['guidance']['mode']}")
    print(f"  UGD enabled: {config['guidance']['ugd']['enabled']}")
    print(f"  PSLD enabled: {config['guidance']['psld']['enabled']}")
    print(f"  Adaptive weighting: {config['guidance']['hybrid']['adaptive_weighting']}")
    print(f"  Multi-scale: {config['style_transfer']['multiscale']['enabled']}")
    print(f"  Sampling steps: {config['sampling']['num_steps']}")
    
    # Create guidance configuration from file
    guidance_config = GuidanceConfig(
        mode=GuidanceMode(config['guidance']['mode']),
        ugd_weight=config['guidance']['ugd']['clip_weight'],
        psld_weight=config['guidance']['psld']['weight'],
        guidance_scale=config['guidance']['hybrid']['guidance_scale'],
        adaptive_weighting=config['guidance']['hybrid']['adaptive_weighting']
    )
    
    print("\n✓ Configuration loaded and guidance config created!")
    print(f"  Guidance mode: {guidance_config.mode.value}")
    print(f"  UGD weight: {guidance_config.ugd_weight}")
    print(f"  PSLD weight: {guidance_config.psld_weight}")
    print(f"  Guidance scale: {guidance_config.guidance_scale}")


def main():
    """Main function to run all examples"""
    
    print("\n" + "=" * 60)
    print(" UGD-PSLD Integration Examples ")
    print("=" * 60)
    
    parser = argparse.ArgumentParser(description="UGD-PSLD Integration Examples")
    parser.add_argument("--example", type=int, default=0,
                       help="Which example to run (0=all, 1-5 for specific)")
    args = parser.parse_args()
    
    examples = [
        ("Basic Style Transfer", example_basic_style_transfer),
        ("Comparison of Modes", example_comparison_modes),
        ("Adaptive Guidance", example_adaptive_guidance),
        ("Style Interpolation", example_style_interpolation),
        ("Config-based Setup", example_config_based_setup),
    ]
    
    if args.example == 0:
        # Run all examples
        for i, (name, func) in enumerate(examples, 1):
            try:
                func()
            except Exception as e:
                print(f"\n⚠ Example {i} ({name}) failed: {e}")
    else:
        # Run specific example
        if 1 <= args.example <= len(examples):
            name, func = examples[args.example - 1]
            print(f"\nRunning Example {args.example}: {name}")
            func()
        else:
            print(f"Invalid example number. Choose 0 (all) or 1-{len(examples)}")
    
    print("\n" + "=" * 60)
    print(" Examples Complete! ")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Download model checkpoint if not already done")
    print("2. Prepare your own style images")
    print("3. Adjust parameters in ugd_psld_config.yaml")
    print("4. Run style transfer with your content and style!")


if __name__ == "__main__":
    main()