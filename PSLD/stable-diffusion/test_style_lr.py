#!/usr/bin/env python3

import torch
import sys
import os

# Add the current directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_style_constraint_with_lr():
    """
    Test the style constraint with different learning rates
    """
    print("Testing style constraint with different learning rates...")
    
    # Test different learning rates
    learning_rates = [0.001, 0.01, 0.05, 0.1, 0.2]
    
    for lr in learning_rates:
        print(f"\n=== Testing with learning rate: {lr} ===")
        
        # Run the inverse script with the learning rate
        cmd = f"python scripts/inverse.py --file_id='00014.png' --task_config='configs/style_extraction_config.yaml' --outdir='outputs/psld-samples-fr-lr{lr}' --prompt='happy dog' --ddim_steps=10"
        
        print(f"Command: {cmd}")
        print("Note: You'll need to manually modify the lr parameter in the PSLD code to test different values.")
        print("Current learning rate in the code is hardcoded to 0.01")
        
        # For now, just show what needs to be changed
        print(f"\nTo test lr={lr}, modify these lines in ldm/models/diffusion/psld.py:")
        print("1. Change the function signature to include lr parameter")
        print("2. Remove the hardcoded 'lr = 0.01' lines")
        print("3. Use the lr parameter in gradient updates")

if __name__ == "__main__":
    test_style_constraint_with_lr()


