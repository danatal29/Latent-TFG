#!/usr/bin/env python3

import torch
import torchvision.utils as vutils
from torch.utils.tensorboard import SummaryWriter
import os
from datetime import datetime
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

class PSLDTensorBoardLogger:
    """
    TensorBoard logger for PSLD style constraint optimization.
    Logs images, metrics, and loss curves during training.
    """
    
    def __init__(self, log_dir=None, experiment_name="psld_style_constraint"):
        """
        Initialize TensorBoard logger.
        
        Args:
            log_dir: Directory to save logs (if None, uses default)
            experiment_name: Name of the experiment
        """
        if log_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_dir = f"runs/{experiment_name}_{timestamp}"
        
        self.log_dir = log_dir
        self.writer = SummaryWriter(log_dir)
        self.step = 0
        self.image_step = 0
        
        print(f"TensorBoard logs will be saved to: {log_dir}")
        print(f"To view logs, run: tensorboard --logdir={log_dir}")
    
    def log_metrics(self, metrics_dict, step=None):
        """
        Log scalar metrics to TensorBoard.
        
        Args:
            metrics_dict: Dictionary of metric names and values
            step: Step number (if None, uses internal counter)
        """
        if step is None:
            step = self.step
        
        for name, value in metrics_dict.items():
            if isinstance(value, torch.Tensor):
                value = value.item()
            self.writer.add_scalar(name, value, step)
        
        self.step += 1
    
    def log_image(self, image_tensor, name="generated_image", step=None, every_n_steps=10):
        """
        Log image to TensorBoard every n steps.
        
        Args:
            image_tensor: Image tensor [C, H, W] or [B, C, H, W]
            name: Name for the image in TensorBoard
            step: Step number (if None, uses internal counter)
            every_n_steps: Log image every n steps
        """
        if step is None:
            step = self.step
        
        # Only log every n steps
        if step % every_n_steps == 0:
            # Ensure image is in the right format for TensorBoard (CHW)
            if image_tensor.dim() == 4:  # [B, C, H, W]
                image_tensor = image_tensor[0]  # Take first batch: [C, H, W]
            elif image_tensor.dim() == 3:  # [C, H, W]
                pass  # Already in correct format
            else:
                print(f"Warning: Unexpected image tensor shape: {image_tensor.shape}")
                return
            
            # Convert from [-1, 1] to [0, 1] range if needed
            if image_tensor.min() < 0:
                image_tensor = (image_tensor + 1.0) / 2.0
            
            # Clamp to valid range
            image_tensor = torch.clamp(image_tensor, 0, 1)
            
            # Log the image with explicit dataformats
            self.writer.add_image(f"{name}_step_{step}", image_tensor, step, dataformats='CHW')
            self.image_step = step
            
            print(f"Logged image to TensorBoard at step {step}")
    
    def log_style_comparison(self, original_image, generated_image, target_style_features, 
                           pred_style_features, step=None, every_n_steps=10):
        """
        Log style comparison including original, generated, and style features.
        
        Args:
            original_image: Original input image tensor
            generated_image: Generated image tensor
            target_style_features: Target style features
            pred_style_features: Predicted style features
            step: Step number
            every_n_steps: Log every n steps
        """
        if step is None:
            step = self.step
        
        if step % every_n_steps == 0:
            # Create a grid of images
            images_to_log = []
            
            # Add original image
            if original_image.dim() == 3:
                original_image = original_image.unsqueeze(0)
            if original_image.min() < 0:
                original_image = (original_image + 1.0) / 2.0
            original_image = torch.clamp(original_image, 0, 1)
            images_to_log.append(original_image)
            
            # Add generated image
            if generated_image.dim() == 3:
                generated_image = generated_image.unsqueeze(0)
            if generated_image.min() < 0:
                generated_image = (generated_image + 1.0) / 2.0
            generated_image = torch.clamp(generated_image, 0, 1)
            images_to_log.append(generated_image)
            
            # Create style feature visualization
            if target_style_features is not None and pred_style_features is not None:
                # Convert style features to heatmap-like visualization
                target_vis = self._features_to_heatmap(target_style_features)
                pred_vis = self._features_to_heatmap(pred_style_features)
                
                images_to_log.extend([target_vis, pred_vis])
            
            # Create grid
            if len(images_to_log) > 1:
                grid = torch.cat(images_to_log, dim=0)
                self.writer.add_images(f"style_comparison_step_{step}", grid, step)
            
            # Log individual images
            self.log_image(original_image, "original_image", step, every_n_steps)
            self.log_image(generated_image, "generated_image", step, every_n_steps)
    
    def _features_to_heatmap(self, features, size=64):
        """
        Convert style features to a heatmap visualization.
        
        Args:
            features: Style feature tensor
            size: Size of the heatmap
            
        Returns:
            Heatmap tensor [1, 3, H, W]
        """
        if features.dim() == 1:
            features = features.unsqueeze(0)  # Add batch dimension
        
        # Reshape features to 2D
        feature_size = int(np.sqrt(features.shape[-1]))
        if feature_size * feature_size != features.shape[-1]:
            feature_size = int(np.sqrt(features.shape[-1] * 2))  # Approximate
        
        # Pad or truncate to make it square
        target_size = feature_size * feature_size
        if features.shape[-1] < target_size:
            # Pad with zeros
            padding = torch.zeros(features.shape[0], target_size - features.shape[-1], 
                                device=features.device)
            features = torch.cat([features, padding], dim=-1)
        else:
            # Truncate
            features = features[..., :target_size]
        
        # Reshape to 2D
        features_2d = features.view(features.shape[0], feature_size, feature_size)
        
        # Normalize to [0, 1]
        features_2d = (features_2d - features_2d.min()) / (features_2d.max() - features_2d.min() + 1e-8)
        
        # Resize to desired size
        features_2d = torch.nn.functional.interpolate(
            features_2d.unsqueeze(1), size=(size, size), mode='bilinear', align_corners=False
        )
        
        # Convert to RGB heatmap
        heatmap = torch.cat([features_2d, features_2d, features_2d], dim=1)
        
        return heatmap
    
    def log_gradients(self, gradients, step=None):
        """
        Log gradient statistics.
        
        Args:
            gradients: Gradient tensor
            step: Step number
        """
        if step is None:
            step = self.step
        
        grad_norm = torch.norm(gradients).item()
        grad_mean = gradients.mean().item()
        grad_std = gradients.std().item()
        grad_max = gradients.max().item()
        grad_min = gradients.min().item()
        
        self.log_metrics({
            'gradients/norm': grad_norm,
            'gradients/mean': grad_mean,
            'gradients/std': grad_std,
            'gradients/max': grad_max,
            'gradients/min': grad_min
        }, step)
    
    def log_latent_stats(self, z_tensor, step=None):
        """
        Log latent space statistics.
        
        Args:
            z_tensor: Latent tensor
            step: Step number
        """
        if step is None:
            step = self.step
        
        z_norm = torch.norm(z_tensor).item()
        z_mean = z_tensor.mean().item()
        z_std = z_tensor.std().item()
        z_max = z_tensor.max().item()
        z_min = z_tensor.min().item()
        
        self.log_metrics({
            'latent/norm': z_norm,
            'latent/mean': z_mean,
            'latent/std': z_std,
            'latent/max': z_max,
            'latent/min': z_min
        }, step)
    
    def close(self):
        """Close the TensorBoard writer."""
        self.writer.close()
        print(f"TensorBoard logs saved to: {self.log_dir}")
    
    def __del__(self):
        """Destructor to ensure writer is closed."""
        if hasattr(self, 'writer'):
            self.close()


# Global logger instance
_tensorboard_logger = None

def get_tensorboard_logger(log_dir=None, experiment_name="psld_style_constraint"):
    """
    Get or create a global TensorBoard logger instance.
    
    Args:
        log_dir: Directory to save logs
        experiment_name: Name of the experiment
        
    Returns:
        PSLDTensorBoardLogger instance
    """
    global _tensorboard_logger
    if _tensorboard_logger is None:
        _tensorboard_logger = PSLDTensorBoardLogger(log_dir, experiment_name)
    return _tensorboard_logger
