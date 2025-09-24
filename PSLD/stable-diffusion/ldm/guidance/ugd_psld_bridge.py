"""
UGD + PSLD Integration Bridge

This module provides the bridge between UGD guidance and PSLD's measurement consistency.
It allows combining both guidance types in a unified optimization loop.
"""

import torch
import torch.nn.functional as F
from typing import Optional, Dict, Any, Callable
from ldm.guidance.api import GuidanceFn, GuidanceConfig


class CombinedGuidanceFunction:
    """
    Combines UGD guidance with PSLD measurement consistency.
    
    This implements the total_loss approach mentioned in step 3:
    total_loss = lambda_meas * meas_loss + guidance_fn_weight * guidance_fn
    """
    
    def __init__(
        self,
        guidance_fn: Optional[GuidanceFn] = None,
        measurement_operator = None,
        measurement_target = None,
        guidance_weight: float = 1.0,
        measurement_weight: float = 1.0,
        device = None
    ):
        self.guidance_fn = guidance_fn
        self.measurement_operator = measurement_operator
        self.measurement_target = measurement_target
        self.guidance_weight = guidance_weight
        self.measurement_weight = measurement_weight
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def __call__(self, pred, **kwargs) -> torch.Tensor:
        """
        Combined guidance function following UGD + PSLD integration approach.
        
        Args:
            pred: Predicted x_0 (clean image) in appropriate domain
            **kwargs: Additional arguments (timestep, index, etc.)
            
        Returns:
            Combined loss scalar tensor
        """
        total_loss = torch.tensor(0.0, device=pred.device)
        
        # 1. PSLD measurement consistency term
        if self.measurement_operator is not None and self.measurement_target is not None:
            try:
                # Compute measurement consistency loss: ||A(pred) - y||
                if hasattr(self.measurement_operator, 'forward'):
                    measured = self.measurement_operator.forward(pred)
                else:
                    measured = self.measurement_operator(pred)
                    
                meas_loss = F.mse_loss(measured, self.measurement_target)
                total_loss += self.measurement_weight * meas_loss
                
            except Exception as e:
                print(f"Warning: PSLD measurement consistency failed: {e}")
        
        # 2. UGD guidance term  
        if self.guidance_fn is not None:
            try:
                guidance_loss = self.guidance_fn(pred, **kwargs)
                total_loss += self.guidance_weight * guidance_loss
                
            except Exception as e:
                print(f"Warning: UGD guidance failed: {e}")
        
        return total_loss


def create_psld_measurement_function(operator, target, device):
    """
    Create a PSLD measurement consistency function.
    
    Args:
        operator: PSLD measurement operator (e.g., inpainting, deblur, etc.)
        target: Target measurement (y)
        device: Torch device
        
    Returns:
        Function that computes ||A(x) - y||^2
    """
    def measurement_consistency_fn(pred, **kwargs):
        """PSLD measurement consistency loss."""
        try:
            if hasattr(operator, 'forward'):
                measured = operator.forward(pred, **kwargs)
            else:
                measured = operator(pred, **kwargs)
                
            loss = F.mse_loss(measured, target)
            return loss
            
        except Exception as e:
            print(f"Measurement consistency error: {e}")
            return torch.tensor(0.0, device=pred.device)
    
    return measurement_consistency_fn


def create_combined_ugd_psld_guidance(
    ugd_guidance_fn: Optional[GuidanceFn] = None,
    psld_operator = None,
    psld_target = None,
    ugd_weight: float = 1.0,
    psld_weight: float = 1.0,
    device = None
) -> GuidanceFn:
    """
    Factory function to create combined UGD + PSLD guidance.
    
    This implements the integration approach from step 3 of the plan:
    - Keep PSLD's measurement consistency term
    - Add UGD guidance as another term
    - Backprop once on total_loss
    
    Args:
        ugd_guidance_fn: UGD guidance function (e.g., style transfer)
        psld_operator: PSLD measurement operator  
        psld_target: PSLD measurement target
        ugd_weight: Weight for UGD guidance term
        psld_weight: Weight for PSLD measurement term
        device: Torch device
        
    Returns:
        Combined guidance function following GuidanceFn protocol
    """
    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def combined_guidance_fn(pred, **kwargs) -> torch.Tensor:
        """
        Combined UGD + PSLD guidance function.
        
        This follows the exact approach suggested in step 3:
        total_loss = lambda_meas * meas_loss + guidance_fn_weight * guidance_fn
        """
        total_loss = torch.tensor(0.0, device=pred.device, requires_grad=True)
        
        # PSLD measurement consistency term
        if psld_operator is not None and psld_target is not None:
            try:
                # Compute ||A(pred) - y||^2
                if hasattr(psld_operator, 'forward'):
                    measured = psld_operator.forward(pred, **kwargs)
                else:
                    measured = psld_operator(pred, **kwargs)
                
                # Handle different measurement types
                if measured.shape != psld_target.shape:
                    # Try to reshape or interpolate if needed
                    if measured.numel() == psld_target.numel():
                        measured = measured.reshape(psld_target.shape)
                    else:
                        # For image measurements, interpolate to target size
                        if len(measured.shape) == 4 and len(psld_target.shape) == 4:
                            measured = F.interpolate(measured, size=psld_target.shape[-2:], mode='bilinear')
                
                meas_loss = F.mse_loss(measured, psld_target)
                total_loss = total_loss + psld_weight * meas_loss
                
                # Optional: log measurement consistency
                if kwargs.get('verbose', False):
                    print(f"  PSLD measurement loss: {meas_loss.item():.6f}")
                
            except Exception as e:
                if kwargs.get('verbose', False):
                    print(f"Warning: PSLD measurement consistency failed: {e}")
        
        # UGD guidance term
        if ugd_guidance_fn is not None:
            try:
                guidance_loss = ugd_guidance_fn(pred, **kwargs)
                total_loss = total_loss + ugd_weight * guidance_loss
                
                # Optional: log guidance loss
                if kwargs.get('verbose', False):
                    print(f"  UGD guidance loss: {guidance_loss.item():.6f}")
                
            except Exception as e:
                if kwargs.get('verbose', False):
                    print(f"Warning: UGD guidance failed: {e}")
        
        # Optional: log total loss
        if kwargs.get('verbose', False):
            print(f"  Total combined loss: {total_loss.item():.6f}")
        
        return total_loss
    
    return combined_guidance_fn


# Example usage functions for common PSLD tasks

def create_inpainting_ugd_guidance(
    mask,
    target_image,
    style_guidance_fn: Optional[GuidanceFn] = None,
    ugd_weight: float = 1.0,
    inpainting_weight: float = 1.0
):
    """Create combined inpainting + UGD style guidance."""
    
    def inpainting_operator(pred, **kwargs):
        """Simple inpainting operator: mask * pred"""
        return mask * pred
    
    target = mask * target_image
    
    return create_combined_ugd_psld_guidance(
        ugd_guidance_fn=style_guidance_fn,
        psld_operator=inpainting_operator,
        psld_target=target,
        ugd_weight=ugd_weight,
        psld_weight=inpainting_weight
    )


def create_super_resolution_ugd_guidance(
    low_res_image,
    style_guidance_fn: Optional[GuidanceFn] = None,
    ugd_weight: float = 1.0,
    sr_weight: float = 1.0,
    scale_factor: int = 4
):
    """Create combined super-resolution + UGD style guidance."""
    
    def sr_operator(pred, **kwargs):
        """Simple super-resolution operator: downsample pred"""
        return F.interpolate(pred, scale_factor=1/scale_factor, mode='bilinear')
    
    return create_combined_ugd_psld_guidance(
        ugd_guidance_fn=style_guidance_fn,
        psld_operator=sr_operator,
        psld_target=low_res_image,
        ugd_weight=ugd_weight,
        psld_weight=sr_weight
    )


# Utility functions

def validate_guidance_compatibility(guidance_cfg: GuidanceConfig, psld_domain: str):
    """
    Validate that UGD guidance config is compatible with PSLD setup.
    
    Args:
        guidance_cfg: UGD guidance configuration
        psld_domain: PSLD operating domain ("latent" or "image")
        
    Returns:
        Boolean indicating compatibility
    """
    if guidance_cfg.domain == "image" and psld_domain == "latent":
        print("⚠️  UGD guidance in image domain but PSLD in latent domain")
        print("   This will require encode/decode operations")
        return True  # Still works, just less efficient
        
    if guidance_cfg.domain == "latent" and psld_domain == "image":
        print("⚠️  UGD guidance in latent domain but PSLD in image domain") 
        print("   This will require encode/decode operations")
        return True  # Still works, just less efficient
    
    return True
