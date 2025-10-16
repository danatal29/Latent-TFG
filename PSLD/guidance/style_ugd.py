"""
Minimal Universal Guidance (UGD) for Style Transfer in PSLD.

This is a POC implementation following the UGD approach:
1. Compute style loss on the denoised image (predicted x_0)
2. Apply gradient guidance to nudge the latent towards style target
3. Keep base model frozen - no retraining needed
"""

import torch
import torch.nn.functional as F
from typing import Optional, List
import logging

logger = logging.getLogger(__name__)


class StyleUGDGuidance:
    """
    Minimal UGD style guidance implementation.
    
    Following UGD paper approach: compute guidance on denoised image (x_0)
    and apply gradient to latent to avoid domain gap.
    """
    
    def __init__(
        self, 
        guidance_weight: float = 1.0,
        guidance_schedule: Optional[List[int]] = None,
        device: Optional[torch.device] = None
    ):
        self.guidance_weight = guidance_weight
        self.guidance_schedule = guidance_schedule or []
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Target style features (computed from style image)
        self.target_style_features: Optional[torch.Tensor] = None
        
        # Use existing PSLD style operators
        self._setup_style_extractor()
        
        logger.info(f"StyleUGDGuidance: weight={guidance_weight}, schedule={guidance_schedule}")
    
    def _setup_style_extractor(self):
        """Setup style extractor using PSLD's existing operators."""
        try:
            # Use PSLD's existing style operator
            import sys
            import os
            sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'diffusion-posterior-sampling'))
            from guided_diffusion.measurements import StyleOperator
            self.style_extractor = CLIPStyleOperator(device=self.device)
            logger.info("Using PSLD's DINOv2 StyleOperator")
        except ImportError:
            # Fallback to simple style extraction
            self.style_extractor = None
            logger.warning("Could not import PSLD StyleOperator, using simple style extraction")
    
    def set_style_target(self, style_image: torch.Tensor):
        """Set target style image and extract its features."""
        if self.style_extractor:
            # Use PSLD's style extractor
            with torch.no_grad():
                self.target_style_features = self.style_extractor.forward(style_image)
        else:
            # Simple fallback: extract color/texture statistics
            self.target_style_features = self._extract_simple_style_features(style_image)
        
        logger.info(f"Style target set, features shape: {self.target_style_features.shape}")
    
    def _extract_simple_style_features(self, image: torch.Tensor) -> torch.Tensor:
        """Simple style feature extraction as fallback."""
        # Convert to [0,1] range
        img = torch.clamp((image + 1.0) / 2.0, 0.0, 1.0)
        
        # Ensure we have batch dimension
        if img.dim() == 3:
            img = img.unsqueeze(0)  # [C, H, W] -> [1, C, H, W]
        
        # Color statistics
        color_mean = img.mean(dim=[2, 3])  # [B, C]
        color_std = img.std(dim=[2, 3])    # [B, C]
        
        # Texture features (gradients)
        grad_x = img[:, :, :, 1:] - img[:, :, :, :-1]
        grad_y = img[:, :, 1:, :] - img[:, :, :-1, :]
        texture_mag = torch.sqrt(grad_x**2 + grad_y**2)
        
        # Combine features
        features = torch.cat([color_mean.flatten(), color_std.flatten(), 
                            texture_mag.mean().unsqueeze(0), texture_mag.std().unsqueeze(0)])
        return F.normalize(features, dim=0)
    
    def should_apply_guidance(self, timestep: int) -> bool:
        """Check if guidance should be applied at this timestep."""
        if not self.guidance_schedule:
            return True  # Apply at all timesteps if no schedule
        return timestep in self.guidance_schedule
    
    def compute_guidance(self, pred_x0: torch.Tensor, timestep: int) -> torch.Tensor:
        """
        Compute UGD style guidance on the denoised image (pred_x0).
        
        This is the core UGD approach: compute style loss gradient on x_0
        and apply it to guide the latent towards style target.
        """
        if not self.should_apply_guidance(timestep):
            return torch.zeros_like(pred_x0)
        
        if self.target_style_features is None:
            logger.warning("No style target set, returning zero guidance")
            return torch.zeros_like(pred_x0)
        
        # Make pred_x0 require gradients for style loss computation
        pred_x0_grad = pred_x0.clone().detach().requires_grad_(True)
        
        try:
            if self.style_extractor:
                # Use PSLD's style extractor
                pred_features = self.style_extractor.forward(pred_x0_grad)
            else:
                # Use simple style extraction
                pred_features = self._extract_simple_style_features(pred_x0_grad)
            
            # Compute style loss (cosine similarity loss)
            # Ensure both features have the same shape
            if pred_features.dim() == 1:
                pred_features = pred_features.unsqueeze(0)
            if self.target_style_features.dim() == 1:
                target_features = self.target_style_features.unsqueeze(0)
            else:
                target_features = self.target_style_features
                
            style_loss = 1.0 - F.cosine_similarity(
                pred_features, 
                target_features, 
                dim=1
            ).mean()
            
            # Compute gradient w.r.t. pred_x0
            guidance_gradient = torch.autograd.grad(
                outputs=style_loss,
                inputs=pred_x0_grad,
                create_graph=False,
                retain_graph=False
            )[0]
            
            # Apply guidance weight
            guidance_gradient = guidance_gradient * self.guidance_weight
            
            logger.debug(f"UGD guidance at t={timestep}, loss={style_loss.item():.4f}")
            return guidance_gradient
            
        except Exception as e:
            logger.error(f"Error computing UGD guidance: {e}")
            return torch.zeros_like(pred_x0)
    
    def apply_guidance_to_latent(self, latent: torch.Tensor, pred_x0: torch.Tensor, timestep: int):
        """
        Apply UGD guidance to the latent representation.
        
        Following UGD: compute guidance on x_0, then apply to latent.
        """
        guidance_gradient = self.compute_guidance(pred_x0, timestep)
        
        # Apply guidance to latent (UGD approach)
        guided_latent = latent - guidance_gradient
        
        return guided_latent.detach()


def create_style_guidance(guidance_weight: float = 1.0, guidance_schedule: Optional[List[int]] = None):
    """Factory function to create StyleUGDGuidance."""
    return StyleUGDGuidance(
        guidance_weight=guidance_weight,
        guidance_schedule=guidance_schedule
    )
