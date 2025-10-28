"""
Integrated UGD-PSLD Style Transfer Implementation
This module combines Universal Guided Diffusion (UGD) with Posterior Sampling
for Latent Diffusion (PSLD) to achieve enhanced style transfer capabilities.
"""

import argparse
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm
from pytorch_lightning import seed_everything
from torchvision import transforms
from pathlib import Path
import clip
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, Tuple, Callable
from functools import partial

# Add paths for both UGD and PSLD
sys.path.append('/workspace/Universal-Guided-Diffusion/stable-diffusion-guided')
sys.path.append('/workspace/PSLD/stable-diffusion')
sys.path.append('/workspace/PSLD/diffusion-posterior-sampling')

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim_with_grad import DDIMSamplerWithGrad


class StyleGuidance(ABC):
    """Abstract base class for style guidance methods"""
    
    @abstractmethod
    def compute_guidance(self, x_t, x_0_hat, target_style, t, **kwargs):
        """Compute style guidance gradient"""
        pass


class CLIPStyleGuidance(StyleGuidance):
    """CLIP-based style guidance from UGD"""
    
    def __init__(self, clip_model_name="RN50", device="cuda"):
        self.device = device
        self.clip_model, _ = clip.load(clip_model_name, device=device)
        self.clip_model.eval()
        for param in self.clip_model.parameters():
            param.requires_grad = False
            
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        
    def encode_image(self, image):
        """Encode image to CLIP feature space"""
        # Assume image is in [-1, 1] range
        image = (image + 1) * 0.5
        image = F.interpolate(image, size=(224, 224), mode='bicubic', align_corners=False)
        image = self.normalize(image)
        
        with torch.no_grad():
            image_features = self.clip_model.encode_image(image)
            image_features = image_features / image_features.norm(dim=1, keepdim=True)
        
        return image_features
    
    def compute_guidance(self, x_t, x_0_hat, target_style, t, **kwargs):
        """Compute CLIP-based style guidance"""
        x_0_hat_norm = (x_0_hat + 1) * 0.5
        x_0_hat_resized = F.interpolate(x_0_hat_norm, size=(224, 224), mode='bicubic', align_corners=False)
        x_0_hat_resized = self.normalize(x_0_hat_resized)
        
        # Compute CLIP features
        pred_features = self.clip_model.encode_image(x_0_hat_resized)
        pred_features = pred_features / pred_features.norm(dim=1, keepdim=True)
        
        # Compute similarity loss
        similarity = 100 * (pred_features @ target_style.t())
        loss = -similarity.mean()
        
        return loss


class PosteriorSamplingGuidance(StyleGuidance):
    """Posterior sampling guidance from PSLD"""
    
    def __init__(self, operator=None, scale=1.0):
        self.operator = operator
        self.scale = scale
        
    def compute_guidance(self, x_t, x_0_hat, measurement, t, **kwargs):
        """Compute posterior sampling guidance"""
        if self.operator is not None:
            # Forward operator (e.g., for style transfer: extract style features)
            pred_measurement = self.operator(x_0_hat)
            
            # Compute distance to target measurement
            difference = measurement - pred_measurement
            loss = torch.linalg.norm(difference)
        else:
            # Direct feature matching
            loss = F.mse_loss(x_0_hat, measurement)
            
        return loss * self.scale


class HybridUGDPSLDSampler(DDIMSamplerWithGrad):
    """
    Hybrid sampler combining UGD and PSLD approaches for style transfer
    """
    
    def __init__(self, model, schedule="linear", guidance_mode="hybrid", **kwargs):
        super().__init__(model, schedule, **kwargs)
        self.guidance_mode = guidance_mode  # "ugd", "psld", or "hybrid"
        self.clip_guidance = None
        self.posterior_guidance = None
        
    def setup_guidance(self, guidance_config):
        """Setup guidance methods based on configuration"""
        if self.guidance_mode in ["ugd", "hybrid"]:
            self.clip_guidance = CLIPStyleGuidance(
                clip_model_name=guidance_config.get("clip_model", "RN50")
            )
            
        if self.guidance_mode in ["psld", "hybrid"]:
            self.posterior_guidance = PosteriorSamplingGuidance(
                operator=guidance_config.get("operator", None),
                scale=guidance_config.get("ps_scale", 1.0)
            )
    
    def sample_with_guidance(
        self,
        S,
        batch_size,
        shape,
        conditioning=None,
        style_image=None,
        style_features=None,
        guidance_config=None,
        eta=0.,
        temperature=1.,
        verbose=True,
        unconditional_guidance_scale=1.,
        unconditional_conditioning=None,
        start_zt=None,
        **kwargs
    ):
        """
        Enhanced sampling with hybrid UGD-PSLD guidance
        
        Args:
            S: Number of DDIM steps
            batch_size: Batch size
            shape: Shape of latent (C, H, W)
            conditioning: Text conditioning
            style_image: Target style image for guidance
            style_features: Pre-computed style features
            guidance_config: Configuration for guidance methods
            eta: DDIM eta parameter
            temperature: Temperature for sampling
            verbose: Whether to show progress
            unconditional_guidance_scale: Scale for classifier-free guidance
            unconditional_conditioning: Unconditional embedding for CFG
            start_zt: Initial noise (if None, will be sampled)
        """
        
        self.make_schedule(ddim_num_steps=S, ddim_eta=eta, verbose=verbose)
        
        # Setup guidance if configured
        if guidance_config is not None:
            self.setup_guidance(guidance_config)
        
        # Prepare style features if using CLIP guidance
        if self.clip_guidance is not None and style_image is not None:
            with torch.no_grad():
                style_features = self.clip_guidance.encode_image(style_image)
        
        C, H, W = shape
        shape = (batch_size, C, H, W)
        device = self.model.module.betas.device if hasattr(self.model, 'module') else self.model.betas.device
        
        # Initialize noise
        if start_zt is None:
            img = torch.randn(shape, device=device)
        else:
            img = start_zt
            
        # DDIM sampling loop
        timesteps = self.ddim_timesteps
        time_range = np.flip(timesteps)
        total_steps = timesteps.shape[0]
        
        iterator = tqdm(time_range, desc='Hybrid UGD-PSLD Sampling', total=total_steps) if verbose else time_range
        
        alphas = self.ddim_alphas
        alphas_prev = self.ddim_alphas_prev
        sqrt_one_minus_alphas = self.ddim_sqrt_one_minus_alphas
        sigmas = self.ddim_sigmas
        
        for i, step in enumerate(iterator):
            index = total_steps - i - 1
            ts = torch.full((batch_size,), step, device=device, dtype=torch.long)
            
            # Get DDIM parameters for current timestep
            a_t = torch.full((batch_size, 1, 1, 1), alphas[index], device=device)
            a_prev = torch.full((batch_size, 1, 1, 1), alphas_prev[index], device=device)
            sigma_t = torch.full((batch_size, 1, 1, 1), sigmas[index], device=device)
            sqrt_one_minus_at = torch.full((batch_size, 1, 1, 1), sqrt_one_minus_alphas[index], device=device)
            
            # Enable gradients for guidance
            torch.set_grad_enabled(True)
            img_in = img.detach().requires_grad_(True)
            
            # Apply model with optional classifier-free guidance
            if unconditional_guidance_scale != 1.0 and unconditional_conditioning is not None:
                x_in = torch.cat([img_in] * 2)
                t_in = torch.cat([ts] * 2)
                c_in = torch.cat([unconditional_conditioning, conditioning])
                
                model_module = self.model.module if hasattr(self.model, 'module') else self.model
                e_t_uncond, e_t = model_module.apply_model(x_in, t_in, c_in).chunk(2)
                e_t = e_t_uncond + unconditional_guidance_scale * (e_t - e_t_uncond)
            else:
                model_module = self.model.module if hasattr(self.model, 'module') else self.model
                e_t = model_module.apply_model(img_in, ts, conditioning)
            
            # Predict x_0
            pred_x0 = (img_in - sqrt_one_minus_at * e_t) / a_t.sqrt()
            
            # Apply guidance based on mode
            total_grad = torch.zeros_like(img_in)
            
            if self.guidance_mode in ["ugd", "hybrid"] and self.clip_guidance is not None and style_features is not None:
                # Decode to image space for CLIP guidance
                model_module = self.model.module if hasattr(self.model, 'module') else self.model
                if hasattr(model_module, 'decode_first_stage_with_grad'):
                    recons_image = model_module.decode_first_stage_with_grad(pred_x0)
                else:
                    # Fallback if method doesn't exist
                    with torch.enable_grad():
                        recons_image = model_module.decode_first_stage(pred_x0)
                
                # Compute CLIP guidance
                clip_loss = self.clip_guidance.compute_guidance(
                    img_in, recons_image, style_features, ts
                )
                
                if clip_loss.requires_grad:
                    clip_grad = torch.autograd.grad(outputs=clip_loss, inputs=img_in, retain_graph=True)[0]
                    total_grad += guidance_config.get("clip_weight", 1.0) * clip_grad
            
            if self.guidance_mode in ["psld", "hybrid"] and self.posterior_guidance is not None:
                # Apply posterior sampling guidance
                if style_image is not None:
                    ps_loss = self.posterior_guidance.compute_guidance(
                        img_in, pred_x0, style_image, ts
                    )
                    
                    if ps_loss.requires_grad:
                        ps_grad = torch.autograd.grad(outputs=ps_loss, inputs=img_in)[0]
                        total_grad += guidance_config.get("ps_weight", 1.0) * ps_grad
            
            # Apply guidance gradient
            if self.guidance_mode != "none":
                img_in = img_in - guidance_config.get("guidance_scale", 1.0) * total_grad
            
            # DDIM update step
            torch.set_grad_enabled(False)
            
            # Direction pointing to x_t
            dir_xt = (1. - a_prev - sigma_t**2).sqrt() * e_t
            noise = sigma_t * torch.randn_like(img_in) * temperature if sigma_t > 0 else 0.
            
            img = a_prev.sqrt() * pred_x0 + dir_xt + noise
            
        return img


class UGDPSLDStyleTransfer:
    """
    Main class for integrated UGD-PSLD style transfer
    """
    
    def __init__(self, config_path, checkpoint_path, device="cuda"):
        self.device = device
        self.config = OmegaConf.load(config_path)
        self.model = self._load_model(checkpoint_path)
        self.sampler = None
        
    def _load_model(self, checkpoint_path):
        """Load the diffusion model"""
        print(f"Loading model from {checkpoint_path}")
        pl_sd = torch.load(checkpoint_path, map_location="cpu")
        if "global_step" in pl_sd:
            print(f"Global Step: {pl_sd['global_step']}")
        sd = pl_sd["state_dict"]
        
        model = instantiate_from_config(self.config.model)
        m, u = model.load_state_dict(sd, strict=False)
        
        if len(m) > 0:
            print(f"Missing keys: {m}")
        if len(u) > 0:
            print(f"Unexpected keys: {u}")
            
        model = model.to(self.device)
        model.eval()
        
        # Use DataParallel if multiple GPUs available
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
            
        return model
    
    def setup_sampler(self, guidance_mode="hybrid"):
        """Setup the hybrid sampler"""
        self.sampler = HybridUGDPSLDSampler(self.model, guidance_mode=guidance_mode)
        
    def transfer_style(
        self,
        content_text,
        style_image_path,
        output_path,
        num_steps=50,
        guidance_config=None,
        seed=42,
        **kwargs
    ):
        """
        Perform style transfer
        
        Args:
            content_text: Text description of content
            style_image_path: Path to style image
            output_path: Path to save output
            num_steps: Number of DDIM steps
            guidance_config: Configuration for guidance
            seed: Random seed
        """
        
        if self.sampler is None:
            self.setup_sampler()
            
        # Set seed for reproducibility
        seed_everything(seed)
        
        # Load and preprocess style image
        style_image = Image.open(style_image_path).convert("RGB")
        style_transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Lambda(lambda t: (t * 2) - 1)
        ])
        style_tensor = style_transform(style_image).unsqueeze(0).to(self.device)
        
        # Encode text
        model_module = self.model.module if hasattr(self.model, 'module') else self.model
        conditioning = model_module.get_learned_conditioning([content_text])
        unconditional_conditioning = model_module.get_learned_conditioning([""])
        
        # Default guidance configuration
        if guidance_config is None:
            guidance_config = {
                "clip_model": "RN50",
                "clip_weight": 1.0,
                "ps_weight": 0.5,
                "guidance_scale": 100.0,
                "ps_scale": 1.0
            }
        
        # Sample with guidance
        shape = [4, 64, 64]  # Latent shape for 512x512 images
        samples = self.sampler.sample_with_guidance(
            S=num_steps,
            batch_size=1,
            shape=shape,
            conditioning=conditioning,
            style_image=style_tensor,
            guidance_config=guidance_config,
            unconditional_guidance_scale=kwargs.get("cfg_scale", 7.5),
            unconditional_conditioning=unconditional_conditioning,
            eta=kwargs.get("eta", 0.0)
        )
        
        # Decode latents to images
        with torch.no_grad():
            images = model_module.decode_first_stage(samples)
            images = torch.clamp((images + 1.0) / 2.0, min=0.0, max=1.0)
            
        # Save output
        from torchvision.utils import save_image
        save_image(images, output_path)
        print(f"Style transfer result saved to {output_path}")
        
        return images


def main():
    parser = argparse.ArgumentParser(description="UGD-PSLD Integrated Style Transfer")
    
    parser.add_argument("--config", type=str, required=True, help="Path to model config")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--content", type=str, required=True, help="Content text description")
    parser.add_argument("--style", type=str, required=True, help="Path to style image")
    parser.add_argument("--output", type=str, default="output.png", help="Output path")
    parser.add_argument("--guidance_mode", type=str, default="hybrid", 
                       choices=["ugd", "psld", "hybrid", "none"],
                       help="Guidance mode to use")
    parser.add_argument("--steps", type=int, default=50, help="Number of DDIM steps")
    parser.add_argument("--clip_weight", type=float, default=1.0, help="Weight for CLIP guidance")
    parser.add_argument("--ps_weight", type=float, default=0.5, help="Weight for posterior sampling")
    parser.add_argument("--guidance_scale", type=float, default=100.0, help="Overall guidance scale")
    parser.add_argument("--cfg_scale", type=float, default=7.5, help="Classifier-free guidance scale")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    
    args = parser.parse_args()
    
    # Initialize style transfer system
    style_transfer = UGDPSLDStyleTransfer(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        device=args.device
    )
    
    style_transfer.setup_sampler(guidance_mode=args.guidance_mode)
    
    # Configure guidance
    guidance_config = {
        "clip_model": "RN50",
        "clip_weight": args.clip_weight,
        "ps_weight": args.ps_weight,
        "guidance_scale": args.guidance_scale,
        "ps_scale": 1.0
    }
    
    # Perform style transfer
    style_transfer.transfer_style(
        content_text=args.content,
        style_image_path=args.style,
        output_path=args.output,
        num_steps=args.steps,
        guidance_config=guidance_config,
        cfg_scale=args.cfg_scale,
        seed=args.seed
    )


if __name__ == "__main__":
    main()