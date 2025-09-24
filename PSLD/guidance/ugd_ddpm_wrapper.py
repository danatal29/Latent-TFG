"""
UGD DDPM Wrapper for minimal UGD integration into PSLD.
Following Hook Point A strategy - intercepts apply_model calls.
"""

import torch
from omegaconf import OmegaConf
from ldm.util import instantiate_from_config
from ldm.modules.diffusionmodules.util import extract_into_tensor


class UGDDDPMWrapper:
    """
    Minimal UGD wrapper for DDPM - intercepts apply_model calls.
    Following the exact suggestion: Hook point A in apply_model.
    """
    
    def __init__(self, ddpm_model, guidance_fn=None):
        self.ddpm_model = ddpm_model
        self.guidance_fn = guidance_fn
        self._current_pred_x0 = None
        
    def __getattr__(self, name):
        # Forward all attributes to wrapped model
        return getattr(self.ddpm_model, name)
    
    def apply_model(self, x_noisy, t, cond, return_ids=False):
        """
        UGD-enhanced apply_model - intercepts the key call from DDIMSampler.
        This is Hook Point A from the suggestion.
        """
        # Step 1: Run standard apply_model
        eps_pred = self.ddpm_model.apply_model(x_noisy, t, cond, return_ids)
        
        # Step 2: Store pred_x0 for UGD (computed same way as DDIMSampler line 206)
        if self.guidance_fn is not None:
            # Compute pred_x0 (matches DDIMSampler logic)
            alphas_cumprod = self.ddpm_model.alphas_cumprod
            sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
            sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
            
            # Extract timestep values
            a_t = extract_into_tensor(sqrt_alphas_cumprod, t, x_noisy.shape)
            sqrt_one_minus_at = extract_into_tensor(sqrt_one_minus_alphas_cumprod, t, x_noisy.shape)
            
            # Compute predicted x_0 (same as DDIMSampler line 206)
            pred_x0 = (x_noisy - sqrt_one_minus_at * eps_pred) / a_t
            
            # Store for potential guidance use
            self._current_pred_x0 = pred_x0
        
        return eps_pred

# Usage in scripts:
def create_ugd_model(config_path, ckpt_path, guidance_fn=None):
    """Create UGD-enhanced model with minimal modification."""
    config = OmegaConf.load(config_path)
    model = instantiate_from_config(config.model)
    model.load_state_dict(torch.load(ckpt_path)["state_dict"], strict=False)
    
    if guidance_fn is not None:
        # Wrap with UGD - transparent to existing code!
        model = UGDDDPMWrapper(model, guidance_fn)
        
    return model