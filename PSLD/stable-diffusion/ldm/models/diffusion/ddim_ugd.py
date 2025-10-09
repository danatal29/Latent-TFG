"""
UGD-enhanced DDIM sampler with inner optimization loop.
Extends the standard DDIMSampler with UGD guidance integration.
"""

import torch
import numpy as np
from tqdm import tqdm
from functools import partial
from typing import Optional

from ldm.modules.diffusionmodules.util import make_ddim_sampling_parameters, make_ddim_timesteps, noise_like, extract_into_tensor
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.guidance.api import GuidanceConfig, GuidanceFn


def _get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class UGDDDIMSampler(DDIMSampler):
    """
    UGD-enhanced DDIM sampler with inner optimization loop.
    Implements step 2 of the UGD integration plan.
    """
    
    def __init__(self, model, schedule="linear", **kwargs):
        super().__init__(model, schedule, **kwargs)
        
        # Initialize TensorBoard logger
        try:
            from tensorboard_logger import get_tensorboard_logger
            self.tensorboard_logger = get_tensorboard_logger(
                experiment_name="ugd_style_guidance"
            )
            self.log_step = 0
        except ImportError:
            print("Warning: tensorboard_logger not available, logging disabled")
            self.tensorboard_logger = None
            self.log_step = 0

    def p_sample_ddim(self, x, c, t, index, repeat_noise=False, use_original_steps=False, quantize_denoised=False,
                      temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                      unconditional_guidance_scale=1., unconditional_conditioning=None,
                      guidance_cfg: Optional[GuidanceConfig] = None,
                      guidance_fn: Optional[GuidanceFn] = None, 
                      **kwargs):
        """
        UGD-enhanced p_sample_ddim with inner optimization loop.
        
        This implements the core UGD algorithm:
        1. Standard epsilon -> pred_x0 computation
        2. Inner optimization loop over x_t using guidance function
        3. Final DDIM step with optimized x_t
        """
        b, *_, device = *x.shape, x.device

        # Standard DDIM setup - use no_grad for initial prediction if no UGD guidance
        if not (guidance_cfg and guidance_cfg.enabled and guidance_fn):
            with torch.no_grad():
                if unconditional_conditioning is None or unconditional_guidance_scale == 1.:
                    e_t = self.model.apply_model(x, t, c)
                else:
                    x_in = torch.cat([x] * 2)
                    t_in = torch.cat([t] * 2)
                    c_in = torch.cat([unconditional_conditioning, c])
                    e_t_uncond, e_t = self.model.apply_model(x_in, t_in, c_in).chunk(2)
                    e_t = e_t_uncond + unconditional_guidance_scale * (e_t - e_t_uncond)
        else:
            # UGD mode - allow gradients
            if unconditional_conditioning is None or unconditional_guidance_scale == 1.:
                e_t = self.model.apply_model(x, t, c)
            else:
                x_in = torch.cat([x] * 2)
                t_in = torch.cat([t] * 2)
                c_in = torch.cat([unconditional_conditioning, c])
                e_t_uncond, e_t = self.model.apply_model(x_in, t_in, c_in).chunk(2)
                e_t = e_t_uncond + unconditional_guidance_scale * (e_t - e_t_uncond)

        if score_corrector is not None:
            assert self.model.parameterization == "eps"
            e_t = score_corrector.modify_score(self.model, e_t, x, t, c, **corrector_kwargs)

        # DDIM parameters
        alphas = self.model.alphas_cumprod if use_original_steps else self.ddim_alphas
        alphas_prev = self.model.alphas_cumprod_prev if use_original_steps else self.ddim_alphas_prev
        sqrt_one_minus_alphas = self.model.sqrt_one_minus_alphas_cumprod if use_original_steps else self.ddim_sqrt_one_minus_alphas
        sigmas = self.model.ddim_sigmas_for_original_num_steps if use_original_steps else self.ddim_sigmas
        
        # Select parameters for current timestep
        a_t = torch.full((b, 1, 1, 1), alphas[index], device=device)
        a_prev = torch.full((b, 1, 1, 1), alphas_prev[index], device=device)
        sigma_t = torch.full((b, 1, 1, 1), sigmas[index], device=device)
        sqrt_one_minus_at = torch.full((b, 1, 1, 1), sqrt_one_minus_alphas[index], device=device)

        # 1) Standard epsilon -> pred_x0 computation
        pred_x0 = (x - sqrt_one_minus_at * e_t) / a_t.sqrt()

        # 2) UGD INNER OPTIMIZATION LOOP
        if guidance_cfg and guidance_cfg.enabled and guidance_fn:
            # Optimize over x_t (or z_t) IN-PLACE for num_steps
            x_t = x.detach().clone().requires_grad_(True)

            # --- BEFORE the loop (once per outer timestep) ---
            # make a fresh, leaf variable for the inner optimization

            for step_idx in range(guidance_cfg.num_steps):
                # Make sure autograd is ON inside the inner loop
                with torch.enable_grad():

                    # ===== UNet forward (with or without CFG) =====
                    if unconditional_conditioning is None or unconditional_guidance_scale == 1.0:
                        e_t_cur = self.model.apply_model(x_t, t, c)
                    else:
                        # Duplicate along batch for CFG; embeddings don't need grads
                        x_in_cur = torch.cat([x_t, x_t], dim=0)
                        t_in     = torch.cat([t,   t],   dim=0)
                        c_in     = torch.cat([unconditional_conditioning.detach(),
                                            c.detach()], dim=0)

                        e_t_uncond_cur, e_t_cond_cur = self.model.apply_model(x_in_cur, t_in, c_in).chunk(2)
                        e_t_cur = e_t_uncond_cur + unconditional_guidance_scale * (e_t_cond_cur - e_t_uncond_cur)

                    # ===== predict x0 at current inner state =====
                    pred_x0_cur = (x_t - sqrt_one_minus_at * e_t_cur) / a_t.sqrt()

                    # ===== build guidance loss (image or latent domain) =====
                    if guidance_cfg.domain == "image":
                        # IMPORTANT: use grad-preserving decode (not decode_first_stage if it wraps no_grad)
                        img = self.model.first_stage_model.decode(pred_x0_cur)

                        # Optional clamp (keeps grads but beware saturation -> zero grad at bounds)
                        if guidance_cfg.decode_kwargs and 'clamp' in guidance_cfg.decode_kwargs:
                            lo, hi = guidance_cfg.decode_kwargs['clamp']
                            img = torch.clamp(img, lo, hi)

                        loss = guidance_fn(img, timestep=int(t[0]), index=index)
                    else:
                        loss = guidance_fn(pred_x0_cur, timestep=int(t[0]), index=index)

                    # force scalar loss
                    if loss.ndim > 0:
                        loss = loss.mean()

                    # quick sanity checks once
                    if step_idx == 0:
                        if not x_t.requires_grad:
                            print("⚠️ x_t.requires_grad is False — check how x_t is created.")
                        if guidance_cfg.domain == "image":
                            print("inner-step: img.requires_grad =", img.requires_grad)
                        print("inner-step: pred_x0_cur.requires_grad =", pred_x0_cur.requires_grad)

                    # ===== take gradient wrt x_t only =====
                    grad = torch.autograd.grad(
                        loss, x_t, retain_graph=False, create_graph=False, allow_unused=False
                    )[0]

                # ===== update x_t without tracking (optimizer-free GD step) =====
                with torch.no_grad():
                    # optional stabilization
                    # grad = grad / (grad.norm().clamp_min(1e-6))
                    x_t_prev = x_t.clone()  # Store previous for step size calculation
                    x_t.add_(- guidance_cfg.step_wt * grad)
                    
                    # Calculate actual step size taken
                    step_size_taken = (x_t - x_t_prev).norm()

                # re-leaf for next inner iteration (except after the last)
                if step_idx < guidance_cfg.num_steps - 1:
                    x_t = x_t.detach().requires_grad_(True)

                # Logging for each inner step
                gnorm = float(grad.norm().detach().cpu())
                lval  = float(loss.detach().cpu())
                x_t_norm = float(x_t.norm().detach().cpu())
                
                print(f"✅ UGD step {step_idx}: loss={lval:.6f}, grad_norm={gnorm:.6f}")
                
                # TensorBoard logging - metrics only during inner loop
                if self.tensorboard_logger is not None:
                    # Log metrics for this inner optimization step
                    metrics_to_log = {
                        'ugd_inner/loss': lval,
                        'ugd_inner/gradient_norm': gnorm,
                        'ugd_inner/parameter_norm': x_t_norm,
                        'ugd_inner/step_size': step_size_taken.item(),
                        'ugd_inner/learning_rate': guidance_cfg.step_wt,
                    }
                    
                    self.tensorboard_logger.log_metrics(metrics_to_log, step=self.log_step)
                    
                    # Log gradient statistics
                    self.tensorboard_logger.log_gradients(grad, step=self.log_step)
                    
                    # Log latent statistics
                    self.tensorboard_logger.log_latent_stats(x_t, step=self.log_step)
                    
                    # NOTE: Not logging images during inner loop - they're intermediate states and look bad
                    # Images will be logged after inner loop completes with the final optimized result
                    
                    self.log_step += 1

            # use improved state for the outer DDIM update
            x = x_t.detach()
            
            # Re-evaluate e_t, pred_x0 with the final optimized x
            if unconditional_conditioning is None or unconditional_guidance_scale == 1.:
                e_t = self.model.apply_model(x, t, c)
            else:
                x_in = torch.cat([x] * 2)
                t_in = torch.cat([t] * 2)
                c_in = torch.cat([unconditional_conditioning, c])
                e_t_uncond, e_t = self.model.apply_model(x_in, t_in, c_in).chunk(2)
                e_t = e_t_uncond + unconditional_guidance_scale * (e_t - e_t_uncond)
            
            # Recompute pred_x0 with optimized x - this is our prediction of clean x_0 from optimized x_t
            pred_x0 = (x - sqrt_one_minus_at * e_t) / a_t.sqrt()
            
            # Log summary and the pred_x0 (our guess at x_0 from the optimized x_t)
            if self.tensorboard_logger is not None:
                with torch.no_grad():
                    final_loss = float(loss.detach().cpu())
                    metrics_summary = {
                        'ugd_outer/final_inner_loss': final_loss,
                        'ugd_outer/num_inner_steps': guidance_cfg.num_steps,
                    }
                    self.tensorboard_logger.log_metrics(metrics_summary, step=self.log_step)
                    
                    # Decode pred_x0 to image space - this is our guess at what x_0 looks like
                    # pred_x0 is in latent space, decode it to RGB image for visualization
                    pred_x0_image = self.model.decode_first_stage(pred_x0)
                    
                    # Log every outer step to see progression
                    self.tensorboard_logger.log_image(pred_x0_image, 
                                                    name=f"ugd_pred_x0_optimized", 
                                                    step=self.log_step, 
                                                    every_n_steps=1)
                    print(f"📸 Logged pred_x0 (optimized) at outer step {index}, timestep {int(t[0])}, log_step {self.log_step}")

        # 6) Continue with standard DDIM direction/noise synthesis
        if quantize_denoised:
            pred_x0, _, *_ = self.model.first_stage_model.quantize(pred_x0)
            
        # Direction pointing to x_t
        dir_xt = (1. - a_prev - sigma_t**2).sqrt() * e_t
        noise = sigma_t * noise_like(x.shape, device, repeat_noise) * temperature
        
        if noise_dropout > 0.:
            noise = torch.nn.functional.dropout(noise, p=noise_dropout)
            
        x_prev = a_prev.sqrt() * pred_x0 + dir_xt + noise
        
        # Log scale parameters only (image already logged after inner loop)
        if self.tensorboard_logger is not None:
            with torch.no_grad():
                # Log scale parameters (similar to PSLD)
                self.tensorboard_logger.log_scale_parameters(
                    unconditional_guidance_scale=unconditional_guidance_scale,
                    diffusion_timestep=int(t[0]) if hasattr(t, '__getitem__') else int(t),
                    step=self.log_step
                )
                
                # Don't log image here - already logged after inner loop optimization
                # This avoids duplicate/redundant images
                
                self.log_step += 1
        
        return x_prev, pred_x0

    # REMOVED @torch.no_grad() - UGD needs gradients for inner optimization
    def sample(self,
               S,
               batch_size,
               shape,
               conditioning=None,
               callback=None,
               normals_sequence=None,
               img_callback=None,
               quantize_x0=False,
               eta=0.,
               mask=None,
               x0=None,
               temperature=1.,
               noise_dropout=0.,
               score_corrector=None,
               corrector_kwargs=None,
               verbose=True,
               x_T=None,
               log_every_t=100,
               unconditional_guidance_scale=1.,
               unconditional_conditioning=None,
               # UGD parameters
               guidance_cfg: Optional[GuidanceConfig] = None,
               guidance_fn: Optional[GuidanceFn] = None,
               **kwargs):
        """
        Enhanced sample method with UGD guidance support.
        
        Additional parameters:
        - guidance_cfg: GuidanceConfig for UGD settings
        - guidance_fn: GuidanceFn for computing guidance loss
        """
        if conditioning is not None:
            if isinstance(conditioning, dict):
                cbs = conditioning[list(conditioning.keys())[0]].shape[0]
                if cbs != batch_size:
                    print(f"Warning: Got {cbs} conditionings but batch-size is {batch_size}")
            else:
                if conditioning.shape[0] != batch_size:
                    print(f"Warning: Got {conditioning.shape[0]} conditionings but batch-size is {batch_size}")

        self.make_schedule(ddim_num_steps=S, ddim_eta=eta, verbose=verbose)
        # sampling
        C, H, W = shape
        size = (batch_size, C, H, W)
        print(f'Data shape for DDIM sampling is {size}, eta {eta}')

        # Use no_grad for standard operations, but allow gradients for UGD
        if guidance_cfg and guidance_cfg.enabled and guidance_fn:
            # UGD mode - gradients needed
            samples, intermediates = self.ddim_sampling(conditioning, size,
                                                        callback=callback,
                                                        img_callback=img_callback,
                                                        quantize_denoised=quantize_x0,
                                                        mask=mask, x0=x0,
                                                        ddim_use_original_steps=False,
                                                        noise_dropout=noise_dropout,
                                                        temperature=temperature,
                                                        score_corrector=score_corrector,
                                                        corrector_kwargs=corrector_kwargs,
                                                        x_T=x_T,
                                                        log_every_t=log_every_t,
                                                        unconditional_guidance_scale=unconditional_guidance_scale,
                                                        unconditional_conditioning=unconditional_conditioning,
                                                        # Pass UGD parameters
                                                        guidance_cfg=guidance_cfg,
                                                        guidance_fn=guidance_fn,
                                                        **kwargs)
        else:
            # Standard mode - can use no_grad for efficiency
            with torch.no_grad():
                samples, intermediates = self.ddim_sampling(conditioning, size,
                                                            callback=callback,
                                                            img_callback=img_callback,
                                                            quantize_denoised=quantize_x0,
                                                            mask=mask, x0=x0,
                                                            ddim_use_original_steps=False,
                                                            noise_dropout=noise_dropout,
                                                            temperature=temperature,
                                                            score_corrector=score_corrector,
                                                            corrector_kwargs=corrector_kwargs,
                                                            x_T=x_T,
                                                            log_every_t=log_every_t,
                                                            unconditional_guidance_scale=unconditional_guidance_scale,
                                                            unconditional_conditioning=unconditional_conditioning,
                                                            # Pass UGD parameters
                                                            guidance_cfg=guidance_cfg,
                                                            guidance_fn=guidance_fn,
                                                            **kwargs)
        return samples, intermediates

    def ddim_sampling(self, cond, shape,
                      x_T=None, ddim_use_original_steps=False,
                      callback=None, timesteps=None, quantize_denoised=False,
                      mask=None, x0=None, img_callback=None, log_every_t=100,
                      temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                      unconditional_guidance_scale=1., unconditional_conditioning=None,
                      # UGD parameters
                      guidance_cfg: Optional[GuidanceConfig] = None,
                      guidance_fn: Optional[GuidanceFn] = None,
                      **kwargs):
        device = self.model.betas.device
        b = shape[0]
        if x_T is None:
            img = torch.randn(shape, device=device)
        else:
            img = x_T

        if timesteps is None:
            timesteps = self.ddpm_num_timesteps if ddim_use_original_steps else self.ddim_timesteps
        elif timesteps is not None and not ddim_use_original_steps:
            subset_end = int(min(timesteps / self.ddim_timesteps.shape[0] * self.ddim_timesteps.shape[0], self.ddim_timesteps.shape[0]))
            timesteps = self.ddim_timesteps[:subset_end]

        intermediates = {'x_inter': [img], 'pred_x0': [img]}
        time_range = reversed(range(0,timesteps)) if ddim_use_original_steps else np.flip(timesteps)
        total_steps = timesteps if ddim_use_original_steps else timesteps.shape[0]
        print(f"Running DDIM Sampling with {total_steps} timesteps")

        iterator = tqdm(time_range, desc='DDIM Sampler', total=total_steps)

        for i, step in enumerate(iterator):
            index = total_steps - i - 1
            ts = torch.full((b,), step, device=device, dtype=torch.long)

            if mask is not None:
                assert x0 is not None
                img_orig = self.model.q_sample(x0, ts)  # TODO: deterministic forward pass?
                img = img_orig * mask + (1. - mask) * img

            # Log outer diffusion step info
            if self.tensorboard_logger is not None:
                with torch.no_grad():
                    # Log diffusion progress
                    self.tensorboard_logger.log_scale_parameters(
                        diffusion_timestep=int(step),
                        total_steps=int(total_steps),
                        current_step=int(index),
                        step=self.log_step
                    )

            # Call UGD-enhanced p_sample_ddim
            outs = self.p_sample_ddim(img, cond, ts, index=index, 
                                      use_original_steps=ddim_use_original_steps,
                                      quantize_denoised=quantize_denoised, 
                                      temperature=temperature,
                                      noise_dropout=noise_dropout, 
                                      score_corrector=score_corrector,
                                      corrector_kwargs=corrector_kwargs,
                                      unconditional_guidance_scale=unconditional_guidance_scale,
                                      unconditional_conditioning=unconditional_conditioning,
                                      # Pass UGD parameters
                                      guidance_cfg=guidance_cfg,
                                      guidance_fn=guidance_fn)
            img, pred_x0 = outs
            if callback: callback(i)
            if img_callback: img_callback(pred_x0, i)

            if index % log_every_t == 0 or index == total_steps - 1:
                intermediates['x_inter'].append(img)
                intermediates['pred_x0'].append(pred_x0)
                
                # Log intermediate pred_x0 at key checkpoints
                if self.tensorboard_logger is not None:
                    with torch.no_grad():
                        decoded_pred_x0 = self.model.decode_first_stage(pred_x0)
                        self.tensorboard_logger.log_image(decoded_pred_x0, 
                                                        name=f"diffusion_pred_x0", 
                                                        step=self.log_step, 
                                                        every_n_steps=1)
                        print(f"📸 Logged pred_x0 at diffusion step {index}/{total_steps}")

        return img, intermediates
