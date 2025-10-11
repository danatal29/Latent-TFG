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
            self.reference_image_logged = False  # Track if we've logged reference image
            self.log_step = 0  # Sequential counter for all logged items
        except ImportError:
            self.tensorboard_logger = None
            self.reference_image_logged = False
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

            # Check if normalization is enabled in the config, default to True if not specified
            normalize_grad = getattr(guidance_cfg, 'normalize_grad', True)

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
                            print(f"&&&&&&&&&&&&&&&&&&&&& Clamping image to {lo} and {hi}")
                            img = torch.clamp(img,-1, 1)

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
                    # <<<<<<< NEW: Conditionally normalize the gradient >>>>>>>
                    # This stabilizes the optimization by ensuring a consistent step size,
                    # preventing large gradients from causing chaotic updates.
                    if normalize_grad:
                        grad = grad / (grad.norm().clamp_min(1e-6))
                    # optional stabilization
                    # grad = grad / (grad.norm().clamp_min(1e-6))
                    x_t.add_(- guidance_cfg.step_wt * grad)

                # re-leaf for next inner iteration (except after the last)
                if step_idx < guidance_cfg.num_steps - 1:
                    x_t = x_t.detach().requires_grad_(True)

                # TensorBoard logging for inner loop
                if step_idx % 1 == 0:
                    gnorm = float(grad.norm().detach().cpu())
                    lval  = float(loss.detach().cpu())
                    pnorm = float(x_t.norm().detach().cpu())
                    
                    # Log inner loop metrics
                    self.tensorboard_logger.log_metrics({
                        'ugd_inner/loss': lval,
                        'ugd_inner/gradient_norm': gnorm,
                        'ugd_inner/parameter_norm': pnorm,
                        'ugd_inner/step_size': guidance_cfg.step_wt,
                        'ugd_inner/learning_rate': guidance_cfg.step_wt,
                        'ugd_inner/k_recur': guidance_cfg.k_recur,
                        'ugd_inner/normalize_grad': float(guidance_cfg.normalize_grad),
                        'ugd_inner/inner_step': step_idx,
                        'ugd_inner/outer_step': index,
                    }, step=self.log_step)
                    self.log_step += 1
                    
                    print(f"✅ UGD step {step_idx}: loss={lval:.6f}, grad_norm={gnorm:.6f}")

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
            
            # Recompute pred_x0 with optimized x
            pred_x0 = (x - sqrt_one_minus_at * e_t) / a_t.sqrt()
            
            # TensorBoard logging for outer loop (after inner optimization completes)
            # Log the final optimized state
            with torch.no_grad():
                # Compute final loss
                decoded_x = self.model.decode_first_stage(pred_x0.detach())
                final_loss = guidance_fn(decoded_x, timestep=int(t[0]), index=index)
                
                # Log outer loop metrics
                self.tensorboard_logger.log_metrics({
                    'ugd_outer/final_inner_loss': float(final_loss.cpu()),
                    'ugd_outer/num_inner_steps': guidance_cfg.num_steps,
                    'ugd_outer/k_recur': guidance_cfg.k_recur,
                    'ugd_outer/normalize_grad': float(guidance_cfg.normalize_grad),
                    'ugd_outer/outer_step_index': index,
                    'ugd_outer/diffusion_timestep': int(t[0]),
                }, step=self.log_step)
                
                # Log the optimized pred_x0 as an image
                # Decode pred_x0 to image space for visualization
                pred_x0_img = self.model.decode_first_stage(pred_x0.detach())
                # Normalize to [0, 1] for visualization
                pred_x0_img_normalized = (pred_x0_img + 1.0) / 2.0
                pred_x0_img_normalized = torch.clamp(pred_x0_img_normalized, 0.0, 1.0)
                
                # Log image (handle batch dimension)
                if pred_x0_img_normalized.dim() == 4:
                    pred_x0_img_normalized = pred_x0_img_normalized[0]
                
                self.tensorboard_logger.log_image(
                    pred_x0_img_normalized,
                    name='ugd_pred_x0_optimized',
                    step=self.log_step,
                    every_n_steps=1
                )
                
                self.log_step += 1

        # 6) Continue with standard DDIM direction/noise synthesis
        if quantize_denoised:
            pred_x0, _, *_ = self.model.first_stage_model.quantize(pred_x0)
            
        # Direction pointing to x_t
        dir_xt = (1. - a_prev - sigma_t**2).sqrt() * e_t
        noise = sigma_t * noise_like(x.shape, device, repeat_noise) * temperature
        
        if noise_dropout > 0.:
            noise = torch.nn.functional.dropout(noise, p=noise_dropout)
            
        x_prev = a_prev.sqrt() * pred_x0 + dir_xt + noise
        
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
               reference_image=None,  # Reference image path or tensor for TensorBoard
               **kwargs):
        """
        Enhanced sample method with UGD guidance support.
        
        Additional parameters:
        - guidance_cfg: GuidanceConfig for UGD settings
        - guidance_fn: GuidanceFn for computing guidance loss
        - reference_image: Reference/style image to log in TensorBoard (path, PIL, or tensor)
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
        
        # Log reference image once at the beginning
        if self.tensorboard_logger is not None and reference_image is not None and not self.reference_image_logged:
            with torch.no_grad():
                # Ensure reference image is in correct format
                if isinstance(reference_image, torch.Tensor):
                    ref_img = reference_image
                else:
                    # If it's a path or PIL image, convert it
                    from PIL import Image
                    import torchvision.transforms as transforms
                    if isinstance(reference_image, str):
                        ref_img = Image.open(reference_image)
                    else:
                        ref_img = reference_image
                    
                    # Convert PIL to tensor if needed
                    if not isinstance(ref_img, torch.Tensor):
                        transform = transforms.Compose([
                            transforms.ToTensor(),
                        ])
                        ref_img = transform(ref_img)
                
                # Move to correct device if needed
                if hasattr(self.model, 'device'):
                    device = self.model.device
                elif hasattr(self, 'device'):
                    device = self.device
                else:
                    device = _get_device()
                    
                if ref_img.device != device:
                    ref_img = ref_img.to(device)
                
                # Ensure 3D (CHW)
                if ref_img.dim() == 4:
                    ref_img = ref_img[0]  # Take first image if batch
                
                # Log the reference image
                self.tensorboard_logger.log_image(ref_img, 
                                                name="reference_image", 
                                                step=0, 
                                                every_n_steps=1)
                self.reference_image_logged = True
                print(f"📸 Logged reference image to TensorBoard")

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

        # Get k from your config for self-recurrence, defaulting to 1 (no recurrence).
        # [cite_start]This is the number of times the denoise/re-noise process is repeated per timestep[cite: 180].
        k_recur = 1
        if guidance_cfg and hasattr(guidance_cfg, 'k_recur'):
             k_recur = guidance_cfg.k_recur
        if k_recur > 1:
            print(f"Using Per-step Self-recurrence with k={k_recur} iterations per step.")
            
        # Log sampling configuration to TensorBoard
        if self.tensorboard_logger is not None:
            sampling_metrics = {
                'sampling/k_recur': k_recur,
                'sampling/total_timesteps': total_steps,
                'sampling/ddim_eta': getattr(self, 'ddim_eta', 0.0),
            }
            if guidance_cfg:
                sampling_metrics.update({
                    'sampling/normalize_grad': float(guidance_cfg.normalize_grad),
                    'sampling/guidance_steps': guidance_cfg.num_steps,
                    'sampling/guidance_weight': guidance_cfg.step_wt,
                })
            # Add unconditional guidance scale if available
            if 'unconditional_guidance_scale' in kwargs:
                sampling_metrics['sampling/guidance_scale'] = kwargs['unconditional_guidance_scale']
            self.tensorboard_logger.log_metrics(sampling_metrics, step=0)


        iterator = tqdm(time_range, desc='DDIM Sampler', total=total_steps)

        # DEBUG: Print initial alpha schedule info
        print(f"🔍 DEBUG: DDIM Alpha Schedule Analysis")
        print(f"   Total steps: {total_steps}")
        print(f"   DDIM alphas shape: {self.ddim_alphas.shape}")
        print(f"   DDIM alphas_prev shape: {self.ddim_alphas_prev.shape}")
        print(f"   First few alphas: {self.ddim_alphas[:5]}")
        print(f"   First few alphas_prev: {self.ddim_alphas_prev[:5]}")
        print(f"   Last few alphas: {self.ddim_alphas[-5:]}")
        print(f"   Last few alphas_prev: {self.ddim_alphas_prev[-5:]}")
        
        for i, step in enumerate(iterator):
            index = total_steps - i - 1
            ts = torch.full((b,), step, device=device, dtype=torch.long)
            
            # DEBUG: Print timestep info for first few steps
            if i < 3:
                a_t_current = self.ddim_alphas[index] if index < len(self.ddim_alphas) else "N/A"
                a_prev_current = self.ddim_alphas_prev[index] if index < len(self.ddim_alphas_prev) else "N/A"
                print(f"🔍 DEBUG Timestep {i}: index={index}, step={step}, a_t={a_t_current}, a_prev={a_prev_current}")

            if mask is not None:
                assert x0 is not None
                img_orig = self.model.q_sample(x0, ts)  # TODO: deterministic forward pass?
                img = img_orig * mask + (1. - mask) * img

            current_k = k_recur if index > 0 else 1


            for k_idx in range(current_k):
                            # 1. DENOISE STEP: Call the UGD-enhanced p_sample_ddim.
                            # It takes the current noisy latent `img` (x_t) and produces a less noisy version (x_{t-1}).
                            outs = self.p_sample_ddim(img, cond, ts, index=index,
                                                    use_original_steps=ddim_use_original_steps,
                                                    quantize_denoised=quantize_denoised,
                                                    temperature=temperature,
                                                    noise_dropout=noise_dropout,
                                                    score_corrector=score_corrector,
                                                    corrector_kwargs=corrector_kwargs,
                                                    unconditional_guidance_scale=unconditional_guidance_scale,
                                                    unconditional_conditioning=unconditional_conditioning,
                                                    guidance_cfg=guidance_cfg,
                                                    guidance_fn=guidance_fn)
                            x_prev, pred_x0 = outs

                            # 2. RE-NOISE STEP: Add noise back to x_{t-1} to bring it to the noise level of x_t.
                            # This step is skipped on the final iteration of the recurrence loop.
                            if k_idx < k_recur - 1:
                                # DEBUG: Verify a_t and a_prev values before re-noising
                                a_t_val = self.ddim_alphas[index]
                                a_prev_val = self.ddim_alphas_prev[index]
                                ratio = a_t_val / a_prev_val
                                sqrt_term = 1 - ratio
                                
                                print(f"🔍 DEBUG Self-Recurrence Step {k_idx+1}/{k_recur}:")
                                print(f"   Index: {index}, Timestep: {step}")
                                print(f"   a_t (α_t): {a_t_val:.6f}")
                                print(f"   a_prev (α_t-1): {a_prev_val:.6f}")
                                print(f"   Ratio (a_t/a_prev): {ratio:.6f}")
                                print(f"   (1 - ratio): {sqrt_term:.6f}")
                                print(f"   sqrt(1 - ratio): {torch.sqrt(torch.tensor(max(sqrt_term, 1e-8))):.6f}")
                                
                                # Check for potential issues
                                if ratio >= 1.0:
                                    print(f"⚠️  WARNING: a_t <= a_prev! This should not happen in normal DDIM sampling.")
                                if sqrt_term >= 0:
                                    print(f"⚠️  WARNING: (1 - a_t/a_prev) <= 0! Using small epsilon to avoid negative sqrt.")
                                    sqrt_term = 1e-8
                                
                                # [cite_start]This is Equation (10) from the UGD paper[cite: 178].
                                a_t = torch.full((b, 1, 1, 1), a_t_val, device=device)
                                a_prev = torch.full((b, 1, 1, 1), a_prev_val, device=device)

                                alpha_ratio = a_t / a_prev

                                #
                                # --- ROBUSTNESS FIX ---
                                # Clamp the ratio to prevent sqrt of a negative number due to float precision.
                                # This is the key change to prevent NaN propagation.
                                #
                                alpha_ratio = torch.clamp(alpha_ratio, min=1e-8, max=1.0 - 1e-8)

                                # Get the coefficient for the previous state (x_{t-1})
                                coeff_x_prev = alpha_ratio.sqrt()
                                # Get the coefficient for the noise
                                coeff_noise = (1.0 - alpha_ratio).sqrt()

                                # Apply the re-noising using the robust coefficients
                                img = coeff_x_prev * x_prev + coeff_noise * torch.randn_like(x_prev)

                            else:
                                # On the final recurrence step, the result becomes the input for the next main DDIM step.
                                img = x_prev
                        # <<<<<<< END of Self-Recurrence Loop >>>>>>>

            if callback: callback(i)
            if img_callback: img_callback(pred_x0, i)

            if index % log_every_t == 0 or index == total_steps - 1:
                intermediates['x_inter'].append(img)
                intermediates['pred_x0'].append(pred_x0)
                
                # TensorBoard logging for intermediate checkpoints
                # Log intermediate pred_x0 images at checkpoints
                with torch.no_grad():
                    # Decode pred_x0 to image space
                    pred_x0_img = self.model.decode_first_stage(pred_x0.detach())
                    # Normalize to [0, 1]
                    pred_x0_img_normalized = (pred_x0_img + 1.0) / 2.0
                    pred_x0_img_normalized = torch.clamp(pred_x0_img_normalized, 0.0, 1.0)
                    
                    # Handle batch dimension
                    if pred_x0_img_normalized.dim() == 4:
                        pred_x0_img_normalized = pred_x0_img_normalized[0]
                    
                    # Log checkpoint image
                    self.tensorboard_logger.log_image(
                        pred_x0_img_normalized,
                        name='diffusion_checkpoint',
                        step=self.log_step,
                        every_n_steps=1
                    )
                    self.log_step += 1

        return img, intermediates
