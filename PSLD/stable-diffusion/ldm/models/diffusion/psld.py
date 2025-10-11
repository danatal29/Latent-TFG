
"""SAMPLING ONLY."""

import torch
import numpy as np
from tqdm import tqdm
from functools import partial
import torch.nn.functional as F

from ldm.modules.diffusionmodules.util import make_ddim_sampling_parameters, make_ddim_timesteps, noise_like, \
    extract_into_tensor

import pdb

def get_device():
    """Get the best available device for MacBook (MPS, CUDA, or CPU)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

class DDIMSampler(object):
    def __init__(self, model, schedule="linear", **kwargs):
        super().__init__()
        self.model = model
        self.ddpm_num_timesteps = model.num_timesteps
        self.schedule = schedule
        self.device = get_device()
        
        # Initialize TensorBoard logger
        try:
            from tensorboard_logger import get_tensorboard_logger
            self.tensorboard_logger = get_tensorboard_logger(
                experiment_name="psld_style_constraint"
            )
            self.log_step = 0
            self.reference_image_logged = False  # Track if we've logged reference image
        except ImportError:
            print("Warning: tensorboard_logger not available, logging disabled")
            self.tensorboard_logger = None
            self.log_step = 0
            self.reference_image_logged = False

    def register_buffer(self, name, attr):
        if type(attr) == torch.Tensor:
            # Convert to float32 first to ensure MPS compatibility
            if attr.dtype == torch.float64:
                attr = attr.float()
            if attr.device != self.device:
                attr = attr.to(self.device)
        setattr(self, name, attr)

    def make_schedule(self, ddim_num_steps, ddim_discretize="uniform", ddim_eta=0., verbose=True):
        self.ddim_timesteps = make_ddim_timesteps(ddim_discr_method=ddim_discretize, num_ddim_timesteps=ddim_num_steps,
                                                  num_ddpm_timesteps=self.ddpm_num_timesteps,verbose=verbose)
        
        alphas_cumprod = self.model.alphas_cumprod
        assert alphas_cumprod.shape[0] == self.ddpm_num_timesteps, 'alphas have to be defined for each timestep'
        to_torch = lambda x: x.clone().detach().to(torch.float32).to(self.device)

        self.register_buffer('betas', to_torch(self.model.betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(self.model.alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod.cpu())))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod.cpu())))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod.cpu())))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod.cpu())))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod.cpu() - 1)))

        # ddim sampling parameters
        ddim_sigmas, ddim_alphas, ddim_alphas_prev = make_ddim_sampling_parameters(alphacums=alphas_cumprod.cpu(),
                                                                                   ddim_timesteps=self.ddim_timesteps,
                                                                                   eta=ddim_eta,verbose=verbose)
        self.register_buffer('ddim_sigmas', ddim_sigmas)
        self.register_buffer('ddim_alphas', ddim_alphas)
        self.register_buffer('ddim_alphas_prev', ddim_alphas_prev)
        # Ensure tensors are float32 for MPS compatibility
        sqrt_one_minus_alphas = torch.sqrt(1. - ddim_alphas).float()
        self.register_buffer('ddim_sqrt_one_minus_alphas', sqrt_one_minus_alphas)
        sigmas_for_original_sampling_steps = ddim_eta * torch.sqrt(
            (1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod) * (
                        1 - self.alphas_cumprod / self.alphas_cumprod_prev))
        # Ensure the tensor is float32 for MPS compatibility
        sigmas_for_original_sampling_steps = sigmas_for_original_sampling_steps.float()
        self.register_buffer('ddim_sigmas_for_original_num_steps', sigmas_for_original_sampling_steps)

    # @torch.no_grad()
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
               ip_mask = None, measurements = None, operator = None, gamma = 1, inpainting = False, omega=1,
               general_inverse = None, noiser=None,
               ffhq256=False,
               reference_image=None,
               # this has to come in the same format as the conditioning, # e.g. as encoded tokens, ...
               **kwargs
               ):
        # Log reference image to TensorBoard (once per sampling session)
        if (reference_image is not None and 
            self.tensorboard_logger is not None and 
            not self.reference_image_logged):
            
            ref_img = reference_image.clone()
            
            # Move to correct device if needed
            if hasattr(self.model, 'device'):
                device = self.model.device
            elif hasattr(self, 'device'):
                device = self.device
            else:
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                
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

        if conditioning is not None:
            if isinstance(conditioning, dict):
                cbs = conditioning[list(conditioning.keys())[0]].shape[0]
                if cbs != batch_size:
                    print(f"Warning: Got {cbs} conditionings but batch-size is {batch_size}")
            else:
                if conditioning.shape[0] != batch_size:
                    print(f"Warning: Got {conditioning.shape[0]} conditionings but batch-size is {batch_size}")
        else:
            print('Running unconditional generation...')
        
        self.make_schedule(ddim_num_steps=S, ddim_eta=eta, verbose=verbose)
        # sampling
        C, H, W = shape
        size = (batch_size, C, H, W)
        print(f'Data shape for DDIM sampling is {size}, eta {eta}')

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
                                                    ip_mask = ip_mask, measurements = measurements, operator = operator,
                                                    gamma = gamma,
                                                    inpainting = inpainting, omega=omega,
                                                    general_inverse = general_inverse, noiser = noiser,
                                                    ffhq256=ffhq256
                                                    )
        return samples, intermediates

    ## lr
    # @torch.no_grad()
    def ddim_sampling(self, cond, shape,
                      x_T=None, ddim_use_original_steps=False,
                      callback=None, timesteps=None, quantize_denoised=False,
                      mask=None, x0=None, img_callback=None, log_every_t=100,
                      temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                      unconditional_guidance_scale=1., unconditional_conditioning=None,
                      ip_mask = None, measurements = None, operator = None, gamma = 1, inpainting=False, omega=1,
                      general_inverse = None, noiser=None,
                      ffhq256=False):
        device = self.device
        b = shape[0]
        if x_T is None:
            img = torch.randn(shape, device=device)
        else:
            img = x_T

        if timesteps is None:
            timesteps = self.ddpm_num_timesteps if ddim_use_original_steps else self.ddim_timesteps
        elif timesteps is not None and not ddim_use_original_steps:
            subset_end = int(min(timesteps / self.ddim_timesteps.shape[0], 1) * self.ddim_timesteps.shape[0]) - 1
            timesteps = self.ddim_timesteps[:subset_end]

        intermediates = {'x_inter': [img], 'pred_x0': [img]}
        time_range = reversed(range(0,timesteps)) if ddim_use_original_steps else np.flip(timesteps)
        total_steps = timesteps if ddim_use_original_steps else timesteps.shape[0]
        print(f"Running DDIM Sampling with {total_steps} timesteps")

        iterator = tqdm(time_range, desc='DDIM Sampler', total=total_steps)

        for i, step in enumerate(iterator):
            index = total_steps - i - 1
            #print('index:', index)
            ts = torch.full((b,), step, device=device, dtype=torch.long)

            if mask is not None:
                assert x0 is not None
                img_orig = self.model.q_sample(x0, ts)  # TODO: deterministic forward pass?
                img = img_orig * mask + (1. - mask) * img

            outs = self.p_sample_ddim(img, cond, ts, index=index, use_original_steps=ddim_use_original_steps,
                                      quantize_denoised=quantize_denoised, temperature=temperature,
                                      noise_dropout=noise_dropout, score_corrector=score_corrector,
                                      corrector_kwargs=corrector_kwargs,
                                      unconditional_guidance_scale=unconditional_guidance_scale,
                                      unconditional_conditioning=unconditional_conditioning,
                                      ip_mask = ip_mask, measurements = measurements, operator = operator, gamma = gamma,
                                      inpainting=inpainting, omega=omega,
                                      gamma_scale = index/total_steps,
                                      general_inverse=general_inverse, noiser=noiser,
                                      ffhq256=ffhq256)
            img, pred_x0 = outs
            if callback: callback(i)
            if img_callback: img_callback(pred_x0, i)

            if index % log_every_t == 0 or index == total_steps - 1:
                intermediates['x_inter'].append(img)
                intermediates['pred_x0'].append(pred_x0)

        return img, intermediates

    ######################
    def p_sample_ddim(self, x, c, t, index, repeat_noise=False, use_original_steps=False, quantize_denoised=False,
                      temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                      unconditional_guidance_scale=1., unconditional_conditioning=None,
                      ip_mask=None, measurements = None, operator = None, gamma=1, inpainting=False,
                      gamma_scale = None, omega = 1e-1,
                      general_inverse=False,noiser=None,
                      ffhq256=False, total_steps=None):
        b, *_, device = *x.shape, x.device
           
        ##########################################
        ## measurment consistency guided diffusion
        ##########################################

        
        if general_inverse:
            print(f"🔍 DEBUG - Running PSLD constraint optimization...")
            print(f"🔍 DEBUG - general_inverse: {general_inverse}")
            print(f"🔍 DEBUG - operator: {operator}")
            print(f"🔍 DEBUG - measurements: {measurements is not None}")
            print(f"🔍 DEBUG - gamma: {gamma}, omega: {omega}")
            
            z_t = torch.clone(x.detach())
            z_t.requires_grad = True
            print(f"🔍 DEBUG - z_t.requires_grad: {z_t.requires_grad}")
            
            if unconditional_conditioning is None or unconditional_guidance_scale == 1.:
                e_t = self.model.apply_model(z_t, t, c)
            else:
                x_in = torch.cat([z_t] * 2)
                t_in = torch.cat([t] * 2)
                c_in = torch.cat([unconditional_conditioning, c])
                e_t_uncond, e_t = self.model.apply_model(x_in, t_in, c_in).chunk(2)
                e_t = e_t_uncond + unconditional_guidance_scale * (e_t - e_t_uncond)
            
            
            if score_corrector is not None:
                assert self.model.parameterization == "eps"
                e_t = score_corrector.modify_score(self.model, e_t, z_t, t, c, **corrector_kwargs)
            
            
            alphas = self.model.alphas_cumprod if use_original_steps else self.ddim_alphas
            alphas_prev = self.model.alphas_cumprod_prev if use_original_steps else self.ddim_alphas_prev
            sqrt_one_minus_alphas = self.model.sqrt_one_minus_alphas_cumprod if use_original_steps else self.ddim_sqrt_one_minus_alphas
            sigmas = self.model.ddim_sigmas_for_original_num_steps if use_original_steps else self.ddim_sigmas
            # select parameters corresponding to the currently considered timestep
            a_t = torch.full((b, 1, 1, 1), alphas[index], device=device)
            a_prev = torch.full((b, 1, 1, 1), alphas_prev[index], device=device)
            sigma_t = torch.full((b, 1, 1, 1), sigmas[index], device=device)
            sqrt_one_minus_at = torch.full((b, 1, 1, 1), sqrt_one_minus_alphas[index],device=device)
            
            # current prediction for x_0
            pred_z_0 = (z_t - sqrt_one_minus_at * e_t) / a_t.sqrt()
            
            
            if quantize_denoised:
                pred_z_0, _, *_ = self.model.first_stage_model.quantize(pred_z_0)
            
            
            # direction pointing to x_t
            dir_zt = (1. - a_prev - sigma_t**2).sqrt() * e_t
            noise = sigma_t * noise_like(x.shape, device, repeat_noise) * temperature
            if noise_dropout > 0.:
                noise = torch.nn.functional.dropout(noise, p=noise_dropout)
            
            z_prev = a_prev.sqrt() * pred_z_0 + dir_zt + noise
            
            
            ##############################################
            image_pred = self.model.differentiable_decode_first_stage(pred_z_0)
            
            # Check if this is a style operator first
            is_style_operator = (hasattr(operator, '__class__') and 
                                ('style' in operator.__class__.__name__.lower() or 
                                 'StyleOperator' in operator.__class__.__name__))
            
            # Debug: Check operator detection
            print(f"🔍 DEBUG - Operator class: {operator.__class__.__name__}")
            print(f"🔍 DEBUG - Is style operator: {is_style_operator}")
            print(f"🔍 DEBUG - Operator name contains 'style': {'style' in operator.__class__.__name__.lower()}")
            print(f"🔍 DEBUG - Operator name contains 'StyleOperator': {'StyleOperator' in operator.__class__.__name__}")
            
            if not is_style_operator:
                # For non-style operators: get measurement prediction and add noise
                print(f"🔍 DEBUG - Using NON-STYLE operator path")
                print(f"🔍 DEBUG - measurements shape: {measurements.shape}")
                print(f"🔍 DEBUG - measurements type: {type(measurements)}")
                
                meas_pred = operator.forward(image_pred)
                meas_pred = noiser(meas_pred)
                
                print(f"🔍 DEBUG - meas_pred shape: {meas_pred.shape}")
                print(f"🔍 DEBUG - meas_pred.requires_grad: {meas_pred.requires_grad}")


            # Handle style operators
            if is_style_operator:
                # For style extraction: use differentiable style loss
                # This maintains the computational graph and allows gradient-based optimization
                
                # Extract target style features from measurements (these are the target style vectors)
                target_style_features = measurements.detach()
                
                # Debug: Check measurements and target features
                print(f"🔍 DEBUG - measurements shape: {measurements.shape}")
                print(f"🔍 DEBUG - measurements type: {type(measurements)}")
                print(f"🔍 DEBUG - target_style_features shape: {target_style_features.shape}")
                print(f"🔍 DEBUG - target_style_features.requires_grad: {target_style_features.requires_grad}")
                
                # Extract style features from the predicted image using the same method
                pred_style_features = operator.forward(image_pred)
                
                # Debug: Check predicted features
                print(f"🔍 DEBUG - pred_style_features shape: {pred_style_features.shape}")
                print(f"🔍 DEBUG - pred_style_features.requires_grad: {pred_style_features.requires_grad}")
                
                # Compute cosine similarity loss for style features
                # Normalize features for cosine similarity - make sure this preserves gradients
                pred_norm = F.normalize(pred_style_features, p=2, dim=-1)
                target_norm = F.normalize(target_style_features, p=2, dim=-1)
                
                # Debug normalization gradients
                print(f"Debug - pred_norm.requires_grad: {pred_norm.requires_grad}")
                
                # Cosine similarity loss (1 - cosine_similarity) - use manual computation to ensure gradients
                # cosine_sim = (pred_norm * target_norm).sum(dim=-1)  # Manual cosine similarity
                cosine_sim = F.cosine_similarity(pred_norm, target_norm, dim=-1)
                cosine_loss = (1.0 - cosine_sim).mean()
                
                # Debug cosine loss gradients
                print(f"Debug - cosine_sim.requires_grad: {cosine_sim.requires_grad}")
                print(f"Debug - cosine_loss.requires_grad: {cosine_loss.requires_grad}")
                
                # Add L2 regularization to prevent overfitting  
                l2_reg = 0.1 * torch.norm(pred_style_features - target_style_features, p=2)
                
                # Combine both losses for better style transfer
                style_loss = cosine_loss #+ l2_reg
                
                # CRITICAL: Ensure style_loss has gradients
                if not style_loss.requires_grad:
                    print(f"❌ WARNING: style_loss lost gradients! cosine_loss.requires_grad = {cosine_loss.requires_grad}")
                    print(f"   pred_style_features.requires_grad = {pred_style_features.requires_grad}")
                    print(f"   pred_norm.requires_grad = {pred_norm.requires_grad}")
                    print(f"   cosine_sim.requires_grad = {cosine_sim.requires_grad}")
                    
                    # FAILSAFE: Create a differentiable style loss
                    if pred_style_features.requires_grad:
                        print("🔧 Attempting to recreate differentiable style loss...")
                        # Use MSE loss as backup that preserves gradients
                        style_loss = torch.nn.functional.mse_loss(
                            F.normalize(pred_style_features, dim=-1),
                            F.normalize(target_style_features, dim=-1)
                        )
                        print(f"   Recreated style_loss.requires_grad = {style_loss.requires_grad}")
                    else:
                        raise RuntimeError("Cannot create differentiable style loss - pred_style_features has no gradients!")
                
                print(f'***Cosine similarity: {cosine_sim.mean().item():.3f}, Cosine loss: {style_loss.item():.3f}, L2 reg: {l2_reg.item():.3f}')

                # # 3) Time schedule on omega (late-strong)
                # omega_t = omega * (1-a_t).sqrt()                               # a_t = alphas[index]
                # error = omega_t * style_loss


                #log SNR
                logsnr_t = torch.log(a_t / (1 - a_t + 1e-8))
                # pick mid around 0 dB -> a_t ≈ 0.5, and a sharpness k in [3,7]
                m = torch.tensor(0.0, device=a_t.device)  # midpoint (logSNR=0)
                k = 5.0                                    # steeper -> later/stronger

                w_t = torch.sigmoid(k * (logsnr_t - m))    # in (0,1), monotone with denoising progress
                omega_t = omega * w_t.clamp(0.0, 1.0)
                error = omega_t * style_loss


                
                # Debug final error tensor
                print(f"Debug - omega_t.requires_grad: {omega_t.requires_grad}")
                print(f"Debug - final error.requires_grad: {error.requires_grad}")
                
                print(f'Style loss: {style_loss.item():.4f}, Omega_t: {omega_t.item():.4f}')
                
                # Continue with gradient computation for style operators
                # (no early return - let it proceed to gradient computation)
                
            else:
                # For other operators: use L2 norm and projection
                meas_error = torch.linalg.norm(meas_pred - measurements)
                
                ortho_project = image_pred - operator.transpose(operator.forward(image_pred))
                parallel_project = operator.transpose(measurements)
                inpainted_image = parallel_project + ortho_project
                
                # encoded_z_0 = self.model.encode_first_stage(inpainted_image) if ffhq256 else self.model.encode_first_stage(inpainted_image).mean  
                encoded_z_0 = self.model.encode_first_stage(inpainted_image)
                encoded_z_0 = self.model.get_first_stage_encoding(encoded_z_0)
                inpaint_error = torch.linalg.norm(encoded_z_0 - pred_z_0)
                
                error = inpaint_error * gamma + meas_error * omega
                
                # Debug gradient flow for non-style operators
                print(f"Debug - meas_error.requires_grad: {meas_error.requires_grad}")
                print(f"Debug - inpaint_error.requires_grad: {inpaint_error.requires_grad}")
                print(f"Debug - final error.requires_grad: {error.requires_grad}")
            
            # Safety check before computing gradients
            if not z_t.requires_grad:
                raise RuntimeError(f"z_t does not require gradients! z_t.requires_grad = {z_t.requires_grad}")
            if not error.requires_grad:
                raise RuntimeError(f"error does not require gradients! error.requires_grad = {error.requires_grad}")
                
            gradients = torch.autograd.grad(error, inputs=z_t, retain_graph=False)[0]
            
            # Adaptive learning rate based on gradient-to-parameter ratio
            grad_norm = gradients.norm()
            normalized_gradients = gradients / (grad_norm + 1e-8)

            # trust-region: move ~k of ||z|| per step, scaled by the same late-strong sched
            k = 0.02                                                     # 2% of ||z||
            step_size_t = (k * a_t.pow(2) * z_prev.norm()).clamp(1e-4, 2e-1)
            z_norm = z_prev.norm()
            print(f'grad_NORM: {normalized_gradients.norm().item():.3f}')
            


            # Debug learning rate calculation

            # Apply gradient update with balanced learning rate for effective but stable constraint enforcement
            lr = (omega_t).clamp(0.1, 10).item()  # Scale up by 100x for balanced style constraint
            z_prev = z_prev - lr * normalized_gradients

            
            
            # Calculate actual step size for monitoring
            actual_step_size = lr * normalized_gradients.norm()
            
            print(f'🔍 DEBUG - PSLD Optimization Step:')
            print(f'   Gradients: {gradients.norm().item():.6f}, Z_PREV: {z_prev.norm().item():.6f}')
            print(f'   Learning Rate: {lr:.3f} (omega_t*100), Actual Step Size: {actual_step_size:.6f}')
            print(f'   Loss: {error.item():.6f}, Omega_t: {omega_t.item():.6f}')
            
            # Log metrics (handle both style and non-style operators)
            if self.tensorboard_logger is not None:
                metrics_to_log = {
                    'loss/total_loss': error.item(),
                    'optimization/learning_rate': lr,
                    'optimization/step_size': step_size_t.item(),
                    'optimization/gradient_norm': grad_norm.item(),
                    'optimization/parameter_norm': z_norm.item(),
                    'optimization/omega': omega,
                    'optimization/omega_t': omega_t,

                }
                
                # Add style loss if it exists (for style operators)
                if 'style_loss' in locals():
                    metrics_to_log['loss/style_loss'] = style_loss.item()
                
                self.tensorboard_logger.log_metrics(metrics_to_log, step=self.log_step)
                
                # Log scale parameters
                self.tensorboard_logger.log_scale_parameters(
                    gamma_scale=gamma_scale,
                    unconditional_guidance_scale=unconditional_guidance_scale,
                    diffusion_timestep=t.item() if hasattr(t, 'item') else t,
                    total_steps=total_steps,
                    current_step=index,
                    step=self.log_step
                )
                # Log images every 10 steps
                if self.log_step % 10 == 0:
                    current_image = self.model.differentiable_decode_first_stage(pred_z_0)
                    self.tensorboard_logger.log_image(current_image, step=self.log_step, every_n_steps=10)

            self.log_step += 1
            
            return z_prev.detach(), pred_z_0.detach()
        
        
        #########################################
        else:
            if unconditional_conditioning is None or unconditional_guidance_scale == 1.:
                with torch.no_grad():
                    e_t = self.model.apply_model(x, t, c)
            else:
                x_in = torch.cat([x] * 2)
                t_in = torch.cat([t] * 2)
                c_in = torch.cat([unconditional_conditioning, c])
                ## lr
                with torch.no_grad():
                    e_t_uncond, e_t = self.model.apply_model(x_in, t_in, c_in).chunk(2)
                e_t = e_t_uncond + unconditional_guidance_scale * (e_t - e_t_uncond)

            if score_corrector is not None:
                assert self.model.parameterization == "eps"
                ## lr
                with torch.no_grad():
                    e_t = score_corrector.modify_score(self.model, e_t, x, t, c, **corrector_kwargs)

            alphas = self.model.alphas_cumprod if use_original_steps else self.ddim_alphas
            alphas_prev = self.model.alphas_cumprod_prev if use_original_steps else self.ddim_alphas_prev
            sqrt_one_minus_alphas = self.model.sqrt_one_minus_alphas_cumprod if use_original_steps else self.ddim_sqrt_one_minus_alphas
            sigmas = self.model.ddim_sigmas_for_original_num_steps if use_original_steps else self.ddim_sigmas
            # select parameters corresponding to the currently considered timestep
            a_t = torch.full((b, 1, 1, 1), alphas[index], device=device)
            a_prev = torch.full((b, 1, 1, 1), alphas_prev[index], device=device)
            sigma_t = torch.full((b, 1, 1, 1), sigmas[index], device=device)
            sqrt_one_minus_at = torch.full((b, 1, 1, 1), sqrt_one_minus_alphas[index],device=device)

            # current prediction for x_0
            pred_x0 = (x - sqrt_one_minus_at * e_t) / a_t.sqrt()
            if quantize_denoised:
                ## 
                with torch.no_grad():
                    pred_x0, _, *_ = self.model.first_stage_model.quantize(pred_x0)
            # direction pointing to x_t
            dir_xt = (1. - a_prev - sigma_t**2).sqrt() * e_t
            noise = sigma_t * noise_like(x.shape, device, repeat_noise) * temperature
            if noise_dropout > 0.:
                noise = torch.nn.functional.dropout(noise, p=noise_dropout)
            x_prev = a_prev.sqrt() * pred_x0 + dir_xt + noise

            # Log scale parameters for regular diffusion path
            if self.tensorboard_logger is not None:
                self.tensorboard_logger.log_scale_parameters(
                    gamma_scale=gamma_scale,
                    unconditional_guidance_scale=unconditional_guidance_scale,
                    diffusion_timestep=t.item() if hasattr(t, 'item') else t,
                    total_steps=total_steps,
                    current_step=index,
                    step=self.log_step
                )
                self.log_step += 1

            return x_prev, pred_x0
    
    ######################

    
    ######################
    
    #@torch.no_grad()
    def stochastic_encode(self, x0, t, use_original_steps=False, noise=None):
        # fast, but does not allow for exact reconstruction
        # t serves as an index to gather the correct alphas
        if use_original_steps:
            sqrt_alphas_cumprod = self.sqrt_alphas_cumprod
            sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod
        else:
            sqrt_alphas_cumprod = torch.sqrt(self.ddim_alphas)
            sqrt_one_minus_alphas_cumprod = self.ddim_sqrt_one_minus_alphas

        if noise is None:
            noise = torch.randn_like(x0)
        return (extract_into_tensor(sqrt_alphas_cumprod, t, x0.shape) * x0 +
                extract_into_tensor(sqrt_one_minus_alphas_cumprod, t, x0.shape) * noise)

    #@torch.no_grad()
    def decode(self, x_latent, cond, t_start, unconditional_guidance_scale=1.0, unconditional_conditioning=None,
               use_original_steps=False):

        timesteps = np.arange(self.ddpm_num_timesteps) if use_original_steps else self.ddim_timesteps
        timesteps = timesteps[:t_start]

        time_range = np.flip(timesteps)
        total_steps = timesteps.shape[0]
        print(f"Running DDIM Sampling with {total_steps} timesteps")

        iterator = tqdm(time_range, desc='Decoding image', total=total_steps)
        x_dec = x_latent
        for i, step in enumerate(iterator):
            index = total_steps - i - 1
            ts = torch.full((x_latent.shape[0],), step, device=x_latent.device, dtype=torch.long)
            x_dec, _ = self.p_sample_ddim(x_dec, cond, ts, index=index, use_original_steps=use_original_steps,
                                          unconditional_guidance_scale=unconditional_guidance_scale,
                                          unconditional_conditioning=unconditional_conditioning,
                                          total_steps=total_steps)
        return x_dec
