"""SAMPLING ONLY."""

import torch
import torch.nn.functional as F
from torch import nn
import math
from functools import partial
import numpy as np
from tqdm import tqdm
from ldm.modules.diffusionmodules.util import make_ddim_sampling_parameters, make_ddim_timesteps, noise_like, extract_into_tensor

import pdb
def _get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def compute_differentiable_style_loss(pred_image, target_style_features, device):
    """
    Compute differentiable style loss using Gram matrices and feature extraction.
    This maintains the computational graph and can be used for gradient-based optimization.
    
    Args:
        pred_image: Predicted image tensor [B, C, H, W] in [-1, 1] range
        target_style_features: Pre-computed target style features from style operator
        device: Device to compute on
    
    Returns:
        style_loss: Differentiable style loss
    """
    # Normalize image to [0, 1] range
    pred_image = torch.clamp((pred_image + 1.0) / 2.0, 0.0, 1.0)
    
    # Simple feature extraction using convolutional layers
    # This is a simplified version that maintains differentiability
    features = extract_simple_features(pred_image)
    
    # Compute Gram matrix for style representation
    gram_pred = compute_gram_matrix(features)
    
    # Normalize Gram matrices
    gram_pred_norm = F.normalize(gram_pred.view(gram_pred.size(0), -1), dim=1)
    target_norm = F.normalize(target_style_features.view(target_style_features.size(0), -1), dim=1)
    
    # Compute cosine similarity loss
    cosine_sim = F.cosine_similarity(gram_pred_norm, target_norm, dim=1)
    style_loss = 1.0 - cosine_sim.mean()
    
    return style_loss

def extract_simple_features(image):
    """
    Extract simple features using basic convolutions.
    This is a differentiable alternative to DINOv2 for style extraction.
    """
    # Simple feature extraction using basic convolutions
    # This maintains the computational graph
    features = []
    
    # Layer 1: Basic edge detection
    kernel1 = torch.tensor([[[[1, 0, -1], [0, 0, 0], [-1, 0, 1]]]], dtype=torch.float32, device=image.device)
    kernel1 = kernel1.repeat(image.size(1), 1, 1, 1)
    conv1 = F.conv2d(image, kernel1, padding=1, groups=image.size(1))
    features.append(conv1)
    
    # Layer 2: Blur filter
    kernel2 = torch.tensor([[[[1/9, 1/9, 1/9], [1/9, 1/9, 1/9], [1/9, 1/9, 1/9]]]], dtype=torch.float32, device=image.device)
    kernel2 = kernel2.repeat(image.size(1), 1, 1, 1)
    conv2 = F.conv2d(image, kernel2, padding=1, groups=image.size(1))
    features.append(conv2)
    
    # Layer 3: High-pass filter
    kernel3 = torch.tensor([[[[0, -1, 0], [-1, 4, -1], [0, -1, 0]]]], dtype=torch.float32, device=image.device)
    kernel3 = kernel3.repeat(image.size(1), 1, 1, 1)
    conv3 = F.conv2d(image, kernel3, padding=1, groups=image.size(1))
    features.append(conv3)
    
    # Concatenate all features
    all_features = torch.cat(features, dim=1)
    
    # Global average pooling to get a fixed-size representation
    pooled_features = F.adaptive_avg_pool2d(all_features, (1, 1)).squeeze(-1).squeeze(-1)
    
    return pooled_features

def compute_gram_matrix(features):
    """
    Compute Gram matrix for style representation.
    """
    # features shape: [B, C]
    gram = torch.bmm(features.unsqueeze(2), features.unsqueeze(1))  # [B, C, C]
    return gram

class DDIMSampler(object):
    def __init__(self, model, schedule="linear", **kwargs):
        super().__init__()
        self.model = model
        self.device = _get_device()
        self.ddpm_num_timesteps = model.num_timesteps
        self.schedule = schedule

    def register_buffer(self, name, attr):
        if isinstance(attr, torch.Tensor):
            # Downcast doubles that slipped in (common when from_numpy)
            if attr.dtype == torch.float64:
                attr = attr.float()
            # Keep everything on the sampler's device
            attr = attr.to(self.device)
        setattr(self, name, attr)


    def make_schedule(self, ddim_num_steps, ddim_discretize="uniform", ddim_eta=0., verbose=True):
        self.ddim_timesteps = make_ddim_timesteps(ddim_discr_method=ddim_discretize, num_ddim_timesteps=ddim_num_steps,
                                                  num_ddpm_timesteps=self.ddpm_num_timesteps,verbose=verbose)
        
        alphas_cumprod = self.model.alphas_cumprod
        assert alphas_cumprod.shape[0] == self.ddpm_num_timesteps, 'alphas have to be defined for each timestep'
        to_torch = lambda x: x.clone().detach().to(torch.float32).to(self.model.device)

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
        self.register_buffer('ddim_sqrt_one_minus_alphas', np.sqrt(1. - ddim_alphas))
        sigmas_for_original_sampling_steps = ddim_eta * torch.sqrt(
            (1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod) * (
                        1 - self.alphas_cumprod / self.alphas_cumprod_prev))
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
               # this has to come in the same format as the conditioning, # e.g. as encoded tokens, ...
               **kwargs
               ):
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
        device = self.model.betas.device
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
    def compute_differentiable_style_loss(self, image_pred, target_measurements, omega):
        """
        Compute a differentiable style loss using image statistics.
        This approximates style without needing DINOv2 gradients.
        """
        # Convert image to [0, 1] range
        img = torch.clamp((image_pred + 1.0) / 2.0, min=0.0, max=1.0)
        
        # Compute color statistics (mean and std per channel)
        color_mean = img.mean(dim=[2, 3])  # [B, C]
        color_std = img.std(dim=[2, 3])    # [B, C]
        
        # Compute texture features using gradients
        grad_x = img[:, :, :, 1:] - img[:, :, :, :-1]  # Horizontal gradients
        grad_y = img[:, :, 1:, :] - img[:, :, :-1, :]  # Vertical gradients
        texture_magnitude = torch.sqrt(grad_x**2 + grad_y**2)
        texture_mean = texture_magnitude.mean()
        texture_std = texture_magnitude.std()
        
        # Combine features into a style vector
        style_features = torch.cat([
            color_mean.flatten(),
            color_std.flatten(),
            texture_mean.unsqueeze(0),
            texture_std.unsqueeze(0)
        ])
        
        # Normalize features
        style_features = F.normalize(style_features, dim=0)
        
        # For now, use a simple L2 loss on the style features
        # In practice, you'd want to compare with target style features
        style_loss = torch.mean(style_features**2) * omega
        
        return style_loss

    def p_sample_ddim(self, x, c, t, index, repeat_noise=False, use_original_steps=False, quantize_denoised=False,
                      temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                      unconditional_guidance_scale=1., unconditional_conditioning=None,
                      ip_mask=None, measurements = None, operator = None, gamma=1, inpainting=False,
                      gamma_scale = None, omega = 1e-1,
                      general_inverse=False,noiser=None,
                      ffhq256=False):
        b, *_, device = *x.shape, x.device
           
        ##########################################
        ## measurment consistency guided diffusion
        ##########################################
        if inpainting:
            # print('Running inpainting module...')
            z_t = x.detach() + 0.0  # Create a differentiable operation
            z_t.requires_grad_(True)
            
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
            meas_pred = operator.forward(image_pred,mask=ip_mask)
            meas_pred = noiser(meas_pred)
            
            # Check if this is a style operator
            # if hasattr(operator, '__class__') and ('style' in operator.__class__.__name__.lower() or 'StyleOperator' in operator.__class__.__name__):
            #     # For style extraction: use differentiable style loss
            #     # This maintains the computational graph and allows gradient-based optimization
                
            #     # Extract target style features from measurements (these are the target style vectors)
            #     target_style_features = measurements
                
            #     # Extract style features from the predicted image using the same method
            #     pred_style_features = operator.forward(image_pred)
                
            #     # Compute L2 loss between predicted and target style features
            #     style_loss = torch.linalg.norm(pred_style_features - target_style_features)
                
            #     # For style extraction: use only the style loss (no inpaint error)
            #     error = style_loss * omega
                
            #     print(f'Style loss: {style_loss.item():.4f}')
                
            #     # Continue with gradient computation for style operators
            #     # (no early return - let it proceed to gradient computation)
                
            # else:
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

            ### tab aboove part of the else
            gradients = torch.autograd.grad(error, inputs=z_t)[0]
            z_prev = z_prev - gradients
            print('Loss: ', error.item())
            
            return z_prev.detach(), pred_z_0.detach()
        
        if general_inverse:
            # print('Running general inverse module...')
            z_t = x.detach() + 0.0  # Create a differentiable operation
            z_t.requires_grad_(True)
            
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
            meas_pred = operator.forward(image_pred)
            meas_pred = noiser(meas_pred)
            
            # Check if this is a style operator
            if hasattr(operator, '__class__') and ('style' in operator.__class__.__name__.lower() or 'StyleOperator' in operator.__class__.__name__):
                # For style extraction: use differentiable style loss
                # This maintains the computational graph and allows gradient-based optimization
                
                # Extract target style features from measurements (these are the target style vectors)
                target_style_features = measurements
                
                # Extract style features from the predicted image using the same method
                pred_style_features = operator.forward(image_pred)
                
                # Compute L2 loss between predicted and target style features
                style_loss = torch.linalg.norm(pred_style_features - target_style_features)
                
                # For style extraction: use only the style loss (no inpaint error)
                error = style_loss * omega
                
                print(f'Style loss: {style_loss.item():.4f}')
                
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
            
            gradients = torch.autograd.grad(error, inputs=z_t)[0]
            z_prev = z_prev - gradients
            print('Loss: ', error.item())
            
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

            return x_prev, pred_x0
    
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
                                          unconditional_conditioning=unconditional_conditioning)
        return x_dec