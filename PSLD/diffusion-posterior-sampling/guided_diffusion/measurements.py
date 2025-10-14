'''This module handles task-dependent operations (A) and noises (n) to simulate a measurement y=Ax+n.'''

from abc import ABC, abstractmethod
from functools import partial
import yaml
import numpy as np
from PIL import Image
from torch.nn import functional as F
import torch
from motionblur.motionblur import Kernel

from util.resizer import Resizer
from util.img_utils import Blurkernel, fft2_m
# For PIL images, convert to tensor first
import torchvision.transforms as T




# =================
# Operation classes
# =================

__OPERATOR__ = {}

def register_operator(name: str):
    def wrapper(cls):
        if __OPERATOR__.get(name, None):
            raise NameError(f"Name {name} is already registered!")
        __OPERATOR__[name] = cls
        return cls
    return wrapper


def get_operator(name: str, **kwargs):
    if __OPERATOR__.get(name, None) is None:
        raise NameError(f"Name {name} is not defined.")
    return __OPERATOR__[name](**kwargs)


class LinearOperator(ABC):
    @abstractmethod
    def forward(self, data, **kwargs):
        # calculate A * X
        pass

    @abstractmethod
    def transpose(self, data, **kwargs):
        # calculate A^T * X
        pass
    
    def ortho_project(self, data, **kwargs):
        # calculate (I - A^T * A)X
        return data - self.transpose(self.forward(data, **kwargs), **kwargs)

    def project(self, data, measurement, **kwargs):
        # calculate (I - A^T * A)Y - AX
        return self.ortho_project(measurement, **kwargs) - self.forward(data, **kwargs)



@register_operator(name='noise')
class DenoiseOperator(LinearOperator):
    def __init__(self, device):
        self.device = device
    
    def forward(self, data):
        return data

    def transpose(self, data):
        return data
    
    def ortho_project(self, data):
        return data

    def project(self, data):
        return data


@register_operator(name='super_resolution')
class SuperResolutionOperator(LinearOperator):
    def __init__(self, in_shape, scale_factor, device):
        self.device = device
        self.up_sample = partial(F.interpolate, scale_factor=scale_factor)
        self.down_sample = Resizer(in_shape, 1/scale_factor).to(device)

    def forward(self, data, **kwargs):
        return self.down_sample(data)

    def transpose(self, data, **kwargs):
        return self.up_sample(data)

    def project(self, data, measurement, **kwargs):
        return data - self.transpose(self.forward(data)) + self.transpose(measurement)

@register_operator(name='motion_blur')
class MotionBlurOperator(LinearOperator):
    def __init__(self, kernel_size, intensity, device):
        self.device = device
        self.kernel_size = kernel_size
        self.conv = Blurkernel(blur_type='motion',
                               kernel_size=kernel_size,
                               std=intensity,
                               device=device).to(device)  # should we keep this device term?

        self.kernel = Kernel(size=(kernel_size, kernel_size), intensity=intensity)
        kernel = torch.tensor(self.kernel.kernelMatrix, dtype=torch.float32)
        self.conv.update_weights(kernel)
    
    def forward(self, data, **kwargs):
        # A^T * A 
        return self.conv(data)

    def transpose(self, data, **kwargs):
        return data

    def get_kernel(self):
        kernel = self.kernel.kernelMatrix.type(torch.float32).to(self.device)
        return kernel.view(1, 1, self.kernel_size, self.kernel_size)


@register_operator(name='gaussian_blur')
class GaussialBlurOperator(LinearOperator):
    def __init__(self, kernel_size, intensity, device):
        self.device = device
        self.kernel_size = kernel_size
        self.conv = Blurkernel(blur_type='gaussian',
                               kernel_size=kernel_size,
                               std=intensity,
                               device=device).to(device)
        self.kernel = self.conv.get_kernel()
        self.conv.update_weights(self.kernel.type(torch.float32))

    def forward(self, data, **kwargs):
        return self.conv(data)

    def transpose(self, data, **kwargs):
        return data

    def get_kernel(self):
        return self.kernel.view(1, 1, self.kernel_size, self.kernel_size)

@register_operator(name='inpainting')
class InpaintingOperator(LinearOperator):
    '''This operator get pre-defined mask and return masked image.'''
    def __init__(self, device):
        self.device = device
    
    def forward(self, data, **kwargs):
        try:
            return data * kwargs.get('mask', None).to(self.device)
        except:
            raise ValueError("Require mask")
    
    def transpose(self, data, **kwargs):
        return data
    
    def ortho_project(self, data, **kwargs):
        return data - self.forward(data, **kwargs)


class NonLinearOperator(ABC):
    @abstractmethod
    def forward(self, data, **kwargs):
        pass

    def project(self, data, measurement, **kwargs):
        return data + measurement - self.forward(data) 

@register_operator(name='style_retrieval')
class StyleOperator(NonLinearOperator):
    def __init__(self, device=None):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available()
            else "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
            else "cpu")
        else:
            self.device = device

        from transformers import AutoImageProcessor, AutoModel
        self.processor = AutoImageProcessor.from_pretrained("facebook/dinov2-large")
        self.model = AutoModel.from_pretrained("facebook/dinov2-large")
        self.model = self.model.eval().to(self.device)

    @staticmethod
    def gram(tokens, offdiag_only=True):
        # tokens: [B, N, C]
        B, N, C = tokens.shape
        X = tokens / (N ** 0.5)
        G = X.transpose(1, 2) @ X     # [B, C, C]
        if offdiag_only:
            G = G - torch.diag_embed(torch.diagonal(G, dim1=1, dim2=2))
        return G.reshape(B, -1)     

    
    def style_vec(self, img_tensor, layers=[-1], use_adain=False):
        """
        pil_img -> normalized style vector from multiple hidden states.
        `layers` are indices into hidden_states (negative = from the end).
        """
        # 1. Ensure batch dimension
        if img_tensor.dim() == 3:
            img_tensor = img_tensor.unsqueeze(0)  # Add batch dimension
        
        # 2. Differentiable Resizing
        # DINOv2 expects a 224x224 input.
        # The image tensor is in the range [0, 1] after the clamp operation.
        # Use F.interpolate instead of T.Resize to avoid MPS issues
        if img_tensor.shape[-1] != 224 or img_tensor.shape[-2] != 224:
            img_tensor = F.interpolate(img_tensor, size=(224, 224), mode='bilinear', align_corners=False)

        # 3. Differentiable Normalization
        # DINOv2's mean and std
        norm_mean = torch.tensor([0.485, 0.456, 0.406], device=img_tensor.device).view(1, 3, 1, 1)
        norm_std = torch.tensor([0.229, 0.224, 0.225], device=img_tensor.device).view(1, 3, 1, 1)
        
        # Apply normalization to the tensor
        inputs = (img_tensor - norm_mean) / norm_std

        # 4. Forward pass through DINOv2
        if inputs.requires_grad:
            out = self.model(inputs, output_hidden_states=True)
        else:
            with torch.no_grad():
                out = self.model(inputs, output_hidden_states=True)
        hs_list = [out.hidden_states[i] for i in layers]   # each: [B, N+1, C]
        parts = []
        for hs in hs_list:
            tok = hs[:, 1:, :]                             # drop CLS
            if use_adain:
                # Simple AdaIN implementation
                mean = tok.mean(dim=1, keepdim=True)
                std = tok.std(dim=1, keepdim=True) + 1e-8
                part = (tok - mean) / std
                part = part.reshape(part.shape[0], -1)
            else:
                part = StyleOperator.gram(tok, offdiag_only=True)
            parts.append(part)
        v = torch.cat(parts, dim=1)
        return F.normalize(v, dim=1)   

        

    def forward(self, data, **kwargs):
        # For differentiable operations, we need to handle tensors directly
        if torch.is_tensor(data):
            # Convert from [-1, 1] to [0, 1] range without reshaping
            data = data.add(1.0).div(2.0).clamp(0.0, 1.0)
            
            # Use a differentiable style extraction method
            style_vec = self.style_vec(data, **kwargs)
            return style_vec
        else:
            # For non-tensor inputs (PIL images), use the original method
            style_vec = self.style_vec(data, **kwargs)
            return style_vec
    


    def transpose(self, data, **kwargs):
        """
        For style operators, the transpose is the same as forward since we're dealing with style vectors.
        This is a simplified implementation - in practice, style operators are typically non-linear.
        """
        return self.forward(data, **kwargs)

    def ortho_project(self, data, **kwargs):
        """
        Orthogonal projection: (I - A^T * A)X
        For style operators, this projects out the style component.
        """
        return data - self.transpose(self.forward(data, **kwargs))

    def project(self, data, measurement, **kwargs):
        """
        Projection: (I - A^T * A)Y - AX
        For style operators, this projects the measurement onto the orthogonal space.
        """
        return self.ortho_project(measurement, **kwargs) - self.forward(data, **kwargs)    
    


@register_operator(name='phase_retrieval')
class PhaseRetrievalOperator(NonLinearOperator):
    def __init__(self, oversample, device):
        self.pad = int((oversample / 8.0) * 256)
        self.device = device
        
    def forward(self, data, **kwargs):
        padded = F.pad(data, (self.pad, self.pad, self.pad, self.pad))
        amplitude = fft2_m(padded).abs()
        return amplitude

@register_operator(name='nonlinear_blur')
class NonlinearBlurOperator(NonLinearOperator):
    def __init__(self, opt_yml_path, device):
        self.device = device
        self.blur_model = self.prepare_nonlinear_blur_model(opt_yml_path)     
         
    def prepare_nonlinear_blur_model(self, opt_yml_path):
        '''
        Nonlinear deblur requires external codes (bkse).
        '''
        from bkse.models.kernel_encoding.kernel_wizard import KernelWizard

        with open(opt_yml_path, "r") as f:
            opt = yaml.safe_load(f)["KernelWizard"]
            model_path = opt["pretrained"]
        blur_model = KernelWizard(opt)
        blur_model.eval()
        blur_model.load_state_dict(torch.load(model_path)) 
        blur_model = blur_model.to(self.device)
        return blur_model
        
    
    def forward(self, data, **kwargs):
        random_kernel = torch.randn(1, 512, 2, 2).to(self.device) * 1.2
        data = (data + 1.0) / 2.0  #[-1, 1] -> [0, 1]
        blurred = self.blur_model.adaptKernel(data, kernel=random_kernel)
        blurred = (blurred * 2.0 - 1.0).clamp(-1, 1) #[0, 1] -> [-1, 1]
        return blurred

# =============
# Noise classes
# =============


__NOISE__ = {}

def register_noise(name: str):
    def wrapper(cls):
        if __NOISE__.get(name, None):
            raise NameError(f"Name {name} is already defined!")
        __NOISE__[name] = cls
        return cls
    return wrapper

def get_noise(name: str, **kwargs):
    if __NOISE__.get(name, None) is None:
        raise NameError(f"Name {name} is not defined.")
    noiser = __NOISE__[name](**kwargs)
    noiser.__name__ = name
    return noiser

class Noise(ABC):
    def __call__(self, data):
        return self.forward(data)
    
    @abstractmethod
    def forward(self, data):
        pass

@register_noise(name='clean')
class Clean(Noise):
    def forward(self, data):
        return data

@register_noise(name='gaussian')
class GaussianNoise(Noise):
    def __init__(self, sigma):
        self.sigma = sigma
    
    def forward(self, data):
        return data + torch.randn_like(data, device=data.device) * self.sigma


@register_noise(name='poisson')
class PoissonNoise(Noise):
    def __init__(self, rate):
        self.rate = rate

    def forward(self, data):
        '''
        Follow skimage.util.random_noise.
        '''

        # TODO: set one version of poisson
       
        # version 3 (stack-overflow)
        import numpy as np
        data = (data + 1.0) / 2.0
        data = data.clamp(0, 1)
        device = data.device
        data = data.detach().cpu()
        data = torch.from_numpy(np.random.poisson(data * 255.0 * self.rate) / 255.0 / self.rate)
        data = data * 2.0 - 1.0
        data = data.clamp(-1, 1)
        return data.to(device)

        # version 2 (skimage)
        # if data.min() < 0:
        #     low_clip = -1
        # else:
        #     low_clip = 0

    
        # # Determine unique values in iamge & calculate the next power of two
        # vals = torch.Tensor([len(torch.unique(data))])
        # vals = 2 ** torch.ceil(torch.log2(vals))
        # vals = vals.to(data.device)

        # if low_clip == -1:
        #     old_max = data.max()
        #     data = (data + 1.0) / (old_max + 1.0)

        # data = torch.poisson(data * vals) / float(vals)

        # if low_clip == -1:
        #     data = data * (old_max + 1.0) - 1.0
       
        # return data.clamp(low_clip, 1.0)
@register_operator(name='clip_style_retrieval')
class CLIPStyleOperator(NonLinearOperator):
    def __init__(self, device=None):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available()
            else "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
            else "cpu")
        else:
            self.device = device

        # Import CLIP model and processor
        import clip
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)
        # DON'T put the full model in eval mode - we need gradients to flow
        # self.model = self.model.eval()
        
        # Get the image encoder part
        self.image_encoder = self.model.visual
        
        # Convert to float32 to avoid MPS type mismatches
        self.image_encoder = self.image_encoder.float()
        
        # Freeze the model parameters BUT allow gradients to flow through
        for param in self.image_encoder.parameters():
            param.requires_grad = False
            
        # CRITICAL: Set model to training mode to allow gradients to flow
        # Even though parameters are frozen, gradients need to flow for style loss
        self.image_encoder.train()

    @staticmethod
    def gram(tokens, offdiag_only=True):
        # tokens: [B, N, C]
        B, N, C = tokens.shape
        X = tokens / (N ** 0.5)
        G = X.transpose(1, 2) @ X     # [B, C, C]
        if offdiag_only:
            G = G - torch.diag_embed(torch.diagonal(G, dim1=1, dim2=2))
        return G.reshape(B, -1)     

    def style_vec(self, img_tensor, layers=[-1], use_adain=False):
        """
        Extract style features using CLIP's image encoder.
        `layers` are indices into transformer layers (negative = from the end).
        """
        # 1. Differentiable Resizing and Preprocessing
        # CLIP expects 224x224 input
        if img_tensor.shape[-1] != 224 or img_tensor.shape[-2] != 224:
            if img_tensor.dim() == 3:
                img_tensor = img_tensor.unsqueeze(0)  # Add batch dimension
            img_tensor = F.interpolate(img_tensor, size=(224, 224), mode='bilinear', align_corners=False)
            if img_tensor.shape[0] == 1:
                img_tensor = img_tensor.squeeze(0)  # Remove batch dimension if it was added

        # 2. CLIP Normalization
        # CLIP uses ImageNet normalization
        norm_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=img_tensor.device).view(1, 3, 1, 1)
        norm_std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=img_tensor.device).view(1, 3, 1, 1)
        
        # Apply normalization
        inputs = (img_tensor - norm_mean) / norm_std
        
        # Ensure inputs are float32 to match CLIP model
        inputs = inputs.float()

        # Ensure we have batch dimension
        if inputs.dim() == 3:
            inputs = inputs.unsqueeze(0)  # [C, H, W] -> [1, C, H, W]
            
        # CRITICAL: Explicitly enable gradients on the input tensor
        if img_tensor.requires_grad:
            inputs = inputs.requires_grad_(True)
            print(f"Debug CLIP - Explicitly enabled gradients on inputs: {inputs.requires_grad}")

        # Extract features from CLIP - ALWAYS preserve gradients for optimization
        # Even though CLIP parameters are frozen, we need gradients to flow through for style loss
        
        # Ensure CLIP encoder is in training mode if we need gradients
        if inputs.requires_grad:
            self.image_encoder.train()
            print(f"Debug CLIP - image_encoder.training: {self.image_encoder.training}")
        
        features = self.image_encoder(inputs)
        
        # Debug gradient flow
        print(f"Debug CLIP - inputs.requires_grad: {inputs.requires_grad}")
        print(f"Debug CLIP - features.requires_grad: {features.requires_grad}")
        
        # For CLIP, we'll use the final output features
        # CLIP doesn't provide intermediate layer access easily
        if features.dim() == 3:
            # If we get [B, N, C] features
            if use_adain:
                # AdaIN normalization
                mean = features.mean(dim=1, keepdim=True)
                std = features.std(dim=1, keepdim=True) + 1e-8
                part = (features - mean) / std
                part = part.reshape(part.shape[0], -1)
            else:
                # Gram matrix
                part = CLIPStyleOperator.gram(features, offdiag_only=True)
        else:
            # If features are 2D [B, C], flatten them
            part = features.reshape(features.shape[0], -1)

        # Return normalized features with gradients preserved
        normalized_part = F.normalize(part, dim=1)
        
        # Debug final output
        print(f"Debug CLIP - normalized_part.requires_grad: {normalized_part.requires_grad}")
        
        # FAILSAFE: If gradients are lost, try to restore them
        if not normalized_part.requires_grad and inputs.requires_grad:
            print("WARNING: Gradients lost in CLIP - attempting to restore")
            normalized_part = normalized_part.requires_grad_(True)
            
        return normalized_part

    def forward(self, data, **kwargs):
        if torch.is_tensor(data):
            # Handle tensors: convert from [-1, 1] to [0, 1] range
            data = data.add(1.0).div(2.0).clamp(0.0, 1.0)
            
            # Ensure we have batch dimension
            if data.dim() == 3:
                data = data.unsqueeze(0)
                
            # Extract style features directly from tensor
            style_vec = self.style_vec(data, **kwargs)
            return style_vec
        else:
            # Handle PIL images: apply transform pipeline
            transform = T.Compose([
                T.Resize((224, 224)),
                T.ToTensor(),
                T.Normalize([0.48145466, 0.4578275, 0.40821073], 
                           [0.26862954, 0.26130258, 0.27577711])
            ])
            img_tensor = transform(data).unsqueeze(0).to(self.device)
            style_vec = self.style_vec(img_tensor, **kwargs)
            return style_vec

    def transpose(self, data, **kwargs):
        """
        For style operators, the transpose is the same as forward.
        """
        return self.forward(data, **kwargs)

    def ortho_project(self, data, **kwargs):
        """
        Orthogonal projection: (I - A^T * A)X
        """
        return data - self.transpose(self.forward(data, **kwargs))

    def project(self, data, measurement, **kwargs):
        """
        Projection: (I - A^T * A)Y - AX
        """
        return self.ortho_project(measurement, **kwargs) - self.forward(data, **kwargs)