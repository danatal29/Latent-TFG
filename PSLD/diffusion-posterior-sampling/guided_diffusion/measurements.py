'''This module handles task-dependent operations (A) and noises (n) to simulate a measurement y=Ax+n.'''

from abc import ABC, abstractmethod
from functools import partial
import yaml
import numpy as np
from PIL import Image
from torch.nn import functional as F
from torchvision import torch
from motionblur.motionblur import Kernel

from util.resizer import Resizer
from util.img_utils import Blurkernel, fft2_m



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

    
    def style_vec(self, pil_img, layers=[-1], use_adain=False):
        """
        pil_img -> normalized style vector from multiple hidden states.
        `layers` are indices into hidden_states (negative = from the end).
        """
        inputs = self.processor(images=pil_img, return_tensors="pt").to(self.device)
        if pil_img.requires_grad:
            out = self.model(**inputs, output_hidden_states=True)
        else:
            with torch.no_grad():
                out = self.model(**inputs, output_hidden_states=True)
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

            #data = torch.clamp((data + 1.0) / 2.0, min=0.0, max=1.0)

            # Convert from [-1, 1] to [0, 1] range without reshaping
            data = data.add(1.0).div(2.0).clamp(0.0, 1.0)
            
            # For differentiable style extraction, we'll use a simplified approach
            # that maintains the computational graph
            if data.dim() == 4:  # Batch dimension
                data = data[0]  # Take first image if batch
            
            # Use a differentiable style extraction method
            style_vec = self.style_vec(data, **kwargs)
            return style_vec
        else:
            # For non-tensor inputs (PIL images), use the original method
            style_vec = self.style_vec(data, **kwargs)
            return style_vec
    
    def differentiable_style_vec(self, tensor_img, layers=[-1], use_adain=False):
        """
        Differentiable version of style extraction that maintains computational graph.
        This uses a simplified approach that works with tensors directly.
        """
        # Normalize the image for processing
        if tensor_img.min() < 0 or tensor_img.max() > 1:
            tensor_img = torch.clamp(tensor_img, 0, 1)
        
        # Resize to expected input size if needed (DINOv2 expects 224x224)
        if tensor_img.shape[-1] != 224 or tensor_img.shape[-2] != 224:
            tensor_img = F.interpolate(tensor_img.unsqueeze(0), size=(224, 224), mode='bilinear', align_corners=False).squeeze(0)
        
        # Convert tensor to PIL for DINOv2 processing
        # We need to do this carefully to maintain gradients
        tensor_img_np = tensor_img.permute(1, 2, 0).detach().cpu().numpy()
        pil_img = Image.fromarray((tensor_img_np * 255).astype(np.uint8))
        
        # Process with DINOv2 model
        inputs = self.processor(images=pil_img, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model(**inputs, output_hidden_states=True)
        
        # Extract features from specified layers
        hs_list = [out.hidden_states[i] for i in layers]
        parts = []
        for hs in hs_list:
            tok = hs[:, 1:, :]  # drop CLS
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