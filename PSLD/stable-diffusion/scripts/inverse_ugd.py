import argparse, os, sys, glob
import cv2
import torch
import numpy as np
import torch.nn.functional as F
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
from imwatermark import WatermarkEncoder
from itertools import islice
from einops import rearrange
from torchvision.utils import make_grid
import time
from pytorch_lightning import seed_everything
from torch import autocast
from contextlib import contextmanager, nullcontext

from ldm.util import instantiate_from_config
from ldm.models.diffusion.psld import DDIMSampler
from ldm.models.diffusion.ddim_ugd import UGDDDIMSampler  # UGD-enhanced sampler
from ldm.models.diffusion.psld_ugd import PSLDUGDSampler  # Unified PSLD+UGD sampler
from ldm.models.diffusion.plms import PLMSSampler
from ldm.models.diffusion.dpm_solver import DPMSolverSampler
from ldm.guidance.api import GuidanceConfig, GuidanceFn

# from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from transformers import AutoFeatureExtractor
import pytorch_lightning.callbacks.model_checkpoint
import torch.serialization
torch.serialization.add_safe_globals([pytorch_lightning.callbacks.model_checkpoint.ModelCheckpoint])


# load safety model
safety_model_id = "CompVis/stable-diffusion-safety-checker"
safety_feature_extractor = AutoFeatureExtractor.from_pretrained(safety_model_id)
# safety_checker = StableDiffusionSafetyChecker.from_pretrained(safety_model_id)

def amp_context(device, use_amp):
    if not use_amp:
        return nullcontext()
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    if device.type == "cpu":
        return torch.autocast(device_type="cpu", dtype=torch.bfloat16)
    # MPS: keep fp32 for stability
    return nullcontext()

def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())


def numpy_to_pil(images):
    """
    Convert a numpy image or a batch of images to a PIL image.
    """
    if images.ndim == 3:
        images = images[None, ...]
    images = (images * 255).round().astype("uint8")
    pil_images = [Image.fromarray(image) for image in images]

    return pil_images

def _get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def load_model_from_config(config, ckpt, verbose=False):
    device = _get_device()
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu", weights_only=False)
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.to(device)
    model.eval()
    return model


def put_watermark(img, wm_encoder=None):
    if wm_encoder is not None:
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        img = wm_encoder.encode(img, 'dwtDct')
        img = Image.fromarray(img[:, :, ::-1])
    return img


def load_replacement(x):
    try:
        hwc = x.shape
        y = Image.open("assets/rick.jpeg").convert("RGB").resize((hwc[1], hwc[0]))
        y = (np.array(y)/255.0).astype(x.dtype)
        assert y.shape == x.shape
        return y
    except Exception:
        return x


def check_safety(x_image):
    safety_checker_input = safety_feature_extractor(numpy_to_pil(x_image), return_tensors="pt")
    # x_checked_image, has_nsfw_concept = safety_checker(images=x_image, clip_input=safety_checker_input.pixel_values)
    x_checked_image, has_nsfw_concept = x_image, False
    for i in range(len(has_nsfw_concept)):
        if has_nsfw_concept[i]:
            x_checked_image[i] = load_replacement(x_checked_image[i])
    return x_checked_image, has_nsfw_concept


def create_style_guidance_function(style_image_path, device, operator):
    """
    Create UGD style guidance function using PSLD's existing operator.
    
    Args:
        style_image_path: Path to style reference image
        device: torch device
        operator: PSLD operator from config (reused for consistency)
    
    Returns:
        style_guidance_fn: Function that computes style loss for UGD
    """
    if not style_image_path:
        print(f"⚠️  No style image path provided")
        return None
    
    if operator is None:
        print(f"⚠️  No operator provided - cannot create guidance")
        return None
        
    # Try to resolve the path relative to current working directory
    if not os.path.exists(style_image_path):
        # Try different path resolutions
        alt_paths = [
            os.path.abspath(style_image_path),
            os.path.join(os.getcwd(), style_image_path),
            style_image_path.replace('../../../', '../../'),
            style_image_path.replace('../../', '../../../')
        ]
        
        resolved_path = None
        for path in alt_paths:
            if os.path.exists(path):
                resolved_path = path
                break
                
        if resolved_path is None:
            print(f"⚠️  Style image not found at any of these paths:")
            for path in [style_image_path] + alt_paths:
                print(f"     {path}")
            return None
        else:
            style_image_path = resolved_path
            print(f"✅ Found style image at: {style_image_path}")
    
    try:
        print(f"🎨 Loading style guidance from: {style_image_path}")
        print(f"🎨 Using PSLD operator: {operator.__class__.__name__}")
        
        # Load and preprocess target style image using torch operations for consistency
        style_img = Image.open(style_image_path).convert('RGB')
        
        # Convert to tensor [-1, 1] range to match PSLD expectations
        import torchvision.transforms.functional as TF
        style_tensor = TF.to_tensor(style_img).to(device)
        style_tensor = style_tensor * 2.0 - 1.0  # [0,1] -> [-1,1]
        style_tensor = style_tensor.unsqueeze(0)  # Add batch dim
        
        # Resize to proper dimensions if needed
        if style_tensor.shape[-1] != 512 or style_tensor.shape[-2] != 512:
            style_tensor = torch.nn.functional.interpolate(
                style_tensor, size=(512, 512), mode='bilinear', align_corners=False
            )
        
        # Extract target style features using PSLD's operator
        with torch.no_grad():
            target_features = operator.forward(style_tensor)
            print(f"✅ Extracted target style features: shape={target_features.shape}")
        
        def style_guidance_fn(pred_img, **kwargs):
            """
            UGD style guidance function using PSLD's operator.
            
            Args:
                pred_img: Predicted image in [-1, 1] range (from decoded latent)
            
            Returns:
                style_loss: Scalar loss for gradient computation
            """
            try:
                # Extract style from predicted image using PSLD operator
                # pred_img should already be in [-1, 1] range from decoder
                pred_features = operator.forward(pred_img)
                
                # Compute style loss using cosine similarity (same as PSLD)
                
                
                # Normalize features for stable cosine similarity
                pred_norm = F.normalize(pred_features, p=2, dim=-1)
                target_norm = F.normalize(target_features, p=2, dim=-1)
                
                # Cosine similarity loss: minimize 1 - cosine_similarity
                cosine_sim = F.cosine_similarity(pred_norm, target_norm, dim=-1)
                style_loss = (1.0 - cosine_sim).mean()
                
                return style_loss
                
            except Exception as e:
                print(f"⚠️  Style guidance error: {e}")
                import traceback
                traceback.print_exc()
                return torch.tensor(0.0, device=pred_img.device, requires_grad=True)
        
        print(f"✅ Style guidance function created successfully")
        return style_guidance_fn  # Return only the guidance function
        
    except Exception as e:
        print(f"❌ Failed to create style guidance: {e}")
        import traceback
        traceback.print_exc()
        return None


def create_ugd_guidance_config(opt, operator):
    """
    Create UGD guidance configuration from CLI arguments.
    
    Args:
        opt: Command line options
        operator: PSLD operator from config (reused for style extraction)
    
    Returns:
        guidance_cfg: GuidanceConfig object or None
        guidance_fn: Guidance function or None
    """
    if not opt.optim_forward_guidance:
        print("📊 UGD guidance disabled - using standard PSLD")
        return None, None
        
    print("🎯 Attempting to create UGD guidance configuration...")
        
    # Create guidance function based on type
    guidance_fn = None
    device = _get_device()
    
    if opt.style_image:
        print(f"🎨 Creating style guidance for: {opt.style_image}")
        guidance_fn = create_style_guidance_function(opt.style_image, device, operator)
    else:
        print("⚠️  UGD guidance enabled but no style_image specified")
        print("📊 Falling back to standard PSLD")
        return None, None
        
    if guidance_fn is None:
        print("❌ Failed to create guidance function")
        print("📊 Falling back to standard PSLD")
        return None, None
    
    # Create guidance config only after function is successfully created
    guidance_cfg = GuidanceConfig(
        enabled=opt.optim_forward_guidance,
        domain=opt.guidance_domain,
        num_steps=opt.optim_num_steps,
        step_wt=opt.optim_forward_guidance_wt,
        k_recur=opt.k_recur,
        normalize_grad=opt.normalize_grad,
        decode_kwargs={'clamp': (-1, 1)} if opt.guidance_domain == "image" else None
    )
        
    print(f"✅ UGD Guidance configured successfully:")
    print(f"  - Domain: {guidance_cfg.domain}")
    print(f"  - Steps: {guidance_cfg.num_steps}")
    print(f"  - Weight: {guidance_cfg.step_wt}")
    print(f"  - Self-recurrence: {guidance_cfg.k_recur}")
    print(f"  - Normalize gradients: {guidance_cfg.normalize_grad}")
    
    return guidance_cfg, guidance_fn


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--prompt",
        type=str,
        nargs="?",
        default="",
        help="the prompt to render"
    )
    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        help="dir to write results to",
        default="outputs/txt2img-samples"
    )
    parser.add_argument(
        "--skip_grid",
        action='store_false',
        help="do not save a grid, only individual samples. Helpful when evaluating lots of samples",
    )
    parser.add_argument(
        "--skip_save",
        action='store_true',
        help="do not save individual samples. For speed measurements.",
    )
    parser.add_argument(
        "--ddim_steps",
        type=int,
        default=1000,
        help="number of ddim sampling steps",
    )
    parser.add_argument(
        "--plms",
        action='store_true',
        help="use plms sampling",
    )
    parser.add_argument(
        "--dpm_solver",
        action='store_true',
        help="use dpm_solver sampling",
    )
    parser.add_argument(
        "--laion400m",
        action='store_true',
        help="uses the LAION400M model",
    )
    parser.add_argument(
        "--fixed_code",
        action='store_true',
        help="if enabled, uses the same starting code across samples ",
    )
    parser.add_argument(
        "--ddim_eta",
        type=float,
        default=0.0,
        help="ddim eta (eta=0.0 corresponds to deterministic sampling",
    )
    parser.add_argument(
        "--n_iter",
        type=int,
        default=1,
        help="sample this often",
    )
    parser.add_argument(
        "--H",
        type=int,
        default=256,
        help="image height, in pixel space",
    )
    parser.add_argument(
        "--W",
        type=int,
        default=256,
        help="image width, in pixel space",
    )
    parser.add_argument(
        "--C",
        type=int,
        default=4,
        help="latent channels",
    )
    parser.add_argument(
        "--f",
        type=int,
        default=8,
        help="downsampling factor",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=1,
        help="how many samples to produce for each given prompt. A.k.a. batch size",
    )
    parser.add_argument(
        "--n_rows",
        type=int,
        default=0,
        help="rows in the grid (default: n_samples)",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=7.5,
        help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
    )
    parser.add_argument(
        "--from-file",
        type=str,
        help="if specified, load prompts from this file",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/stable-diffusion/v1-inference.yaml",
        help="path to config which constructs model",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="models/ldm/stable-diffusion-v1/model.ckpt",
        help="path to checkpoint of model",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="the seed (for reproducible sampling)",
    )
    parser.add_argument(
        "--precision",
        type=str,
        help="evaluate at this precision",
        choices=["full", "autocast"],
        default="autocast"
    )
    ## 
    parser.add_argument(
        "--dps_path",
        type=str,
        default='../diffusion-posterior-sampling/',
        help="DPS codebase path",
    )
    parser.add_argument(
        "--task_config",
        type=str,
        default='configs/inpainting_config.yaml',
        help="task config yml file",
    )
    parser.add_argument(
        "--diffusion_config",
        type=str,
        default='configs/diffusion_config.yaml',
        help="diffusion config yml file",
    )
    parser.add_argument(
        "--model_config",
        type=str,
        default='configs/model_config.yaml',
        help="model config yml file",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=1e-1,
        help="inpainting error",
    )
    parser.add_argument(
        "--omega",
        type=float,
        default=1,
        help="measurement error",
    )
    parser.add_argument(
        "--inpainting",
        type=int,
        default=0,
        help="inpainting",
    )
    parser.add_argument(
        "--general_inverse",
        type=int,
        default=1,
        help="general inverse",
    )
    parser.add_argument(
        "--file_id",
        type=str,
        default='00014.png',
        help='input image',
    )
    parser.add_argument(
        "--skip_low_res",
        action='store_true',
        help='downsample result to 256',
    )
    parser.add_argument(
        "--ffhq256",
        action='store_true',
        help='load SD weights trained on FFHQ',
    )
    
    # =====================================
    # UGD GUIDANCE CLI FLAGS (Step 3)
    # =====================================
    parser.add_argument(
        "--optim_forward_guidance",
        action='store_true',
        help="Enable UGD forward guidance (equivalent to UGD's --optim_forward_guidance)"
    )
    parser.add_argument(
        "--optim_num_steps",
        type=int,
        default=5,
        help="Number of UGD inner optimization steps (equivalent to UGD's --optim_num_steps)"
    )
    parser.add_argument(
        "--optim_forward_guidance_wt",
        type=float,
        default=5.0,
        help="UGD guidance weight (equivalent to UGD's --optim_forward_guidance_wt)"
    )
    parser.add_argument(
        "--guidance_domain",
        type=str,
        choices=["latent", "image"],
        default="image",
        help="Domain for guidance computation: 'latent' for z-space, 'image' for decoded images"
    )
    parser.add_argument(
        "--k_recur",
        type=int,
        default=1,
        help="Number of self-recurrence iterations per timestep (default: 1, no recurrence)"
    )
    parser.add_argument(
        "--normalize_grad",
        action='store_true',
        default=True,
        help="Normalize gradients during optimization for stability (default: True)"
    )
    parser.add_argument(
        "--no_normalize_grad",
        action='store_false',
        dest='normalize_grad',
        help="Disable gradient normalization (may cause instability)"
    )
    parser.add_argument(
        "--style_image",
        type=str,
        help="Path to style reference image for style transfer guidance"
    )
    parser.add_argument(
        "--guidance_weight",
        type=float,
        default=1.0,
        help="Weight for guidance loss vs measurement consistency loss"
    )
    parser.add_argument(
        "--measurement_weight", 
        type=float,
        default=1.0,
        help="Weight for PSLD measurement consistency loss"
    )
    parser.add_argument(
        "--use_hybrid_sampler",
        action='store_true',
        help="Use unified PSLD+UGD sampler that combines both methods for superior style transfer"
    )
    
    # =====================================
    # UNIFIED SAMPLER CLI FLAGS
    # =====================================
    parser.add_argument(
        "--use_unified_sampler",
        action='store_true',
        help="Use unified PSLD+UGD sampler with alternating timesteps"
    )
    parser.add_argument(
        "--schedule_mode",
        type=str,
        default="pattern",
        choices=["pattern", "early_late", "custom"],
        help="Method scheduling mode: pattern (alternating), early_late (split), or custom"
    )
    parser.add_argument(
        "--pattern_type",
        type=str,
        default="even_odd",
        choices=["even_odd", "odd_even"],
        help="Pattern type for alternating: even_odd (PSLD on even, UGD on odd) or odd_even"
    )
    parser.add_argument(
        "--split_timestep",
        type=int,
        default=None,
        help="For early_late mode: timestep index to switch from one method to another"
    )
    parser.add_argument(
        "--psld_weight",
        type=float,
        default=1.0,
        help="Weight for PSLD gradient contribution (scales PSLD gradient)"
    )
    parser.add_argument(
        "--ugd_weight",
        type=float,
        default=1.0,
        help="Weight for UGD gradient contribution (scales UGD gradient)"
    )
    ##

    opt = parser.parse_args()
    # pdb.set_trace()

    if opt.laion400m:
        print("Falling back to LAION 400M model...")
        opt.config = "configs/latent-diffusion/txt2img-1p4B-eval.yaml"
        opt.ckpt = "models/ldm/text2img-large/model.ckpt"
        
    ## 
    if opt.ffhq256:
        print("Using FFHQ 256 finetuned model...")
        opt.config = "models/ldm/ffhq256/config.yaml"
        opt.ckpt = "models/ldm/ffhq256/model.ckpt"
    ##
    
    seed_everything(opt.seed)

    config = OmegaConf.load(f"{opt.config}")
    model = load_model_from_config(config, f"{opt.ckpt}")

    device = _get_device()
    model = model.to(device)

    #pdb.set_trace()

    if device.type != "cuda":
        model.float()  # MPS/CPU stay fp32

    # ---- ensure CLIP is on the same device and dtype ----
    if hasattr(model, 'cond_stage_model') and hasattr(model.cond_stage_model, 'transformer'):
        model.cond_stage_model.transformer.to(device)
        if device.type != "cuda":
            model.cond_stage_model.transformer.float()
    
    # ---- ensure text encoder device is set correctly ----
    if hasattr(model, 'cond_stage_model'):
        # Set device for the conditioning model
        model.cond_stage_model.device = device
        model.cond_stage_model.to(device)
        
        # For FrozenCLIPEmbedder specifically, update device attribute
        if hasattr(model.cond_stage_model, 'device'):
            model.cond_stage_model.device = device
        
        # Note: Tokenizers don't need to be moved to device (they're not PyTorch modules)
            
        # Ensure transformer is on correct device too
        if hasattr(model.cond_stage_model, 'transformer'):
            model.cond_stage_model.transformer.to(device)
            
        print(f"✅ Text encoder moved to {device}")

    # Setup PSLD components FIRST - need operator before creating UGD guidance
    sys.path.append(opt.dps_path)

    import yaml
    from guided_diffusion.measurements import get_noise, get_operator
    from util.img_utils import clear_color, mask_generator
    import torch.nn.functional as f
    import matplotlib.pyplot as plt

    def load_yaml(file_path: str) -> dict:
        with open(file_path) as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        return config

    model_config=opt.dps_path+opt.model_config
    diffusion_config=opt.dps_path+opt.diffusion_config
    task_config=opt.dps_path+opt.task_config

    # Load configurations
    model_config = load_yaml(model_config)
    diffusion_config = load_yaml(diffusion_config)
    task_config = load_yaml(task_config)
    
    # Only set mask_opt image_size if mask_opt exists (for inpainting tasks)
    if 'mask_opt' in task_config['measurement']:
        task_config['measurement']['mask_opt']['image_size']=opt.H
    
    # Prepare Operator and noise
    measure_config = task_config['measurement']
    operator = get_operator(device=device, **measure_config['operator'])
    noiser = get_noise(**measure_config['noise'])
    
    print(f"✅ Loaded PSLD operator: {operator.__class__.__name__}")

    # Create UGD guidance configuration using PSLD's operator
    guidance_cfg, guidance_fn = create_ugd_guidance_config(opt, operator)

    # Create sampler based on mode
    if opt.use_unified_sampler:
        from ldm.models.diffusion.ugd_psld_sampler import UnifiedPSLDUGDSampler
        from ldm.guidance.api import UnifiedConfig
        
        print("🌟 Using UNIFIED PSLD+UGD sampler (alternating timesteps)")
        print(f"   - Schedule mode: {opt.schedule_mode}")
        if opt.schedule_mode == "pattern":
            print(f"   - Pattern: {opt.pattern_type}")
        elif opt.schedule_mode == "early_late":
            print(f"   - Split timestep: {opt.split_timestep}")
        print(f"   - PSLD weight: {opt.psld_weight}, UGD weight: {opt.ugd_weight}")
        print(f"   - Self-recurrence (k): {guidance_cfg.k_recur if guidance_cfg else 1}")
        
        # Create unified config
        unified_cfg = UnifiedConfig(
            schedule_mode=opt.schedule_mode,
            pattern_type=opt.pattern_type,
            split_timestep=opt.split_timestep,
            custom_schedule=None,  # Can be extended later
            psld_weight=opt.psld_weight,
            ugd_weight=opt.ugd_weight,
            gamma=opt.gamma,
            omega=opt.omega,
            guidance_cfg=guidance_cfg,
            guidance_fn=guidance_fn
        )
        
        sampler = UnifiedPSLDUGDSampler(model)
        # Store unified_cfg for use in sampling
        sampler.unified_cfg = unified_cfg
    elif opt.use_hybrid_sampler:
        print("🌟 Using UNIFIED PSLD+UGD sampler (hybrid mode)")
        print("   - Combines UGD inner optimization with PSLD measurement consistency")
        print("   - Best quality: strong style matching + structural preservation")
        sampler = PSLDUGDSampler(model)
    elif guidance_cfg and guidance_cfg.enabled:
        print("🚀 Using UGD-only DDIM sampler")
        print("   - Inner optimization loop for style guidance")
        sampler = UGDDDIMSampler(model)
    else:
        print("📊 Using standard PSLD DDIM sampler")
        print("   - Measurement consistency with adaptive learning rate")
        if opt.plms:
            sampler = PLMSSampler(model)
        elif opt.dpm_solver:
            sampler = DPMSolverSampler(model)
        else:
            sampler = DDIMSampler(model)

    # Exception) In case of inpainting, we need to generate a mask 
    if measure_config['operator']['name'] == 'inpainting':
        mask_gen = mask_generator(
        **measure_config['mask_opt']
        )

    print("Creating invisible watermark encoder (see https://github.com/ShieldMnt/invisible-watermark)...")
    wm = "StableDiffusionV1"
    wm_encoder = WatermarkEncoder()
    wm_encoder.set_watermark('bytes', wm.encode('utf-8'))

    os.makedirs(opt.outdir, exist_ok=True)
    outpath = opt.outdir

    batch_size = opt.n_samples
    n_rows = opt.n_rows if opt.n_rows > 0 else batch_size
    if not getattr(opt, 'from-file', None):
        prompt = opt.prompt
        assert prompt is not None
        data = [batch_size * [prompt]]
    else:
        print(f"reading prompts from {getattr(opt, 'from-file')}")
        with open(getattr(opt, 'from-file'), "r") as f:
            data = f.read().splitlines()
            data = list(chunk(data, batch_size))

    sample_path = os.path.join(outpath, "samples")
    os.makedirs(sample_path, exist_ok=True)
    base_count = len(os.listdir(sample_path))
    grid_count = len(os.listdir(outpath)) - 1

    # Load input image
    file_path = f'{opt.dps_path}data/samples/' + opt.file_id

    # Fix for MPS compatibility - ensure float32 dtype
    img_array = np.asarray(Image.open(file_path), dtype=np.float32) / 255.0
    img_normalized = 2 * img_array - 1
    org_image = torch.tensor(img_normalized, device=device)
    org_image = org_image.permute(2, 0, 1).unsqueeze(0)

    if opt.skip_low_res:
        # Resize to 256x256 if specified  
        org_image = torch.nn.functional.interpolate(org_image, size=[256, 256])
        opt.H, opt.W = 256, 256

    # Generate measurement
    # Exception) In case of inpainting,
    if measure_config['operator'] ['name'] == 'inpainting':
        dps_mask = mask_gen(org_image)
        dps_mask = dps_mask[:, 0, :, :].unsqueeze(dim=0)
        # Forward measurement model (Ax + n)
        y = operator.forward(org_image, mask=dps_mask)
        y_n = noiser(y)

    elif (measure_config['operator']['name'] == 'clip_style_retrieval') or (measure_config['operator']['name'] == 'style_retrieval'):
        # Style operators: NO noise should be added to style features
        y = operator.forward(org_image)
        y_n = y  # Keep original style features without noise
        dps_mask = None
    else: 
        # Forward measurement model (Ax + n)
        y = operator.forward(org_image)
        y_n = noiser(y)

    start_code = None
    if opt.fixed_code:
        start_code = torch.randn([opt.n_samples, opt.C, opt.H // opt.f, opt.W // opt.f], device=device).float()

    precision_scope = autocast if opt.precision == "autocast" else nullcontext
    with torch.no_grad():
        with precision_scope("cuda" if device.type == "cuda" else "cpu"):
            with model.ema_scope():
                tic = time.time()
                all_samples = []
                for n in trange(opt.n_iter, desc="Sampling"):
                    for prompts in tqdm(data, desc="data"):
                        uc = None
                        if opt.scale != 1.0:
                            uc = model.get_learned_conditioning(batch_size * [""])
                        if isinstance(prompts, tuple):
                            prompts = list(prompts)
                        c = model.get_learned_conditioning(prompts)

                        shape = [opt.C, opt.H // opt.f, opt.W // opt.f]

                        # Prepare config dictionary for logging
                        config_dict = {
                            # Sampling parameters
                            'prompt': prompts[0] if prompts else "",
                            'ddim_steps': opt.ddim_steps,
                            'ddim_eta': opt.ddim_eta,
                            'scale': opt.scale,
                            'seed': opt.seed,
                            'n_samples': opt.n_samples,
                            'H': opt.H,
                            'W': opt.W,
                            # Sampler mode
                            'use_unified_sampler': opt.use_unified_sampler if hasattr(opt, 'use_unified_sampler') else False,
                            'use_hybrid_sampler': opt.use_hybrid_sampler if hasattr(opt, 'use_hybrid_sampler') else False,
                            # PSLD parameters
                            'gamma': opt.gamma,
                            'omega': opt.omega,
                            'general_inverse': opt.general_inverse,
                            'inpainting': opt.inpainting,
                            # UGD parameters
                            'optim_forward_guidance': opt.optim_forward_guidance if hasattr(opt, 'optim_forward_guidance') else False,
                            'optim_num_steps': opt.optim_num_steps if hasattr(opt, 'optim_num_steps') else 0,
                            'optim_forward_guidance_wt': opt.optim_forward_guidance_wt if hasattr(opt, 'optim_forward_guidance_wt') else 0,
                            'k_recur': opt.k_recur if hasattr(opt, 'k_recur') else 1,
                            'normalize_grad': opt.normalize_grad if hasattr(opt, 'normalize_grad') else True,
                            # Unified sampler parameters
                            'schedule_mode': opt.schedule_mode if hasattr(opt, 'schedule_mode') else 'none',
                            'pattern_type': opt.pattern_type if hasattr(opt, 'pattern_type') else 'none',
                            'split_timestep': opt.split_timestep if hasattr(opt, 'split_timestep') else 0,
                            'psld_weight': opt.psld_weight if hasattr(opt, 'psld_weight') else 1.0,
                            'ugd_weight': opt.ugd_weight if hasattr(opt, 'ugd_weight') else 1.0,
                            # Task configuration
                            'task_config': opt.task_config if hasattr(opt, 'task_config') else 'none',
                            'operator': operator.__class__.__name__ if operator else 'none',
                            'style_image': opt.style_image if hasattr(opt, 'style_image') else 'none',
                            'file_id': opt.file_id if hasattr(opt, 'file_id') else 'none',
                        }

                        # =====================================
                        # SAMPLING: PSLD / UGD / HYBRID / UNIFIED
                        # =====================================
                        if opt.use_unified_sampler:
                            print("🌟 Running UNIFIED PSLD+UGD sampling (alternating timesteps)...")
                            # Unified mode: pass unified_cfg along with both PSLD and UGD parameters
                            samples_ddim, _ = sampler.sample(S=opt.ddim_steps,
                                                            conditioning=c,
                                                            batch_size=opt.n_samples,
                                                            shape=shape,
                                                            verbose=False,
                                                            unconditional_guidance_scale=opt.scale,
                                                            unconditional_conditioning=uc,
                                                            eta=opt.ddim_eta,
                                                            x_T=start_code,
                                                            # Unified config
                                                            unified_cfg=sampler.unified_cfg,
                                                            # PSLD parameters
                                                            ip_mask = dps_mask if measure_config['operator']['name'] == 'inpainting' else None,
                                                            measurements = y_n,
                                                            operator = operator,
                                                            gamma = opt.gamma,
                                                            inpainting = opt.inpainting,
                                                            omega = opt.omega,
                                                            general_inverse=opt.general_inverse,
                                                            noiser=noiser,
                                                            # UGD parameters
                                                            guidance_cfg=guidance_cfg,
                                                            guidance_fn=guidance_fn,
                                                            reference_image=org_image,
                                                            # Logging parameters
                                                            prompt=prompts[0] if prompts else "",
                                                            config_dict=config_dict)
                        elif opt.use_hybrid_sampler:
                            print("🌟 Running UNIFIED PSLD+UGD sampling (hybrid mode)...")
                            # Hybrid mode: pass both UGD and PSLD parameters
                            samples_ddim, _ = sampler.sample(S=opt.ddim_steps,
                                                            conditioning=c,
                                                            batch_size=opt.n_samples,
                                                            shape=shape,
                                                            verbose=False,
                                                            unconditional_guidance_scale=opt.scale,
                                                            unconditional_conditioning=uc,
                                                            eta=opt.ddim_eta,
                                                            x_T=start_code,
                                                            # UGD parameters
                                                            guidance_cfg=guidance_cfg,
                                                            guidance_fn=guidance_fn,
                                                            # PSLD parameters
                                                            ip_mask = dps_mask if measure_config['operator']['name'] == 'inpainting' else None,
                                                            measurements = y_n,
                                                            operator = operator,
                                                            gamma = opt.gamma,
                                                            inpainting = opt.inpainting,
                                                            omega = opt.omega,
                                                            general_inverse=opt.general_inverse,
                                                            noiser=noiser,
                                                            reference_image=org_image)
                        elif guidance_cfg and guidance_cfg.enabled:
                            print("🎯 Running UGD-only sampling...")
                            # UGD-only mode: just guidance parameters
                            samples_ddim, _ = sampler.sample(S=opt.ddim_steps,
                                                            conditioning=c,
                                                            batch_size=opt.n_samples,
                                                            shape=shape,
                                                            verbose=False,
                                                            unconditional_guidance_scale=opt.scale,
                                                            unconditional_conditioning=uc,
                                                            eta=opt.ddim_eta,
                                                            x_T=start_code,
                                                            # UGD parameters
                                                            guidance_cfg=guidance_cfg,
                                                            guidance_fn=guidance_fn,
                                                            reference_image=org_image)
                        else:
                            print("📊 Running standard PSLD sampling...")
                            # PSLD-only mode: just measurement parameters
                            samples_ddim, _ = sampler.sample(S=opt.ddim_steps,
                                                            conditioning=c,
                                                            batch_size=opt.n_samples,
                                                            shape=shape,
                                                            verbose=False,
                                                            unconditional_guidance_scale=opt.scale,
                                                            unconditional_conditioning=uc,
                                                            eta=opt.ddim_eta,
                                                            x_T=start_code,
                                                            ip_mask = dps_mask if measure_config['operator']['name'] == 'inpainting' else None,
                                                            measurements = y_n,
                                                            operator = operator,
                                                            gamma = opt.gamma,
                                                            inpainting = opt.inpainting,
                                                            omega = opt.omega,
                                                            general_inverse=opt.general_inverse,
                                                            noiser=noiser,
                                                            reference_image=org_image)

                        x_samples_ddim = model.decode_first_stage(samples_ddim)
                        x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)

                        if not opt.skip_save:
                            for x_sample in x_samples_ddim:
                                x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                                img = Image.fromarray(x_sample.astype(np.uint8))
                                img = put_watermark(img, wm_encoder)
                                img.save(os.path.join(sample_path, f"{base_count:05}.png"))
                                base_count += 1

                        if not opt.skip_grid:
                            all_samples.append(x_samples_ddim)

                if not opt.skip_grid:
                    # additionally, save as grid
                    grid = torch.stack(all_samples, 0)
                    grid = rearrange(grid, 'n b c h w -> (n b) c h w')
                    grid = make_grid(grid, nrow=n_rows)

                    # to image
                    grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()
                    img = Image.fromarray(grid.astype(np.uint8))
                    img = put_watermark(img, wm_encoder)
                    img.save(os.path.join(outpath, f'grid-{grid_count:04}.png'))
                    grid_count += 1

                toc = time.time()

    print(f"Your samples are ready and waiting for you here: \n{outpath} \n"
          f" \nEnjoy.")


if __name__ == "__main__":
    main()
