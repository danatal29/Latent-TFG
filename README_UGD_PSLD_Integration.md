# UGD-PSLD Integration for Style Transfer

## Overview

This implementation integrates Universal Guided Diffusion (UGD) with Posterior Sampling for Latent Diffusion (PSLD) to create a powerful hybrid approach for style transfer. The integration combines the strengths of both methods:

- **UGD**: Provides universal guidance without retraining using CLIP-based style matching
- **PSLD**: Offers posterior sampling guidance for measurement consistency

## Key Features

### 1. Hybrid Guidance System
- **UGD Mode**: Uses CLIP features for style guidance
- **PSLD Mode**: Uses posterior sampling with style operators
- **Hybrid Mode**: Combines both approaches with configurable weights

### 2. Advanced Features
- **Adaptive Weighting**: Dynamically adjusts guidance weights during sampling
- **Multi-scale Processing**: Computes style loss at multiple scales
- **Style Interpolation**: Blend multiple style images
- **Multiple Operators**: Support for style, texture, color, and structure operators

### 3. Flexible Configuration
- YAML-based configuration system
- Easy switching between guidance modes
- Fine-tuned control over all parameters

## Installation

```bash
# Ensure you have both repositories
git clone https://github.com/arpitbansal297/Universal-Guided-Diffusion.git
# PSLD should already be in your workspace

# Install dependencies
pip install torch torchvision clip omegaconf pytorch-lightning
pip install transformers diffusers
```

## Usage

### Basic Style Transfer

```python
from ugd_psld_style_transfer import UGDPSLDStyleTransfer

# Initialize the system
style_transfer = UGDPSLDStyleTransfer(
    config_path="path/to/config.yaml",
    checkpoint_path="path/to/model.ckpt",
    device="cuda"
)

# Set up hybrid sampler
style_transfer.setup_sampler(guidance_mode="hybrid")

# Configure guidance
guidance_config = {
    "clip_model": "RN50",
    "clip_weight": 1.0,      # Weight for UGD guidance
    "ps_weight": 0.5,        # Weight for PSLD guidance
    "guidance_scale": 100.0,  # Overall guidance strength
    "ps_scale": 1.0
}

# Perform style transfer
style_transfer.transfer_style(
    content_text="A beautiful landscape",
    style_image_path="style.jpg",
    output_path="output.png",
    num_steps=50,
    guidance_config=guidance_config,
    cfg_scale=7.5,
    seed=42
)
```

### Command Line Interface

```bash
python ugd_psld_style_transfer.py \
    --config configs/stable-diffusion/v1-inference.yaml \
    --checkpoint models/ldm/stable-diffusion-v1/model.ckpt \
    --content "A serene mountain landscape" \
    --style path/to/style_image.jpg \
    --output result.png \
    --guidance_mode hybrid \
    --steps 50 \
    --clip_weight 1.0 \
    --ps_weight 0.5 \
    --guidance_scale 100.0 \
    --cfg_scale 7.5 \
    --seed 42
```

### Advanced Usage with Configuration File

```python
from ugd_psld_advanced import create_hybrid_sampler, load_config
from ugd_psld_advanced import GuidanceConfig, GuidanceMode

# Load configuration
config = load_config("ugd_psld_config.yaml")

# Create guidance configuration
guidance_config = GuidanceConfig(
    mode=GuidanceMode.HYBRID,
    ugd_weight=1.0,
    psld_weight=0.5,
    guidance_scale=100.0,
    adaptive_weighting=True,
    timestep_threshold=0.5
)

# Create enhanced sampler with advanced features
sampler = EnhancedHybridSampler(
    model=model,
    guidance_config=guidance_config,
    use_multiscale=True,
    use_style_interpolation=True
)
```

## Configuration Options

### Guidance Modes

1. **UGD Mode** (`--guidance_mode ugd`)
   - Pure Universal Guided Diffusion
   - Best for: Strong style transfer with CLIP guidance
   - Pros: No training required, works with any CLIP model
   - Cons: May lose some content details

2. **PSLD Mode** (`--guidance_mode psld`)
   - Pure Posterior Sampling
   - Best for: Maintaining content structure
   - Pros: Better content preservation
   - Cons: May require tuning for different styles

3. **Hybrid Mode** (`--guidance_mode hybrid`)
   - Combines both approaches
   - Best for: Balanced style transfer
   - Pros: Best of both worlds
   - Cons: More parameters to tune

4. **None Mode** (`--guidance_mode none`)
   - No guidance, pure diffusion
   - Best for: Text-to-image generation without style

### Key Parameters

- `clip_weight`: Controls strength of UGD guidance (0.5-2.0 recommended)
- `ps_weight`: Controls strength of PSLD guidance (0.3-1.0 recommended)
- `guidance_scale`: Overall guidance strength (50-200 recommended)
- `cfg_scale`: Classifier-free guidance scale (5-15 recommended)
- `steps`: Number of DDIM steps (25-100, higher = better quality)

### Advanced Configuration

Edit `ugd_psld_config.yaml` for fine-grained control:

```yaml
guidance:
  hybrid:
    guidance_scale: 100.0
    balance_factor: 0.5  # 0=full PSLD, 1=full UGD
    adaptive_weighting: true  # Dynamic weight adjustment
    
style_transfer:
  multiscale:
    enabled: true
    scales: [0.5, 1.0, 2.0]  # Multi-scale processing
    
  augmentation:
    enabled: true  # Style augmentation for robustness
```

## Architecture

### Integration Design

```
Input Image/Text
      ↓
[Encoding Stage]
      ↓
[Diffusion Process]
      ↓
┌─────────────────────────────┐
│   Hybrid Guidance Module     │
│  ┌──────────┐  ┌──────────┐ │
│  │   UGD    │  │   PSLD   │ │
│  │  (CLIP)  │  │(Posterior)│ │
│  └─────┬────┘  └────┬─────┘ │
│        └──────┬──────┘       │
│         [Weighted Sum]       │
│               ↓              │
│      [Adaptive Scheduler]    │
└──────────────┬───────────────┘
               ↓
        [Guided Update]
               ↓
         [Next Step]
```

### Key Components

1. **UGD Component**
   - CLIP encoder for style features
   - Gradient computation through decoder
   - Forward/backward guidance options

2. **PSLD Component**
   - Style operator (VGG-based or custom)
   - Posterior sampling gradient
   - Measurement consistency

3. **Hybrid Controller**
   - Weight balancing
   - Adaptive scheduling
   - Multi-scale processing

## Experimental Results

### Recommended Settings by Use Case

| Use Case | Mode | CLIP Weight | PS Weight | Guidance Scale | Steps |
|----------|------|-------------|-----------|----------------|-------|
| Artistic Style | Hybrid | 1.0 | 0.5 | 100 | 50 |
| Photo Style | Hybrid | 0.7 | 0.8 | 75 | 50 |
| Abstract Art | UGD | 1.5 | - | 150 | 75 |
| Subtle Style | PSLD | - | 1.0 | 50 | 50 |
| Fast Preview | Hybrid | 1.0 | 0.3 | 100 | 25 |

## Troubleshooting

### Common Issues

1. **Out of Memory**
   - Reduce batch size to 1
   - Enable FP16: Set `advanced.use_fp16: true` in config
   - Reduce image resolution

2. **Poor Style Transfer**
   - Increase `guidance_scale` (try 150-200)
   - Adjust `clip_weight` and `ps_weight` ratio
   - Try different CLIP models (ViT-B/32 often works better)

3. **Content Loss**
   - Reduce `guidance_scale`
   - Increase `ps_weight` relative to `clip_weight`
   - Lower `cfg_scale` (try 5-7)

4. **Slow Generation**
   - Reduce steps (minimum 25 recommended)
   - Disable multi-scale processing
   - Use single GPU mode

## Advanced Techniques

### Style Interpolation

```python
# Blend multiple styles
style_images = ["style1.jpg", "style2.jpg", "style3.jpg"]
style_weights = [0.5, 0.3, 0.2]  # Must sum to 1

# The system will automatically interpolate between styles
```

### Progressive Style Transfer

```python
# Start with low guidance, increase over time
for strength in [50, 100, 150]:
    guidance_config["guidance_scale"] = strength
    style_transfer.transfer_style(...)
```

### Multi-Resolution Processing

```python
# Process at multiple resolutions for better results
resolutions = [256, 512, 768]
for res in resolutions:
    # Adjust configuration for resolution
    # Higher res = lower guidance scale typically
    pass
```

## Citation

If you use this integration in your research, please cite both original papers:

```bibtex
@article{bansal2023universal,
  title={Universal Guidance for Diffusion Models},
  author={Bansal, Arpit and others},
  journal={arXiv preprint},
  year={2023}
}

@article{chung2022diffusion,
  title={Diffusion Posterior Sampling for General Noisy Inverse Problems},
  author={Chung, Hyungjin and others},
  journal={arXiv preprint},
  year={2022}
}
```

## Future Improvements

- [ ] Support for video style transfer
- [ ] Real-time style transfer optimization
- [ ] AutoML for parameter tuning
- [ ] Support for more measurement operators
- [ ] Integration with ControlNet
- [ ] Web UI interface

## License

This integration follows the licenses of both UGD and PSLD repositories. Please refer to their respective LICENSE files.