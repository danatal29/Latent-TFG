# PSLD Style Extraction & Transfer

A powerful system for extracting and transferring artistic styles using Stable Diffusion and CLIP. This repository implements Posterior Sampling with Latent Diffusion (PSLD) for high-quality style transfer with precise control over style adherence.

## 🎨 What This Does

Transform any image to match a specific artistic style by:
- **Style Extraction**: Automatically extract style features from reference images
- **Style Transfer**: Apply extracted styles to new images with text guidance
- **Quality Control**: Fine-tune style strength and generation quality
- **Real-time Monitoring**: Track progress with TensorBoard visualization

## 🚀 Quick Start

### 1. Setup
```bash
# Clone the repository
git clone <repository-url>
cd projectv2

# Install dependencies (if needed)
pip install torch transformers clip-by-openai tensorboard omegaconf pillow opencv-python
```

### 2. Prepare Your Images
Place your input images in the `pics/` directory:
```bash
cp your_image.jpg pics/
```

### 3. Run Style Transfer
```bash
cd PSLD/stable-diffusion

# Basic style transfer
python scripts/inverse.py \
    --file_id='your_image.jpg' \
    --task_config='../diffusion-posterior-sampling/configs/style_extraction_config.yaml' \
    --outdir='outputs/my_style_results' \
    --prompt='van gogh painting style' \
    --omega=10 \
    --ddim_steps=50
```

### 4. View Results
- Generated images: `outputs/my_style_results/samples/`
- Progress tracking: `tensorboard --logdir=runs --port=6006`

## 📁 Project Structure

```
projectv2/
├── PSLD/
│   ├── stable-diffusion/           # Main Stable Diffusion implementation
│   │   ├── scripts/inverse.py     # Main script for style transfer
│   │   ├── run/inverse.sh         # Example commands
│   │   └── ldm/models/diffusion/psld.py  # PSLD sampler
│   ├── diffusion-posterior-sampling/
│   │   └── configs/style_extraction_config.yaml  # Configuration
│   └── notebooks/
│       └── extract_style.ipynb    # Jupyter examples
├── pics/                          # Your input images go here
└── outputs/                       # Generated results appear here
```

## ⚙️ Configuration Options

### Key Parameters

| Parameter | Description | Typical Values | Effect |
|-----------|-------------|----------------|---------|
| `--omega` | Style constraint strength | 1-20 | Higher = stronger style adherence |
| `--ddim_steps` | Sampling steps | 20-100 | More steps = better quality, slower |
| `--scale` | Text guidance strength | 3-15 | Higher = better prompt following |
| `--ddim_eta` | Sampling randomness | 0-1 | 0 = deterministic, 1 = more random |

### Style Strength Guide
- **`omega=1-3`**: Subtle style influence
- **`omega=5-10`**: Balanced style transfer
- **`omega=10-20`**: Strong style dominance

### Quality vs Speed
- **Fast**: `ddim_steps=20`, `omega=5`
- **Balanced**: `ddim_steps=50`, `omega=10`
- **High Quality**: `ddim_steps=100`, `omega=15`

## 🎯 Example Commands

### Artistic Style Transfer
```bash
# Van Gogh style
python scripts/inverse.py \
    --file_id='portrait.jpg' \
    --task_config='../diffusion-posterior-sampling/configs/style_extraction_config.yaml' \
    --outdir='outputs/van_gogh' \
    --prompt='van gogh painting with thick brushstrokes' \
    --omega=12 \
    --ddim_steps=75

# Watercolor style
python scripts/inverse.py \
    --file_id='landscape.jpg' \
    --task_config='../diffusion-posterior-sampling/configs/style_extraction_config.yaml' \
    --outdir='outputs/watercolor' \
    --prompt='soft watercolor painting' \
    --omega=8 \
    --ddim_steps=60
```

### Quick Testing
```bash
# Fast test run
python scripts/inverse.py \
    --file_id='test_image.jpg' \
    --task_config='../diffusion-posterior-sampling/configs/style_extraction_config.yaml' \
    --outdir='outputs/test' \
    --prompt='oil painting' \
    --omega=5 \
    --ddim_steps=25
```

## 📊 Monitoring Progress

### TensorBoard Visualization
```bash
# Start TensorBoard
tensorboard --logdir=runs --port=6006

# Open in browser: http://localhost:6006
```

### What You'll See
- **Loss Curves**: Style constraint loss over time
- **Learning Rates**: How the model adjusts during generation
- **Image Progress**: Step-by-step generation process
- **Metrics**: Gradient norms, parameter magnitudes

## 🔧 Troubleshooting

### Common Issues

**"File not found" error**
- Make sure your image is in the `pics/` directory
- Check the filename matches exactly (case-sensitive)

**Out of memory error**
- Reduce `ddim_steps` (try 20-30)
- Lower `omega` value (try 5-8)
- Use CPU: add `--device=cpu` (slower but works)

**Poor style transfer results**
- Increase `omega` value (try 15-20)
- Use more descriptive prompts
- Increase `ddim_steps` for better quality

**TensorBoard not showing data**
- Check that `runs/` directory exists
- Wait a few minutes for logs to appear
- Try refreshing the browser

### Performance Tips
- **GPU**: Much faster than CPU (if available)
- **Memory**: 4-8GB GPU memory recommended
- **Storage**: Each run creates ~100MB of output files

## 🎨 Style Examples

Try these prompt styles for different effects:

### Artistic Styles
- `"van gogh painting with thick brushstrokes"`
- `"picasso cubist style"`
- `"monet impressionist painting"`
- `"salvador dali surrealist art"`

### Medium-Specific
- `"watercolor painting with soft edges"`
- `"oil painting with rich colors"`
- `"pencil sketch with fine details"`
- `"digital art with vibrant colors"`

### Mood-Based
- `"dark gothic art style"`
- `"bright cheerful illustration"`
- `"minimalist modern design"`
- `"vintage retro aesthetic"`

## 📈 Advanced Usage

### Custom Configuration
Edit `../diffusion-posterior-sampling/configs/style_extraction_config.yaml`:
```yaml
measurement:
  operator:
    name: clip_style_retrieval  # Style extraction method
  noise: 
    name: gaussian
    sigma: 0.05  # Noise level (lower = cleaner)
```

### Batch Processing
```bash
# Process multiple images
for image in pics/*.jpg; do
    python scripts/inverse.py \
        --file_id="$(basename "$image")" \
        --task_config='../diffusion-posterior-sampling/configs/style_extraction_config.yaml' \
        --outdir="outputs/batch_$(basename "$image" .jpg)" \
        --prompt='your style description' \
        --omega=10 \
        --ddim_steps=50
done
```

## 🤝 Contributing

This repository implements PSLD (Posterior Sampling with Latent Diffusion) for style extraction. Key components:

- **Style Extraction**: Uses CLIP to extract style features
- **Diffusion Sampling**: Custom DDIM sampler with style constraints
- **Constraint Optimization**: Cosine similarity loss for style matching
- **Time Scheduling**: Late-strong scheduling for better results

## 📄 License

Please check the individual component licenses in the repository.

## 🆘 Support

If you encounter issues:
1. Check the troubleshooting section above
2. Verify your image is in the `pics/` directory
3. Try reducing `omega` and `ddim_steps` values
4. Check TensorBoard logs for detailed error information

Happy style transferring! 🎨✨


