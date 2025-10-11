# Reference Image Only - TensorBoard Integration

## What's Added

**ONLY** the reference image logging to TensorBoard. No other metrics.

## Code Changes

### 1. In `ddim_ugd.py` - Lines 31-43 (Minimal Setup)
```python
def __init__(self, model, schedule="linear", **kwargs):
    super().__init__(model, schedule, **kwargs)
    
    # Initialize TensorBoard logger for reference image only
    try:
        from tensorboard_logger import get_tensorboard_logger
        self.tensorboard_logger = get_tensorboard_logger(
            experiment_name="ugd_style_guidance"
        )
        self.reference_image_logged = False
    except ImportError:
        self.tensorboard_logger = None
        self.reference_image_logged = False
```

### 2. In `ddim_ugd.py` - Lines 284-331 (Reference Image Logging)
```python
def sample(self, ..., reference_image=None, ...):
    ...
    # Log reference image once at the beginning
    if self.tensorboard_logger is not None and reference_image is not None and not self.reference_image_logged:
        with torch.no_grad():
            # Load and convert image (handles path, PIL, or tensor)
            ...
            # Log the reference image
            self.tensorboard_logger.log_image(ref_img, 
                                            name="reference_image", 
                                            step=0, 
                                            every_n_steps=1)
            self.reference_image_logged = True
            print(f"📸 Logged reference image to TensorBoard")
```

### 3. In `inverse_ugd.py` - Line 761 (Pass Reference)
```python
samples_ddim, _ = sampler.sample(
    ...
    reference_image=opt.style_image,  # Uses --style_image from command line
    ...
)
```

## What You'll See in TensorBoard

**Images Tab:**
- **`reference_image`** at step 0 - Your style reference image (from `--style_image`)

**That's it!** No other metrics, no other images.

## Command Line Usage

```bash
python scripts/inverse_ugd.py \
    --prompt "a dog" \
    --config configs/style_extraction_config.yaml \
    --file_id im1.jpg \                    # Input image (not logged)
    --style_image ../../pics/im2.jpg \     # Reference image (logged to TB)
    --ddim_steps 20
```

## Key Points

1. **`--file_id`**: The input image for PSLD reconstruction (not logged to TensorBoard)
2. **`--style_image`**: The reference style image (logged to TensorBoard at step 0)
3. **Logged once**: Reference image appears only once, not repeated
4. **No metrics**: No loss curves, gradients, or other metrics logged

## View in TensorBoard

```bash
cd PSLD/stable-diffusion
tensorboard --logdir=runs/
# Open http://localhost:6006
```

Go to **Images** tab → You'll see **`reference_image`** at step 0.

## Benefits

- **Minimal overhead**: Only one image logged, no performance impact
- **Visual comparison**: Easy to compare your results against the reference
- **Clean TensorBoard**: No clutter with metrics you don't need

## If You Want More Later

If you decide you want metrics too, just let me know and I can add them back!

