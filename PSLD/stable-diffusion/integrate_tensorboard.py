#!/usr/bin/env python3

"""
Integration guide for TensorBoard logging in PSLD.
This shows how to modify the PSLD code to log images and metrics to TensorBoard.
"""

from tensorboard_logger import get_tensorboard_logger

def integrate_tensorboard_into_psld():
    """
    Example of how to integrate TensorBoard logging into the PSLD p_sample_ddim function.
    """
    
    # 1. Initialize TensorBoard logger in the DDIMSampler __init__ method
    init_code = '''
    def __init__(self, model, schedule="linear", beta_start=0.0001, beta_end=0.02, device=None):
        # ... existing code ...
        
        # Initialize TensorBoard logger
        self.tensorboard_logger = get_tensorboard_logger(
            experiment_name="psld_style_constraint"
        )
        self.log_step = 0
    '''
    
    # 2. Add TensorBoard logging in p_sample_ddim
    logging_code = '''
    # After computing the style loss and before gradient computation
    if hasattr(operator, '__class__') and ('style' in operator.__class__.__name__.lower() or 'StyleOperator' in operator.__class__.__name__):
        # ... existing style loss computation ...
        
        # Log metrics to TensorBoard
        self.tensorboard_logger.log_metrics({
            'loss/style_loss': style_loss.item(),
            'loss/total_loss': error.item(),
            'optimization/learning_rate': lr
        }, step=self.log_step)
        
        # Log gradients
        gradients = torch.autograd.grad(error, inputs=z_t)[0]
        self.tensorboard_logger.log_gradients(gradients, step=self.log_step)
        
        # Log latent statistics
        self.tensorboard_logger.log_latent_stats(z_prev, step=self.log_step)
        
        # Log images every 10 steps
        if self.log_step % 10 == 0:
            # Decode the current prediction to image space
            current_image = self.model.differentiable_decode_first_stage(pred_z_0)
            
            # Log the generated image
            self.tensorboard_logger.log_image(
                current_image, 
                name="generated_image", 
                step=self.log_step, 
                every_n_steps=10
            )
            
            # Log style comparison (if you have original image)
            if hasattr(self, 'original_image'):
                self.tensorboard_logger.log_style_comparison(
                    original_image=self.original_image,
                    generated_image=current_image,
                    target_style_features=measurements,
                    pred_style_features=pred_style_features,
                    step=self.log_step,
                    every_n_steps=10
                )
        
        self.log_step += 1
    '''
    
    # 3. Add cleanup in the main script
    cleanup_code = '''
    # At the end of your main script, close the TensorBoard logger
    if hasattr(sampler, 'tensorboard_logger'):
        sampler.tensorboard_logger.close()
    '''
    
    # 4. Example usage in the main inverse.py script
    main_integration = '''
    # In your main inverse.py script, add this after creating the sampler:
    
    # Store original image for comparison
    sampler.original_image = org_image  # Store the original image
    
    # After sampling is complete:
    if hasattr(sampler, 'tensorboard_logger'):
        sampler.tensorboard_logger.close()
        print(f"TensorBoard logs saved. Run: tensorboard --logdir={sampler.tensorboard_logger.log_dir}")
    '''
    
    print("=== TensorBoard Integration for PSLD ===")
    print("\n1. Add to DDIMSampler.__init__:")
    print(init_code)
    
    print("\n2. Add logging in p_sample_ddim:")
    print(logging_code)
    
    print("\n3. Add cleanup:")
    print(cleanup_code)
    
    print("\n4. Main script integration:")
    print(main_integration)
    
    print("\n=== What Gets Logged ===")
    print("📊 Metrics:")
    print("  - Style loss over time")
    print("  - Total loss over time")
    print("  - Learning rate changes")
    print("  - Gradient statistics (norm, mean, std, max, min)")
    print("  - Latent space statistics")
    
    print("\n🖼️ Images (every 10 steps):")
    print("  - Generated images during optimization")
    print("  - Original vs generated comparison")
    print("  - Style feature heatmaps")
    
    print("\n📈 How to View:")
    print("  1. Run your PSLD script")
    print("  2. Open terminal and run: tensorboard --logdir=runs/")
    print("  3. Open browser to http://localhost:6006")
    print("  4. Navigate to 'Images' and 'Scalars' tabs")

if __name__ == "__main__":
    integrate_tensorboard_into_psld()
