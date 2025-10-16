#!/usr/bin/env python3

"""
Integration guide for the learning rate scheduler in PSLD.
This shows how to modify the PSLD code to use the adaptive learning rate scheduler.
"""

# Import the scheduler
from ldm.models.diffusion.lr_scheduler import AdaptiveLRScheduler

def integrate_scheduler_into_psld():
    """
    Example of how to integrate the scheduler into the PSLD p_sample_ddim function.
    """
    
    # 1. Initialize the scheduler in the DDIMSampler __init__ method
    scheduler_init_code = '''
    def __init__(self, model, schedule="linear", beta_start=0.0001, beta_end=0.02, device=None):
        # ... existing code ...
        
        # Initialize the adaptive learning rate scheduler
        self.lr_scheduler = AdaptiveLRScheduler(
            initial_lr=0.1,      # Start with higher LR
            min_lr=1e-6,         # Minimum LR
            max_lr=50.0,         # Maximum LR (high for your case)
            patience=3,          # Reduce LR after 3 steps without improvement
            factor=0.7,          # Multiply LR by 0.7 when reducing
            warmup_steps=5,      # Warm up for 5 steps
            gradient_threshold=1e-4,  # Threshold for small gradients
            loss_threshold=1e-3       # Threshold for loss improvement
        )
    '''
    
    # 2. Modify the gradient computation in p_sample_ddim
    gradient_update_code = '''
    # Replace the hardcoded learning rate with adaptive scheduler
    gradients = torch.autograd.grad(error, inputs=z_t)[0]
    
    # Get gradient and z norms for the scheduler
    gradient_norm = torch.norm(gradients).item()
    z_norm = torch.norm(z_prev).item()
    
    # Get adaptive learning rate
    lr = self.lr_scheduler.get_lr(
        current_loss=error.item(),
        gradient_norm=gradient_norm,
        z_norm=z_norm
    )
    
    # Apply gradient update with adaptive LR
    z_prev = z_prev - lr * gradients
    
    # Print debug info
    print(f'Style loss: {style_loss.item():.4f}')
    print(f'Gradients: {gradients}')
    print(f'LR: {lr:.6f}, Grad norm: {gradient_norm:.6f}, Z norm: {z_norm:.2f}')
    print(f'Loss: {error.item()}')
    
    # Optional: Print scheduler stats every 10 steps
    if self.lr_scheduler.step_count % 10 == 0:
        stats = self.lr_scheduler.get_stats()
        print(f"Scheduler stats: LR={stats['current_lr']:.6f}, "
              f"Best loss={stats['best_loss']:.6f}, "
              f"Avg grad={stats['avg_gradient']:.6f}")
    '''
    
    # 3. Alternative: Use different schedulers
    alternative_schedulers = '''
    # For cosine annealing (good for longer runs)
    self.lr_scheduler = CosineAnnealingScheduler(
        initial_lr=0.1,
        min_lr=1e-6,
        max_lr=50.0,
        total_steps=100,  # Total DDIM steps
        warmup_steps=10
    )
    
    # For exponential decay (good for aggressive early learning)
    self.lr_scheduler = ExponentialScheduler(
        initial_lr=0.1,
        min_lr=1e-6,
        max_lr=50.0,
        decay_rate=0.95  # Decay by 5% each step
    )
    '''
    
    print("=== PSLD Learning Rate Scheduler Integration ===")
    print("\n1. Add to DDIMSampler.__init__:")
    print(scheduler_init_code)
    
    print("\n2. Replace gradient computation in p_sample_ddim:")
    print(gradient_update_code)
    
    print("\n3. Alternative schedulers:")
    print(alternative_schedulers)
    
    print("\n=== Key Benefits ===")
    print("✅ Automatically adapts to small gradients (your case)")
    print("✅ Handles large latent values (120+)")
    print("✅ Prevents learning rate from getting too small")
    print("✅ Tracks progress and reduces LR when stuck")
    print("✅ Provides warmup for stable early training")

if __name__ == "__main__":
    integrate_scheduler_into_psld()


