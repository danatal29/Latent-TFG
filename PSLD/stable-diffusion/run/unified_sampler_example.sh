#!/bin/bash
# Example script for running the unified PSLD+UGD sampler
# This demonstrates the alternating timestep strategy
#
# USAGE: Run this script from the PSLD/stable-diffusion directory:
#   cd PSLD/stable-diffusion
#   bash run/unified_sampler_example.sh
#
# IMPORTANT: For style transfer tasks:
#   - file_id: The reference style image (PSLD extracts measurements from this)
#   - style_image: The reference style image (UGD creates guidance from this)
#   - Both should be THE SAME image (the style you want to extract/transfer)

# Make sure we're in the right directory
cd "$(dirname "$0")/.." || exit 1
echo "Running from: $(pwd)"

# Configuration paths (relative to PSLD/stable-diffusion)
CONFIG_DIR="../diffusion-posterior-sampling/configs"

# Style reference image (used for both PSLD measurements and UGD guidance)
STYLE_REFERENCE="../../../pics/starry_night_full.jpg"

# Model paths
MODEL_CKPT="models/ldm/stable-diffusion-v1/model.ckpt"
MODEL_CONFIG="configs/stable-diffusion/v1-inference.yaml"

# Output directory
OUTPUT_DIR="outputs/unified_psld_ugd"




# ==============================================
# EXAMPLE 6: TEXTURE-OPTIMIZED (STABLE)
# Early-late mode with conservative parameters for stability
# PSLD for structure (high noise), UGD for texture (low noise)
# ==============================================
# echo "Running Unified Sampler - Texture Optimized (Stable)"
# python scripts/inverse_ugd.py \
#     --use_unified_sampler \
#     --schedule_mode pattern \
#     --psld_weight 1.5 \
#     --ugd_weight 2.0 \
#     --prompt "An ancient temple surrounded by cherry blossoms in morning mist" \
#     --style_image "${STYLE_REFERENCE}" \
#     --file_id "starry_night_full.jpg" \
#     --task_config "${CONFIG_DIR}/style_extraction_config.yaml" \
#     --diffusion_config "${CONFIG_DIR}/diffusion_config.yaml" \
#     --model_config "${CONFIG_DIR}/model_config.yaml" \
#     --config "${MODEL_CONFIG}" \
#     --ckpt "${MODEL_CKPT}" \
#     --ddim_steps 100 \
#     --ddim_eta 0.0 \
#     --scale 10.0 \
#     --omega 1.0 \
#     --gamma 0.08 \
#     --general_inverse 1 \
#     --optim_forward_guidance \
#     --optim_num_steps 10 \
#     --optim_forward_guidance_wt 8.0 \
#     --k_recur 4 \
#     --normalize_grad \
#     --outdir "${OUTPUT_DIR}/texture_stable" \
#     --seed 42

# echo "All unified sampler examples completed!"
# echo "Results saved to: ${OUTPUT_DIR}"




# # ==============================================
# # EXAMPLE 5: UGD-only mode - STYLE FOCUSED
# # Use only UGD at all timesteps for maximum style transfer
# # ==============================================
# echo "Running Unified Sampler - UGD Only (Style Focused)"
# python scripts/inverse_ugd.py \
#     --use_unified_sampler \
#     --schedule_mode pattern \
#     --pattern_type all_ugd \
#     --ugd_weight 2.5 \
#     --prompt "An ancient temple surrounded by cherry blossoms in morning mist" \
#     --style_image "${STYLE_REFERENCE}" \
#     --file_id "starry_night_full.jpg" \
#     --task_config "${CONFIG_DIR}/style_extraction_config.yaml" \
#     --diffusion_config "${CONFIG_DIR}/diffusion_config.yaml" \
#     --model_config "${CONFIG_DIR}/model_config.yaml" \
#     --config "${MODEL_CONFIG}" \
#     --ckpt "${MODEL_CKPT}" \
#     --ddim_steps 100 \
#     --ddim_eta 0.0 \
#     --scale 10.0 \
#     --general_inverse 0 \
#     --optim_forward_guidance \
#     --optim_num_steps 8 \
#     --optim_forward_guidance_wt 6.0 \
#     --k_recur 4 \
#     --omega 1.0 \
#     --gamma 0.08 \
#     --normalize_grad \
#     --outdir "${OUTPUT_DIR}/ugd_only_style" \
#     --seed 42


# ==============================================
# EXAMPLE 1: Pattern mode (even-odd alternating) - STYLE FOCUSED
# PSLD on even timesteps, UGD on odd timesteps with strong UGD
# ==============================================
echo "Running Unified Sampler - Pattern Mode (even-odd) Style Focused"
python scripts/inverse_ugd.py \
    --use_unified_sampler \
    --schedule_mode pattern \
    --pattern_type even_odd \
    --psld_weight 0.8 \
    --ugd_weight 2.5 \
    --prompt "A gondola gliding through the canals of Venice at night" \
    --style_image "${STYLE_REFERENCE}" \
    --file_id "starry_night_full.jpg" \
    --task_config "${CONFIG_DIR}/style_extraction_config.yaml" \
    --diffusion_config "${CONFIG_DIR}/diffusion_config.yaml" \
    --model_config "${CONFIG_DIR}/model_config.yaml" \
    --config "${MODEL_CONFIG}" \
    --ckpt "${MODEL_CKPT}" \
    --ddim_steps 100 \
    --ddim_eta 0.0 \
    --scale 10.0 \
    --omega 1.0 \
    --gamma 0.08 \
    --general_inverse 1 \
    --optim_forward_guidance \
    --optim_num_steps 8 \
    --optim_forward_guidance_wt 6.0 \
    --k_recur 4 \
    --normalize_grad \
    --outdir "${OUTPUT_DIR}/pattern_even_odd_style" \
    --seed 42

# # ==============================================
# # EXAMPLE 2: Pattern mode (odd-even alternating) - STYLE FOCUSED
# # UGD on even timesteps (more frequently), PSLD on odd timesteps
# # ==============================================
# echo "Running Unified Sampler - Pattern Mode (odd-even) Style Focused"
# python scripts/inverse_ugd.py \
#     --use_unified_sampler \
#     --schedule_mode pattern \
#     --pattern_type odd_even \
#     --psld_weight 0.7 \
#     --ugd_weight 2.8 \
#     --prompt "An ancient temple surrounded by cherry blossoms in morning mist" \
#     --style_image "${STYLE_REFERENCE}" \
#     --file_id "starry_night_full.jpg" \
#     --task_config "${CONFIG_DIR}/style_extraction_config.yaml" \
#     --diffusion_config "${CONFIG_DIR}/diffusion_config.yaml" \
#     --model_config "${CONFIG_DIR}/model_config.yaml" \
#     --config "${MODEL_CONFIG}" \
#     --ckpt "${MODEL_CKPT}" \
#     --ddim_steps 100 \
#     --ddim_eta 0.0 \
#     --scale 10.0 \
#     --omega 1.0 \
#     --gamma 0.08 \
#     --general_inverse 1 \
#     --optim_forward_guidance \
#     --optim_num_steps 8 \
#     --optim_forward_guidance_wt 6.0 \
#     --k_recur 4 \
#     --normalize_grad \
#     --outdir "${OUTPUT_DIR}/pattern_odd_even_style" \
#     --seed 42

# # ==============================================
# # EXAMPLE 3: Early-late mode - STYLE FOCUSED  
# # PSLD for early structure, UGD for late texture (70% UGD time)
# # ==============================================
# echo "Running Unified Sampler - Early-Late Mode Style Focused"
# python scripts/inverse_ugd.py \
#     --use_unified_sampler \
#     --schedule_mode early_late \
#     --split_timestep 30 \
#     --psld_weight 0.8 \
#     --ugd_weight 2.2 \
#     --prompt "An ancient temple surrounded by cherry blossoms in morning mist" \
#     --style_image "${STYLE_REFERENCE}" \
#     --file_id "starry_night_full.jpg" \
#     --task_config "${CONFIG_DIR}/style_extraction_config.yaml" \
#     --diffusion_config "${CONFIG_DIR}/diffusion_config.yaml" \
#     --model_config "${CONFIG_DIR}/model_config.yaml" \
#     --config "${MODEL_CONFIG}" \
#     --ckpt "${MODEL_CKPT}" \
#     --ddim_steps 100 \
#     --ddim_eta 0.0 \
#     --scale 10.0 \
#     --omega 1.0 \
#     --gamma 0.08 \
#     --general_inverse 1 \
#     --optim_forward_guidance \
#     --optim_num_steps 8 \
#     --optim_forward_guidance_wt 6.0 \
#     --k_recur 4 \
#     --normalize_grad \
#     --outdir "${OUTPUT_DIR}/early_late_style" \
#     --seed 42

# # ==============================================
# # EXAMPLE 4: Weighted contributions - STYLE FOCUSED
# # Emphasize UGD heavily for maximum style transfer
# # ==============================================
# echo "Running Unified Sampler - Weighted (UGD-heavy Style)"
# python scripts/inverse_ugd.py \
#     --use_unified_sampler \
#     --schedule_mode pattern \
#     --pattern_type even_odd \
#     --psld_weight 0.5 \
#     --ugd_weight 3.0 \
#     --prompt "An ancient temple surrounded by cherry blossoms in morning mist" \
#     --style_image "${STYLE_REFERENCE}" \
#     --file_id "starry_night_full.jpg" \
#     --task_config "${CONFIG_DIR}/style_extraction_config.yaml" \
#     --diffusion_config "${CONFIG_DIR}/diffusion_config.yaml" \
#     --model_config "${CONFIG_DIR}/model_config.yaml" \
#     --config "${MODEL_CONFIG}" \
#     --ckpt "${MODEL_CKPT}" \
#     --ddim_steps 100 \
#     --ddim_eta 0.0 \
#     --scale 10.0 \
#     --omega 1.0 \
#     --gamma 0.08 \
#     --general_inverse 1 \
#     --optim_forward_guidance \
#     --optim_num_steps 10 \
#     --optim_forward_guidance_wt 7.0 \
#     --k_recur 5 \
#     --normalize_grad \
#     --outdir "${OUTPUT_DIR}/weighted_ugd_heavy_style" \
#     --seed 42

