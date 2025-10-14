#!/bin/bash
# Example script for running the unified PSLD+UGD sampler with DINOv2
# This demonstrates texture-focused style transfer using DINOv2's superior texture extraction
#
# DINOv2 Advantages:
#   - Better at capturing fine-grained textures (brush strokes, paint texture)
#   - Excels at low-level visual features without semantic bias
#   - Superior for distinguishing artistic styles (e.g., Van Gogh vs Cartoon)
#   - Captures spatial patterns and directional textures
#
# USAGE: Run this script from the PSLD/stable-diffusion directory:
#   cd PSLD/stable-diffusion
#   bash run/dinov2_sampler_example.sh
#
# IMPORTANT: For DINOv2 style transfer tasks:
#   1. Config must use: operator.name = "style_retrieval" (NOT "clip_style_retrieval")
#   2. file_id: The reference style image (PSLD extracts DINOv2 features from this)
#   3. style_image: The reference style image (UGD creates DINOv2 guidance from this)
#   4. Both should be THE SAME image (the style you want to extract/transfer)

# Make sure we're in the right directory
cd "$(dirname "$0")/.." || exit 1
echo "🎨 Running DINOv2 Style Transfer Examples from: $(pwd)"

# Configuration paths (relative to PSLD/stable-diffusion)
CONFIG_DIR="../diffusion-posterior-sampling/configs"

# Style reference images - DINOv2 excels with texture-rich styles
VANGOGH_STYLE="../../../pics/starry_night_full.jpg"
CUBISM_STYLE="../../../pics/cubism_picasso_three-musicians.jpg"

# Model paths
MODEL_CKPT="models/ldm/stable-diffusion-v1/model.ckpt"
MODEL_CONFIG="configs/stable-diffusion/v1-inference.yaml"

# Output directory
OUTPUT_DIR="outputs/dinov2_style_transfer"

# ==============================================
# IMPORTANT: Make sure your style_extraction_config.yaml uses:
#   operator:
#     name: style_retrieval  # DINOv2 (NOT clip_style_retrieval)
# ==============================================

# ==============================================
# EXAMPLE 1: Van Gogh Texture Transfer - Pattern Mode
# DINOv2 captures brush strokes and impasto texture beautifully
# ==============================================
echo ""
echo "🌟 Example 1: Van Gogh Texture Transfer with DINOv2"
echo "   - Using pattern mode for balanced style + content"
echo "   - DINOv2 captures swirling brush strokes and paint texture"
python scripts/inverse_ugd_dinov2.py \
    --use_unified_sampler \
    --schedule_mode pattern \
    --pattern_type even_odd \
    --psld_weight 0.8 \
    --ugd_weight 2.5 \
    --prompt "A gondola gliding through the canals of Venice at night" \
    --style_image "${VANGOGH_STYLE}" \
    --file_id "starry_night_full.jpg" \
    --task_config "${CONFIG_DIR}/style_extraction_config.yaml" \
    --diffusion_config "${CONFIG_DIR}/diffusion_config.yaml" \
    --model_config "${CONFIG_DIR}/model_config.yaml" \
    --config "${MODEL_CONFIG}" \
    --ckpt "${MODEL_CKPT}" \
    --ddim_steps 100 \
    --ddim_eta 0.0 \
    --scale 7.0 \
    --omega 1.0 \
    --gamma 0.08 \
    --general_inverse 1 \
    --optim_forward_guidance \
    --optim_num_steps 8 \
    --optim_forward_guidance_wt 6.0 \
    --k_recur 4 \
    --normalize_grad \
    --outdir "${OUTPUT_DIR}/vangogh_pattern" \
    --seed 42

echo "✅ Example 1 Complete!"
echo ""

# ==============================================
# EXAMPLE 2: Strong Texture Focus - High UGD Weight
# Maximize texture extraction with DINOv2's texture features
# ==============================================
# echo ""
# echo "🌟 Example 2: Van Gogh - Maximum Texture Transfer"
# echo "   - Higher UGD weight to emphasize DINOv2 texture guidance"
# echo "   - Perfect for capturing heavy impasto and directional strokes"
# python scripts/inverse_ugd_dinov2.py \
#     --use_unified_sampler \
#     --schedule_mode pattern \
#     --pattern_type even_odd \
#     --psld_weight 0.5 \
#     --ugd_weight 3.5 \
#     --prompt "A peaceful countryside with rolling hills and cypress trees" \
#     --style_image "${VANGOGH_STYLE}" \
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
#     --k_recur 5 \
#     --normalize_grad \
#     --outdir "${OUTPUT_DIR}/vangogh_texture_max" \
#     --seed 42

# echo "✅ Example 2 Complete!"
# echo ""

# ==============================================
# EXAMPLE 3: Cubist Texture - Geometric Patterns
# DINOv2 captures angular, faceted texture patterns
# ==============================================
# echo ""
# echo "🌟 Example 3: Cubist Style with DINOv2"
# echo "   - DINOv2 captures geometric facets and angular patterns"
# echo "   - Better than CLIP at texture without semantic bias"
# python scripts/inverse_ugd_dinov2.py \
#     --use_unified_sampler \
#     --schedule_mode pattern \
#     --pattern_type even_odd \
#     --psld_weight 1.0 \
#     --ugd_weight 2.0 \
#     --prompt "Musicians playing instruments in a room" \
#     --style_image "${CUBISM_STYLE}" \
#     --file_id "cubism_picasso_three-musicians.jpg" \
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
#     --outdir "${OUTPUT_DIR}/cubist_pattern" \
#     --seed 42

# echo "✅ Example 3 Complete!"
# echo ""

# ==============================================
# EXAMPLE 4: Hybrid Mode - Combined PSLD+UGD
# Use hybrid sampler for strongest texture matching
# ==============================================
# echo ""
# echo "🌟 Example 4: Hybrid Mode with DINOv2"
# echo "   - Combines UGD inner optimization with PSLD consistency"
# echo "   - Best quality for texture-heavy style transfer"
# python scripts/inverse_ugd_dinov2.py \
#     --use_hybrid_sampler \
#     --prompt "A mystical forest with ancient trees under moonlight" \
#     --style_image "${VANGOGH_STYLE}" \
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
#     --outdir "${OUTPUT_DIR}/vangogh_hybrid" \
#     --seed 42

# echo "✅ Example 4 Complete!"
# echo ""

# ==============================================
# EXAMPLE 5: Early-Late Split - Structure then Texture
# PSLD early for structure, DINOv2 UGD late for fine texture
# ==============================================
# echo ""
# echo "🌟 Example 5: Early-Late Split Strategy"
# echo "   - PSLD early for basic structure"
# echo "   - DINOv2 UGD late for fine texture details"
# python scripts/inverse_ugd_dinov2.py \
#     --use_unified_sampler \
#     --schedule_mode early_late \
#     --split_timestep 50 \
#     --psld_weight 1.5 \
#     --ugd_weight 2.5 \
#     --prompt "A lighthouse on rocky cliffs during a storm" \
#     --style_image "${VANGOGH_STYLE}" \
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
#     --outdir "${OUTPUT_DIR}/vangogh_early_late" \
#     --seed 42

# echo "✅ Example 5 Complete!"
# echo ""

# ==============================================
# EXAMPLE 6: Conservative Parameters - Stable Results
# Lower weights for more conservative, stable results
# ==============================================
# echo ""
# echo "🌟 Example 6: Conservative DINOv2 Style Transfer"
# echo "   - Lower weights for stable, subtle texture transfer"
# echo "   - Good starting point for experimentation"
# python scripts/inverse_ugd_dinov2.py \
#     --use_unified_sampler \
#     --schedule_mode pattern \
#     --pattern_type even_odd \
#     --psld_weight 1.2 \
#     --ugd_weight 1.8 \
#     --prompt "A serene Japanese garden with a koi pond" \
#     --style_image "${VANGOGH_STYLE}" \
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
#     --optim_num_steps 5 \
#     --optim_forward_guidance_wt 4.0 \
#     --k_recur 3 \
#     --normalize_grad \
#     --outdir "${OUTPUT_DIR}/vangogh_conservative" \
#     --seed 42

# echo "✅ Example 6 Complete!"
# echo ""

echo ""
echo "🎉 All DINOv2 style transfer examples completed!"
echo "📁 Results saved to: ${OUTPUT_DIR}"
echo ""
echo "💡 Tips for using DINOv2:"
echo "   - Works best with texture-rich styles (Van Gogh, Monet, Impressionism)"
echo "   - Excellent for distinguishing artistic techniques (brush strokes vs flat colors)"
echo "   - Higher UGD weights = stronger texture transfer"
echo "   - Use hybrid mode for maximum quality"
echo "   - Ensure config uses 'style_retrieval' not 'clip_style_retrieval'"

