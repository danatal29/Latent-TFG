#!/bin/bash

# Test script for comparing PSLD-only, UGD-only, and Unified PSLD+UGD methods
# This script runs style transfer with all three approaches and saves results to separate folders

export CUDA_VISIBLE_DEVICES='0'

# Configuration
INPUT_IMAGE='im1.jpg'
STYLE_IMAGE='../../pics/im2.jpg'
PROMPT='A tropic island with a volcano'
CONFIG='configs/style_extraction_config.yaml'
DDIM_STEPS=100
DDIM_ETA=0.5
SCALE_PSLD=5      # Lower scale for PSLD (more conservative)
SCALE_UGD=7.5     # Higher scale for UGD (more aggressive)
OMEGA=10          # PSLD measurement weight
UGD_STEPS=5       # UGD inner optimization steps
UGD_WEIGHT=5.0    # UGD step weight

echo "=========================================="
echo "  Style Transfer Method Comparison Test"
echo "=========================================="
echo ""
echo "Configuration:"
echo "  Input Image: $INPUT_IMAGE"
echo "  Style Image: $STYLE_IMAGE"
echo "  Prompt: $PROMPT"
echo "  DDIM Steps: $DDIM_STEPS"
echo ""

# ========================================
# TEST 1: PSLD-ONLY (Baseline)
# ========================================
echo "=========================================="
echo "TEST 1: PSLD-ONLY (Baseline)"
echo "=========================================="
echo "Method: Measurement consistency with adaptive learning rate"
echo "Expected: Good structural preservation, moderate style transfer"
echo ""

python scripts/inverse.py \
    --file_id="$INPUT_IMAGE" \
    --task_config="$CONFIG" \
    --outdir='outputs/test-psld-only' \
    --prompt="$PROMPT" \
    --ddim_eta=$DDIM_ETA \
    --omega=$OMEGA \
    --scale=$SCALE_PSLD \
    --ddim_steps=$DDIM_STEPS \
    --general_inverse=1

echo ""
echo "✅ PSLD-only test complete. Results in: outputs/test-psld-only/"
echo ""
sleep 2

# ========================================
# TEST 2: UGD-ONLY (Fixed Implementation)
# ========================================
echo "=========================================="
echo "TEST 2: UGD-ONLY (Fixed Implementation)"
echo "=========================================="
echo "Method: Inner optimization loop with style guidance"
echo "Expected: Strong style matching, may have artifacts"
echo ""

python scripts/inverse_ugd.py \
    --file_id="$INPUT_IMAGE" \
    --task_config="$CONFIG" \
    --outdir='outputs/test-ugd-only' \
    --prompt="$PROMPT" \
    --ddim_eta=$DDIM_ETA \
    --scale=$SCALE_UGD \
    --ddim_steps=$DDIM_STEPS \
    --optim_forward_guidance \
    --style_image="$STYLE_IMAGE" \
    --optim_num_steps=$UGD_STEPS \
    --optim_forward_guidance_wt=$UGD_WEIGHT \
    --guidance_domain='image' \
    --general_inverse=0

echo ""
echo "✅ UGD-only test complete. Results in: outputs/test-ugd-only/"
echo ""
sleep 2

# ========================================
# TEST 3: UNIFIED PSLD+UGD (Hybrid)
# ========================================
echo "=========================================="
echo "TEST 3: UNIFIED PSLD+UGD (Hybrid - Best)"
echo "=========================================="
echo "Method: UGD inner optimization + PSLD measurement consistency"
echo "Expected: Best quality - strong style + good structure"
echo ""

python scripts/inverse_ugd.py \
    --file_id="$INPUT_IMAGE" \
    --task_config="$CONFIG" \
    --outdir='outputs/test-hybrid' \
    --prompt="$PROMPT" \
    --ddim_eta=$DDIM_ETA \
    --scale=$SCALE_UGD \
    --ddim_steps=$DDIM_STEPS \
    --optim_forward_guidance \
    --style_image="$STYLE_IMAGE" \
    --optim_num_steps=$UGD_STEPS \
    --optim_forward_guidance_wt=$UGD_WEIGHT \
    --guidance_domain='image' \
    --omega=$OMEGA \
    --general_inverse=1 \
    --use_hybrid_sampler

echo ""
echo "✅ Hybrid test complete. Results in: outputs/test-hybrid/"
echo ""

# ========================================
# Summary
# ========================================
echo "=========================================="
echo "  ALL TESTS COMPLETE!"
echo "=========================================="
echo ""
echo "Results summary:"
echo "  1. PSLD-only:  outputs/test-psld-only/samples/"
echo "  2. UGD-only:   outputs/test-ugd-only/samples/"
echo "  3. Hybrid:     outputs/test-hybrid/samples/"
echo ""
echo "Compare the results to evaluate:"
echo "  - PSLD: Good structure, moderate style"
echo "  - UGD:  Strong style, may have artifacts"
echo "  - Hybrid: Best of both - strong style + good structure"
echo ""
echo "Hyperparameter tuning recommendations:"
echo "  - If UGD too strong: decrease --optim_forward_guidance_wt"
echo "  - If PSLD too weak: increase --omega"
echo "  - For more style: increase --optim_num_steps"
echo "  - For more structure: increase --omega in hybrid mode"
echo ""
echo "=========================================="

