# Fix UGD + PSLD Style Transfer Integration

## 📍 Quick Reference - Current Line Numbers

### inverse_ugd.py
- Line 121: `create_style_guidance_function` (fixed to use PSLD operator)
- Line 234: `create_ugd_guidance_config` (accepts operator parameter)
- Line 534: `--use_hybrid_sampler` CLI flag definition
- Lines 586-623: PSLD operator loading
- Line 626: Call to `create_ugd_guidance_config(opt, operator)`
- Lines 629-647: Sampler selection logic (PSLD/UGD/Hybrid)
- Lines 735-794: Sampling call supporting all three modes

### measurements.py
- Line 171: `StyleOperator` class (DINOv2-based)
- Line 426: `CLIPStyleOperator` class (CLIP-based)

### psld_ugd.py (NEW)
- Lines 1-370: Complete unified sampler implementation
- Lines 68-361: `p_sample_ddim` method with UGD+PSLD integration

### test_methods.sh (NEW)
- Lines 1-140: Comprehensive test script for all three modes

---

## Problem Analysis

Current issues identified:

1. **UGD standalone produces terrible results** - much worse than the paper
2. **Root cause**: In `inverse_ugd.py` lines 684-697, UGD mode doesn't pass PSLD's measurement parameters (`measurements`, `operator`, `omega`) to the sampler
3. **Architecture mismatch**: UGD creates its own StyleOperator in `create_style_guidance_function`, separate from PSLD's operator loaded from config
4. **Missing integration**: The two methods run completely independently instead of complementing each other

## Phase 1: Fix UGD Standalone (Priority)

### 1.1 Fix UGD Style Guidance Function ✅ COMPLETED

**File**: `PSLD/stable-diffusion/scripts/inverse_ugd.py`

**Location**: Line 121 - `create_style_guidance_function`

**What was fixed**: Function now accepts `operator` parameter and uses PSLD's existing operator instead of creating a new one.

```python
def create_style_guidance_function(style_image_path, device, operator):
    """Use PSLD's existing operator instead of creating new one"""
    # Extract target style from reference image using PSLD operator
    # Create guidance function that computes style loss in image domain
```

### 1.2 Connect UGD Guidance to PSLD Operator ✅ COMPLETED

**File**: `PSLD/stable-diffusion/scripts/inverse_ugd.py`

**Location**: 
- Line 234: `create_ugd_guidance_config` function definition
- Line 626: Call to `create_ugd_guidance_config(opt, operator)`

**What was fixed**: PSLD operator is now loaded first (lines 586-623) and passed to UGD guidance configuration

### 1.3 Verify Style Operator Gradient Flow ✅ VERIFIED

**File**: `PSLD/diffusion-posterior-sampling/guided_diffusion/measurements.py`

**Locations**: 
- Line 426: `CLIPStyleOperator` class (CLIP-based style features)
- Line 171: `StyleOperator` class (DINOv2-based style features)

**Verified**: Both operators properly preserve gradients for UGD inner optimization:
- Model set to training mode when gradients needed
- Differentiable resize and normalization operations
- No `torch.no_grad()` blocks that would break gradient flow

## Phase 2: Ensure PSLD Continues Working

### 2.1 Test PSLD Style Transfer

**File**: `PSLD/stable-diffusion/scripts/inverse.py`

**Verify**:

- PSLD's gradient-based optimization in `psld.py` lines 233-414
- Style loss computation (lines 283-360)
- Adaptive learning rate and trust-region step size (lines 390-413)

### 2.2 Document PSLD Parameters

**File**: `PSLD/diffusion-posterior-sampling/configs/style_extraction_config.yaml`

Current settings work but should be documented:

- `omega`: measurement error weight (default: 10 from command)
- `scale`: unconditional guidance scale (5 for PSLD, 7.5 for UGD)
- `ddim_eta`: stochasticity (0.5)

## Phase 3: Create Unified PSLD+UGD Sampler

### 3.1 Design Hybrid Sampler ✅ COMPLETED

**New File**: `PSLD/stable-diffusion/ldm/models/diffusion/psld_ugd.py` (370 lines)

**Architecture**:

```python
class PSLDUGDSampler(DDIMSampler):
    """
    Unified sampler combining:
    - PSLD: Outer loop measurement consistency 
    - UGD: Inner loop style guidance with multiple gradient steps
    
    At each diffusion timestep:
    1. Predict x_0 from x_t
    2. UGD inner loop: optimize x_t with style guidance (5 steps)
    3. PSLD outer loop: apply measurement consistency with operator
    4. Compute next x_{t-1} with combined gradients
    """
```

### 3.2 Implement Hybrid p_sample_ddim ✅ COMPLETED

**Location**: `psld_ugd.py` lines 68-361

**Key integration points**:

```python
def p_sample_ddim(self, x, c, t, ...):
    # 1. Initial prediction
    e_t = self.model.apply_model(x, t, c)
    pred_x0 = (x - sqrt_one_minus_at * e_t) / a_t.sqrt()
    
    # 2. UGD INNER OPTIMIZATION (if enabled)
    if guidance_cfg and guidance_cfg.enabled:
        x_t = x.detach().clone().requires_grad_(True)
        for inner_step in range(guidance_cfg.num_steps):
            # Optimize x_t using guidance function
            # This is the UGD contribution
            ...
        x = x_t.detach()
        
    # 3. PSLD MEASUREMENT CONSISTENCY (if enabled)
    if general_inverse and operator:
        # Compute measurement error using operator
        # Apply gradient with adaptive learning rate
        # This is the PSLD contribution
        ...
    
    # 4. Standard DDIM step
    x_prev = a_prev.sqrt() * pred_x0 + dir_xt + noise
    return x_prev, pred_x0
```

### 3.3 Balance Weights Between Methods

**Parameters to tune**:

- `--optim_forward_guidance_wt`: UGD inner step size (default: 5.0)
- `--omega`: PSLD measurement error weight (default: 10)
- `--guidance_weight`: Balance between UGD and PSLD (new parameter, default: 1.0)

## Phase 4: Update CLI and Scripts

### 4.1 Extend inverse_ugd.py ✅ COMPLETED

**Location**: Line 534 in `inverse_ugd.py`

Added hybrid mode flag:

```bash
--use_hybrid_sampler  # Use unified PSLD+UGD sampler
```

**Sampler Selection Logic**: Lines 629-647
- Hybrid mode: Uses `PSLDUGDSampler`
- UGD-only mode: Uses `UGDDDIMSampler`
- PSLD-only mode: Uses standard `DDIMSampler`

**Sampling Call**: Lines 735-794 (supports all three modes)

### 4.2 Create Test Script ✅ COMPLETED

**New file**: `PSLD/stable-diffusion/run/test_methods.sh` (140 lines)

Comprehensive test script that runs all three modes:

```bash
# 1. PSLD only (baseline) - outputs/test-psld-only/
# 2. UGD only (fixed) - outputs/test-ugd-only/
# 3. Hybrid PSLD+UGD (best quality) - outputs/test-hybrid/
```

Includes hyperparameter recommendations and result comparison guide.

## Expected Outcomes

1. **UGD standalone**: Should produce reasonable style transfer results (comparable to paper)
2. **PSLD standalone**: Continues to work as before
3. **Hybrid PSLD+UGD**: Best quality by combining:
   - UGD's powerful inner optimization for style matching
   - PSLD's measurement consistency for structural preservation

## Key Files to Modify

1. `PSLD/stable-diffusion/scripts/inverse_ugd.py` - Fix guidance function
2. `PSLD/stable-diffusion/ldm/models/diffusion/ddim_ugd.py` - Verify gradient flow
3. `PSLD/stable-diffusion/ldm/models/diffusion/psld_ugd.py` - NEW unified sampler
4. `PSLD/diffusion-posterior-sampling/guided_diffusion/measurements.py` - Verify operators
5. `PSLD/stable-diffusion/run/test_methods.sh` - NEW test script

## Implementation Order

**Priority 1**: Fix UGD standalone (sections 1.1-1.3, 2.2)

- Most critical since it's currently broken
- Should take ~2-3 hours

**Priority 2**: Test and verify PSLD still works (section 2.1)

- Quick validation
- Should take ~30 minutes

**Priority 3**: Create unified sampler (sections 3.1-3.3)

- Main integration work
- Should take ~3-4 hours

**Priority 4**: Testing and refinement (section 4)

- Create comprehensive tests
- Tune hyperparameters
- Should take ~1-2 hours

## To-dos

- [x] Fix create_style_guidance_function to use PSLD's operator instead of creating new one
- [x] Pass PSLD operator to UGD guidance configuration and ensure same style extraction
- [x] Verify StyleOperator gradient flow for UGD inner optimization
- [ ] Test PSLD standalone to ensure it still works correctly
- [x] Create PSLDUGDSampler that combines PSLD measurement consistency with UGD inner optimization
- [x] Implement hybrid p_sample_ddim method with both UGD inner loop and PSLD outer loop
- [x] Add --use_hybrid_sampler flag and weight balancing parameters to CLI
- [x] Create test script that compares PSLD-only, UGD-only, and hybrid modes
- [ ] Tune weight balance between UGD and PSLD for optimal style transfer quality

