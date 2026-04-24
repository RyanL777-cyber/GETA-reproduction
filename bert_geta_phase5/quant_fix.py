"""
Phase 4.5 quantization fix: calibration + STE backward repair.

Root causes identified in Exp C diagnostic:

  H1. `initialize_quant_layer` in `geta/only_train_once/quantization/quant_layers.py`
      sets activation quant params (`q_m_act`, `d_quant_act`) from WEIGHT
      statistics, not activation statistics. For BERT this makes `q_m_act`
      ~O(0.1) while real activations are O(1-10) → massive saturation
      (avg 32%, up to 76% for qa_outputs input).

  H2. `SymQuantizerNonLinear.backward` only zeros gradient where
      `|x| >= clip_val` (hardcoded 2.0), NOT where forward already saturates
      (`|x| >= q_m`). So saturated region's forward is constant but backward
      acts like identity → gradient imbalance of ~15000x across layers.

This module provides two fixes applied together, without modifying the
upstream GETA source:

  fix 1 (STE): monkey-patch `SymQuantizerNonLinear.backward` (and the linear
  variant) to ALSO zero grad where `|x| >= q_m`.

  fix 2 (calibration): after `model_to_quantize_model`, run a few forward
  passes on real data with activation-quant temporarily disabled, record
  per-layer `max|activation|`, set `q_m_act = observed_max * safety_margin`
  and derive `d_quant_act` from num_bits.
"""
import logging

import torch

from only_train_once.quantization.quant_layers import (
    QuantizationMode,
    QuantizeLinear,
    SymQuantizerLinear,
    SymQuantizerNonLinear,
)

logger = logging.getLogger(__name__)


# =========================================================================
# Fix 1 — STE backward repair
# =========================================================================
_STE_FIX_APPLIED = False


def _nonlinear_backward_fixed(ctx, grad_output):
    """Replacement for SymQuantizerNonLinear.backward.

    Delta vs upstream: also zero grad_x where `|x| >= q_m` or `|x| <= q_s`.
    The rest (gradients for d_quant, q_m, t_quant) is kept identical.
    """
    input, d_quant, q_m, t_quant, clip_val, q_s = ctx.saved_tensors
    device = input.device
    input_abs = torch.abs(input)

    grad_x = grad_output.clone()
    grad_x[input.ge(clip_val[1])] = 0
    grad_x[input.le(clip_val[0])] = 0
    # FIX: forward output is CONSTANT where |x|>=q_m (line 67 of quant_layers)
    # and ZERO where |x|<=q_s (line 66). Both regions have dy/dx = 0.
    grad_x[input_abs >= q_m] = 0
    grad_x[input_abs <= q_s] = 0

    range_pow = torch.exp(t_quant * torch.log(torch.abs(q_m - q_s) + 1e-6))
    range_pow_low = torch.exp((t_quant - 1) * torch.log(torch.abs(q_m - q_s) + 1e-6))
    input_pow = torch.exp(t_quant * torch.log(input_abs - q_s))

    grad_d_xq = torch.round(input_pow.div(d_quant)) - input_pow.div(d_quant)
    grad_d_xq[input_abs >= q_m] = torch.round(range_pow.div(d_quant)) - range_pow.div(d_quant)
    grad_d_xq[input_abs <= q_s] = 0
    grad_d_xq = torch.sign(input) * grad_d_xq
    grad_d = torch.tensor([torch.sum(grad_output * grad_d_xq)], device=device)

    grad_qm_xq = torch.sign(input) * ((t_quant * range_pow_low).expand_as(input))
    grad_qm_xq[input_abs <= q_m] = 0
    grad_qm = torch.tensor([torch.sum(grad_output * grad_qm_xq)], device=device)

    grad_t_xq = input_pow * (torch.log(input_abs - q_s))
    grad_t_xq[input_abs >= q_m] = range_pow * torch.log(torch.abs(q_m - q_s) + 1e-6)
    grad_t_xq[input_abs <= q_s] = 0
    grad_t_xq = torch.sign(input) * grad_t_xq
    grad_t = torch.tensor([torch.sum(grad_output * grad_t_xq)], device=device)

    return grad_x, grad_d, grad_qm, grad_t, None, None


def _linear_backward_fixed(ctx, grad_output):
    """Replacement for SymQuantizerLinear.backward (mirror of the nonlinear fix)."""
    input, d_quant, q_m, clip_val, q_s = ctx.saved_tensors
    device = input.device
    input_abs = torch.abs(input)

    grad_x = grad_output.clone()
    grad_x[input.ge(clip_val[1])] = 0
    grad_x[input.le(clip_val[0])] = 0
    grad_x[input_abs >= q_m] = 0
    grad_x[input_abs <= q_s] = 0

    range_pow = torch.abs(q_m - q_s)
    input_pow = input_abs - q_s

    grad_d_xq = torch.round(input_pow.div(d_quant)) - input_pow.div(d_quant)
    grad_d_xq[input_abs >= q_m] = torch.round(range_pow.div(d_quant)) - range_pow.div(d_quant)
    grad_d_xq[input_abs <= q_s] = 0
    grad_d_xq = torch.sign(input) * grad_d_xq
    grad_d = torch.tensor([torch.sum(grad_output * grad_d_xq)], device=device)

    grad_qm_xq = torch.sign(input)
    grad_qm_xq[input_abs <= q_m] = 0
    grad_qm = torch.tensor([torch.sum(grad_output * grad_qm_xq)], device=device)

    return grad_x, grad_d, grad_qm, None, None


def apply_ste_fix():
    """Monkey-patch the upstream quantizer classes. Idempotent."""
    global _STE_FIX_APPLIED
    if _STE_FIX_APPLIED:
        return
    SymQuantizerNonLinear.backward = staticmethod(_nonlinear_backward_fixed)
    SymQuantizerLinear.backward = staticmethod(_linear_backward_fixed)
    _STE_FIX_APPLIED = True
    logger.info("STE fix applied to SymQuantizerNonLinear and SymQuantizerLinear")


# =========================================================================
# Fix 2 — activation calibration
# =========================================================================
def calibrate_quant_layers(
    model,
    calib_batches,
    num_bits=16,
    safety_margin=1.05,
    log=None,
):
    """Calibrate `q_m_act` / `d_quant_act` on every QuantizeLinear.

    Args:
        model: model already passed through `model_to_quantize_model`.
        calib_batches: iterable of dicts; each dict is forwarded as
                       `model(**batch)`. Typically 4-8 batches is enough.
        num_bits: target bit width (must match `num_bits` passed to
                  `model_to_quantize_model`; default 16).
        safety_margin: multiply observed max by this to leave headroom
                       for unseen activations.
        log: logger-like object with `.info(msg)`. Optional.

    Returns:
        dict {layer_name: observed_max_abs} for inspection.
    """
    _log = (log.info if log is not None else print)

    quant_layers = [(n, m) for n, m in model.named_modules()
                    if isinstance(m, QuantizeLinear)
                    and m.quant_mode == QuantizationMode.WEIGHT_AND_ACTIVATION]
    if not quant_layers:
        _log("[calib] no QuantizeLinear layers with WEIGHT_AND_ACTIVATION mode; nothing to do")
        return {}

    # Temporarily disable activation quantization so we observe the TRUE
    # activation distribution, not one that has already been clipped by a
    # badly-initialized q_m_act.
    saved_modes = {}
    for name, mod in quant_layers:
        saved_modes[name] = mod.quant_mode
        mod.quant_mode = QuantizationMode.WEIGHT_ONLY

    # Hooks record per-layer max|input|.
    layer_maxabs = {name: 0.0 for name, _ in quant_layers}

    def make_hook(layer_name):
        def hook(module, args, out):
            x = args[0].detach()
            cur = x.abs().max().item()
            if cur > layer_maxabs[layer_name]:
                layer_maxabs[layer_name] = cur
        return hook

    handles = [mod.register_forward_hook(make_hook(name)) for name, mod in quant_layers]

    was_training = model.training
    model.eval()
    n_seen = 0
    with torch.no_grad():
        for batch in calib_batches:
            if isinstance(batch, dict):
                model(**batch)
            else:
                model(*batch)
            n_seen += 1
    if was_training:
        model.train()

    for h in handles:
        h.remove()

    # Restore quant modes
    for name, mod in quant_layers:
        mod.quant_mode = saved_modes[name]

    # Apply new q_m_act / d_quant_act
    qmax_divisor = (2 ** (num_bits - 1)) - 1
    skipped = 0
    for name, mod in quant_layers:
        obs_max = layer_maxabs[name]
        if obs_max <= 0:
            skipped += 1
            continue
        new_qm = obs_max * safety_margin
        new_d = new_qm / qmax_divisor
        with torch.no_grad():
            mod.q_m_act.fill_(new_qm)
            mod.d_quant_act.fill_(new_d)

    _log(f"[calib] {n_seen} batches over {len(quant_layers)} layers  "
         f"(skipped={skipped})  num_bits={num_bits}  safety={safety_margin}")
    vals = [v for v in layer_maxabs.values() if v > 0]
    if vals:
        _log(f"[calib] observed max|x|: min={min(vals):.3f}  "
             f"median={sorted(vals)[len(vals)//2]:.3f}  max={max(vals):.3f}")
    return layer_maxabs


# =========================================================================
# Convenience wrapper
# =========================================================================
def wrap_quant_fixed(model, device, calib_batches, num_bits=16, log=None):
    """Apply STE fix, wrap model, then calibrate activation quant params.

    Args:
        model: un-wrapped nn.Module on `device`.
        device: 'cuda' | 'cpu'.
        calib_batches: iterable of dicts for calibration forward passes.
        num_bits: weight/activation bit width (default 16 matches upstream).
        log: logger-like. Optional.

    Returns:
        Wrapped + calibrated model (same object, quant_mode preserved).
    """
    from _common import wrap_quant
    apply_ste_fix()
    model = wrap_quant(model, device)
    calibrate_quant_layers(model, calib_batches, num_bits=num_bits, log=log)
    return model
