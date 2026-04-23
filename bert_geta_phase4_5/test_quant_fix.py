"""
Phase 4.5 — fast unit tests for quant_fix.py.

Run:  python3 test_quant_fix.py
Takes <20s. No SQuAD data needed, uses synthetic inputs.

Tests:
  T1. STE backward: grad_x = 0 where |x| >= q_m   (verifies STE fix)
  T2. STE backward: grad_x != 0 where |x| < q_m   (identity STE preserved)
  T3. Calibration sets q_m_act to observed activation max (within safety margin)
  T4. Calibration skips WEIGHT_ONLY layers (no-op for non-WA mode)
"""
import os
import sys
import traceback

_GETA_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "geta")
if os.path.isdir(_GETA_ROOT) and _GETA_ROOT not in sys.path:
    sys.path.insert(0, _GETA_ROOT)

os.environ["CUDA_VISIBLE_DEVICES"] = ""  # force CPU for tests; fast + deterministic

import torch
import torch.nn as nn

from only_train_once.quantization.quant_layers import (
    QuantizationMode,
    QuantizationType,
    QuantizeLinear,
    SymQuantizerNonLinear,
)

from quant_fix import apply_ste_fix, calibrate_quant_layers


PASS, FAIL = "\033[92mPASS\033[0m", "\033[91mFAIL\033[0m"


def run(name, fn):
    try:
        fn()
        print(f"  {PASS}  {name}")
        return True
    except AssertionError as e:
        print(f"  {FAIL}  {name}")
        print(f"         {e}")
        return False
    except Exception:
        print(f"  {FAIL}  {name}")
        print("         " + traceback.format_exc().replace("\n", "\n         "))
        return False


# =========================================================================
# T1 + T2: STE backward correctness
# =========================================================================
def _run_ste_backward(x_vals, q_m_val=1.0):
    """Helper: forward + backward SymQuantizerNonLinear on explicit input."""
    apply_ste_fix()
    x = torch.tensor(x_vals, requires_grad=True)
    d_quant = torch.tensor([0.01])
    q_m = torch.tensor([q_m_val])
    t_quant = torch.tensor([1.0])
    clip_val = torch.tensor([-10.0, 10.0])  # wider than q_m so clip_val is not what matters
    q_s = torch.tensor(0.0)
    out = SymQuantizerNonLinear.apply(x, d_quant, q_m, t_quant, clip_val, q_s)
    out.sum().backward()
    return x.grad.detach()


def test_ste_zero_in_saturated():
    """|x| >= q_m  →  grad_x = 0"""
    grads = _run_ste_backward([1.5, 2.0, -2.5, 5.0], q_m_val=1.0)
    for i, g in enumerate(grads):
        assert g.abs().item() < 1e-6, f"idx {i}: expected 0, got {g.item()}"


def test_ste_passthrough_below_qm():
    """|x| < q_m  →  grad_x passes through (identity STE)"""
    grads = _run_ste_backward([0.1, 0.3, -0.5, 0.7], q_m_val=1.0)
    for i, g in enumerate(grads):
        assert g.abs().item() > 0.9, f"idx {i}: expected ~1, got {g.item()}"


# =========================================================================
# T3: calibration sets q_m_act correctly
# =========================================================================
def test_calibration_sets_qm_act():
    linear = nn.Linear(64, 32)
    q_linear = QuantizeLinear.from_module(
        module=linear,
        quant_type=QuantizationType.SYMMETRIC_NONLINEAR,
        quant_mode=QuantizationMode.WEIGHT_AND_ACTIVATION,
        num_bits=16,
    )
    # Before calibration: q_m_act derived from weight stats (~max|W| ≈ O(0.1))
    initial_qm = q_linear.q_m_act.item()

    # Feed inputs with max|x| ≈ 5.0
    torch.manual_seed(0)
    batches = [(torch.randn(2, 64) * 2.0,) for _ in range(4)]
    observed_max = max(b[0].abs().max().item() for b in batches)

    calibrate_quant_layers(q_linear, batches, num_bits=16, safety_margin=1.05)

    new_qm = q_linear.q_m_act.item()
    assert new_qm != initial_qm, "q_m_act was not updated"
    assert new_qm >= observed_max, f"q_m_act={new_qm} < observed_max={observed_max}"
    assert new_qm <= observed_max * 1.1, f"q_m_act={new_qm} too high vs observed={observed_max}"

    # d_quant_act must match: new_qm / (2^15 - 1)
    expected_d = new_qm / (2**15 - 1)
    actual_d = q_linear.d_quant_act.item()
    assert abs(actual_d - expected_d) < 1e-8, f"d_quant_act={actual_d} expected {expected_d}"


# =========================================================================
# T4: calibration skips weight-only layers
# =========================================================================
def test_calibration_skips_weight_only():
    linear = nn.Linear(64, 32)
    q_linear = QuantizeLinear.from_module(
        module=linear,
        quant_type=QuantizationType.SYMMETRIC_NONLINEAR,
        quant_mode=QuantizationMode.WEIGHT_ONLY,  # <-- weight-only
        num_bits=16,
    )
    # WEIGHT_ONLY mode does not create d_quant_act / q_m_act params at all,
    # so calibration must be a no-op (no AttributeError).
    torch.manual_seed(0)
    batches = [(torch.randn(2, 64),) for _ in range(4)]
    observed = calibrate_quant_layers(q_linear, batches, num_bits=16)
    assert observed == {}, f"expected empty, got {observed}"
    assert not hasattr(q_linear, "q_m_act"), "WEIGHT_ONLY layer should not have q_m_act"


# =========================================================================
# Runner
# =========================================================================
def main():
    print("Phase 4.5 quant_fix unit tests")
    print("=" * 60)
    results = [
        run("T1 STE grad_x = 0 for |x| >= q_m", test_ste_zero_in_saturated),
        run("T2 STE grad_x != 0 for |x| < q_m", test_ste_passthrough_below_qm),
        run("T3 calibration sets q_m_act to observed max", test_calibration_sets_qm_act),
        run("T4 calibration skips WEIGHT_ONLY layers", test_calibration_skips_weight_only),
    ]
    print("=" * 60)
    if all(results):
        print(f"  ALL {len(results)} TESTS PASSED")
        sys.exit(0)
    else:
        print(f"  {sum(results)}/{len(results)} passed, {sum(1 for r in results if not r)} failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
