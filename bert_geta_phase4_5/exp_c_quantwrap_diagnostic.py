"""
Phase 4.5 — Exp C: Quant wrap diagnostic.

Exp A + B both showed F1 stuck at ~27-29 — quant wrap is the culprit.
This script verifies WHY without running full training (<30s).

Hypotheses from code review:

  H1. `initialize_quant_layer` sets q_m_act = max(|weight|), but activations
      have very different scale (BERT weights ~0.1-0.5, activations ~1-10).
      → almost all activations fall in the `|x| >= q_m_act` branch and
      get saturated to a single quantized value (effectively 1-bit).

  H2. `weight_clip_val` and `act_clip_val` are hardcoded (-2, 2). BERT
      activations routinely exceed ±2 → backward zeroes out those grads.

Script prints:
  1. Logit diff: same input through model before vs after wrap
  2. Per-layer: q_m_wt / max|weight|,  q_m_act / max|activation|
  3. Per-layer: fraction of activations saturated (|x| >= q_m_act)
  4. Per-layer: fraction of activations grad-zeroed (|x| >= clip_val)
  5. qa_outputs.weight.grad norm after 1 backward step (STE sanity check)
"""
import os
import sys

# --- GETA source path ---
_GETA_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "geta")
if os.path.isdir(_GETA_ROOT) and _GETA_ROOT not in sys.path:
    sys.path.insert(0, _GETA_ROOT)

from _common import select_idle_gpu
select_idle_gpu()

os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch

from transformers import AutoModelForQuestionAnswering, AutoTokenizer

from _common import MODEL_NAME, wrap_quant
from only_train_once.quantization.quant_layers import QuantizeLinear

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def hline(title):
    print(f"\n{'='*72}\n  {title}\n{'='*72}")


def main():
    # =========================================================================
    # Build identical inputs
    # =========================================================================
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    enc = tokenizer(
        "Who wrote Hamlet?", "Hamlet was written by William Shakespeare in 1600.",
        max_length=64, padding="max_length", truncation="only_second", return_tensors="pt",
    )
    inputs = {k: v.to(DEVICE) for k, v in enc.items()}

    # =========================================================================
    # Build dense + wrapped models (identical weights)
    # =========================================================================
    torch.manual_seed(42)
    dense = AutoModelForQuestionAnswering.from_pretrained(MODEL_NAME).to(DEVICE).eval()
    torch.manual_seed(42)
    wrapped = AutoModelForQuestionAnswering.from_pretrained(MODEL_NAME).to(DEVICE)
    wrapped = wrap_quant(wrapped, DEVICE).eval()

    # =========================================================================
    # (1) Logit diff: dense vs wrapped on same input
    # =========================================================================
    hline("(1) Logit diff: dense BERT vs quant-wrapped BERT (same input)")
    with torch.no_grad():
        out_d = dense(**inputs)
        out_w = wrapped(**inputs)

    def stats(t):
        return f"mean={t.mean().item():+.3f}  std={t.std().item():.3f}  min={t.min().item():+.3f}  max={t.max().item():+.3f}"

    print(f"  dense   start_logits: {stats(out_d.start_logits)}")
    print(f"  wrapped start_logits: {stats(out_w.start_logits)}")
    diff = (out_w.start_logits - out_d.start_logits).abs()
    print(f"  |diff|  start_logits: {stats(diff)}")
    if diff.max().item() > 1.0:
        print("  >> Wrapped logits differ SIGNIFICANTLY from dense. Forward is distorted.")

    # =========================================================================
    # (2/3/4) Per-layer quant-param sanity: capture real activation stats
    # =========================================================================
    hline("(2-4) Per-layer quant parameters vs observed activation stats")
    print("  Columns: layer | q_m_wt / max|W| | q_m_act / max|x| | sat% | gradZero%")
    print(f"  q_m_act / max|x| < 1  means activation range EXCEEDS quant range → saturation")

    # Hook wrapped QuantizeLinear layers to capture their input activation.
    act_stats = {}  # name -> dict(max_abs, sat_frac, grad_zero_frac)

    def make_hook(name):
        def hook(module, args, out):
            x = args[0].detach()
            x_abs = x.abs()
            q_m_act = module.q_m_act.detach().abs().item()
            clip_val = module.act_clip_val[1]  # (-2, 2) so symmetric
            sat = (x_abs >= q_m_act).float().mean().item()
            gzero = ((x >= clip_val) | (x <= -clip_val)).float().mean().item()
            act_stats[name] = {
                "max_abs": x_abs.max().item(),
                "sat_frac": sat,
                "grad_zero_frac": gzero,
                "q_m_act": q_m_act,
                "q_m_wt": module.q_m_wt.detach().abs().item(),
                "max_w": module.weight.detach().abs().max().item(),
                "clip_val": clip_val,
            }
        return hook

    handles = []
    for name, module in wrapped.named_modules():
        if isinstance(module, QuantizeLinear):
            handles.append(module.register_forward_hook(make_hook(name)))

    with torch.no_grad():
        wrapped(**inputs)

    for h in handles:
        h.remove()

    # Print first 8 and last 4 for brevity, plus aggregate stats
    items = list(act_stats.items())
    shortlist = items[:8] + [("... (middle layers omitted) ...", None)] + items[-4:]
    for name, s in shortlist:
        if s is None:
            print(f"  {name}")
            continue
        short_name = name.replace("bert.encoder.layer.", "L").replace(".attention.self.", ".att.").replace(".attention.output.", ".attout.").replace(".intermediate.", ".ffn1.").replace(".output.", ".ffn2.")
        if len(short_name) > 32:
            short_name = short_name[:29] + "..."
        print(f"  {short_name:<32}  "
              f"qm_wt/max|W|={s['q_m_wt']/max(s['max_w'],1e-9):.2f}  "
              f"qm_act/max|x|={s['q_m_act']/max(s['max_abs'],1e-9):.3f}  "
              f"sat={s['sat_frac']*100:5.1f}%  "
              f"gz={s['grad_zero_frac']*100:4.1f}%")

    # Aggregate
    vals_sat = [s["sat_frac"] for s in act_stats.values()]
    vals_gz = [s["grad_zero_frac"] for s in act_stats.values()]
    vals_ratio = [s["q_m_act"] / max(s["max_abs"], 1e-9) for s in act_stats.values()]
    print(f"\n  AGGREGATE over {len(act_stats)} QuantizeLinear layers:")
    print(f"    sat%       : mean={sum(vals_sat)/len(vals_sat)*100:5.1f}%  "
          f"max={max(vals_sat)*100:5.1f}%")
    print(f"    gradZero%  : mean={sum(vals_gz)/len(vals_gz)*100:5.1f}%  "
          f"max={max(vals_gz)*100:5.1f}%")
    print(f"    qm_act/max|x|: mean={sum(vals_ratio)/len(vals_ratio):.3f}  "
          f"min={min(vals_ratio):.3f}  max={max(vals_ratio):.3f}")

    if sum(vals_sat) / len(vals_sat) > 0.5:
        print("  >> H1 CONFIRMED: >50% of activations are SATURATED on average.")
    if sum(vals_gz) / len(vals_gz) > 0.1:
        print("  >> H2 CONFIRMED: >10% of activation gradients are zeroed by clip_val.")

    # =========================================================================
    # (5) STE sanity: grad on qa_outputs and a mid-encoder layer after 1 step
    # =========================================================================
    hline("(5) STE sanity: one backward pass, grad norms on selected weights")
    wrapped.train()
    wrapped.zero_grad()
    # Need labels for QA loss
    bs, seq = inputs["input_ids"].shape
    labels = {
        "start_positions": torch.zeros(bs, dtype=torch.long, device=DEVICE),
        "end_positions": torch.zeros(bs, dtype=torch.long, device=DEVICE),
    }
    out_w = wrapped(**inputs, **labels)
    loss = out_w.loss
    print(f"  loss = {loss.item():.4f}")
    loss.backward()

    # Inspect grads on a few strategic params
    for pname in [
        "qa_outputs.weight",
        "bert.encoder.layer.0.output.dense.weight",
        "bert.encoder.layer.5.intermediate.dense.weight",
        "bert.encoder.layer.11.attention.self.query.weight",
    ]:
        p = dict(wrapped.named_parameters()).get(pname)
        if p is None or p.grad is None:
            print(f"  {pname:<55}  grad = (none)")
            continue
        gn = p.grad.norm().item()
        wn = p.detach().norm().item()
        print(f"  {pname:<55}  |grad|={gn:.3e}  |W|={wn:.3e}  |grad|/|W|={gn/max(wn,1e-9):.3e}")

    print("\n[done]")


if __name__ == "__main__":
    main()
