"""
Phase 4.5 — Exp C2: same diagnostic as Exp C, but on the FIXED quant wrap.

Acceptance criteria (vs Exp C baseline):
  * logit std ratio wrapped/dense  >  0.7     (Exp C: 0.07 — 10× compressed)
  * aggregate sat% mean            <  5%       (Exp C: 32.6%)
  * aggregate qm_act/max|x| min    ≥  0.9      (Exp C: 0.004)
  * grad |grad|/|W| ratio across layers within ~100×  (Exp C: 15,000×)
"""
import os
import sys

_GETA_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "geta")
if os.path.isdir(_GETA_ROOT) and _GETA_ROOT not in sys.path:
    sys.path.insert(0, _GETA_ROOT)

from _common import select_idle_gpu
select_idle_gpu()

os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["DATASETS_VERBOSITY"] = "error"

import torch

from transformers import AutoModelForQuestionAnswering, AutoTokenizer

from _common import MODEL_NAME, load_squad, setup_logger
from quant_fix import apply_ste_fix, calibrate_quant_layers
from only_train_once.quantization.quant_layers import QuantizeLinear
from only_train_once.quantization.quant_model import model_to_quantize_model
from only_train_once.quantization.quant_layers import QuantizationMode

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def hline(title):
    print(f"\n{'='*72}\n  {title}\n{'='*72}")


def stats(t):
    return f"mean={t.mean().item():+.3f}  std={t.std().item():.3f}  min={t.min().item():+.3f}  max={t.max().item():+.3f}"


def main():
    log = setup_logger("./exp_c2_diagnostic.log", "exp_c2")
    log.info(f"[env] device={DEVICE}")

    # =========================================================================
    # Data (need real SQuAD batches for calibration)
    # =========================================================================
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    _, train_ds, _ = load_squad(tokenizer, log)

    # Build 8 calibration batches of bs=4
    CALIB_BS = 4
    CALIB_N = 8
    calib_batches = []
    indices = torch.arange(CALIB_N * CALIB_BS).tolist()
    for i in range(CALIB_N):
        idx = indices[i * CALIB_BS:(i + 1) * CALIB_BS]
        batch = train_ds[idx]
        calib_batches.append({
            k: torch.as_tensor(batch[k]).to(DEVICE)
            for k in ("input_ids", "attention_mask", "token_type_ids")
            if k in batch
        })

    # =========================================================================
    # Build dense + fixed-wrapped models
    # =========================================================================
    torch.manual_seed(42)
    dense = AutoModelForQuestionAnswering.from_pretrained(MODEL_NAME).to(DEVICE).eval()
    torch.manual_seed(42)
    wrapped = AutoModelForQuestionAnswering.from_pretrained(MODEL_NAME).to(DEVICE)

    # Apply STE fix + wrap + calibrate
    apply_ste_fix()
    wrapped = model_to_quantize_model(
        wrapped, quant_mode=QuantizationMode.WEIGHT_AND_ACTIVATION
    ).to(DEVICE)
    calibrate_quant_layers(wrapped, calib_batches, num_bits=16, log=log)
    wrapped.eval()

    # =========================================================================
    # Same diagnostic structure as Exp C, with acceptance checks
    # =========================================================================
    # Use a simple probe input (same as Exp C) for the comparison
    enc = tokenizer(
        "Who wrote Hamlet?", "Hamlet was written by William Shakespeare in 1600.",
        max_length=64, padding="max_length", truncation="only_second", return_tensors="pt",
    )
    inputs = {k: v.to(DEVICE) for k, v in enc.items()}

    hline("(1) Logit parity")
    with torch.no_grad():
        out_d = dense(**inputs)
        out_w = wrapped(**inputs)
    print(f"  dense   start_logits: {stats(out_d.start_logits)}")
    print(f"  wrapped start_logits: {stats(out_w.start_logits)}")
    std_ratio = out_w.start_logits.std().item() / max(out_d.start_logits.std().item(), 1e-9)
    print(f"  std ratio wrapped/dense = {std_ratio:.3f}  "
          f"({'PASS' if std_ratio > 0.7 else 'FAIL'} threshold 0.7)")

    hline("(2-4) Per-layer quant-param sanity")
    act_stats = {}

    def make_hook(name):
        def hook(module, args, out):
            x = args[0].detach()
            x_abs = x.abs()
            q_m_act = module.q_m_act.detach().abs().item()
            clip_val = module.act_clip_val[1]
            act_stats[name] = {
                "max_abs": x_abs.max().item(),
                "sat_frac": (x_abs >= q_m_act).float().mean().item(),
                "grad_zero_frac": ((x >= clip_val) | (x <= -clip_val)).float().mean().item(),
                "q_m_act": q_m_act,
            }
        return hook

    handles = [m.register_forward_hook(make_hook(n))
               for n, m in wrapped.named_modules()
               if isinstance(m, QuantizeLinear)]
    with torch.no_grad():
        wrapped(**inputs)
    for h in handles:
        h.remove()

    items = list(act_stats.items())
    shortlist = items[:6] + [("... (middle omitted) ...", None)] + items[-3:]
    for name, s in shortlist:
        if s is None:
            print(f"  {name}")
            continue
        short = (name.replace("bert.encoder.layer.", "L")
                     .replace(".attention.self.", ".att.")
                     .replace(".attention.output.", ".attout.")
                     .replace(".intermediate.", ".ffn1.")
                     .replace(".output.", ".ffn2."))
        if len(short) > 32:
            short = short[:29] + "..."
        ratio = s["q_m_act"] / max(s["max_abs"], 1e-9)
        print(f"  {short:<32}  qm_act/max|x|={ratio:.3f}  "
              f"sat={s['sat_frac']*100:5.1f}%")

    vals_sat = [s["sat_frac"] for s in act_stats.values()]
    vals_ratio = [s["q_m_act"] / max(s["max_abs"], 1e-9) for s in act_stats.values()]
    mean_sat = sum(vals_sat) / len(vals_sat)
    max_sat = max(vals_sat)
    min_ratio = min(vals_ratio)
    print(f"\n  AGGREGATE over {len(act_stats)} layers:")
    print(f"    sat%            : mean={mean_sat*100:5.2f}%  max={max_sat*100:5.2f}%  "
          f"({'PASS' if mean_sat < 0.05 else 'FAIL'} threshold mean<5%)")
    print(f"    qm_act/max|x|   : min={min_ratio:.3f}  "
          f"({'PASS' if min_ratio >= 0.9 else 'FAIL'} threshold >=0.9)")

    hline("(5) Gradient balance")
    wrapped.train()
    wrapped.zero_grad()
    bs, seq = inputs["input_ids"].shape
    labels = {
        "start_positions": torch.zeros(bs, dtype=torch.long, device=DEVICE),
        "end_positions": torch.zeros(bs, dtype=torch.long, device=DEVICE),
    }
    out_w = wrapped(**inputs, **labels)
    print(f"  loss = {out_w.loss.item():.4f}")
    out_w.loss.backward()

    ratios = []
    for pname in [
        "qa_outputs.weight",
        "bert.encoder.layer.0.output.dense.weight",
        "bert.encoder.layer.5.intermediate.dense.weight",
        "bert.encoder.layer.11.attention.self.query.weight",
    ]:
        p = dict(wrapped.named_parameters()).get(pname)
        if p is None or p.grad is None:
            print(f"  {pname:<55}  (no grad)")
            continue
        gn = p.grad.norm().item()
        wn = p.detach().norm().item()
        r = gn / max(wn, 1e-9)
        ratios.append(r)
        print(f"  {pname:<55}  |grad|/|W|={r:.3e}")

    if ratios:
        spread = max(ratios) / max(min(ratios), 1e-30)
        print(f"\n  grad/weight spread = {spread:.1e}×  "
              f"({'PASS' if spread < 100 else 'FAIL'} threshold <100×)")

    print("\n[done]")


if __name__ == "__main__":
    main()
