# bert_geta_clean

Clean reference implementation for reproducing **GETA (CVPR 2025) joint pruning + quantization on BERT-base SQuAD**.

The upstream GETA / `only_train_once` codebase has 3 issues that only surface when
joint pruning + quantization runs on BERT (CV tutorials don't trigger them).
This folder packages everything needed to bypass them with minimal effort.

## Setup (~5 min)

```bash
# 1. Clone GETA / OTO upstream into ./geta/
git clone <oto-upstream-url> geta

# 2. Apply 2 source patches
cd geta
git apply ../docs/patches/01-graph-dedup.patch
git apply ../docs/patches/02-pruning-compression.patch
cd ..

# 3. Edit launchers/run.sh (KNOBS section) and run it
bash launchers/run.sh
```

That's it. No further GETA edits needed. `quant_fix.py` is auto-loaded by
`src/run_experiment.py`; nothing to copy by hand.

## What's in here

| Path | Status | Purpose |
|---|---|---|
| `docs/BERT_GETA_PATCH_GUIDE.md` | ✅ | All 3 fixes: symptom, root cause, fix, verify |
| `docs/EXPERIMENT_RESULTS.md` | ✅ | Reproduced numbers vs paper |
| `docs/patches/01-graph-dedup.patch` | ✅ | Source patch #1 (graph.py) |
| `docs/patches/02-pruning-compression.patch` | ✅ | Source patch #2 (pruning_compression.py) |
| `src/run_experiment.py` | ✅ | Training driver |
| `src/quant_fix.py` | ✅ | Runtime monkey-patch (no upstream edit) |
| `src/build_table.py` | ✅ | Build paper-Table-3-style summary from results |
| `launchers/run.sh` | ✅ | Editable launcher template (KNOBS/PLUMBING split) |
| `launchers/PRESETS.md` | ✅ | Best-result / paper-faithful / our actual runs (full CLIs) |

## Why 3 fixes?

| # | Where | Type |
|---|---|---|
| 1 | `graph.py` dedup order | upstream bug |
| 2 | `pruning_compression.py` orphaned in-dim | upstream bug |
| 3 | `quant_fix.py` (calibration + STE) | runtime patch (1 upstream bug + 1 design issue) |

Read [docs/BERT_GETA_PATCH_GUIDE.md](docs/BERT_GETA_PATCH_GUIDE.md) for the full story.
