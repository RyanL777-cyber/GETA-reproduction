"""
Phase 3 — BERT × GETA 介面打通 smoke test
只做 minimum forward/backward/step，不 eval、不匯出、不調超參。
每個 milestone 單行 log，不印 tqdm / verbose，避免 VS Code OOM。
"""
import logging
import os
import subprocess
import sys
import traceback
from datetime import datetime


# --- 自動挑閒置 GPU（必須在 import torch 之前） ---
def select_idle_gpu(max_used_mem_mb=2000, max_util=10):
    """用 nvidia-smi 挑最閒的 GPU 並設 CUDA_VISIBLE_DEVICES。沿用 phase 1 baseline 實作。"""
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        print(f"[gpu] CUDA_VISIBLE_DEVICES already set: {os.environ['CUDA_VISIBLE_DEVICES']}")
        return
    try:
        proc = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True,
        )
    except Exception as exc:
        print(f"[gpu] nvidia-smi not available, skip auto GPU selection: {exc}")
        return
    cands = []
    for line in proc.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 3:
            continue
        try:
            cands.append((int(parts[0]), int(parts[1]), int(parts[2])))
        except ValueError:
            continue
    if not cands:
        print("[gpu] no GPUs found in nvidia-smi output")
        return
    cands.sort(key=lambda x: (x[2], x[1]))
    best_idx, best_mem, best_util = cands[0]
    status = "idle" if (best_util <= max_util and best_mem <= max_used_mem_mb) else "busy-but-best"
    print(f"[gpu] selected GPU {best_idx} ({status}): util={best_util}%, mem_used={best_mem}MiB")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(best_idx)


select_idle_gpu()

# --- 關掉吵雜輸出 ---
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["DATASETS_VERBOSITY"] = "error"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch  # noqa: E402

# --- Logger: 一行一個 milestone，同時寫檔與 stdout ---
_LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
os.makedirs(_LOG_DIR, exist_ok=True)
_LOG_PATH = os.path.join(_LOG_DIR, f"smoke_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

log = logging.getLogger("phase3")
log.setLevel(logging.INFO)
log.propagate = False
_fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S")
_fh = logging.FileHandler(_LOG_PATH, mode="w", encoding="utf-8")
_fh.setFormatter(_fmt)
_sh = logging.StreamHandler(sys.stdout)
_sh.setFormatter(_fmt)
log.addHandler(_fh)
log.addHandler(_sh)

# 把 HF/datasets 的 logger 降噪
import transformers  # noqa: E402
import datasets as hf_datasets  # noqa: E402
transformers.logging.set_verbosity_error()
hf_datasets.logging.set_verbosity_error()


def step(name):
    """milestone 包裝器：失敗時印一行 + traceback 前 10 行後直接 sys.exit(1)。"""
    def deco(fn):
        def wrapped(*args, **kwargs):
            log.info(f"[M] start: {name}")
            try:
                out = fn(*args, **kwargs)
            except Exception as e:
                tb = traceback.format_exc().strip().splitlines()
                log.error(f"[X] FAIL : {name} -> {type(e).__name__}: {e}")
                for line in tb[-10:]:
                    log.error(f"    {line}")
                log.error(f"[log] saved to {_LOG_PATH}")
                sys.exit(1)
            log.info(f"[v] done : {name}")
            return out
        return wrapped
    return deco


# =========================================================================
# Config — 全部來自 README 預設 / Phase 1 baseline，不調
# =========================================================================
MODEL_NAME = "bert-base-uncased"
MAX_LENGTH = 384
DOC_STRIDE = 128
BATCH_SIZE = 2          # smoke，只要能 backward 就好
NUM_SMOKE_BATCHES = 2
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

GETA_KWARGS = dict(
    variant="adam",
    lr=3e-5,
    lr_quant=3e-5,
    target_group_sparsity=0.5,
    bit_reduction=2,
    min_bit_wt=4,
    max_bit_wt=16,
)

# =========================================================================
# M1. load BERT QA + tokenizer
# =========================================================================
@step("M1 load bert-base-uncased")
def m1_load_model():
    from transformers import AutoTokenizer, AutoModelForQuestionAnswering
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    mdl = AutoModelForQuestionAnswering.from_pretrained(MODEL_NAME)
    mdl.to(DEVICE)
    return tok, mdl


# =========================================================================
# M2. model_to_quantize_model
# =========================================================================
@step("M2 model_to_quantize_model (WEIGHT_AND_ACTIVATION)")
def m2_quantize_wrap(model):
    from only_train_once.quantization.quant_model import model_to_quantize_model
    from only_train_once.quantization.quant_layers import QuantizationMode
    wrapped = model_to_quantize_model(model, quant_mode=QuantizationMode.WEIGHT_AND_ACTIVATION)
    return wrapped


# =========================================================================
# M3. 建 dummy_input + OTO
# =========================================================================
@step("M3 build OTO with dict dummy_input")
def m3_build_oto(model, tokenizer):
    # 用真實 tokenize 結果當 dummy_input，避免 id=0 mask 行為奇怪
    sample_q = "What is the capital of France?"
    sample_c = "Paris is the capital and most populous city of France."
    enc = tokenizer(
        sample_q, sample_c,
        max_length=MAX_LENGTH, truncation="only_second",
        padding="max_length", return_tensors="pt",
    )
    dummy_input = {k: v.to(DEVICE) for k, v in enc.items()}
    log.info(f"    dummy keys={list(dummy_input.keys())}  input_ids.shape={tuple(dummy_input['input_ids'].shape)}")

    model.eval()  # tracing 時關 dropout，避免隨機性干擾圖建構
    from only_train_once import OTO
    oto = OTO(model=model, dummy_input=dummy_input)
    model.train()
    return oto


# =========================================================================
# M4. geta optimizer
# =========================================================================
@step("M4 oto.geta(...)")
def m4_build_optimizer(oto):
    optimizer = oto.geta(**GETA_KWARGS)
    log.info(f"    optimizer type={type(optimizer).__name__}")
    return optimizer


# =========================================================================
# M5. 跑 2 個 real batch 的 forward+backward+step
# =========================================================================
def _make_real_batches(tokenizer, n_batches, batch_size):
    """抓 SQuAD 前 n_batches*batch_size 筆跑 phase1 相同 preprocess，回傳 tensor batches。"""
    from datasets import load_dataset
    raw = load_dataset("squad", split=f"train[:{n_batches * batch_size}]")

    def preprocess(examples):
        questions = [q.strip() for q in examples["question"]]
        inputs = tokenizer(
            questions, examples["context"],
            max_length=MAX_LENGTH, truncation="only_second",
            stride=DOC_STRIDE,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
        )
        offset_mapping = inputs.pop("offset_mapping")
        sample_map = inputs.pop("overflow_to_sample_mapping")
        answers = examples["answers"]
        sp, ep = [], []
        for i, offsets in enumerate(offset_mapping):
            a = answers[sample_map[i]]
            if len(a["answer_start"]) == 0:
                sp.append(0); ep.append(0); continue
            sc = a["answer_start"][0]; ec = sc + len(a["text"][0])
            seq_ids = inputs.sequence_ids(i)
            idx = 0
            while seq_ids[idx] != 1: idx += 1
            cs = idx
            idx = len(seq_ids) - 1
            while seq_ids[idx] != 1: idx -= 1
            ce = idx
            if offsets[cs][0] > ec or offsets[ce][1] < sc:
                sp.append(0); ep.append(0)
            else:
                idx = cs
                while idx <= ce and offsets[idx][0] <= sc: idx += 1
                sp.append(idx - 1)
                idx = ce
                while idx >= cs and offsets[idx][1] >= ec: idx -= 1
                ep.append(idx + 1)
        inputs["start_positions"] = sp
        inputs["end_positions"] = ep
        return inputs

    feats = raw.map(preprocess, batched=True, remove_columns=raw.column_names)
    feats.set_format(type="torch")
    batches = []
    for i in range(n_batches):
        b = feats[i * batch_size : (i + 1) * batch_size]
        batches.append({k: v.to(DEVICE) for k, v in b.items()})
    return batches


@step("M5 2-batch forward + backward + step")
def m5_train_loop(model, optimizer, tokenizer):
    model.train()
    batches = _make_real_batches(tokenizer, NUM_SMOKE_BATCHES, BATCH_SIZE)
    for i, batch in enumerate(batches, 1):
        optimizer.zero_grad()
        out = model(**batch)
        loss = out.loss if hasattr(out, "loss") else out[0]
        if loss is None:
            raise RuntimeError(f"batch {i}: model returned no loss (check start/end_positions)")
        loss.backward()
        optimizer.step()
        log.info(f"    batch {i}/{NUM_SMOKE_BATCHES}  loss={loss.item():.4f}")


# =========================================================================
# main
# =========================================================================
def main():
    log.info(f"[log] writing to {_LOG_PATH}")
    log.info(f"[env] torch={torch.__version__}  device={DEVICE}  cuda={torch.cuda.is_available()}")
    tokenizer, model = m1_load_model()
    model = m2_quantize_wrap(model)
    oto = m3_build_oto(model, tokenizer)
    optimizer = m4_build_optimizer(oto)
    m5_train_loop(model, optimizer, tokenizer)
    log.info("[OK] phase 3 smoke test PASSED — BERT × GETA wired up")


if __name__ == "__main__":
    main()
