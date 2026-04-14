import collections
import os
import sys
from datetime import datetime

import numpy as np
from datasets import load_dataset
import evaluate
from transformers import (
    AutoTokenizer,
    AutoModelForQuestionAnswering,
    TrainingArguments,
    Trainer,
    DefaultDataCollator,
)

# =========================
# stdout / stderr tee → logs/run_<timestamp>.log
# 同步寫到終端與檔案，跑完不用自己複製
# =========================
class _Tee:
    def __init__(self, *streams):
        self.streams = streams
    def write(self, data):
        for s in self.streams:
            try:
                s.write(data)
            except Exception:
                pass
    def flush(self):
        for s in self.streams:
            try:
                s.flush()
            except Exception:
                pass
    def isatty(self):
        return False
    def fileno(self):
        return self.streams[0].fileno()
    def __getattr__(self, name):
        return getattr(self.streams[0], name)

_LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
os.makedirs(_LOG_DIR, exist_ok=True)
_LOG_PATH = os.path.join(_LOG_DIR, f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
_log_file = open(_LOG_PATH, "w", encoding="utf-8", buffering=1)
sys.stdout = _Tee(sys.__stdout__, _log_file)
sys.stderr = _Tee(sys.__stderr__, _log_file)
print(f"[log] writing to {_LOG_PATH}")

# =========================
# Config
# =========================
MODEL_NAME = "bert-base-uncased"
MAX_LENGTH = 384
DOC_STRIDE = 128

# 資料規模：None = 使用全部 (server 跑 full baseline 用)
# 本機 smoke test: TRAIN_SAMPLES = 64, VAL_SAMPLES = 32
# Server full run : TRAIN_SAMPLES = None, VAL_SAMPLES = None
TRAIN_SAMPLES = 64
VAL_SAMPLES = 32

# 訓練超參：smoke 與 full 都用這組，只差資料量
NUM_TRAIN_EPOCHS = 2
TRAIN_BATCH_SIZE = 12
EVAL_BATCH_SIZE = 32
LEARNING_RATE = 3e-5
WEIGHT_DECAY = 0.01

OUTPUT_DIR = "./bert_squad_baseline"

# =========================
# Load dataset / tokenizer / model
# =========================
raw_datasets = load_dataset("squad")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForQuestionAnswering.from_pretrained(MODEL_NAME)

train_dataset = (
    raw_datasets["train"]
    if TRAIN_SAMPLES is None
    else raw_datasets["train"].select(range(TRAIN_SAMPLES))
)
eval_dataset = (
    raw_datasets["validation"]
    if VAL_SAMPLES is None
    else raw_datasets["validation"].select(range(VAL_SAMPLES))
)
print(f"[data] train={len(train_dataset)}  val={len(eval_dataset)}")

# =========================
# Preprocessing for training
# =========================
def preprocess_training_examples(examples):
    questions = [q.strip() for q in examples["question"]]

    inputs = tokenizer(
        questions,
        examples["context"],
        max_length=MAX_LENGTH,
        truncation="only_second",
        stride=DOC_STRIDE,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    offset_mapping = inputs.pop("offset_mapping")
    sample_map = inputs.pop("overflow_to_sample_mapping")
    answers = examples["answers"]

    start_positions = []
    end_positions = []

    for i, offsets in enumerate(offset_mapping):
        sample_idx = sample_map[i]
        answer = answers[sample_idx]

        if len(answer["answer_start"]) == 0:
            start_positions.append(0)
            end_positions.append(0)
            continue

        start_char = answer["answer_start"][0]
        end_char = start_char + len(answer["text"][0])

        sequence_ids = inputs.sequence_ids(i)

        # 找 context 的起點與終點 token index
        idx = 0
        while sequence_ids[idx] != 1:
            idx += 1
        context_start = idx

        idx = len(sequence_ids) - 1
        while sequence_ids[idx] != 1:
            idx -= 1
        context_end = idx

        # 若答案不在這個 span 裡，標成 CLS(0)
        if offsets[context_start][0] > end_char or offsets[context_end][1] < start_char:
            start_positions.append(0)
            end_positions.append(0)
        else:
            # 找 start token
            idx = context_start
            while idx <= context_end and offsets[idx][0] <= start_char:
                idx += 1
            start_positions.append(idx - 1)

            # 找 end token
            idx = context_end
            while idx >= context_start and offsets[idx][1] >= end_char:
                idx -= 1
            end_positions.append(idx + 1)

    inputs["start_positions"] = start_positions
    inputs["end_positions"] = end_positions
    return inputs

# =========================
# Preprocessing for evaluation
# =========================
def preprocess_validation_examples(examples):
    questions = [q.strip() for q in examples["question"]]

    inputs = tokenizer(
        questions,
        examples["context"],
        max_length=MAX_LENGTH,
        truncation="only_second",
        stride=DOC_STRIDE,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    sample_map = inputs.pop("overflow_to_sample_mapping")
    example_ids = []

    for i in range(len(inputs["input_ids"])):
        sample_idx = sample_map[i]
        example_ids.append(examples["id"][sample_idx])

        sequence_ids = inputs.sequence_ids(i)
        offset = inputs["offset_mapping"][i]
        inputs["offset_mapping"][i] = [
            o if sequence_ids[k] == 1 else None for k, o in enumerate(offset)
        ]

    inputs["example_id"] = example_ids
    return inputs

train_dataset = train_dataset.map(
    preprocess_training_examples,
    batched=True,
    remove_columns=train_dataset.column_names,
)

eval_examples = eval_dataset
eval_dataset = eval_dataset.map(
    preprocess_validation_examples,
    batched=True,
    remove_columns=eval_dataset.column_names,
)

data_collator = DefaultDataCollator()
metric = evaluate.load("squad")

# =========================
# Post-processing predictions
# =========================
def postprocess_qa_predictions(examples, features, raw_predictions, n_best_size=20, max_answer_length=30):
    all_start_logits, all_end_logits = raw_predictions

    example_id_to_index = {k: i for i, k in enumerate(examples["id"])}
    features_per_example = collections.defaultdict(list)
    for i, feature in enumerate(features):
        features_per_example[example_id_to_index[feature["example_id"]]].append(i)

    predictions = collections.OrderedDict()

    for example_index, example in enumerate(examples):
        feature_indices = features_per_example[example_index]
        context = example["context"]

        valid_answers = []

        for feature_index in feature_indices:
            start_logits = all_start_logits[feature_index]
            end_logits = all_end_logits[feature_index]
            offset_mapping = features[feature_index]["offset_mapping"]

            start_indexes = np.argsort(start_logits)[-1 : -n_best_size - 1 : -1].tolist()
            end_indexes = np.argsort(end_logits)[-1 : -n_best_size - 1 : -1].tolist()

            for start_index in start_indexes:
                for end_index in end_indexes:
                    if start_index >= len(offset_mapping) or end_index >= len(offset_mapping):
                        continue
                    if offset_mapping[start_index] is None or offset_mapping[end_index] is None:
                        continue
                    if end_index < start_index:
                        continue
                    if end_index - start_index + 1 > max_answer_length:
                        continue

                    start_char = offset_mapping[start_index][0]
                    end_char = offset_mapping[end_index][1]

                    valid_answers.append(
                        {
                            "score": start_logits[start_index] + end_logits[end_index],
                            "text": context[start_char:end_char],
                        }
                    )

        if valid_answers:
            best_answer = max(valid_answers, key=lambda x: x["score"])
            predictions[example["id"]] = best_answer["text"]
        else:
            predictions[example["id"]] = ""

    return predictions

# =========================
# TrainingArguments
# Phase 1: 訓練期間不做 QA metric 評估（post-process 與 Trainer 介面不相容），
# 訓練結束後再手動跑一次完整 eval + EM/F1。
# =========================
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    eval_strategy="no",
    save_strategy="no",
    logging_strategy="steps",
    logging_steps=50,
    learning_rate=LEARNING_RATE,
    per_device_train_batch_size=TRAIN_BATCH_SIZE,
    per_device_eval_batch_size=EVAL_BATCH_SIZE,
    num_train_epochs=NUM_TRAIN_EPOCHS,
    weight_decay=WEIGHT_DECAY,
    report_to="none",
)

# =========================
# Trainer
# =========================
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
)

# =========================
# Train
# =========================
trainer.train()

# =========================
# Evaluate
# Trainer.predict 無法處理 string / None 欄位，必須先拿掉 example_id 與 offset_mapping；
# post-process 時仍需要原版 eval_dataset（保留這兩個欄位）來還原 answer span。
# =========================
eval_dataset_for_model = eval_dataset.remove_columns(["example_id", "offset_mapping"])
raw_predictions = trainer.predict(eval_dataset_for_model).predictions
final_predictions = postprocess_qa_predictions(eval_examples, eval_dataset, raw_predictions)

formatted_predictions = [
    {"id": k, "prediction_text": v} for k, v in final_predictions.items()
]
references = [
    {"id": ex["id"], "answers": ex["answers"]} for ex in eval_examples
]

results = metric.compute(predictions=formatted_predictions, references=references)
print("\nFinal Evaluation Results:")
print(results)