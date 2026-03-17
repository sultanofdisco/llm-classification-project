# =========================================================
# KC-BERT로 LLM 종류 분류
# target: model_source (gpt / gemini / claude / deepseek)
# text: reply_text 또는 transformed_text
# split 컬럼(train/val/test) 그대로 사용
# =========================================================

# 필요하면 설치
# !pip install -U transformers datasets accelerate scikit-learn pandas numpy

import os
import random
import inspect
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
)

warnings.filterwarnings("ignore")

# =========================
# 0) 설정
# =========================
CSV_PATH = "dataset_merged_2_fixed_safe_transform (1).csv"   # 파일명 맞게 수정
MODEL_NAME = "beomi/kcbert-base"

# 사용할 텍스트 컬럼
# "reply_text" 또는 "transformed_text"
TEXT_COL = "reply_text"

LABEL_COL = "model_source"
SPLIT_COL = "split"

# 사람 제외하고 LLM끼리만 구분
TARGET_CLASSES = ["gpt", "gemini", "claude", "deepseek"]

MAX_LEN = 256
BATCH_SIZE = 16
LR = 2e-5
EPOCHS = 5
WEIGHT_DECAY = 0.01
SEED = 42
OUTPUT_DIR = "./kcbert_llm_classifier"

# =========================
# 1) 시드 고정
# =========================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(SEED)

# =========================
# 2) 데이터 로드 및 검증
# =========================
df = pd.read_csv(CSV_PATH)

required_cols = [TEXT_COL, LABEL_COL, SPLIT_COL]
for col in required_cols:
    if col not in df.columns:
        raise ValueError(f"필수 컬럼 없음: {col}")

# 문자열/결측 처리
df[TEXT_COL] = df[TEXT_COL].fillna("").astype(str)
df[LABEL_COL] = df[LABEL_COL].fillna("").astype(str)
df[SPLIT_COL] = df[SPLIT_COL].fillna("").astype(str)

# LLM 4개만 남김
df = df[df[LABEL_COL].isin(TARGET_CLASSES)].copy()

# 빈 텍스트 제거
df = df[df[TEXT_COL].str.strip() != ""].copy()

# split 값 검증
valid_splits = {"train", "val", "test"}
bad_splits = set(df[SPLIT_COL].unique()) - valid_splits
if bad_splits:
    raise ValueError(f"이상한 split 값 발견: {bad_splits}")

# 라벨 인코딩
label_list = sorted(df[LABEL_COL].unique().tolist())
label2id = {label: i for i, label in enumerate(label_list)}
id2label = {i: label for label, i in label2id.items()}
df["labels"] = df[LABEL_COL].map(label2id)

print("===== 전체 데이터 확인 =====")
print("label mapping:", label2id)
print("\n[전체 클래스 개수]")
print(df[LABEL_COL].value_counts())

print("\n[split 분포]")
print(df[SPLIT_COL].value_counts())

print("\n[split x class]")
print(pd.crosstab(df[SPLIT_COL], df[LABEL_COL]))

# train / val / test 분리
train_df = df[df[SPLIT_COL] == "train"].copy()
val_df   = df[df[SPLIT_COL] == "val"].copy()
test_df  = df[df[SPLIT_COL] == "test"].copy()

if len(train_df) == 0 or len(val_df) == 0 or len(test_df) == 0:
    raise ValueError("train/val/test 중 하나가 비어 있음")

# =========================
# 3) Hugging Face Dataset 변환
# =========================
def to_hf_dataset(input_df):
    use_df = input_df[[TEXT_COL, "labels"]].rename(columns={TEXT_COL: "text"})
    return Dataset.from_pandas(use_df, preserve_index=False)

train_ds = to_hf_dataset(train_df)
val_ds = to_hf_dataset(val_df)
test_ds = to_hf_dataset(test_df)

# =========================
# 4) 토크나이저
# =========================
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize_fn(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=MAX_LEN,
    )

train_ds = train_ds.map(tokenize_fn, batched=True)
val_ds = val_ds.map(tokenize_fn, batched=True)
test_ds = test_ds.map(tokenize_fn, batched=True)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# =========================
# 5) 클래스 가중치
# =========================
class_weights_np = compute_class_weight(
    class_weight="balanced",
    classes=np.array(sorted(train_df["labels"].unique())),
    y=train_df["labels"].values
)
class_weights = torch.tensor(class_weights_np, dtype=torch.float)

print("\n[class weights]")
for i, w in enumerate(class_weights.tolist()):
    print(f"{id2label[i]}: {w:.4f}")

# =========================
# 6) 모델 로드
# =========================
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(label_list),
    id2label=id2label,
    label2id=label2id,
)

# 참고:
# 여기서 classifier head 관련 MISSING / UNEXPECTED 경고가 나올 수 있는데,
# base checkpoint에 downstream classification head가 없어서 새로 초기화되는 경우 흔하다.
# 분류 태스크 학습 시에는 보통 정상이다.

# =========================
# 7) metric
# =========================
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)

    return {
        "accuracy": accuracy_score(labels, preds),
        "macro_f1": f1_score(labels, preds, average="macro"),
        "weighted_f1": f1_score(labels, preds, average="weighted"),
    }

# =========================
# 8) weighted loss Trainer
# 최신 Trainer 문서에는 compute_loss에 num_items_in_batch가 들어간다.
# 그래서 그 인자까지 받아서 버전 호환되게 처리한다.
# Trainer 생성자도 processing_class/tokenizer 자동 분기한다.
# =========================
class WeightedTrainer(Trainer):
    def __init__(self, class_weights=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")

        if self.class_weights is not None:
            loss_fct = nn.CrossEntropyLoss(weight=self.class_weights.to(logits.device))
        else:
            loss_fct = nn.CrossEntropyLoss()

        loss = loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss

# =========================
# 9) TrainingArguments 버전 호환
# 최신 문서 기준 eval_strategy 사용. 일부 구버전은 evaluation_strategy 사용.
# 최신 문서에서 load_best_model_at_end는 eval/save strategy 정합성이 필요하다.
# =========================
ta_sig = inspect.signature(TrainingArguments.__init__)
ta_params = ta_sig.parameters

training_kwargs = dict(
    output_dir=OUTPUT_DIR,
    save_strategy="epoch",
    logging_strategy="steps",
    logging_steps=50,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    learning_rate=LR,
    num_train_epochs=EPOCHS,
    weight_decay=WEIGHT_DECAY,
    load_best_model_at_end=True,
    metric_for_best_model="macro_f1",
    greater_is_better=True,
    save_total_limit=2,
    report_to="none",
    seed=SEED,
)

# fp16은 CUDA일 때만
if torch.cuda.is_available():
    training_kwargs["fp16"] = True

# 버전별 전략 인자 자동 분기
if "eval_strategy" in ta_params:
    training_kwargs["eval_strategy"] = "epoch"
elif "evaluation_strategy" in ta_params:
    training_kwargs["evaluation_strategy"] = "epoch"
else:
    raise RuntimeError("TrainingArguments에서 eval_strategy/evaluation_strategy 둘 다 찾지 못함")

training_args = TrainingArguments(**training_kwargs)

# =========================
# 10) Trainer 생성자 버전 호환
# 최신 문서 기준 processing_class 사용
# 일부 환경은 tokenizer 인자 사용
# =========================
trainer_sig = inspect.signature(Trainer.__init__)
trainer_params = trainer_sig.parameters

trainer_kwargs = dict(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    class_weights=class_weights,
)

if "processing_class" in trainer_params:
    trainer_kwargs["processing_class"] = tokenizer
elif "tokenizer" in trainer_params:
    trainer_kwargs["tokenizer"] = tokenizer
# 둘 다 없으면 그냥 생략

trainer = WeightedTrainer(**trainer_kwargs)

# =========================
# 11) 학습
# =========================
train_result = trainer.train()

print("\n===== 학습 완료 =====")
print(train_result)

# =========================
# 12) 검증/테스트 평가
# =========================
print("\n===== Validation =====")
val_metrics = trainer.evaluate(eval_dataset=val_ds)
print(val_metrics)

print("\n===== Test =====")
test_metrics = trainer.evaluate(eval_dataset=test_ds)
print(test_metrics)

# =========================
# 13) 테스트 상세 결과
# =========================
pred_output = trainer.predict(test_ds)
test_logits = pred_output.predictions
test_labels = pred_output.label_ids
test_preds = np.argmax(test_logits, axis=-1)

print("\n===== Classification Report (Test) =====")
print(classification_report(
    test_labels,
    test_preds,
    target_names=[id2label[i] for i in range(len(id2label))],
    digits=4
))

print("\n===== Confusion Matrix (Test) =====")
cm = confusion_matrix(test_labels, test_preds)
cm_df = pd.DataFrame(
    cm,
    index=[f"true_{id2label[i]}" for i in range(len(id2label))],
    columns=[f"pred_{id2label[i]}" for i in range(len(id2label))]
)
print(cm_df)

# =========================
# 14) 예측 결과 저장
# =========================
os.makedirs(OUTPUT_DIR, exist_ok=True)

# softmax confidence
probs = torch.softmax(torch.tensor(test_logits), dim=-1).numpy()

result_df = test_df[[TEXT_COL, LABEL_COL]].copy().reset_index(drop=True)
result_df["gold_id"] = test_labels
result_df["pred_id"] = test_preds
result_df["gold_label"] = result_df["gold_id"].map(id2label)
result_df["pred_label"] = result_df["pred_id"].map(id2label)
result_df["correct"] = (result_df["gold_id"] == result_df["pred_id"]).astype(int)

for i in range(len(id2label)):
    result_df[f"prob_{id2label[i]}"] = probs[:, i]

result_df.to_csv(
    os.path.join(OUTPUT_DIR, "test_predictions.csv"),
    index=False,
    encoding="utf-8-sig"
)

cm_df.to_csv(
    os.path.join(OUTPUT_DIR, "test_confusion_matrix.csv"),
    encoding="utf-8-sig"
)

# 가장 헷갈린 오답 몇 개 보기
wrong_df = result_df[result_df["correct"] == 0].copy()
wrong_df["max_prob"] = wrong_df[[f"prob_{id2label[i]}" for i in range(len(id2label))]].max(axis=1)
wrong_df = wrong_df.sort_values("max_prob", ascending=False)

wrong_df.to_csv(
    os.path.join(OUTPUT_DIR, "test_misclassified.csv"),
    index=False,
    encoding="utf-8-sig"
)

print(f"\n저장 완료: {OUTPUT_DIR}")
print(" - test_predictions.csv")
print(" - test_confusion_matrix.csv")
print(" - test_misclassified.csv")
