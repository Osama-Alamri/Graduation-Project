import os
import time
import pandas as pd
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

import transformers
from transformers import (
    RobertaTokenizer,
    RobertaForSequenceClassification,
    Trainer,
    TrainingArguments,
)

# ============================================================ #
#   AI Detector Training Script using RoBERTa                  #   
# ============================================================ # 
#   Setup                                                      #
# ============================================================ #
#   to start training, run: (in your terminal)                 #
#   python ai_detector/train_ai_detector_roberta.py            #   
# ============================================================ #

# =========================
# 1) Load data
# =========================
train_path = "ai_detector/data/train.csv"
test_path  = "ai_detector/data/test.csv"

train_df = pd.read_csv(train_path)
test_df  = pd.read_csv(test_path)

print(f"📊 Loaded train: {len(train_df)} | test: {len(test_df)}")

# =========================
# 2) Tokenizer
# =========================
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

MAX_LENGTH = 256  # max token length

def tokenize_function(example):
    return tokenizer(
        example["text"],
        padding="max_length",
        truncation=True,
        max_length=MAX_LENGTH
    )

train_dataset = Dataset.from_pandas(train_df)
test_dataset  = Dataset.from_pandas(test_df)

train_dataset = train_dataset.map(tokenize_function, batched=True)
test_dataset  = test_dataset.map(tokenize_function, batched=True)

train_dataset = train_dataset.rename_column("label", "labels")
test_dataset  = test_dataset.rename_column("label", "labels")

train_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
test_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

# =========================
# 3) Model
# =========================
model = RobertaForSequenceClassification.from_pretrained("roberta-base", num_labels=2)

# =========================
# 4) Metrics
# =========================
def compute_metrics(pred):
    labels = pred.label_ids
    preds  = pred.predictions.argmax(-1)
    acc = accuracy_score(labels, preds)
    f1  = f1_score(labels, preds)
    prec = precision_score(labels, preds)
    rec  = recall_score(labels, preds)
    return {
        "accuracy": round(acc, 4),
        "f1": round(f1, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4)
    }

# =========================
# 5) TrainingArguments (robust to old/new transformers)
# =========================
output_dir = "ai_detector/ai_detector_roberta_base"
os.makedirs(output_dir, exist_ok=True)

common_kwargs = dict(
    output_dir=output_dir,
    learning_rate=2e-5,
    per_device_train_batch_size=8,   # as requested
    per_device_eval_batch_size=8,
    num_train_epochs=3,              # as requested
    weight_decay=0.01,
    logging_dir=f"{output_dir}/logs",
    save_total_limit=2,
    report_to="none",
)

# Try modern keys first; if TypeError, fallback to older-compatible keys
training_args = None
try:
    training_args = TrainingArguments(
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        **common_kwargs,
    )
except TypeError:
    # Older transformers: remove the newer strategy keys & best-model keys
    training_args = TrainingArguments(
        # Older versions don’t accept evaluation/save/logging_strategy or load_best_model_at_end/metric_for_best_model
        do_eval=True,
        logging_steps=500,     # periodic logs
        save_steps=1000,       # periodic save
        **common_kwargs,
    )

# =========================
# 6) Trainer
# =========================
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

# =========================
# 7) Train + Timer
# =========================
print("\n🚀 Starting training...")
start_time = time.time()

trainer.train()

elapsed = int(time.time() - start_time)
minutes = elapsed // 60
seconds = elapsed % 60
print(f"\n⏱️ Training finished in {minutes} minutes {seconds} seconds")

# =========================
# 8) Save model
# =========================
trainer.save_model(output_dir)
print("\n✅ Training completed! Model saved to:", output_dir)

# =========================
# 9) Final eval
# =========================
metrics = trainer.evaluate()
print("\n📈 Final Evaluation on Test Set:")
for k, v in metrics.items():
    try:
        print(f"{k}: {v:.4f}")
    except Exception:
        print(f"{k}: {v}")
