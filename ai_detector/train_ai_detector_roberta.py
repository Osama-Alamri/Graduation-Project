import os
import time
import torch
import pandas as pd
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# نحاول استيراد matplotlib للرسم، ولو غير متوفر نكمل بدون رسم
try:
    import matplotlib.pyplot as plt
    HAS_MPL = True
except Exception:
    HAS_MPL = False

from transformers import (
    RobertaTokenizer,
    RobertaForSequenceClassification,
    Trainer,
    TrainingArguments,
)
import transformers

# ============================================================
# 🚀 AI Text Detector (v2 - GPU Optimized + Resume Checkpoint)
# ============================================================

print(f"[info] transformers version: {transformers.__version__}")

# =========================
# 1️⃣ Load Data
# =========================
train_path = "ai_detector/data/train.csv"
test_path  = "ai_detector/data/test.csv"

train_df = pd.read_csv(train_path)
test_df  = pd.read_csv(test_path)
print(f"📊 Loaded train: {len(train_df)} | test: {len(test_df)}")

# =========================
# 2️⃣ Tokenizer
# =========================
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
MAX_LENGTH = 256

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
# 3️⃣ Model (stronger dropout to reduce overconfidence)
# =========================
model = RobertaForSequenceClassification.from_pretrained(
    "roberta-base",
    num_labels=2,
    hidden_dropout_prob=0.2,
    attention_probs_dropout_prob=0.2,
)

# =========================
# 4️⃣ Metrics
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
# 5️⃣ Training Arguments
# =========================
output_dir = "ai_detector/ai_detector_roberta_v2"
os.makedirs(output_dir, exist_ok=True)

common_kwargs = dict(
    output_dir=output_dir,
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=2,
    weight_decay=0.1,
    logging_dir=f"{output_dir}/logs",
    save_steps=500,  # ✅ احفظ تقدمك كل 500 خطوة
    save_total_limit=2,
    report_to="none",
    fp16=True,  # ✅ تدريب أسرع باستخدام Mixed Precision (على GPU فقط)
)

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
    training_args = TrainingArguments(
        do_eval=True,
        logging_steps=500,
        **common_kwargs,
    )
    print("[warn] Using legacy TrainingArguments (no *strategy keys).")

# =========================
# 6️⃣ Trainer
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
# 7️⃣ Resume from last checkpoint (NEW)
# =========================
last_checkpoint = transformers.trainer_utils.get_last_checkpoint(output_dir)
if last_checkpoint:
    print(f"🔄 Found checkpoint, resuming from: {last_checkpoint}")
else:
    print("🆕 No checkpoint found, starting fresh training.")

# =========================
# 8️⃣ Train + GPU Setup
# =========================
print("\n🚀 Starting training...")
start_time = time.time()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

if device.type == "cuda":
    print("✅ GPU in use:", torch.cuda.get_device_name(0))
    torch.backends.cudnn.benchmark = True
    torch.cuda.empty_cache()
    print("⚡ GPU ready with cuDNN acceleration.")
else:
    print("⚠️ CUDA not available, using CPU instead.")

# 🔁 GPU memory monitor (every 10s)
from threading import Thread
import time as t

def gpu_monitor():
    while True:
        if torch.cuda.is_available():
            mem = torch.cuda.memory_allocated(0) / 1024**3
            print(f"📊 GPU Memory: {mem:.2f} GB", end="\r")
        t.sleep(10)

if device.type == "cuda":
    Thread(target=gpu_monitor, daemon=True).start()

# 🚀 Train (resume if checkpoint exists)
train_result = trainer.train(resume_from_checkpoint=last_checkpoint if last_checkpoint else None)

elapsed = int(time.time() - start_time)
minutes, seconds = divmod(elapsed, 60)
print(f"\n⏱️ Training finished in {minutes} minutes {seconds} seconds")

# =========================
# 9️⃣ Save model + labels
# =========================
trainer.save_model(output_dir)
model.config.id2label = {0: "Human", 1: "AI"}
model.config.label2id = {"Human": 0, "AI": 1}
model.config.save_pretrained(output_dir)
print("\n✅ Model saved to:", output_dir)

# =========================
# 🔟 Evaluation + Plot
# =========================
metrics = trainer.evaluate()
print("\n📈 Final Evaluation on Test Set:")
for k, v in metrics.items():
    try:
        print(f"{k}: {v:.4f}")
    except Exception:
        print(f"{k}: {v}")

if HAS_MPL:
    loss_values = [x["loss"] for x in trainer.state.log_history if "loss" in x]
    if loss_values:
        plt.figure(figsize=(8, 4))
        plt.plot(range(len(loss_values)), loss_values, label="Training Loss")
        plt.xlabel("Steps")
        plt.ylabel("Loss")
        plt.title("📉 Training Loss Curve (v2 GPU + Resume)")
        plt.legend()
        plt.tight_layout()
        out_png = f"{output_dir}/training_loss_curve.png"
        plt.savefig(out_png)
        try:
            plt.show()
        except Exception:
            pass
        print(f"\n📊 Loss curve saved to: {out_png}")
    else:
        print("\n[info] No loss logs were captured for plotting.")
else:
    print("\n[info] matplotlib not installed; skipping loss plot. Install via: pip install matplotlib")
