import json
import pandas as pd
import numpy as np
import os

# =====================================================
# 1️⃣ Setup file paths
# =====================================================
input_path = "ai_detector/data/HC3_all.jsonl"
output_dir = "ai_detector/data"
os.makedirs(output_dir, exist_ok=True)

print("📂 Reading:", input_path)

# =====================================================
# 2️⃣ Read JSONL file line by line
# =====================================================
rows = []
with open(input_path, "r", encoding="utf-8") as f:
    for line in f:
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            q = row.get("question", "").strip()
            for human in row.get("human_answers", []):
                rows.append({"question": q, "text": f"{q} {human}", "label": 0})
            for ai in row.get("chatgpt_answers", []):
                rows.append({"question": q, "text": f"{q} {ai}", "label": 1})
        except json.JSONDecodeError as e:
            print("⚠️ Skipped bad line:", e)
            continue

# =====================================================
# 3️⃣ Clean and create DataFrame
# =====================================================
df = pd.DataFrame(rows)
df = df.dropna().drop_duplicates(subset="text").reset_index(drop=True)
print(f"✅ Loaded total rows: {len(df)}")

# =====================================================
# 4️⃣ Balance dataset (equal number of samples per class)
# =====================================================
min_count = df["label"].value_counts().min()
balanced = (
    df.groupby("label")
    .sample(n=min_count, random_state=42)
    .reset_index(drop=True)
)
print(f"✅ Balanced dataset: {len(balanced)} rows ({min_count} per class)")

# =====================================================
# 5️⃣ Split by question (to prevent data leakage)
# =====================================================
unique_questions = balanced["question"].unique()
np.random.seed(42)
test_q = np.random.choice(unique_questions, size=int(0.2 * len(unique_questions)), replace=False)

train = balanced[~balanced["question"].isin(test_q)].reset_index(drop=True)
test = balanced[balanced["question"].isin(test_q)].reset_index(drop=True)

# =====================================================
# 6️⃣ Save train and test CSV files
# =====================================================
train_path = os.path.join(output_dir, "train.csv")
test_path = os.path.join(output_dir, "test.csv")

train.to_csv(train_path, index=False)
test.to_csv(test_path, index=False)

# =====================================================
# 7️⃣ Print summary info
# =====================================================
print("✅ Saved:")
print(f"   Train: {len(train)} rows, class distribution: {train['label'].value_counts().to_dict()}")
print(f"   Test : {len(test)} rows, class distribution: {test['label'].value_counts().to_dict()}")
print(f"   Unique questions → Train: {train['question'].nunique()} | Test: {test['question'].nunique()}")
print("🚀 Done! No question appears in both sets.")
