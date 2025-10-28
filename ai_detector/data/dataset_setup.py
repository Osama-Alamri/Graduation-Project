import json
import pandas as pd
import os

# =====================================================
# 1️ Setup paths
# =====================================================
input_path = "ai_detector/data/HC3_all.jsonl"
output_dir = "ai_detector/data"
os.makedirs(output_dir, exist_ok=True)

print("📂 Reading:", input_path)

# =====================================================
# 2️ Read JSONL file line by line
# =====================================================
rows = []
with open(input_path, "r", encoding="utf-8") as f:
    for line in f:
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            q = row.get("question", "")
            for human in row.get("human_answers", []):
                rows.append({"text": f"{q} {human}", "label": 0})
            for ai in row.get("chatgpt_answers", []):
                rows.append({"text": f"{q} {ai}", "label": 1})
        except json.JSONDecodeError as e:
            print("⚠️ Skipped bad line:", e)
            continue

# =====================================================
# 3️ Clean and convert to DataFrame
# =====================================================
df = pd.DataFrame(rows)
df = df.dropna().drop_duplicates(subset="text").reset_index(drop=True)

print(f"✅ Loaded total rows: {len(df)}")

# =====================================================
# 4️ Balance dataset (equal number of Human & AI)
# =====================================================
min_count = df["label"].value_counts().min()
balanced = (
    df.groupby("label")
    .sample(n=min_count, random_state=42)
    .sample(frac=1, random_state=42)
    .reset_index(drop=True)
)

print(f"✅ Balanced dataset: {len(balanced)} rows ({min_count} per class)")

# =====================================================
# 5️ Split into Train/Test sets
# =====================================================
test = balanced.sample(frac=0.2, random_state=42)
train = balanced.drop(test.index)

train_path = os.path.join(output_dir, "train.csv")
test_path = os.path.join(output_dir, "test.csv")

train.to_csv(train_path, index=False)
test.to_csv(test_path, index=False)

print("✅ Saved:")
print("   Train:", train_path, "→", len(train), "rows")
print("   Test :", test_path, "→", len(test), "rows")
