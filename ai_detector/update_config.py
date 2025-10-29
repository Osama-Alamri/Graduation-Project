# ============================================================
#  Update Model Config to include correct class labels
# ============================================================
#   Run this once (no need to retrain the model)
#   Command:
#       python ai_detector/update_config.py
# ============================================================

from transformers import RobertaForSequenceClassification

# Path to your saved model
model_path = "ai_detector/ai_detector_roberta_base"

# Load the trained model
model = RobertaForSequenceClassification.from_pretrained(model_path)

# ✅ Add proper label mapping
model.config.id2label = {0: "Human", 1: "AI"}
model.config.label2id = {"Human": 0, "AI": 1}

# Save the updated config to the same folder
model.config.save_pretrained(model_path)

print("\n✅ Model config updated successfully!")
print("🔹 id2label:", model.config.id2label)
print("🔹 label2id:", model.config.label2id)
print("📁 Saved to:", model_path)
