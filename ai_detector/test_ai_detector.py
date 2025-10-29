from transformers import RobertaTokenizer, RobertaForSequenceClassification
import torch
import os

# لتفعيل safetensors
os.environ["SAFETENSORS_FAST_GPU"] = "1"

# =====================================================
# 1️⃣ Load Model + Tokenizer
# =====================================================
from transformers import RobertaTokenizer, RobertaForSequenceClassification
import torch

model_path = r"ai_detector/ai_detector_roberta_v2"

tokenizer = RobertaTokenizer.from_pretrained(model_path, local_files_only=True)
model = RobertaForSequenceClassification.from_pretrained(
    model_path,
    ignore_mismatched_sizes=True,
    local_files_only=True
)

model.eval()


# =====================================================
# 2️⃣ Prediction Function
# =====================================================
def detect_ai_text(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=256)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        ai_score = probs[0][1].item() * 100  # class 1 = AI
        human_score = probs[0][0].item() * 100  # class 0 = Human

    print(f"\n🧾 Text: {text}")
    print(f"🤖 AI Probability: {ai_score:.2f}%")
    print(f"👤 Human Probability: {human_score:.2f}%")

    if ai_score > 80:
        print("🟥 Likely AI-generated")
    else:
        print("🟩 Likely Human-written")

# =====================================================
# 3️⃣ Test Samples
# =====================================================
samples = [
    # 👤 Human-like examples
    "I was sitting in a quiet café watching the rain fall, wondering how life might have turned out if I’d taken that one unexpected opportunity years ago.",
    "Sometimes I feel like growing up isn’t about age, but about realizing how much responsibility you carry for every choice you make.",

    # 🤖 AI-like examples
    "Artificial intelligence is transforming global industries by automating workflows, improving predictive accuracy, and optimizing decision-making at unprecedented scales.",
    "The integration of deep learning models into natural language processing pipelines has significantly advanced contextual understanding and content generation."
]


print(model.config.id2label)
for text in samples:
    detect_ai_text(text)
