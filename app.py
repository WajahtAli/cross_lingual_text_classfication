from flask import Flask, render_template, request
import joblib
import torch
import numpy as np
from transformers import (
    BertForSequenceClassification,
    BertTokenizer,
    XLMRobertaForSequenceClassification,
    XLMRobertaTokenizer,
)

app = Flask(__name__)

# --------- Config ---------
ARTIFACT_PATH = "crosslingual_artifact.pkl"
DEVICE = torch.device("cpu")  # force CPU

# --------- Load Model Artifact ---------
print("Loading model artifact...")
artifact = joblib.load(ARTIFACT_PATH)

saved_model_name = artifact["model_name"]
saved_state = artifact["state_dict"]
label_encoder = artifact["label_encoder"]

# --------- Detect model type ---------
if "bert" in saved_model_name.lower():
    print(f"Detected BERT model: {saved_model_name}")
    model = BertForSequenceClassification.from_pretrained(
        saved_model_name,
        num_labels=len(label_encoder.classes_)
    )
    tokenizer = BertTokenizer.from_pretrained(saved_model_name)

elif "xlm" in saved_model_name.lower():
    print(f"Detected XLM-Roberta model: {saved_model_name}")
    model = XLMRobertaForSequenceClassification.from_pretrained(
        saved_model_name,
        num_labels=len(label_encoder.classes_)
    )
    tokenizer = XLMRobertaTokenizer.from_pretrained(saved_model_name)

else:
    raise ValueError(f"Unknown model type for {saved_model_name}")

# Load weights
model.load_state_dict(saved_state, strict=False)
model.to(DEVICE)
model.eval()

print("Model and tokenizer loaded successfully!")

# --------- Flask Routes ---------
@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    user_text = ""

    if request.method == "POST":
        user_text = request.form.get("user_text", "").strip()
        if user_text:
            try:
                # Tokenize input
                enc = tokenizer(
                    [user_text],
                    truncation=True,
                    padding="max_length",
                    max_length=128,
                    return_tensors="pt"
                )

                with torch.no_grad():
                    outputs = model(
                        input_ids=enc["input_ids"].to(DEVICE),
                        attention_mask=enc["attention_mask"].to(DEVICE)
                    )
                    probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()[0]
                    pred_id = np.argmax(probs)
                    pred_label = label_encoder.inverse_transform([pred_id])[0]
                    confidence = probs[pred_id] * 100

                    prediction = f"{pred_label} (Confidence: {confidence:.1f}%)"

            except Exception as e:
                prediction = f"Error during prediction: {e}"

    return render_template("index.html", prediction=prediction, user_text=user_text)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)

