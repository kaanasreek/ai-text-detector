import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# -----------------------------
# Load Model
# -----------------------------
model_path = "model/checkpoint-8662"

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return tokenizer, model, device

tokenizer, model, device = load_model()

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("🧠 AI vs Human Text Detector")

st.write("Enter text below to check whether it is AI-generated or Human-written.")

user_input = st.text_area("Enter text here:")

if st.button("Detect"):

    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        inputs = tokenizer(
            user_input,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=256
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1)
            score = probs[0][1].item()

        st.write(f"AI Probability: {score:.4f}")

        if score > 0.5:
            st.error("⚠️ This text is likely AI Generated.")
        else:
            st.success("✅ This text is likely Human Written.")