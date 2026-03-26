import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F
from utils import LABELS


@st.cache_resource
def load_model(path="outputs/best_model"):
    try:
        tokenizer = AutoTokenizer.from_pretrained(path)
        model = AutoModelForSequenceClassification.from_pretrained(path)
    except Exception as e:
        st.warning(f"Could not load local model at '{path}': {e}. Loading 'bert-base-uncased' instead. Train model and restart to use local model.")
        model_name = "bert-base-uncased"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=4)

    model.eval()
    return tokenizer, model


st.title("AG News Topic Classifier (BERT)")
st.write("Enter a news headline to classify it into a topic.")

tokenizer, model = load_model()

text = st.text_area("News headline", height=120)

if st.button("Classify"):
    if not text.strip():
        st.warning("Please enter a headline.")
    else:
        inputs = tokenizer(text, return_tensors="pt", truncation=True)
        with torch.no_grad():
            outputs = model(**inputs)
            probs = F.softmax(outputs.logits, dim=-1).squeeze().cpu().numpy()
            pred = int(probs.argmax())

        st.markdown(f"**Predicted topic:** {LABELS.get(pred, str(pred))}")
        st.markdown(f"**Confidence:** {probs[pred]:.3f}")
        st.bar_chart(probs)
