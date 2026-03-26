# News Topic Classifier (AG News) — BERT fine-tuning

This project fine-tunes `bert-base-uncased` on the AG News dataset to classify news headlines into four topics, evaluates accuracy and F1, and provides a lightweight Streamlit UI for live inference.

Quick setup

1. Create a Python virtual environment and activate it.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r "requirements.txt"
```

2. Train / fine-tune the model

```powershell
python train.py
```

Trained model and tokenizer will be saved to `outputs/best_model`.

3. Run the Streamlit app

```powershell
streamlit run app.py
```

Notes

- Training uses the Hugging Face `Trainer` API; adjust `TrainingArguments` in `train.py` for batch size, epochs, or learning rate.
- Evaluation metrics printed by `Trainer` include `accuracy` and `f1` (weighted).
- Label mapping is in `utils.py`.
