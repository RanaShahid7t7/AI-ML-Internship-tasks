from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
)
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
import os


def preprocess(tokenizer, max_samples=None):
    ds = load_dataset("ag_news")
    if max_samples:
        ds = ds.map(lambda x, idx: x, with_indices=True)
        ds["train"] = ds["train"].select(range(min(max_samples, len(ds["train"]))))
        ds["test"] = ds["test"].select(range(min(max_samples // 5, len(ds["test"]))))

    def tokenize_fn(examples):
        return tokenizer(examples["text"], truncation=True)

    tokenized = ds.map(tokenize_fn, batched=True)

    # set format for Trainer (torch)
    for split in tokenized.keys():
        tokenized[split].set_format(type="torch", columns=[c for c in ["input_ids", "attention_mask", "label"] if c in tokenized[split].column_names])

    return tokenized


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="weighted")
    return {"accuracy": acc, "f1": f1}


def main():
    model_name = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    datasets = preprocess(tokenizer)

    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=4)

    data_collator = DataCollatorWithPadding(tokenizer)

    training_args = TrainingArguments(
        output_dir="outputs",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=64,
        num_train_epochs=3,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        logging_dir="./logs",
        logging_strategy="epoch",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=datasets["train"],
        eval_dataset=datasets["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    # save best model and tokenizer
    save_dir = os.path.join("outputs", "best_model")
    # ensure output directory exists before saving
    os.makedirs(save_dir, exist_ok=True)
    trainer.save_model(save_dir)
    tokenizer.save_pretrained(save_dir)


if __name__ == "__main__":
    main()
