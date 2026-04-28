"""
train_model.py
--------------
Fine-tunes a BERT model on the LIAR dataset for fake news detection.
Run this ONCE to train and save the model locally.

Usage:
    python train_model.py

Output:
    Saves trained model to ./models/fake_news_bert/
"""

import os
import torch
import pandas as pd
from datasets import load_dataset
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)
from sklearn.metrics import accuracy_score, f1_score
import numpy as np

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_NAME = "bert-base-uncased"
OUTPUT_DIR = "./models/fake_news_bert"
MAX_LENGTH = 128
NUM_LABELS = 2  # 0 = Fake, 1 = Real

# ── Label mapping ─────────────────────────────────────────────────────────────
# LIAR dataset has 6 labels — we simplify to binary: fake vs real
LIAR_TO_BINARY = {
    "pants-fire": 0,   # fake
    "false": 0,        # fake
    "barely-true": 0,  # fake
    "half-true": 1,    # real (borderline)
    "mostly-true": 1,  # real
    "true": 1,         # real
}

def load_and_prepare_data():
    """Load LIAR dataset and convert to binary labels."""
    print("Loading LIAR dataset...")
    dataset = load_dataset("liar")

    def map_labels(example):
        label_str = example["label"]
        # liar uses integer indices — map them
        label_map = {0: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1}
        example["label"] = label_map.get(label_str, 0)
        return example

    dataset = dataset.map(map_labels)
    return dataset

def tokenize_data(dataset, tokenizer):
    """Tokenize the statement field."""
    def tokenize(example):
        return tokenizer(
            example["statement"],
            truncation=True,
            max_length=MAX_LENGTH,
            padding=False,
        )

    tokenized = dataset.map(tokenize, batched=True)
    tokenized = tokenized.remove_columns(
        [col for col in tokenized["train"].column_names
         if col not in ["input_ids", "attention_mask", "token_type_ids", "label"]]
    )
    tokenized.set_format("torch")
    return tokenized

def compute_metrics(eval_pred):
    """Compute accuracy and F1 score."""
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, predictions),
        "f1": f1_score(labels, predictions, average="weighted"),
    }

def train():
    print(f"Using device: {'cuda' if torch.cuda.is_available() else 'cpu'}")

    # Load tokenizer and model
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
    model = BertForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=NUM_LABELS
    )

    # Load and tokenize data
    dataset = load_and_prepare_data()
    tokenized = tokenize_data(dataset, tokenizer)

    # Training config
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        warmup_steps=200,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        logging_dir="./logs",
        logging_steps=50,
        report_to="none",
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    print("Starting training...")
    trainer.train()

    # Save model and tokenizer
    print(f"Saving model to {OUTPUT_DIR}...")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    # Evaluate on test set
    print("Evaluating on test set...")
    results = trainer.evaluate(tokenized["test"])
    print(f"\nTest Results: {results}")
    print("\nModel training complete!")

if __name__ == "__main__":
    train()
