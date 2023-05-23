import os

from data_loader import load_data
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)
from utils import ID_2_LABEL, LABEL_2_ID, STORAGE_PATH, compute_metrics


def train() -> None:
    # Load data, models and config
    training_dataset, test_dataset = load_data()
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    model = AutoModelForSequenceClassification.from_pretrained(
        "distilbert-base-uncased",
        num_labels=len(ID_2_LABEL),
        id2label=ID_2_LABEL,
        label2id=LABEL_2_ID,
    )
    training_args = TrainingArguments(
        output_dir=os.path.join(STORAGE_PATH, "abstract_categorizer_model"),
        learning_rate=2.88e-05,  # from hyperparameter tuning
        per_device_train_batch_size=8,  # from hyperparameter tuning
        per_device_eval_batch_size=8,  # from hyperparameter tuning
        num_train_epochs=3,  # from hyperparameter tuning
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        push_to_hub=False,
    )

    # Tokenize data
    def _preprocess_function(examples):
        return tokenizer(examples["text"], truncation=True)

    tokenized_training_dataset = training_dataset.map(
        _preprocess_function, batched=True
    )
    tokenized_test_dataset = test_dataset.map(_preprocess_function, batched=True)

    # Model training
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_training_dataset,
        eval_dataset=tokenized_test_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.train()


if __name__ == "__main__":
    train()
