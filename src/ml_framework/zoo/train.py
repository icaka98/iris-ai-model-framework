import os
import pickle

from datasets import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)
from utils import ID_2_LABEL, LABEL_2_ID, STORAGE_PATH, compute_metrics

TOKENIZER = AutoTokenizer.from_pretrained("distilbert-base-uncased")


def _preprocess_function(examples):
    return TOKENIZER(examples["text"], truncation=True)


def train():
    with open(os.path.join(STORAGE_PATH, "training_data.dat"), "rb") as data_file:
        training_data = pickle.load(data_file)

    with open(os.path.join(STORAGE_PATH, "test_data.dat"), "rb") as data_file:
        test_data = pickle.load(data_file)

    training_dataset = Dataset.from_list(training_data)
    test_dataset = Dataset.from_list(test_data)

    tokenized_training_dataset = training_dataset.map(
        _preprocess_function, batched=True
    )
    tokenized_test_dataset = test_dataset.map(_preprocess_function, batched=True)

    data_collator = DataCollatorWithPadding(tokenizer=TOKENIZER)

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

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_training_dataset,
        eval_dataset=tokenized_test_dataset,
        tokenizer=TOKENIZER,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()


if __name__ == "__main__":
    train()
