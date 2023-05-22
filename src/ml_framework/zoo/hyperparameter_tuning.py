import os
import pickle

from datasets import Dataset
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.hyperopt import HyperOptSearch
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


def model_init():
    model = AutoModelForSequenceClassification.from_pretrained(
        "distilbert-base-uncased",
        num_labels=len(ID_2_LABEL),
        id2label=ID_2_LABEL,
        label2id=LABEL_2_ID,
    )

    return model


def hypertune():
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

    training_args = TrainingArguments(
        "test", evaluation_strategy="steps", eval_steps=5, disable_tqdm=False
    )

    trainer = Trainer(
        model_init=model_init,
        args=training_args,
        train_dataset=tokenized_training_dataset,
        eval_dataset=tokenized_test_dataset,
        tokenizer=TOKENIZER,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    best_trial = trainer.hyperparameter_search(
        direction="maximize",
        backend="ray",
        search_alg=HyperOptSearch(metric="objective", mode="max"),
        scheduler=ASHAScheduler(metric="objective", mode="max"),
    )

    print("Optimal hyperparameters:")
    print(best_trial.hyperparameters)


if __name__ == "__main__":
    hypertune()
