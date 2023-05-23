import logging
from typing import Any

from data_loader import load_data
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.hyperopt import HyperOptSearch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)
from utils import ID_2_LABEL, LABEL_2_ID, compute_metrics

LOGGER = logging.getLogger(__name__)


def model_init() -> Any:
    model = AutoModelForSequenceClassification.from_pretrained(
        "distilbert-base-uncased",
        num_labels=len(ID_2_LABEL),
        id2label=ID_2_LABEL,
        label2id=LABEL_2_ID,
    )

    return model


def hypertune() -> None:
    # Load data and config
    training_dataset, test_dataset = load_data()
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    training_args = TrainingArguments(
        "test", evaluation_strategy="steps", eval_steps=5, disable_tqdm=False
    )

    # Tokenize data
    def _preprocess_function(examples):
        return tokenizer(examples["text"], truncation=True)

    tokenized_training_dataset = training_dataset.map(
        _preprocess_function, batched=True
    )
    tokenized_test_dataset = test_dataset.map(_preprocess_function, batched=True)

    # Hyperparameter search
    trainer = Trainer(
        model_init=model_init,
        args=training_args,
        train_dataset=tokenized_training_dataset,
        eval_dataset=tokenized_test_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    best_trial = trainer.hyperparameter_search(
        direction="maximize",
        backend="ray",
        search_alg=HyperOptSearch(metric="objective", mode="max"),
        scheduler=ASHAScheduler(metric="objective", mode="max"),
    )

    LOGGER.info(f"Optimal hyperparameters: {best_trial.hyperparameters}")


if __name__ == "__main__":
    hypertune()
