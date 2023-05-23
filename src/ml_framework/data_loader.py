import os
import pickle
from typing import Tuple

from datasets import Dataset
from utils import STORAGE_PATH


def load_data() -> Tuple[Dataset, Dataset]:
    with open(os.path.join(STORAGE_PATH, "training_data.dat"), "rb") as data_file:
        training_data = pickle.load(data_file)

    with open(os.path.join(STORAGE_PATH, "test_data.dat"), "rb") as data_file:
        test_data = pickle.load(data_file)

    training_dataset = Dataset.from_list(training_data)
    test_dataset = Dataset.from_list(test_data)

    return training_dataset, test_dataset
