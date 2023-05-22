import json
import os
import pickle
import random
import string

import nltk
import tqdm
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Download only if not already present
nltk.download("stopwords", quiet=True)
nltk.download("punkt", quiet=True)
nltk.download("wordnet", quiet=True)

LEMMATIZER = WordNetLemmatizer()
STOR_WORDS = set(stopwords.words("english"))

MIN_ABSTRACT_WORDS = 50
MAX_ABSTRACT_WORDS = 250

ID_2_LABEL = {0: "ph", 1: "math", 2: "cs", 3: "bio", 4: "soc", 5: "chem"}
LABEL_2_ID = {v: k for k, v in ID_2_LABEL.items()}

ORIGINAL_DATA_PATH = os.path.join(
    "..", "..", "storage", "arxiv-metadata-oai-snapshot.json"
)

CATEGORY_MAPPING = {
    "math": "math",
    "physics": "ph",
    "cs": "cs",
    "astro-ph": "ph",
    "cond-mat": "ph",
    "hep-ph": "ph",
    "hep-th": "ph",
    "quant-ph": "ph",
    "stat": "math",
    "gr-qc": "ph",
    "math-ph": "ph",
    "eess": "cs",
    "nucl-th": "ph",
    "hep-ex": "ph",
    "q-bio": "bio",
    "nlin": "ph",
    "hep-lat": "ph",
    "nucl-ex": "ph",
    "q-fin": "soc",
    "econ": "soc",
    "chao-dyn": "ph",
    "q-alg": "soc",
    "alg-geom": "math",
    "solv-int": "math",
    "cmp-lg": "cs",
    "dg-ga": "math",
    "patt-sol": "ph",
    "adap-org": "ph",
    "funct-an": "math",
    "mtrl-th": "chem",
    "chem-ph": "chem",
    "comp-gas": "chem",
    "supr-con": "ph",
    "atom-ph": "ph",
    "acc-phys": "ph",
    "plasm-ph": "ph",
    "ao-sci": "bio",
    "bayes-an": "math",
}


def _load_data_dynamically():
    with open(ORIGINAL_DATA_PATH, "r") as input_file:
        for line in input_file:
            yield json.loads(line)


def _filter_data(data):
    data = [
        entry
        for entry in data
        if MAX_ABSTRACT_WORDS > len(entry["abstract"].split()) > MIN_ABSTRACT_WORDS
    ]

    print(len(data))

    return [
        {
            "text": entry["abstract"],
            "cat": [
                CATEGORY_MAPPING[cat.split(".")[0]]
                for cat in entry["categories"].split()
            ],
        }
        for entry in data
    ]


def _text_preprocessing(text):
    text = text.strip().lower()

    text = text.translate(str.maketrans("", "", string.punctuation))

    tokens = word_tokenize(text)
    tokens = [
        LEMMATIZER.lemmatize(token) for token in tokens if token not in STOR_WORDS
    ]

    return " ".join(tokens)


def _clean_text_data(data):
    return [
        {
            "text": _text_preprocessing(entry["text"]),
            "label": LABEL_2_ID[entry["cat"][0]],
        }
        for entry in tqdm.tqdm(data, desc="Preprocessing text data")
    ]


def preprocess():
    print("Loading data...")
    data = list(_load_data_dynamically())

    # Limiting data to 4_000 examples
    data = data[:4_000]

    print("Filtering data...")
    data = _filter_data(data)

    data = _clean_text_data(data)

    random.seed(42)
    random.shuffle(data)

    data_size = len(data)
    percentile_80 = int(data_size * 0.8)
    training_data = data[:percentile_80]
    test_data = data[percentile_80:]

    with open(
        os.path.join("..", "..", "storage", "training_data.dat"), "wb"
    ) as output_file:
        pickle.dump(training_data, output_file)

    with open(
        os.path.join("..", "..", "storage", "test_data.dat"), "wb"
    ) as output_file:
        pickle.dump(test_data, output_file)


if __name__ == "__main__":
    preprocess()
