import os

from transformers import pipeline

STORAGE_PATH = os.path.join("..", "..", "storage")


def inference(text):
    classifier = pipeline(
        "text-classification",
        model=os.path.join(
            STORAGE_PATH, "abstract_categorizer_model", "checkpoint-507"
        ),
    )

    return classifier(text)


def inference_multiple(texts):
    classifier = pipeline(
        "text-classification",
        model=os.path.join(
            STORAGE_PATH, "abstract_categorizer_model", "checkpoint-507"
        ),
    )

    return [classifier(text) for text in texts]


if __name__ == "__main__":
    text = "paper fourth series lowfrequency search technosignatures using murchison widefield array "
    "two night integrate 7 hour data toward galactic centre centred position sagittarius total fieldofview "
    "200 deg2 present targeted search toward 144 exoplanetary system best yet angular resolution 75 arc second "
    "first technosignature search frequency 155 mhz toward galactic centre previous central frequency lower blind "
    "search toward excess 3 million star toward galactic centre galactic bulge also completed placing equivalent "
    "isotropic power limit 11x1019w distance galactic centre plausible technosignatures detected"
    res = inference(text=text)

    print(res)
