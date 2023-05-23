import os
from typing import Any, Dict

from django.http import HttpResponse
from django.template import loader
from transformers import pipeline

STORAGE_PATH = os.path.join("..", "..", "storage")


def inference(text: str) -> Dict[str, Any]:
    classifier = pipeline(
        "text-classification",
        model=os.path.join(
            STORAGE_PATH, "abstract_categorizer_model", "checkpoint-507"
        ),
    )

    return classifier(text)


def index(request):
    template = loader.get_template("categorizer/index.html")

    if request.method == "GET":
        return HttpResponse(template.render({}, request))

    abstract_text_input = request.POST.get("abstract_text", "")
    inference_result = inference(abstract_text_input)[0]
    context = {
        "category": inference_result["label"],
        "confidence": int(inference_result["score"] * 100),
    }
    return HttpResponse(template.render(context, request))
