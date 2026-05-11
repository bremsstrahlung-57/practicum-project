import io
import time

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

CIFAR10_CLASSES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]

TRANSFORM = transforms.Compose(
    [
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]
        ),
    ]
)


def run_inference(model: nn.Module, image_bytes: bytes) -> dict:
    t0 = time.perf_counter()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    tensor = TRANSFORM(image).unsqueeze(0)
    t1 = time.perf_counter()

    with torch.no_grad():
        logits = model(tensor)
    t2 = time.perf_counter()

    probs = torch.softmax(logits, dim=1).squeeze()
    top_prob, top_idx = torch.max(probs, dim=0)
    t3 = time.perf_counter()

    probabilities = sorted(
        (
            {
                "class_name": cls,
                "probability": round(probs[i].item() * 100, 2),
            }
            for i, cls in enumerate(CIFAR10_CLASSES)
        ),
        key=lambda item: item["probability"],
        reverse=True,
    )

    for rank, item in enumerate(probabilities, start=1):
        item["rank"] = rank
        item["is_top"] = item["class_name"] == CIFAR10_CLASSES[top_idx.item()]

    return {
        "predicted_class": CIFAR10_CLASSES[top_idx.item()],
        "confidence": round(top_prob.item() * 100, 2),
        "probabilities": probabilities,
        "all_probs": {
            cls: round(probs[i].item() * 100, 2)
            for i, cls in enumerate(CIFAR10_CLASSES)
        },
        "timings_ms": {
            "preprocess": round((t1 - t0) * 1000, 2),
            "forward": round((t2 - t1) * 1000, 2),
            "postprocess": round((t3 - t2) * 1000, 2),
            "total": round((t3 - t0) * 1000, 2),
        },
    }
