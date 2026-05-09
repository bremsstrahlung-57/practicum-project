# backend/inference.py
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import io
from model_def import get_pruned_architecture

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

TRANSFORM = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.2023, 0.1994, 0.2010]
    )
])


def load_fp32_model(path: str, pruning_ratio: float) -> nn.Module:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    
    if isinstance(checkpoint, dict):
        state_dict = checkpoint['model_state_dict']
        model = get_pruned_architecture(pruning_ratio=pruning_ratio)
        model.load_state_dict(state_dict)
    else:
        # Full model saved directly
        model = checkpoint

    model.eval()
    return model
   


def run_inference(model: nn.Module, image_bytes: bytes) -> dict:
    import time

    t0 = time.perf_counter()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    t1 = time.perf_counter()

    tensor = TRANSFORM(image).unsqueeze(0)
    t2 = time.perf_counter()

    with torch.no_grad():
        logits = model(tensor)
    t3 = time.perf_counter()

    probs = torch.softmax(logits, dim=1).squeeze()
    top_prob, top_idx = torch.max(probs, dim=0)
    t4 = time.perf_counter()

    print(f"Image open : {(t1-t0)*1000:.1f} ms")
    print(f"Transform  : {(t2-t1)*1000:.1f} ms")
    print(f"Forward    : {(t3-t2)*1000:.1f} ms")
    print(f"Postprocess: {(t4-t3)*1000:.1f} ms")

    return {
        "predicted_class": CIFAR10_CLASSES[top_idx.item()],
        "confidence": round(top_prob.item() * 100, 2),
        "all_probs": {
            cls: round(probs[i].item() * 100, 2)
            for i, cls in enumerate(CIFAR10_CLASSES)
        }
    }

    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    tensor = TRANSFORM(image).unsqueeze(0)

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1).squeeze()

    top_prob, top_idx = torch.max(probs, dim=0)

    return {
        "predicted_class": CIFAR10_CLASSES[top_idx.item()],
        "confidence": round(top_prob.item() * 100, 2),
        "all_probs": {
            cls: round(probs[i].item() * 100, 2)
            for i, cls in enumerate(CIFAR10_CLASSES)
        }
    }