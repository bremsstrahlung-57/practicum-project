# backend/main.py
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import uvicorn
from inference import load_fp32_model, run_inference

MODEL_PATH = { "pruned_70_fp32":"../models/resnet18_pruned70_distilled.pth", "pruned_50_fp32":"../models/structured_pruned_50pct_distilled.pth"  }

ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp"}

ml_models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading 70% pruned FP32 model...")
    ml_models["pruned_70_fp32"] = load_fp32_model(MODEL_PATH["pruned_70_fp32"], pruning_ratio=0.7)
    print("Loading 50% pruned FP32 model...")
    ml_models["pruned_50_fp32"] = load_fp32_model(MODEL_PATH["pruned_50_fp32"], pruning_ratio=0.5)
    print("Models ready.")
    yield
    ml_models.clear()

app = FastAPI(title="CNN Compression Demo", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"status": "ok"}

@app.post("/predict")
async def predict(file: UploadFile = File(...), model_id: str = Query(
        default="pruned_70_fp32",
        enum=["pruned_70_fp32", "pruned_50_fp32"]
    )):
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(status_code=400, detail=f"Unsupported type: {file.content_type}")

    image_bytes = await file.read()
    result = run_inference(ml_models[model_id], image_bytes)
    result["model_used"] = model_id
    return result

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)