from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from inference import run_inference
from model_registry import MODEL_SPEC_BY_ID, MODEL_SPECS, load_model

ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp"}

ml_models = {}
model_errors = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    for spec in MODEL_SPECS:
        try:
            print(f"Loading {spec.id} from {spec.path}...")
            ml_models[spec.id] = load_model(spec)
        except Exception as exc:
            model_errors[spec.id] = str(exc)
            print(f"Failed to load {spec.id}: {exc}")

    print(f"Models ready: {len(ml_models)}/{len(MODEL_SPECS)} loaded.")
    yield
    ml_models.clear()
    model_errors.clear()


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


@app.get("/models")
def models():
    return [
        spec.to_api(
            available=spec.id in ml_models,
            error=model_errors.get(spec.id),
        )
        for spec in MODEL_SPECS
    ]


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    model_id: str = Query(default="pruned_50_fp32"),
):
    if model_id not in MODEL_SPEC_BY_ID:
        raise HTTPException(status_code=400, detail=f"Unknown model: {model_id}")

    if model_id not in ml_models:
        detail = model_errors.get(model_id, "model is not loaded")
        raise HTTPException(status_code=503, detail=detail)

    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=400, detail=f"Unsupported type: {file.content_type}"
        )

    image_bytes = await file.read()
    result = run_inference(ml_models[model_id], image_bytes)
    result["model_used"] = model_id
    result["model"] = MODEL_SPEC_BY_ID[model_id].to_api(available=True)
    return result


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
