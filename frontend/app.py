# frontend/app.py
import streamlit as st
import requests
from PIL import Image
import io
import time

API_URL = "http://localhost:8000"

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

st.set_page_config(
    page_title="CNN Compression Demo",
    layout="wide"
)

st.title("CNN compression demo")
st.caption("Upload an image — both models predict side by side")
st.divider()

uploaded = st.file_uploader(
    "Upload image",
    type=["jpg", "jpeg", "png", "webp"],
    label_visibility="collapsed"
)

if uploaded:
    image_bytes = uploaded.read()
    image = Image.open(io.BytesIO(image_bytes))

    st.divider()
    col1, col2 = st.columns(2)

    def call_model(model_id: str) -> tuple[dict, float]:
        start = time.perf_counter()
        response = requests.post(
            f"{API_URL}/predict",
            params={"model_id": model_id},
            files={"file": (uploaded.name, image_bytes, uploaded.type)},
        )
        elapsed = (time.perf_counter() - start) * 1000
        return response.json(), elapsed

    def render_result(col, model_id: str, label: str, subtitle: str):
        with col:
            st.markdown(f"**{label}**")
            st.caption(subtitle)

            img_col, info_col = st.columns([1, 1])
            with img_col:
                st.image(image, use_container_width=True)

            result, latency = call_model(model_id)
            predicted = result.get("predicted_class", "error")
            confidence = result.get("confidence", 0)
            all_probs = result.get("all_probs", {})

            with info_col:
                st.metric("prediction", predicted)
                st.metric("confidence", f"{confidence}%")
                st.caption(f"latency: {latency:.1f} ms")

            st.caption("all classes")
            sorted_probs = sorted(all_probs.items(), key=lambda x: x[1], reverse=True)
            for cls, prob in sorted_probs:
                st.progress(
                    int(prob),
                    text=f"{cls}: {prob}%"
                )

    render_result(col1, "pruned_70_fp32", "70% pruned · FP32", "structured pruned + distilled")
    render_result(col2, "pruned_50_fp32", "50% pruned · FP32", "structured pruned + distilled")

    st.divider()
    st.caption(f"backend: fastapi · {API_URL}")