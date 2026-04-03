import json
import time
import requests
import pandas as pd
import streamlit as st

FASTAPI_BASE_URL = "http://127.0.0.1:8000"
HEALTH_URL = f"{FASTAPI_BASE_URL}/health"
PREDICT_URL = f"{FASTAPI_BASE_URL}/predict"
MODEL_INFO_URL = f"{FASTAPI_BASE_URL}/model-info"

st.set_page_config(page_title="Iris Inference Dashboard", layout="wide")

def ping_backend():
    try:
        r = requests.get(HEALTH_URL, timeout=3)
        return r.status_code == 200, r.json()
    except Exception as e:
        return False, {"error": str(e)}

def fetch_model_info():
    try:
        r = requests.get(MODEL_INFO_URL, timeout=3)
        if r.status_code == 200:
            return True, r.json()
        return False, r.json()
    except Exception as e:
        return False, {"error": str(e)}

def call_predict(payload: dict):
    start = time.perf_counter()
    r = requests.post(PREDICT_URL, json=payload, timeout=10)
    latency_ms = (time.perf_counter() - start) * 1000
    return r, latency_ms

def validate_payload(payload: dict):
    required = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
    missing = [k for k in required if k not in payload]
    if missing:
        return False, f"Missing keys: {missing}"

    for k in required:
        if not isinstance(payload[k], (int, float)):
            return False, f"'{k}' must be a number."
        if payload[k] <= 0:
            return False, f"'{k}' must be > 0."

    return True, None

if "history" not in st.session_state:
    st.session_state.history = []

st.title("🌿 Iris ML Inference Dashboard (FastAPI + Streamlit)")
st.caption("Custom enhancements: dual input modes, model info panel, latency tracking, and prediction history.")

colA, colB = st.columns([1.2, 1])

with colA:
    st.subheader("Backend Status")
    ok, health_payload = ping_backend()
    if ok:
        st.success(f"Backend is UP ✅ — {health_payload}")
    else:
        st.error(f"Backend is DOWN ❌ — {health_payload}")

with colB:
    st.subheader("Model Info")
    info_ok, model_info = fetch_model_info()
    if info_ok:
        with st.expander("View model metadata", expanded=True):
            st.json(model_info)
    else:
        st.warning(f"Model info unavailable — {model_info}")

st.divider()

st.subheader("1) Provide Input")
mode = st.radio("Choose input mode:", ["Manual (Sliders)", "Upload JSON"], horizontal=True)

payload = None

if mode == "Manual (Sliders)":
    c1, c2 = st.columns(2)
    with c1:
        sepal_length = st.slider("Sepal Length", 0.1, 10.0, 5.1, 0.1)
        sepal_width  = st.slider("Sepal Width",  0.1, 10.0, 3.5, 0.1)
    with c2:
        petal_length = st.slider("Petal Length", 0.1, 10.0, 1.4, 0.1)
        petal_width  = st.slider("Petal Width",  0.1, 10.0, 0.2, 0.1)

    payload = {
        "sepal_length": float(sepal_length),
        "sepal_width": float(sepal_width),
        "petal_length": float(petal_length),
        "petal_width": float(petal_width),
    }

else:
    uploaded = st.file_uploader("Upload a JSON file with iris features", type=["json"])
    if uploaded is not None:
        try:
            payload = json.load(uploaded)
            st.info("Uploaded payload:")
            st.json(payload)
        except Exception as e:
            st.error(f"Invalid JSON file: {e}")
            payload = None

st.divider()

st.subheader("2) Predict")
left, right = st.columns([1, 1])

with left:
    run_btn = st.button("Run Prediction", type="primary", disabled=(payload is None or not ok))

with right:
    conf_threshold = st.slider("Low-confidence threshold", 0.0, 1.0, 0.70, 0.01)

if run_btn and payload is not None:
    valid, err = validate_payload(payload)
    if not valid:
        st.error(err)
    else:
        try:
            r, latency_ms = call_predict(payload)
            resp_json = r.json() if r.content else {}

            st.write("### Request payload")
            st.json(payload)

            st.write("### Response")
            st.json(resp_json)

            st.metric("Latency (ms)", f"{latency_ms:.2f}")

            pred_class = resp_json.get("prediction")
            confidence = resp_json.get("confidence")

            if isinstance(confidence, (int, float)) and confidence < conf_threshold:
                st.warning("Low confidence prediction — consider re-checking inputs.")

            row = {
                "timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
                "sepal_length": payload["sepal_length"],
                "sepal_width": payload["sepal_width"],
                "petal_length": payload["petal_length"],
                "petal_width": payload["petal_width"],
                "prediction": pred_class,
                "confidence": confidence,
                "latency_ms": round(latency_ms, 2),
                "status_code": r.status_code,
            }
            st.session_state.history.insert(0, row)

        except Exception as e:
            st.error(f"Prediction call failed: {e}")

st.divider()

st.subheader("3) Prediction History (Session)")
hcol1, hcol2, hcol3 = st.columns([1, 1, 2])

with hcol1:
    if st.button("Clear History"):
        st.session_state.history = []

with hcol2:
    max_rows = st.selectbox("Show last N rows", [5, 10, 20, 50], index=1)

with hcol3:
    st.caption("History is stored per browser session (Streamlit session_state).")

if st.session_state.history:
    df = pd.DataFrame(st.session_state.history[:max_rows])
    st.dataframe(df, use_container_width=True)

    latencies = [r["latency_ms"] for r in st.session_state.history if isinstance(r.get("latency_ms"), (int, float))]
    if latencies:
        st.metric("Average latency (ms)", f"{sum(latencies)/len(latencies):.2f}")
else:
    st.info("No predictions yet. Run a prediction to populate history.")