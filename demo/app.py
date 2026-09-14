"""Streamlit showcase for the subject-independent ECG emotion baseline."""

from __future__ import annotations

import io
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from ecg_emotion.inference import load_baseline, predict_baseline

LABEL_NAMES = {
    "0": "Baseline",
    "1": "Stress",
    "2": "Amusement",
}
MODEL_PRESETS = {
    "Conservative benchmark": {
        "path": "artifacts/demo-baseline/model.joblib",
        "model_name": "shrinkage-lda",
        "description": "Empirical class priors; more conservative overall behavior.",
    },
    "Class-balanced showcase": {
        "path": "artifacts/demo-balanced-lda/model.joblib",
        "model_name": "balanced-shrinkage-lda",
        "description": "Equal class priors; improved coverage of the Amusement class.",
    },
    "Custom bundle": {
        "path": "",
        "model_name": None,
        "description": "Load any compatible local model.joblib bundle.",
    },
}


def display_label(label: str) -> str:
    """Map a numeric class id to a human-readable research label."""

    return LABEL_NAMES.get(str(label), f"Class {label}")


st.set_page_config(page_title="ECG Emotion Recognition", page_icon="❤️", layout="wide")
st.title("ECG Emotion Recognition")
st.caption("A local research demo for subject-independent, confidence-aware ECG classification.")
st.warning("This demo is for research and engineering evaluation only, not medical diagnosis.")

st.sidebar.header("Model configuration")
selected_mode = st.sidebar.selectbox("Operating mode", list(MODEL_PRESETS))
preset = MODEL_PRESETS[selected_mode]
model_path_text = st.sidebar.text_input("Model bundle", value=preset["path"])
model_path = Path(model_path_text) if model_path_text else None

bundle = None
load_error = None
if model_path is not None and model_path.exists():
    try:
        bundle = load_baseline(model_path)
    except (OSError, ValueError, KeyError, TypeError) as error:
        load_error = str(error)

if load_error:
    st.sidebar.error(f"Could not load model bundle: {load_error}")

if bundle is not None:
    bundle_sample_rate = int(round(float(bundle["sample_rate"])))
    sample_rate = bundle_sample_rate
    st.sidebar.caption(f"Using the bundle sample rate: {bundle_sample_rate} Hz")
else:
    sample_rate = st.sidebar.number_input("Signal sample rate (Hz)", min_value=1, value=128, step=1)

if bundle is not None and preset["model_name"] is not None:
    actual_model_name = bundle.get("model_name")
    if actual_model_name != preset["model_name"]:
        st.sidebar.warning(
            f"The selected preset expects {preset['model_name']}, but this bundle contains "
            f"{actual_model_name}."
        )
st.sidebar.caption(preset["description"])

uploaded = st.file_uploader("Upload one ECG window as a .npy file", type=["npy"])

if uploaded is None:
    time = np.arange(sample_rate * 10) / sample_rate
    signal = np.sin(2 * np.pi * 1.1 * time) + 0.08 * np.sin(2 * np.pi * 0.2 * time)
    st.info("Showing a synthetic signal. Upload a prepared ECG window to run inference.")
else:
    try:
        signal = np.load(io.BytesIO(uploaded.getvalue()), allow_pickle=False)
        signal = np.asarray(signal, dtype=np.float32).reshape(-1)
        if signal.size == 0:
            raise ValueError("the uploaded array is empty")
    except (ValueError, TypeError, OSError) as error:
        st.error(f"Could not read the uploaded ECG window: {error}")
        st.stop()

figure, axis = plt.subplots(figsize=(12, 3))
axis.plot(np.arange(len(signal)) / sample_rate, signal, linewidth=0.8)
axis.set_xlabel("Time (seconds)")
axis.set_ylabel("Amplitude")
axis.set_title("Input ECG window")
axis.grid(alpha=0.2)
st.pyplot(figure, clear_figure=True)
plt.close(figure)

if bundle is None:
    if model_path is None:
        st.info("Choose a model bundle path in the sidebar to run inference.")
    else:
        st.info("Train a baseline first, then point the sidebar to its model.joblib file.")
else:
    try:
        result = predict_baseline(bundle, signal)
    except (ValueError, KeyError, TypeError) as error:
        st.error(f"Inference failed: {error}")
        st.stop()

    st.subheader("Prediction")
    metric_columns = st.columns(3)
    metric_columns[0].metric("Predicted state", display_label(result["label"]))
    metric_columns[1].metric("Top-1 confidence", f"{result['confidence']:.1%}")
    metric_columns[2].metric("Bundle threshold", f"{result['confidence_threshold']:.1%}")

    if result["accepted"]:
        st.success("Accepted for review under the training-derived confidence policy.")
    else:
        st.warning("Abstain: confidence is below the training-derived threshold.")

    threshold_selection = bundle.get("confidence_threshold_selection", {})
    if threshold_selection.get("training_only"):
        minimum_coverage = threshold_selection.get("minimum_coverage", "n/a")
        st.caption(
            "Threshold provenance: selected from training-only cross-fitted OOF predictions "
            f"with minimum coverage {minimum_coverage}."
        )
    else:
        st.caption("Threshold provenance: legacy bundle without training-only threshold metadata.")

    probability_frame = pd.DataFrame(
        {
            "Class": [display_label(label) for label in result["probabilities"]],
            "Probability": list(result["probabilities"].values()),
        }
    )
    st.markdown("#### Class probabilities")
    st.dataframe(
        probability_frame.style.format({"Probability": "{:.1%}"}),
        use_container_width=True,
    )
    st.bar_chart(probability_frame.set_index("Class"), y="Probability")
