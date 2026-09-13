"""Streamlit showcase for a serialized feature baseline."""

from __future__ import annotations

import io
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from ecg_emotion.inference import load_baseline, predict_baseline


st.set_page_config(page_title="ECG Emotion Recognition", page_icon="❤️", layout="wide")
st.title("ECG Emotion Recognition")
st.caption("A local research demo for a lightweight feature-based baseline.")
st.warning("This demo is for research and engineering evaluation only, not medical diagnosis.")

model_path = Path(st.sidebar.text_input("Model bundle", "artifacts/demo-baseline/model.joblib"))
sample_rate = st.sidebar.number_input("Sample rate (Hz)", min_value=1, value=128, step=1)
uploaded = st.file_uploader("Upload one ECG window as a .npy file", type=["npy"])

if uploaded is None:
    time = np.arange(sample_rate * 10) / sample_rate
    signal = np.sin(2 * np.pi * 1.1 * time) + 0.08 * np.sin(2 * np.pi * 0.2 * time)
    st.info("Showing a synthetic signal. Upload a prepared ECG window to run inference.")
else:
    signal = np.load(io.BytesIO(uploaded.getvalue()))
    signal = np.asarray(signal).reshape(-1)

figure, axis = plt.subplots(figsize=(12, 3))
axis.plot(np.arange(len(signal)) / sample_rate, signal, linewidth=0.8)
axis.set_xlabel("Time (seconds)")
axis.set_ylabel("Amplitude")
axis.set_title("Input ECG window")
axis.grid(alpha=0.2)
st.pyplot(figure, clear_figure=True)

if model_path.exists():
    bundle = load_baseline(model_path)
    result = predict_baseline(bundle, signal)
    st.subheader(f"Predicted class: {result['label']}")
    st.bar_chart(result["probabilities"])
else:
    st.info("Train a baseline first, then point the sidebar to its model.joblib file.")

