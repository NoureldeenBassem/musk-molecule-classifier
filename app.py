import streamlit as st
import numpy as np
import joblib

# ---------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------
st.set_page_config(page_title="Musk Molecule Classifier", page_icon="🧪", layout="centered")

st.title("🧪 Musk Molecule Classifier")
st.caption("Made by Noureldin Bassem — AI & Computer Engineer")

# ---------------------------------------------------------------
# Load the single saved pipeline, then pull out just the classifier step.
# The classifier is what was actually trained on PCA-transformed data,
# so we predict with it directly using PCA values as input.
# ---------------------------------------------------------------
@st.cache_resource
def load_classifier():
    pipeline = joblib.load("best_pca_pipeline.joblib")
    classifier = pipeline.named_steps["model"]
    n_components = pipeline.named_steps["pca"].n_components_
    return classifier, n_components

classifier, n_components = load_classifier()

st.write(
    f"This app predicts whether a molecule is **musk** or **non-musk**. "
    f"The deployed model was trained on **{n_components} PCA components**, so it expects "
    f"exactly the same {n_components} values here — the same inputs it saw during training."
)

st.divider()

# ---------------------------------------------------------------
# Input — one editable field per PCA component
# ---------------------------------------------------------------
st.subheader("1. Enter PCA Component Values")
st.caption(f"Each of the {n_components} components has its own field below — edit them individually.")

pca_values = []
columns = st.columns(4)

for i in range(n_components):
    col = columns[i % 4]
    value = col.number_input(f"PC{i + 1}", value=0.0, format="%.4f", key=f"pc_{i}")
    pca_values.append(value)

pca_values = np.array(pca_values).reshape(1, -1)

# ---------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------
st.divider()
st.subheader("2. Predict")

if st.button("Run Prediction", type="primary"):
    prediction = classifier.predict(pca_values)[0]
    probability = classifier.predict_proba(pca_values)[0, 1]  # probability of class 1 (musk)

    if prediction == 1:
        st.success(f"🔴 Predicted: **Musk** (confidence: {probability:.1%})")
    else:
        st.info(f"🔵 Predicted: **Non-musk** (confidence: {1 - probability:.1%})")

st.divider()
st.caption(
    "Model: the best-performing PCA-based model, intentionally selected during evaluation "
    "(see Section 10.6 of musk_classification.ipynb). Loaded from the single saved pipeline "
    "(best_pca_pipeline.joblib); this app uses only the pipeline's fitted classifier step, "
    "since inputs here are already the PCA-transformed values the classifier was trained on."
)
