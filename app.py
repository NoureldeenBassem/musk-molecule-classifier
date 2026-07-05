import streamlit as st
import numpy as np
import joblib

# ---------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------
st.set_page_config(page_title="Musk Molecule Classifier", page_icon="🧪", layout="centered")

st.title("🧪 Musk Molecule Classifier")

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
# Input
# ---------------------------------------------------------------
st.subheader("1. Enter PCA Component Values")
st.caption(f"Paste {n_components} comma-separated numbers (PC1 ... PC{n_components}), in order.")

text_input = st.text_area(f"Principal component values ({n_components} numbers)", height=100)

pca_values = None

if text_input.strip():
    try:
        values = [float(v.strip()) for v in text_input.split(",")]
        if len(values) != n_components:
            st.error(f"Expected {n_components} values, but got {len(values)}.")
        else:
            pca_values = np.array(values).reshape(1, -1)
            st.success("Values parsed successfully.")
    except ValueError:
        st.error("Could not parse the values — make sure they are all numbers separated by commas.")

# ---------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------
st.divider()
st.subheader("2. Predict")

if pca_values is not None:
    if st.button("Run Prediction", type="primary"):
        prediction = classifier.predict(pca_values)[0]
        probability = classifier.predict_proba(pca_values)[0, 1]  # probability of class 1 (musk)

        if prediction == 1:
            st.success(f"🔴 Predicted: **Musk** (confidence: {probability:.1%})")
        else:
            st.info(f"🔵 Predicted: **Non-musk** (confidence: {1 - probability:.1%})")
else:
    st.info(f"Enter {n_components} PCA values above to enable prediction.")

st.divider()
st.caption(
    "Model: the best-performing PCA-based model, intentionally selected during evaluation "
    "(see Section 10.6 of musk_classification.ipynb). Loaded from the single saved pipeline "
    "(best_pca_pipeline.joblib); this app uses only the pipeline's fitted classifier step, "
    "since inputs here are already the PCA-transformed values the classifier was trained on."
)
