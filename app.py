import streamlit as st
import pandas as pd
import joblib

# ---------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------
st.set_page_config(page_title="Musk Molecule Classifier", page_icon="🧪", layout="centered")

st.title("🧪 Musk Molecule Classifier")
st.write(
    "This app predicts whether a molecule is **musk** or **non-musk** based on its "
    "166 numerical geometry features (`f1` to `f166`). "
    "The deployed model uses PCA (Principal Component Analysis) internally to reduce these "
    "166 features before classifying — but since scaling and PCA are built directly into the "
    "saved pipeline, you only ever need to provide the original raw features below."
)

# ---------------------------------------------------------------
# Load the single deployed pipeline (scaler + PCA + classifier, all in one)
# ---------------------------------------------------------------
FEATURE_COLS = [f"f{i}" for i in range(1, 167)]

@st.cache_resource
def load_pipeline():
    return joblib.load("best_pca_pipeline.joblib")

pipeline = load_pipeline()

st.divider()

# ---------------------------------------------------------------
# Input
# ---------------------------------------------------------------
st.subheader("1. Provide Molecule Data")
st.caption(
    "Upload a CSV containing columns `f1` through `f166` (one or more rows/conformations). "
    "Extra columns like `molecule_name` or `conformation_name` are fine — they'll be ignored."
)

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
input_df = None

if uploaded_file is not None:
    raw_df = pd.read_csv(uploaded_file)
    missing_cols = [c for c in FEATURE_COLS if c not in raw_df.columns]

    if missing_cols:
        st.error(
            f"The uploaded file is missing {len(missing_cols)} required column(s), "
            f"for example: {missing_cols[:5]}"
        )
    else:
        input_df = raw_df[FEATURE_COLS]
        st.success(f"Loaded {len(input_df)} row(s) successfully.")
        st.dataframe(raw_df.head())

# ---------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------
st.divider()
st.subheader("2. Predict")

if input_df is not None:
    if st.button("Run Prediction", type="primary"):
        # The pipeline handles scaling and PCA internally before classifying,
        # so raw feature values can be passed in directly.
        predictions = pipeline.predict(input_df)
        probabilities = pipeline.predict_proba(input_df)[:, 1]  # probability of class 1 (musk)

        results = pd.DataFrame({
            "Prediction": ["Musk" if p == 1 else "Non-musk" for p in predictions],
            "Confidence (Musk probability)": [f"{prob:.1%}" for prob in probabilities]
        })

        st.write("### Results")
        st.dataframe(results)

        if len(results) == 1:
            label = results.iloc[0]["Prediction"]
            conf = probabilities[0] if predictions[0] == 1 else 1 - probabilities[0]
            if label == "Musk":
                st.success(f"🔴 Predicted: **Musk** (confidence: {conf:.1%})")
            else:
                st.info(f"🔵 Predicted: **Non-musk** (confidence: {conf:.1%})")
else:
    st.info("Upload a CSV above to enable prediction.")

st.divider()
st.caption(
    "Model: the best-performing PCA-based model, intentionally selected during evaluation "
    "(see Section 10.6 of musk_classification.ipynb) to demonstrate PCA in a full deployed workflow. "
    "Scaling, PCA, and the classifier are all bundled into a single saved pipeline "
    "(best_pca_pipeline.joblib) so preprocessing is always applied exactly as it was during training."
)
