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
    "166 numerical geometry features (`f1` to `f166`)."
)

# ---------------------------------------------------------------
# Load the trained pipeline and expected feature columns
# ---------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load("musk_best_model.joblib")
    feature_cols = joblib.load("musk_feature_columns.joblib")
    return model, feature_cols

model, feature_cols = load_artifacts()

st.divider()

# ---------------------------------------------------------------
# Input method
# ---------------------------------------------------------------
st.subheader("1. Provide Molecule Data")

input_method = st.radio(
    "Choose an input method:",
    ["Upload a CSV file", "Paste a single row manually"]
)

input_df = None

if input_method == "Upload a CSV file":
    st.caption(
        "Upload a CSV containing columns `f1` through `f166` (one or more rows/conformations). "
        "Extra columns like `molecule_name` or `conformation_name` are fine — they'll be ignored."
    )
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded_file is not None:
        raw_df = pd.read_csv(uploaded_file)
        missing_cols = [c for c in feature_cols if c not in raw_df.columns]

        if missing_cols:
            st.error(
                f"The uploaded file is missing {len(missing_cols)} required column(s), "
                f"for example: {missing_cols[:5]}"
            )
        else:
            input_df = raw_df[feature_cols]
            st.success(f"Loaded {len(input_df)} row(s) successfully.")
            st.dataframe(raw_df.head())

else:
    st.caption(
        "Paste 166 comma-separated numeric values (in order f1, f2, ..., f166), "
        "for example copied from a spreadsheet row."
    )
    text_input = st.text_area("Feature values (comma-separated)", height=120)

    if text_input.strip():
        try:
            values = [float(v.strip()) for v in text_input.split(",")]
            if len(values) != len(feature_cols):
                st.error(f"Expected {len(feature_cols)} values, but got {len(values)}.")
            else:
                input_df = pd.DataFrame([values], columns=feature_cols)
                st.success("Values parsed successfully.")
        except ValueError:
            st.error("Could not parse the values — make sure they are all numbers separated by commas.")

# ---------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------
st.divider()
st.subheader("2. Predict")

if input_df is not None:
    if st.button("Run Prediction", type="primary"):
        predictions = model.predict(input_df)
        probabilities = model.predict_proba(input_df)[:, 1]  # probability of class 1 (musk)

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
    st.info("Provide molecule data above to enable prediction.")

st.divider()
st.caption(
    "Model: trained and selected via cross-validated F1-score in `musk_classification.ipynb`. "
    "The saved pipeline includes all preprocessing steps (scaling, and PCA if applicable), "
    "so raw f1–f166 values can be passed in directly."
)
