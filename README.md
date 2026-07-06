# 🧪 Musk Molecule Classifier

Predicting whether a molecule is **musk** 🌸 or **non-musk** 🚫 based on its molecular geometry — powered by PCA and Machine Learning.

🔗 **Live App:** [musk-molecule-classifier.streamlit.app](https://musk-molecule-classifier-rxpc8sdehjgczlg8lrczb8.streamlit.app/)

---

## 📖 About This Project

Musk compounds are highly valued in the **perfumery industry** 🧴, but testing whether a molecule smells like musk experimentally is slow and expensive. This project builds a **machine learning classifier** that predicts musk vs. non-musk directly from **166 numerical features** describing a molecule's 3D geometry.

## ✨ Features

- 📊 **Exploratory Data Analysis** — class balance, feature scales, and a deep dive into the dataset's molecule-level structure
- 🧬 **PCA (Principal Component Analysis)** — reduces 166 raw features down to **38 principal components**, capturing 95% of the original variance
- 🤖 **4 Classification Models Compared** — Logistic Regression, Random Forest, SVM, and XGBoost — each trained **with** and **without** PCA
- 🎯 **Group-Aware Train/Test Split** — prevents data leakage by keeping all conformations of the same molecule together
- 🛠️ **Hyperparameter Tuning** — via `GridSearchCV` for every model
- 📈 **Full Evaluation Suite** — Accuracy, Precision, Recall, F1-score, and Confusion Matrices for every model
- 🚀 **Live Deployment** — an interactive Streamlit app for real-time predictions

## 🧠 Model

| Detail | Value |
|---|---|
| 🏆 Deployed Model | SVM (trained on PCA components) |
| 🔢 Input Features | 38 PCA components |
| 📉 Dimensionality Reduction | 166 → 38 features (95% variance retained) |
| ⚖️ Class Imbalance Handling | `class_weight='balanced'` |

> 💡 The PCA-based model was **intentionally chosen for deployment**, even though a non-PCA model scored marginally higher — because demonstrating PCA in a full deployed workflow was a core goal of this project, and it makes the app far simpler to use (38 inputs instead of 166!).

## 🖥️ Try It Yourself

1. Open the 🔗 [live app](https://musk-molecule-classifier-rxpc8sdehjgczlg8lrczb8.streamlit.app/)
2. Enter the 38 PCA component values in the input fields
3. Click **Run Prediction** 🔮
4. See the result: 🔴 Musk or 🔵 Non-musk, with a confidence score!

## 📂 Repository Contents

```
📁 musk-molecule-classifier
├── 📓 musk_classification.ipynb   → Full ML workflow: EDA, PCA, modeling, evaluation
├── 🐍 app.py                      → Streamlit deployment app
├── 📦 best_pca_pipeline.joblib    → Saved pipeline (scaler + PCA + model, all-in-one)
├── 📄 requirements.txt            → Python dependencies
└── 📘 README.md                   → You are here!
```

## 🛠️ Tech Stack

`Python` 🐍 · `scikit-learn` 🤖 · `XGBoost` 🚀 · `pandas` 🐼 · `Streamlit` ⚡ · `Matplotlib` / `Seaborn` 📊

## 👨‍💻 Author

**Noureldin Bassem**
🎓 AI & Computer Engineer

---

⭐ If you found this project interesting, feel free to star the repo!
