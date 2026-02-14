import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import requests


from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

st.set_page_config(page_title="AI Fraud Detection", layout="wide")

# ---------------- SIDEBAR ---------------- #
st.sidebar.title("💳 AI Fraud Detection")
page = st.sidebar.radio(
    "Navigate",
    ["Upload & Train Model", "Visualizations", "Single Prediction", "About Project"],
)

# ---------------- SESSION STATE ---------------- #
if "model" not in st.session_state:
    st.session_state.model = None
    st.session_state.X_columns = None


# ---------------- PAGE 1: UPLOAD & TRAIN ---------------- #
if page == "Upload & Train Model":

    st.title("🚀 AI-Based Fraud Detection System")
    st.write("Upload a dataset, choose a model, and detect fraudulent transactions.")

    uploaded_file = st.file_uploader("Upload CSV dataset", type=["csv"])

    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)

        st.subheader("📊 Dataset Preview")
        st.dataframe(data.head())

        target_column = st.selectbox("Select Target Column", data.columns)

        model_name = st.selectbox(
            "Choose ML Model",
            ["Random Forest", "Logistic Regression", "Decision Tree"],
        )

        if st.button("Train Model"):

            X = data.drop(columns=[target_column])
            y = data[target_column]

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            if model_name == "Random Forest":
                model = RandomForestClassifier()
            elif model_name == "Logistic Regression":
                model = LogisticRegression(max_iter=2000)
            else:
                model = DecisionTreeClassifier()

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            st.session_state.model = model
            st.session_state.X_columns = X.columns

            st.success("✅ Model trained successfully!")

            # Classification report
            st.subheader("📈 Classification Report")
            report = classification_report(y_test, y_pred, output_dict=True)
            st.dataframe(pd.DataFrame(report).transpose())

            # Confusion matrix
            st.subheader("🔍 Confusion Matrix")
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
            st.pyplot(fig)

            # ROC Curve
            if hasattr(model, "predict_proba"):
                y_prob = model.predict_proba(X_test)[:, 1]
                fpr, tpr, _ = roc_curve(y_test, y_prob)
                roc_auc = auc(fpr, tpr)

                st.subheader("📉 ROC Curve")
                fig2, ax2 = plt.subplots()
                ax2.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
                ax2.plot([0, 1], [0, 1], linestyle="--")
                ax2.set_xlabel("False Positive Rate")
                ax2.set_ylabel("True Positive Rate")
                ax2.legend()
                st.pyplot(fig2)


# ---------------- PAGE 2: VISUALIZATION ---------------- #
elif page == "Visualizations":

    st.title("📊 Data Visualizations")

    uploaded_file = st.file_uploader("Upload CSV for visualization", type=["csv"])

    if uploaded_file:
        data = pd.read_csv(uploaded_file)

        st.subheader("Correlation Heatmap")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(data.corr(), cmap="coolwarm")
        st.pyplot(fig)

        st.subheader("Class Distribution")
        target = st.selectbox("Select Target Column", data.columns)
        fig2, ax2 = plt.subplots()
        sns.countplot(x=data[target], ax=ax2)
        st.pyplot(fig2)


# ---------------- PAGE 3: SINGLE PREDICTION ---------------- #
# ---------------- PAGE: SINGLE PREDICTION ---------------- #
elif page == "Single Prediction":

    st.title("🔮 Predict Fraud for Single Transaction")

    st.write("Enter feature values:")

    # Example: assuming 10 features (we can auto-detect later)
    features = []

    for i in range(10):
        val = st.number_input(f"Feature {i+1}", value=0.0)
        features.append(val)

    if st.button("Predict"):

        try:
            response = requests.post(
                "http://127.0.0.1:8000/predict",
                json={"features": features},
                timeout=5
            )

            result = response.json()

            if result["prediction"] == 1:
                st.error("🚨 Fraudulent Transaction Detected!")
            else:
                st.success("✅ Legitimate Transaction")

        except Exception as e:
            st.error("❌ Could not connect to FastAPI backend. Is the server running?")


# ---------------- PAGE 4: ABOUT ---------------- #
else:
    st.title("📘 About This Project")

    st.write("""
    **AI-Based Fraud Detection System**

    This project uses machine learning algorithms to detect fraudulent
    financial transactions.

    ### Features
    - Upload real datasets
    - Train multiple ML models
    - Visualize performance metrics
    - Predict fraud for single transaction

    ### Developed by
    **Vedika Kapoor**
    """)
