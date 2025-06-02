import streamlit as st
import shap
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import pyrebase
from firebase_config import firebase_config

# Initialize Firebase
firebase = pyrebase.initialize_app(firebase_config)
auth = firebase.auth()

# Session state
if 'user' not in st.session_state:
    st.session_state.user = None

# Authentication Functions
def login():
    st.title("🔐 Login")
    email = st.text_input("📧 Email")
    password = st.text_input("🔑 Password", type="password")
    if st.button("Login"):
        try:
            user = auth.sign_in_with_email_and_password(email, password)
            st.session_state.user = user
            st.success("✅ Logged in successfully!")
            st.rerun()
        except:
            st.error("❌ Invalid credentials.")

def signup():
    st.title("🆕 Sign Up")
    email = st.text_input("📧 Email")
    password = st.text_input("🔑 Password", type="password")
    if st.button("Sign Up"):
        try:
            auth.create_user_with_email_and_password(email, password)
            st.success("✅ Account created! Please login.")
        except:
            st.error("❌ Could not sign up. Try again.")

def logout():
    st.session_state.user = None
    st.success("🚪 Logged out.")
    st.rerun()

# Main App Logic
if st.session_state.user:
    st.set_page_config(page_title="SHAP Explainability", layout="wide")

    st.sidebar.success(f"Logged in as: {st.session_state.user['email']}")
    if st.sidebar.button("Logout"):
        logout()

    st.title("🔍 SHAP+WIT Explainability Demo")
    st.markdown("Upload a dataset, train an XGBoost model, and explore feature attributions using SHAP.")
    st.markdown("""
        SHAP (SHapley Additive exPlanations) is a method to explain the output of machine learning models. 
        It provides a way to understand the importance of individual features, both globally and locally, 
        by computing how each feature contributes to the prediction.
    """)

    uploaded_file = st.file_uploader("📂 Upload a CSV dataset", type=["csv"])

    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            st.subheader("📊 Uploaded Data")
            st.dataframe(df.head())

            target_col = st.selectbox("🎯 Select target variable (label)", df.columns)

            X = df.drop(columns=[target_col])
            y = df[target_col]

            # Encode categorical columns
            for col in X.select_dtypes(include=['object', 'category']).columns:
                X[col] = LabelEncoder().fit_transform(X[col])

            if y.dtype == object or y.dtype.name == 'category':
                y = LabelEncoder().fit_transform(y)

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            model = xgb.XGBClassifier(use_label_encoder=False, eval_metric="logloss")
            model.fit(X_train, y_train)

            st.success("✅ Model trained successfully.")

            # SHAP analysis
            explainer = shap.Explainer(model, X_train)
            shap_values = explainer(X_test)

            st.subheader("📌 Global Feature Importance")
            st.markdown("""
                **Global Feature Importance** shows which features are most important across all predictions. 
                A higher impact means the feature has a greater influence on the model’s output for all samples.
            """)
            fig1, ax1 = plt.subplots(figsize=(10, 6))
            shap.plots.bar(shap_values, max_display=10, show=False)
            st.pyplot(fig1, bbox_inches='tight', dpi=300, pad_inches=0.1)

            st.subheader("📈 SHAP Beeswarm Plot (Feature Impact Across All Samples)")
            st.markdown("""
                **Beeswarm Plot** visualizes how features affect the predictions across all samples in the dataset. 
                Each point in the plot represents a SHAP value for a feature and sample. The color indicates whether the feature's value is high or low, and the width of the plot reflects the density of SHAP values.
            """)
            fig2, ax2 = plt.subplots(figsize=(10, 6))
            shap.plots.beeswarm(shap_values, show=False)
            st.pyplot(fig2, bbox_inches='tight', dpi=300, pad_inches=0.1)

            st.subheader("🔬 Explain a Specific Prediction")
            row_idx = st.slider("Select Row Index", 0, len(X_test)-1, 0)
            st.write("Prediction Breakdown for this sample:")

            row_data = X_test.iloc[row_idx]
            feature_contributions = shap_values[row_idx].values

            explanation = "### Explanation for Prediction Breakdown: \n"
            explanation += f"**Predicted Class**: {model.predict([row_data])[0]}\n"
            explanation += "\n### Feature Contributions (How each feature influenced the prediction):\n"

            for feature, contribution in zip(X.columns, feature_contributions):
                if contribution > 0:
                    explanation += f"- **{feature}** contributed positively with a SHAP value of {contribution:.2f}\n"
                else:
                    explanation += f"- **{feature}** contributed negatively with a SHAP value of {contribution:.2f}\n"

            st.markdown(explanation)

            st.markdown("""
                **Waterfall Plot**: This plot helps visualize how each feature contributes to the specific prediction. 
                Positive contributions push the prediction towards the positive class, while negative contributions push it towards the negative class.
            """)
            fig3, ax3 = plt.subplots(figsize=(10, 6))
            shap.plots.waterfall(shap_values[row_idx], show=False)
            st.pyplot(fig3, bbox_inches='tight', dpi=300, pad_inches=0.1)

            st.subheader("📤 Model Prediction")
            prediction = model.predict([X_test.iloc[row_idx].values])[0]
            st.info(f"Predicted class: **{prediction}**")

        except Exception as e:
            st.error("❌ Error while processing. Ensure target is numerical and dataset is clean.")
    else:
        st.warning("📂 Upload a dataset to begin.")

else:
    st.sidebar.title("🔐 Authentication Required")
    mode = st.sidebar.radio("Choose Mode", ["Login", "Sign Up"])
    if mode == "Login":
        login()
    else:
        signup()
