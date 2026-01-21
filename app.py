# 1 -> Good (lower risk)
# 0 -> Bad (higher risk)

import streamlit as st
import joblib
import pandas as pd

try:
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except Exception:
    PLOTLY_AVAILABLE = False
    import matplotlib.pyplot as plt

SAFE_COLORS = ["#2ecc71", "#e74c3c"]  # Green = Good, Red = Bad

try:
    model = joblib.load("extra_trees_model.pkl")
    encoders = {}
    for col in ["Sex", "Housing", "Saving accounts", "Checking account", "Purpose"]:
        encoders[col] = joblib.load(f"{col}_encoder.pkl")
    st.sidebar.success("Model and encoders loaded successfully ✅")
except Exception as e:
    st.sidebar.error(f"Error loading model/encoders: {e}")
    st.stop()
try:
    data_df = pd.read_csv("german_credit_data.csv")
    st.sidebar.success("Dataset loaded successfully ✅")
except Exception as e:
    data_df = None
    st.sidebar.warning(f"Dataset not available: {e}")

st.title("Credit Risk Prediction App")
st.write("Enter the details below to predict credit risk:")

age = st.number_input("Age", min_value=18, max_value=80, value=30)
sex = st.selectbox("Sex", ["male", "female"])
job = st.number_input("Job (0-3)", min_value=0, max_value=3, value=1)
housing = st.selectbox("Housing", ["own", "rent", "free"])
saving_accounts = st.selectbox("Saving accounts", ["little", "moderate", "rich", "quite rich"])
checking_account = st.selectbox("Checking account", ["little", "moderate", "rich"])
purpose = st.selectbox(
    "Purpose",
    ["business", "car", "domestic appliances", "education",
     "furniture/equipment", "radio/TV", "repairs", "vacation/others"]
)

credit_amount = st.number_input("Credit Amount", min_value=0.0, value=1000.0)
duration = st.number_input("Duration (months)", min_value=1, value=12)

input_df = pd.DataFrame({
    "Age": [age],
    "Sex": [encoders["Sex"].transform([sex])[0]],
    "Job": [job],
    "Housing": [encoders["Housing"].transform([housing])[0]],
    "Saving accounts": [encoders["Saving accounts"].transform([saving_accounts])[0]],
    "Checking account": [encoders["Checking account"].transform([checking_account])[0]],
    "Purpose": [encoders["Purpose"].transform([purpose])[0]],
    "Credit amount": [credit_amount],
    "Duration": [duration]
})

try:
    input_df = input_df[list(model.feature_names_in_)]
except Exception as e:
    st.warning(f"Could not align input features: {e}")

st.write("Input data (aligned to model features):", input_df)

if st.button("Predict"):
    try:
        prediction = model.predict(input_df)[0]

        if prediction == 1:
            st.success("The credit risk is: **LOW (Good)**")
        else:
            st.error("The credit risk is: **HIGH (Bad)**")

        try:
            proba = model.predict_proba(input_df)[0]
            labels = ["Good (Low risk)" if c == 1 else "Bad (High risk)" for c in model.classes_]

            if PLOTLY_AVAILABLE:
                proba_df = pd.DataFrame({
                    "Outcome": labels,
                    "Probability": proba
                })
                fig = px.pie(
                    proba_df,
                    names="Outcome",
                    values="Probability",
                    title="Model predicted probabilities",
                    color_discrete_sequence=SAFE_COLORS
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                fig, ax = plt.subplots()
                ax.pie(proba, labels=labels, autopct='%1.1f%%', startangle=90)
                ax.set_title("Model predicted probabilities")
                st.pyplot(fig)

        except Exception as e:
            st.warning(f"Could not compute probabilities or draw pie chart: {e}")

        if data_df is not None:
            try:
                counts = data_df["Risk"].value_counts()
                dist_df = pd.DataFrame({
                    "Outcome": ["Good (Low risk)", "Bad (High risk)"],
                    "Count": [
                        int(counts.get("good", 0)),
                        int(counts.get("bad", 0))
                    ]
                })

                if PLOTLY_AVAILABLE:
                    fig2 = px.pie(
                        dist_df,
                        names="Outcome",
                        values="Count",
                        title="Dataset distribution (german_credit_data.csv)",
                        color_discrete_sequence=SAFE_COLORS
                    )
                    st.plotly_chart(fig2, use_container_width=True)
                else:
                    fig2, ax2 = plt.subplots()
                    ax2.pie(
                        dist_df["Count"],
                        labels=dist_df["Outcome"],
                        autopct='%1.1f%%',
                        startangle=90
                    )
                    ax2.set_title("Dataset distribution (german_credit_data.csv)")
                    st.pyplot(fig2)

            except Exception as e:
                st.warning(f"Could not create dataset pie chart: {e}")

    except Exception as e:
        st.error(f"Prediction error: {e}")
