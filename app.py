import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# -----------------------
# Load Data & Models
# -----------------------
@st.cache_data
def load_data():
  df = pd.read_csv("data/train.csv")
  return df

@st.cache_resource
def load_model():
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    return model

df = load_data()
model = load_model()

# -----------------------
# Sidebar Navigation
# -----------------------
st.sidebar.title("Navigation")
menu = st.sidebar.radio(
    "Go to",
    ["Home", "Data Exploration", "Visualisations", "Model Prediction", "Model Performance"]
)

# -----------------------
# Home Section
# -----------------------
if menu == "Home":
    st.title("📱 Phone Price Prediction App")
    st.markdown("""
    This web application predicts **phone price ranges** based on technical specifications.  
    It allows:
    - Dataset exploration  
    - Interactive visualisations  
    - Real-time predictions using a trained model  
    - Model performance evaluation
    """)

# -----------------------
# Data Exploration Section
# -----------------------
elif menu == "Data Exploration":
    st.header("📊 Data Exploration")

    st.subheader("Dataset Overview")
    st.write(f"Shape: {df.shape}")
    st.write(df.dtypes)

    st.subheader("Sample Data")
    st.dataframe(df.head())

    st.subheader("Interactive Filtering")
    col_to_filter = st.selectbox("Select column to filter", df.columns)
    unique_vals = df[col_to_filter].unique()
    val = st.selectbox("Select value", unique_vals)
    st.write(df[df[col_to_filter] == val])

# -----------------------
# Visualisations Section
# -----------------------
elif menu == "Visualisations":
    st.header("📈 Visualisations")

    # Chart 1: Distribution of target variable
    st.subheader("Price Range Distribution")
    fig, ax = plt.subplots()
    sns.countplot(x="price_range", data=df, ax=ax)
    st.pyplot(fig)

    # Chart 2: Correlation heatmap
    st.subheader("Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

    # Chart 3: Feature vs Target
    feature = st.selectbox("Select feature", df.columns[:-1])
    fig, ax = plt.subplots()
    sns.boxplot(x="price_range", y=feature, data=df, ax=ax)
    st.pyplot(fig)


# Model Prediction Section

# -----------------------
# Model Prediction Section
# -----------------------
elif menu == "Model Prediction":
    st.header("🤖 Model Prediction")

    st.markdown("Enter phone specifications below:")

    # Map numeric predictions to price category names + ranges
    price_labels = {
        0: "Low Cost (Rs 10,000 - Rs 20,000)",
        1: "Medium Cost (Rs 20,000 - Rs 40,000)",
        2: "High Cost (Rs 40,000 - Rs 60,000)",
        3: "Very High Cost (Rs 60,000+)"
    }

    input_data = {}
    for col in df.columns[:-1]:  # assuming last col is target
        if df[col].dtype in [np.int64, np.float64]:
            input_data[col] = st.number_input(f"{col}", value=float(df[col].mean()))
        else:
            input_data[col] = st.selectbox(f"{col}", df[col].unique())

    if st.button("Predict"):
        input_df = pd.DataFrame([input_data])
        prediction = model.predict(input_df)[0]
        category = price_labels.get(int(prediction), "Unknown")

        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(input_df)
            st.write(f"Prediction Probability: {np.max(proba)*100:.2f}%")

        st.success(f"Predicted Price Range: {category}")


# -----------------------
# Model Performance Section
# -----------------------
elif menu == "Model Performance":
    st.header("📉 Model Performance")

    X = df.drop("price_range", axis=1)
    y = df["price_range"]

    preds = model.predict(X)
    acc = accuracy_score(y, preds)
    st.write(f"**Accuracy:** {acc:.2f}")

    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y, preds)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    st.pyplot(fig)

    st.subheader("Classification Report")
    st.text(classification_report(y, preds))
