import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import os

st.set_page_config(page_title="Netflix EDA & Predictor", layout="centered")

st.title("🎬 Netflix Prize Data Exploration")

# Load Data
if os.path.exists("processed_data.csv"):
    df = pd.read_csv("processed_data.csv")
    st.subheader("📊 Sample of Dataset")
    st.dataframe(df.head(20))

    # Show columns
    st.markdown("### 🧾 Available Columns:")
    st.write(df.columns.tolist())

    # Rating distribution
    if 'rating' in df.columns:
        st.subheader("⭐ Rating Distribution")
        fig, ax = plt.subplots()
        df['rating'].value_counts().sort_index().plot(kind='bar', ax=ax)
        ax.set_xlabel("Rating")
        ax.set_ylabel("Frequency")
        st.pyplot(fig)
    else:
        st.warning("Column 'rating' not found in dataset.")
else:
    st.error("❌ 'processed_data.csv' not found. Please ensure the file is in the same directory as app.py.")

# Load Model
if os.path.exists("netflix_model.pkl"):
    model = joblib.load("netflix_model.pkl")
    st.success("✅ Model loaded successfully.")
else:
    model = None
    st.warning("⚠️ 'netflix_model.pkl' not found. Prediction will be disabled.")

# Prediction Interface
if model:
    st.subheader("🎯 Predict User-Movie Rating")
    user_input = st.number_input("Enter User ID", min_value=1, max_value=500000, value=1)
    movie_input = st.number_input("Enter Movie ID", min_value=1, max_value=20000, value=1)

    if st.button("Predict Rating"):
        try:
            pred = model.predict([[user_input, movie_input]])
            st.success(f"Predicted Rating: {pred[0]:.2f}")
        except Exception as e:
            st.error(f"Prediction failed: {e}")
else:
    st.info("Upload the model to enable prediction feature.")
