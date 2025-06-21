import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# Load data and model
df = pd.read_csv("processed_data.csv")
model = joblib.load("netflix_model.pkl")  # if you have a model

st.title("🎬 Netflix Prize Data Exploration")

# EDA - show data
st.subheader("Sample of Dataset")
st.dataframe(df.head(20))

# Add a rating distribution chart
st.subheader("Rating Distribution")
fig, ax = plt.subplots()
df['Rating'].value_counts().sort_index().plot(kind='bar', ax=ax)
st.pyplot(fig)

# (Optional) Simple prediction interface
st.subheader("Try Prediction (optional)")
user_input = st.number_input("Enter User ID", min_value=1, max_value=500000)
movie_input = st.number_input("Enter Movie ID", min_value=1, max_value=20000)
if st.button("Predict Rating"):
    pred = model.predict([[user_input, movie_input]])
    st.success(f"Predicted Rating: {pred[0]:.2f}")
