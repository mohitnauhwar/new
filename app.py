import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Load data
st.title("🎬 TMDB Movie Dataset EDA")
st.markdown("Explore the TMDB 5000 Movie dataset with interactive charts.")

# Upload CSV file
uploaded_file = st.file_uploader("Upload your TMDB Movie CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    
    # Show raw data
    st.subheader("📊 Raw Data Sample")
    st.dataframe(df.head())

    # Display available columns
    st.markdown("### 📋 Available Columns")
    st.write(df.columns.tolist())

    # Plot: Vote Average Distribution
    if 'vote_average' in df.columns:
        st.subheader("⭐ Vote Average Distribution")
        fig1, ax1 = plt.subplots()
        df['vote_average'].value_counts().sort_index().plot(kind='bar', ax=ax1)
        ax1.set_xlabel("Vote Average")
        ax1.set_ylabel("Number of Movies")
        st.pyplot(fig1)
    else:
        st.warning("Column `vote_average` not found in your dataset.")

    # Plot: Vote Count Distribution
    if 'vote_count' in df.columns:
        st.subheader("🗳 Vote Count Distribution")
        fig2, ax2 = plt.subplots()
        df['vote_count'].hist(bins=30, ax=ax2)
        ax2.set_xlabel("Vote Count")
        ax2.set_ylabel("Number of Movies")
        st.pyplot(fig2)
    else:
        st.warning("Column `vote_count` not found in your dataset.")

    # Optional: Movie filter by rating
    st.subheader("🎯 Top Rated Movies Filter")
    min_votes = st.slider("Minimum Vote Count", 0, int(df['vote_count'].max()), 1000)
    min_rating = st.slider("Minimum Vote Average", 0.0, 10.0, 7.5)

    filtered = df[(df['vote_average'] >= min_rating) & (df['vote_count'] >= min_votes)]
    st.write(f"Found {len(filtered)} movies with vote average ≥ {min_rating} and vote count ≥ {min_votes}.")
    st.dataframe(filtered[['title', 'vote_average', 'vote_count']].sort_values(by='vote_average', ascending=False))

else:
    st.info("📥 Please upload your `tmdb_5000_movies.csv` file to begin.")
