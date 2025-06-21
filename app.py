# prompt: give me app py for this trained model to deploy it to streamlit

import streamlit as st
import pandas as pd
import joblib
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load the preprocessed data and the model
try:
    movie_df = pd.read_csv('processed_data.csv')
    # Re-apply the cleaning function to list columns as CSV saves them as strings
    json_columns_to_clean = {'cast', 'keywords', 'genres', 'production_companies', 'production_countries'}
    for col in json_columns_to_clean:
        # Convert string representation of list back to list
        movie_df[col] = movie_df[col].apply(lambda x: eval(x) if pd.notna(x) and x.startswith('[') and x.endswith(']') else [])
        movie_df[col] = movie_df[col].apply(lambda x: [item.strip("'") for item in x]) # Clean single quotes if any

    # Re-create the 'join_features' column from the loaded data
    features_to_join = ['cast', 'keywords', 'genres', 'director', "writer", "producer", "production_companies", "production_countries"]
    def create_joined_features_streamlit(x):
        return ' '.join([' '.join(x[f]) if isinstance(x[f], list) else str(x[f]) for f in features_to_join])
    movie_df["join_features"] = movie_df.apply(create_joined_features_streamlit, axis=1)


    # Re-calculate the cosine similarity matrix
    count = CountVectorizer(stop_words='english')
    count_matrix = count.fit_transform(movie_df['join_features'])
    cosine_sim2 = cosine_similarity(count_matrix, count_matrix)

    # Re-create the indices Series
    movie_df = movie_df.reset_index()
    indices = pd.Series(movie_df.index, index=movie_df['title'])

except FileNotFoundError:
    st.error("Required data files ('processed_data.csv') not found. Please ensure they are in the same directory.")
    st.stop()
except Exception as e:
    st.error(f"An error occurred while loading or processing data: {e}")
    st.stop()


# Recommendation function (same as before, adapted slightly)
def get_recommendations(title, cosine_sim=cosine_sim2):
    if title not in indices.index:
        return pd.DataFrame([["Movie title '{}' not found in the database.".format(title), 0]], columns=["title", "similarity"])

    idx = indices[title]
    if isinstance(idx, pd.Series):
        idx = idx.iloc[0] # Take the first index if there are duplicates

    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:6] # Get the top 5 recommendations (excluding the input movie itself)
    movie_indices = [i[0] for i in sim_scores]
    movie_similarity = [i[1] for i in sim_scores]

    return pd.DataFrame(zip(movie_df['title'].iloc[movie_indices], movie_similarity), columns=["title", "similarity"])

# Streamlit App
st.title("Movie Recommendation System")

# User input
movie_title = st.text_input("Enter a movie title:")

if movie_title:
    recommendations = get_recommendations(movie_title)

    if recommendations.empty or recommendations.iloc[0]['similarity'] == 0:
         st.write(recommendations.iloc[0]['title']) # Display the "not found" message
    else:
        st.subheader("Top 5 Recommendations:")
        st.table(recommendations)
