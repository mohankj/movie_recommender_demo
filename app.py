import streamlit as st
from recommender import MovieRecommender

# Initialize recommender
recommender = MovieRecommender('data/movies1.csv', 'data/ratings1.csv')

# Streamlit UI
st.title('🎬 Movie Recommendation System')
st.markdown("""
Get personalized movie recommendations based on your ratings!
""")

# Get unique user IDs from ratings data
user_ids = sorted(recommender.ratings['user_id'].unique())

# User selection dropdown
user_id = st.selectbox('Select User ID', user_ids)

# Display user's perfection ratings
st.subheader('Your Perfection Ratings')
print("st",st.__version__)
perfection_movies = recommender.get_user_perfection_ratings(user_id)

# Check if we got any movies and remove duplicates
if not perfection_movies.empty:
    # Drop duplicates based on title (you can use 'movie_id' if preferred)
    perfection_movies = perfection_movies.drop_duplicates(subset=['title'])
    perfection_movies.reset_index(drop=True,inplace=True)
    # Display the dataframe
    st.dataframe(perfection_movies[['title', 'genres']])
else:
    st.info("This user hasn't rated any movies as 'Perfection' yet.")
    
    
# Get recommendations
if st.button('Get Recommendations'):
    st.subheader('Recommended For You')
    recommendations = recommender.get_hybrid_recommendations(user_id)
    
    for idx, row in recommendations.iterrows():
        with st.expander(f"🌟 {row['title']} (Predicted rating: {row.get('recommendation_category', 'N/A')})"):
            st.markdown(f"""
            **Genres:** {row['genres']}  
            **Director:** {row['director']}  
            **Why recommended:** Similar to your perfection-rated movies
            """)

