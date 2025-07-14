import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import sklearn

class MovieRecommender:
    def __init__(self, movies_path, ratings_path):
        self.movies = pd.read_csv(movies_path)
        self.ratings = pd.read_csv(ratings_path)
        self.cosine_sim = None  # Initialize as None
        self.tfidf_matrix = None
        self.user_item_matrix = None
        self.prepare_data()
        print(np.__version__)
        print(pd.__version__)
        print(sklearn.__version__)
    def prepare_data(self):
        # Ensure consistent data types
        self.movies['movie_id'] = self.movies['movie_id'].astype(int)
        self.ratings['movie_id'] = self.ratings['movie_id'].astype(int)
        
        # Create combined features
        self.movies['combined_features'] = self.movies.apply(
            lambda x: f"{x['genres']} {x['director']} {x['keywords']}", axis=1)
        
        # TF-IDF Vectorization
        self.tfidf = TfidfVectorizer(stop_words='english')
        self.tfidf_matrix = self.tfidf.fit_transform(self.movies['combined_features'])
        
        # Calculate cosine similarity between movies
        self.cosine_sim = cosine_similarity(self.tfidf_matrix, self.tfidf_matrix)
        
        # Create user-item matrix
        all_movies = sorted(self.movies['movie_id'].unique())
        self.user_item_matrix = self.ratings.pivot_table(
            index='user_id',
            columns='movie_id',
            values='rating',
            aggfunc='first'
        ).reindex(columns=all_movies).fillna(0)
        
        # Calculate and store average ratings
        self.avg_ratings = self.ratings.groupby('movie_id')['rating'].mean().rename('avg_rating')
        
    

    def get_content_based_recommendations(self, movie_titles, top_n=5):
        """Get recommendations based on movie content similarity"""
        movie_indices = [self.movies[self.movies['title'] == title].index[0] for title in movie_titles]
        sim_scores = np.sum(self.cosine_sim[movie_indices], axis=0)
        sim_scores = list(enumerate(sim_scores))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        
        # Get top N recommendations
        top_indices = [i[0] for i in sim_scores[1:top_n+1]]
        top_scores = [i[1] for i in sim_scores[1:top_n+1]]
        
        # Create results DataFrame with similarity scores
        recommendations = self.movies.iloc[top_indices].copy()
        recommendations['similarity_score'] = top_scores
        
        # Normalize similarity scores to 0-1 range
        recommendations['similarity_score'] = recommendations['similarity_score'] / recommendations['similarity_score'].max()
        
        return recommendations
        
    
    def predict_ratings(self, user_id):
        try:
            # Initialize empty result with expected columns
            empty_result = pd.DataFrame(columns=['movie_id', 'title', 'genres', 'director', 'predicted_rating'])
            
            # Handle unknown users
            if user_id not in self.user_item_matrix.index:
                return empty_result
                    
            # Get user vector and calculate similarities
            user_vec = self.user_item_matrix.loc[user_id].values.reshape(1, -1)
            user_sim = cosine_similarity(user_vec, self.user_item_matrix.values)[0]
            
            # Calculate weighted ratings (add small epsilon to avoid zero)
            weighted_ratings = np.dot(user_sim + 1e-8, self.user_item_matrix.values)
            
            # Create predictions DataFrame
            predictions = pd.DataFrame({
                'movie_id': self.user_item_matrix.columns,
                'predicted_rating': weighted_ratings
            })
            
            # Merge with movie data
            predictions = predictions.merge(
                self.movies[['movie_id', 'title', 'genres', 'director']],
                on='movie_id',
                how='left'
            )
            
            # Filter out already rated movies
            rated_movies = self.ratings[self.ratings['user_id'] == user_id]['movie_id']
            predictions = predictions[~predictions['movie_id'].isin(rated_movies)]
            
            # Normalize ratings to 1-4 scale if there's variation
            if predictions['predicted_rating'].nunique() > 1:
                scaler = MinMaxScaler(feature_range=(1, 4))
                predictions['predicted_rating'] = scaler.fit_transform(
                    predictions[['predicted_rating']])
            
            return predictions.sort_values('predicted_rating', ascending=False)
            
        except Exception as e:
            print(f"Prediction error: {str(e)}")
            return pd.DataFrame(columns=['movie_id', 'title', 'genres', 'director', 'predicted_rating'])
    
    def get_user_perfection_ratings(self, user_id):
        """Get movies rated as 'Perfection' (4) by user"""
        perfection_movies = self.ratings[
            (self.ratings['user_id'] == user_id) & (self.ratings['rating'] == 4)]
        return perfection_movies.merge(self.movies, on='movie_id')
    
    def get_user_all_ratings(self, user_id):
        """Get all movies rated by the user"""
        user_ratings = self.ratings[self.ratings['user_id'] == user_id]
        return user_ratings.merge(self.movies, on='movie_id')   

    
    def get_hybrid_recommendations(self, user_id, top_n=5):
        """Combine content-based and predicted ratings for hybrid recommendations"""
        # Get perfection-rated movies
        perfection_movies = self.get_user_perfection_ratings(user_id)
        
        if len(perfection_movies) == 0:
            return self.movies.sample(top_n)  # Fallback for new users
        
        # Content-based recommendations
        cb_recs = self.get_content_based_recommendations(
            perfection_movies['title'].tolist(), top_n*2)
        
        # Rating predictions
        pred_ratings = self.predict_ratings(user_id)
        
        # Get list of already rated movies
        rated_movies = self.ratings[self.ratings['user_id'] == user_id]['movie_id'].tolist()
        
        # Filter content-based recs to only unwatched movies
        cb_recs_unwatched = cb_recs[~cb_recs['movie_id'].isin(rated_movies)]
        
        # Merge predictions with content-based recs on movie_id
        hybrid_recs = pd.merge(
            cb_recs_unwatched,
            pred_ratings,
            on='movie_id',
            how='inner',  # Only keep movies that appear in both
            suffixes=('', '_pred')
        )
        print(hybrid_recs.columns)
        
        # Add average ratings (no merge - use pre-calculated series)
        hybrid_recs['avg_rating'] = hybrid_recs['movie_id'].map(self.avg_ratings)
        
        # Fill any missing average ratings with neutral value
        hybrid_recs['avg_rating'] = hybrid_recs['avg_rating'].fillna(2.5)
       
        # Calculate combined score (60% prediction, 30% similarity, 10% average)
        hybrid_recs['combined_score'] = (
            0.6 * hybrid_recs['predicted_rating'] +
            0.3 * hybrid_recs['similarity_score'] +
            0.1 * hybrid_recs['avg_rating']
        )
        # Normalize final score to 1-4 scale for consistency
        scaler = MinMaxScaler(feature_range=(1, 4))
        hybrid_recs['combined_score'] = scaler.fit_transform(
            hybrid_recs[['combined_score']])
        
        # Convert scores to rating categories
        def score_to_category(score):
            if score <= 1.5:
                return "Skip It"
            elif score <= 2.5:
                return "Time Pass"
            elif score <= 3.5:
                return "Go For It"
            else:
                return "Perfection"
        
        # Apply the conversion
        hybrid_recs['recommendation_category'] = hybrid_recs['combined_score'].apply(score_to_category)
        
        # Select and order the columns for output
        result_cols = ['movie_id', 'title', 'genres', 'director', 
                      'recommendation_category', 'combined_score']
        
        
        # Return top N recommendations
        return hybrid_recs[result_cols].sort_values('combined_score', ascending=False).head(top_n)
        