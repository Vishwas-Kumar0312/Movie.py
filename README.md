import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

# Sample dataset
data = {
    "Title": [
        "The Shawshank Redemption", "The Godfather", "The Dark Knight",
        "Pulp Fiction", "Schindler's List", "Inception", "Fight Club",
        "Forrest Gump", "The Matrix", "Goodfellas"
    ],
    "Genre": [
        "Drama", "Crime|Drama", "Action|Crime|Drama", "Crime|Drama",
        "Biography|Drama|History", "Action|Adventure|Sci-Fi", "Drama",
        "Comedy|Drama|Romance", "Action|Sci-Fi", "Biography|Crime|Drama"
    ]
}

# Load the data into a Pandas DataFrame
df = pd.DataFrame(data)

# Step 2: Process the Genre column
# Convert genre text into a bag-of-words format
count_vectorizer = CountVectorizer(tokenizer=lambda x: x.split('|'))
genre_matrix = count_vectorizer.fit_transform(df['Genre'])

# Step 3: Calculate the cosine similarity
cosine_sim = cosine_similarity(genre_matrix)

# Function to recommend movies
def recommend_movies(title, cosine_sim=cosine_sim):
    # Get the index of the movie that matches the title
    idx = df[df['Title'] == title].index[0]

    # Get similarity scores for all movies with the input movie
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort movies based on similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the top 5 most similar movies (excluding the input movie itself)
    sim_scores = sim_scores[1:6]

    # Get movie indices
    movie_indices = [i[0] for i in sim_scores]

    # Return the top 5 most similar movies
    return df['Title'].iloc[movie_indices]

# Example usage
movie_title = "The Godfather"
recommended_movies = recommend_movies(movie_title)
print(f"Movies similar to '{movie_title}':\n{recommended_movies}")
