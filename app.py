import streamlit as st
#  To fetch movie using api of TMDB
import requests
def fetchMoviePoster(movie_id):
    #  Hit the api here with using the movie id and api key
    response = requests.get('https://api.themoviedb.org/3/movie/{}?api_key=8265bd1679663a7ea12ac168da84d2e8&language=en-US'.format(movie_id))
    data = response.json();
    return "https://image.tmdb.org/t/p/w500" + data["poster_path"]


def recommend(movie):
    movie_index = movies[movies['title'] == movie].index[0]
    movie_list = sorted(list(enumerate(similarity[movie_index])), reverse=True, key=lambda x: x[1])[1:6]
    recommended_movies = []
    recommended_movies_poster = []
    #  actually movie index is not movie id so we have to get movie id first

    for movie in movie_list:
        recommended_movies.append(movies.iloc[movie[0]].title)
        recommended_movies_poster.append(fetchMoviePoster(movies.iloc[movie[0]].movie_id))
    return  recommended_movies,recommended_movies_poster

st.text('Movie recommendation system')

import pickle
movies = pickle.load(open('movies.pkl', 'rb'))
similarity = pickle.load(open('similarity.pkl', 'rb'))

movies_list = movies['title'].values

selected_movie_name = st.selectbox(
    "How would you like to be contacted?",
    movies_list,
)

# if st.button("Recommend"):
#     recommendations, recommended_posters = recommend(selected_movie_name)
#     for i in range(0, len(recommendations)):
#         st.write(recommendations[i])
#         st.image(recommended_posters[i])

if st.button("Recommend"):
    recommendations, recommended_posters = recommend(selected_movie_name)

    col1, col2, col3, col4, col5 = st.columns(5)

    cols = [col1, col2, col3, col4, col5]

    for i in range(len(recommendations)):
        with cols[i]:
            st.text(recommendations[i])
            st.image(recommended_posters[i])