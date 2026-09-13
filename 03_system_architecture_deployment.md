# Module 3: System Architecture, Deployment & Production Engineering

## 1. End-to-End System Architecture

The CineRecommend system is built around a lightweight, modular deployment architecture that integrates offline model preparation with a dynamic web user interface.

```
+-----------------------------------------------------------------------------------+
|                               OFFLINE PIPELINE                                    |
|                                                                                   |
|  [tmdb_5000_movies.csv]                                                           |
|           +            ---> [EDA & Feature Eng.] ---> [BoW & Cosine Sim Matrix]   |
|  [tmdb_5000_credits.csv]                                     |                    |
|                                                              v                    |
|                                                     [movies.pkl] (2.2 MB)         |
|                                                     [similarity.pkl] (184 MB)     |
+-----------------------------------------------------------------------------------+
                                                               |
                                                               v Uploaded to Cloud
+-----------------------------------------------------------------------------------+
|                              ONLINE STREAMLIT UI                                  |
|                                                                                   |
|  User Selection ──> [app.py] ──> Load Pickles (Auto-Download via gdown if missing) |
|                         │                                                         |
|                         ├── Lookup Similarity Row [1:6]                           |
|                         ├── Fetch Posters via TMDB REST API                       |
|                         └── Render 5-Column Grid Output                           |
+-----------------------------------------------------------------------------------+
```

---

## 2. Dynamic Web Interface & API Integration

### Streamlit UI Implementation (`app.py`)
The frontend is constructed using **Streamlit**, providing a clean, reactive single-page app interface.

#### 1. Page Configuration & Layout setup
```python
import streamlit as st
import pickle
import requests
import os
import gdown

st.set_page_config(page_title="CineRecommend", page_icon="🎬")
st.title('CineRecommend')
st.markdown("##### Personalised Movie Recommendation System")
```

#### 2. Auto-Downloading Model Artifacts (`gdown`)
Because the similarity matrix `similarity.pkl` is ~184 MB (exceeding standard GitHub file size limits of 100 MB), artifacts are hosted on Google Drive and automatically downloaded on cold start if not present locally:
```python
if not os.path.exists("movies.pkl"):
    gdown.download(
        "https://drive.google.com/uc?id=1hbHmTynyluS-lYwCM2XOVNnpbJcD_1UD",
        "movies.pkl",
        quiet=False
    )

if not os.path.exists("similarity.pkl"):
    gdown.download(
        "https://drive.google.com/uc?id=1T8iieekvBoPgXWR5l1WY6wZp5oYgcrmr",
        "similarity.pkl",
        quiet=False
    )
```

#### 3. TMDB REST API Integration for Poster Fetching
When recommendations are calculated, the application queries TMDB's REST API using the movie's unique `movie_id` to retrieve high-resolution poster images:

```python
def fetchMoviePoster(movie_id):
    url = f'https://api.themoviedb.org/3/movie/{movie_id}?api_key=8265bd1679663a7ea12ac168da84d2e8&language=en-US'
    response = requests.get(url)
    data = response.json()
    return "https://image.tmdb.org/t/p/w500" + data["poster_path"]
```

#### 4. Recommendation & Multi-Column Rendering
```python
def recommend(movie):
    movie_index = movies[movies['title'] == movie].index[0]
    movie_list = sorted(list(enumerate(similarity[movie_index])), reverse=True, key=lambda x: x[1])[1:6]
    
    recommended_movies = []
    recommended_movies_poster = []

    for item in movie_list:
        movie_id = movies.iloc[item[0]].movie_id
        recommended_movies.append(movies.iloc[item[0]].title)
        recommended_movies_poster.append(fetchMoviePoster(movie_id))
        
    return recommended_movies, recommended_movies_poster

selected_movie_name = st.selectbox(
    "Search or select a movie to get recommendations:",
    movies['title'].values
)

if st.button("Recommend"):
    recommendations, recommended_posters = recommend(selected_movie_name)
    col1, col2, col3, col4, col5 = st.columns(5)
    cols = [col1, col2, col3, col4, col5]

    for i in range(len(recommendations)):
        with cols[i]:
            st.text(recommendations[i])
            st.image(recommended_posters[i])
```

---

## 3. Production Deployment Configuration Files

### Deployment Stack Overview
- **Hosting Platform**: Streamlit Community Cloud
- **Live Deployment URL**: [https://movierecommendarsystem-3wiq5yjnan2xqjszeiabof.streamlit.app/](https://movierecommendarsystem-3wiq5yjnan2xqjszeiabof.streamlit.app/)
- **Process Manager**: `Procfile`
- **Port & Shell Setup**: `setup.sh`

#### `Procfile`
Specifies the execution command for Heroku dynos:
```makefile
web: sh setup.sh && streamlit run app.py
```

#### `setup.sh`
Configures Streamlit credentials and dynamic port allocation:
```bash
mkdir -p ~/.streamlit/
echo "\
[server]\n\
headless = true\n\
port = $PORT\n\
enableCORS = false\n\
\n\
" > ~/.streamlit/config.toml
```

---

## 4. Engineering Bottlenecks & Production Scalability Analysis

In technical interviews (especially for Senior DS / ML Engineer roles), interviewers love to ask: *"What breaks in your architecture if the catalog grows from 5,000 movies to 1,000,000 movies?"*

### 1. Memory Complexity Bottleneck: $O(N^2)$
- In this implementation, we precompute and store a dense pair-wise similarity matrix $S \in \mathbb{R}^{N \times N}$.
- For $N = 4,806$ movies (float64):
  $$\text{Memory} = 4,806 \times 4,806 \times 8 \text{ bytes} \approx 184.8 \text{ MB}$$
- If catalog grows to $N = 100,000$ movies:
  $$\text{Memory} = 100,000 \times 100,000 \times 8 \text{ bytes} = 80,000,000,000 \text{ bytes} \approx 80 \text{ GB}$$
- If catalog grows to $N = 1,000,000$ movies:
  $$\text{Memory} = 10^{12} \times 8 \text{ bytes} \approx 8 \text{ Terabytes}$$
- **Conclusion**: Storing an exact $N \times N$ matrix in RAM does not scale past $N \approx 20,000$.

### 2. Real-Time API Latency Bottleneck
- Currently, `fetchMoviePoster()` executes 5 synchronous HTTP requests (`requests.get`) sequentially inside a Python loop.
- Network latency per request $\approx 200\text{ ms} \implies 5 \times 200\text{ ms} = 1.0\text{ second}$ UI blocking time.
- **Production Solution**: Use asynchronous HTTP requests (`aiohttp` / `asyncio`) or pre-cache image poster URLs directly in database / Redis cache.

### 3. Production Architecture Migration Strategy (Scaling to $1M+$ items)

To scale this recommendation system to millions of items, transition from **Exact Pairwise Precomputation** to **Vector Indexing & Approximate Nearest Neighbors (ANN)**:

```
[New Movie Document] ──> [Transformer / Embedding Pipeline] ──> [512-D Dense Vector]
                                                                        │
                                                                        v
[User Query] ───────> [Vector Database Index (Faiss / Pinecone)] <──────┘
                              │ (HNSW / IVF Index)
                              v
                   [Sub-10ms ANN Query Result]
```

1. **Vector Database**: Use **Faiss** (Facebook AI Similarity Search), **Pinecone**, or **Milvus**.
2. **Index Structure**: Use **HNSW (Hierarchical Navigable Small World)** graphs or **IVF-PQ (Inverted File with Product Quantization)**.
3. **Query Time Complexity**: Reduces lookup time from $O(N)$ brute-force linear search to **$O(\log N)$** sub-millisecond graph traversal.
4. **Memory Optimization**: Avoids precomputing an $N \times N$ matrix altogether; items are queried dynamically on-demand.
