# Module 4: Quick Revision Cheat Sheet (Night Before Interview)

> [!TIP]
> Read this 5-minute cheat sheet on the morning or night before your interview to refresh all key metrics, formulas, code logic, and architectural defenses.

---

## 1. 60-Second Interview Elevator Pitch

> *"I built **CineRecommend**, a personalized Content-Based Movie Recommendation System designed to solve the **User Cold-Start Problem** by recommending top-5 similar movies based on textual and metadata attributes rather than user history.*
>
> *I merged the TMDB 5000 Movies and Credits datasets, performed feature selection to focus on plot overview, genres, keywords, top 3 cast members, and the director. I engineered a custom entity collapsing technique to eliminate space ambiguities in actor and director names (e.g., converting 'Sam Worthington' to 'SamWorthington' to distinguish from 'Sam Raimi').*
>
> *After lowercasing and applying Porter Stemming via NLTK, I built a 5,000-dimensional Bag-of-Words vector space using Scikit-Learn’s `CountVectorizer`. I evaluated movie similarity using **Cosine Similarity**, which is length-invariant and avoids the curse of dimensionality inherent in Euclidean distance.*
>
> *Finally, I deployed the application via **Streamlit**, integrating TMDB’s REST API to dynamically fetch movie posters, and configured automated Google Drive downloading via `gdown` for seamless cloud model artifact management."*

---

## 2. Quick Fact Matrix

| Component | Exact Detail / Metric |
| :--- | :--- |
| **Dataset Source** | TMDB 5000 (`tmdb_5000_movies.csv` + `tmdb_5000_credits.csv`) |
| **Clean Row Count** | **4,806 rows** (after dropping 3 null overviews) |
| **Selected Features (7)** | `movie_id`, `title`, `overview`, `genres`, `keywords`, `cast`, `crew` |
| **Dropped Features (16)**| `budget`, `revenue`, `popularity`, `vote_average`, `vote_count`, `original_language`, `homepage`, `release_date`, `runtime`, `production_companies`, etc. |
| **Parsing Library** | `ast.literal_eval` (parsed JSON strings into Python lists) |
| **Whitespace Removal** | Replaced `" "` with `""` inside entities (prevented name confusion) |
| **Stemmer Used** | `nltk.stem.porter.PorterStemmer` |
| **Vectorizer** | `CountVectorizer(max_features=5000, stop_words='english')` |
| **Vector Matrix Shape**| **$4,806 \times 5,000$** (Sparse Count Matrix) |
| **Similarity Metric** | **Cosine Similarity** ($\cos(\theta) = \frac{\mathbf{A} \cdot \mathbf{B}}{\|\mathbf{A}\| \|\mathbf{B}\|}$) |
| **Similarity Shape** | **$4,806 \times 4,806$** ($\approx 184.8\text{ MB}$ pickle file) |
| **Frontend UI** | Streamlit (`app.py` with 5-column layout) |
| **External API** | TMDB REST API (`https://api.themoviedb.org/3/movie/{id}`) |
| **Cloud Storage** | `gdown` (Google Drive pickle downloader) |

---

## 3. Core Mathematical Formulas

### 1. Cosine Similarity
$$\text{Sim}(\mathbf{A}, \mathbf{B}) = \cos(\theta) = \frac{\mathbf{A} \cdot \mathbf{B}}{\|\mathbf{A}\|_2 \|\mathbf{B}\|_2} = \frac{\sum_{i=1}^{D} A_i B_i}{\sqrt{\sum_{i=1}^{D} A_i^2} \sqrt{\sum_{i=1}^{D} B_i^2}}$$

### 2. Euclidean Distance
$$d_E(\mathbf{A}, \mathbf{B}) = \sqrt{\sum_{i=1}^{D} (A_i - B_i)^2}$$

### 3. Memory Complexity of Exact Precomputed Matrix
$$\text{Memory} = N^2 \times 8 \text{ bytes}$$

---

## 4. Crucial Code Snippets to Remember

### 1. Parse JSON & Extract Director
```python
import ast

def fetchDirector(obj):
    L = []
    for i in ast.literal_eval(obj):
        if i['job'] == 'Director':
            L.append(i['name'])
            break
    return L
```

### 2. Space Removal (Entity Collapsing)
```python
movies['cast'] = movies['cast'].apply(lambda x: [i.replace(" ", "") for i in x])
movies['crew'] = movies['crew'].apply(lambda x: [i.replace(" ", "") for i in x])
```

### 3. Stemming & CountVectorizer
```python
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer

ps = PorterStemmer()

def stem(text):
    return " ".join([ps.stem(word) for word in text.split()])

new_df['tags'] = new_df['tags'].apply(stem)

cv = CountVectorizer(max_features=5000, stop_words='english')
vectors = cv.fit_transform(new_df['tags']).toarray()
```

### 4. Cosine Similarity & Index Enumerate Lookup
```python
from sklearn.metrics.pairwise import cosine_similarity

similarity = cosine_similarity(vectors)

def recommend(movie):
    index = new_df[new_df['title'] == movie].index[0]
    distances = similarity[index]
    # Enumerate keeps track of original movie index after sorting by similarity
    movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]
    return [new_df.iloc[i[0]].title for i in movies_list]
```

---

## 5. Top 5 Interview "Gotchas" & Instant Answers

1. **Q: Why did you drop budget and revenue?**
   - *A: Budget reflects financial scale, not content similarity. A \$200M sci-fi movie and a \$200M animated comedy share zero thematic overlap.*
2. **Q: Why collapse 'Sam Worthington' to 'SamWorthington'?**
   - *A: To treat full names as single atomic tokens. Without this, 'Sam Worthington' and 'Sam Raimi' share the term 'Sam', creating false similarity across unrelated movies.*
3. **Q: Why Cosine Similarity over Euclidean Distance?**
   - *A: Cosine similarity measures angular direction regardless of document length (vector magnitude). Euclidean distance is distorted by plot summary length differences.*
4. **Q: Why Content-Based over Collaborative Filtering?**
   - *A: Recommending items purely based on content metadata solves the User Cold-Start Problem (works for new users with zero history) and TMDB lacks user interaction matrices.*
5. **Q: How would you scale this to 1 Million movies?**
   - *A: Replace precomputed $O(N^2)$ similarity matrix with dense embeddings (e.g. SBERT) stored in a Vector Database (Faiss/Pinecone) using HNSW Approximate Nearest Neighbors (ANN) indexing for $O(\log N)$ sub-10ms lookup.*
