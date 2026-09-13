# Module 2: NLP Pipeline, Vector Space Modeling & Distance Mathematics

## 1. Entity Space Collapse Transformation

### The Entity Ambiguity Problem
In natural language processing, entity names often share tokens (e.g., first names or last names). Consider two distinct individuals:
- Actor **Sam Worthington** (Star of *Avatar*)
- Director **Sam Raimi** (Director of *Spider-Man*)

If tokens are left un-collapsed as `"Sam" "Worthington"` and `"Sam" "Raimi"`, the term `"Sam"` will trigger false similarity scores between *Avatar* and *Spider-Man*. The model would falsely conclude that these two movies are related simply because both creators share the first name `"Sam"`.

### Solution: Whitespace Elimination (Entity Collapsing)
We remove all spaces inside entities across genres, keywords, cast, and crew:
```python
movies['genres'] = movies['genres'].apply(lambda x: [i.replace(" ", "") for i in x])
movies['keywords'] = movies['keywords'].apply(lambda x: [i.replace(" ", "") for i in x])
movies['cast'] = movies['cast'].apply(lambda x: [i.replace(" ", "") for i in x])
movies['crew'] = movies['crew'].apply(lambda x: [i.replace(" ", "") for i in x])
```

#### Outcome:
- `"Sam Worthington"` $\rightarrow$ `"SamWorthington"` (Single unique entity token)
- `"Sam Raimi"` $\rightarrow$ `"SamRaimi"` (Single unique entity token)
- `"Science Fiction"` $\rightarrow$ `"ScienceFiction"`

---

## 2. Tag Aggregation & Text Normalization

### Creating the Unified Tag Vector Source
All tokenized columns (`overview`, `genres`, `keywords`, `cast`, `crew`) are concatenated into a single master feature list `tags`:
```python
movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']

new_df = movies[['movie_id', 'title', 'tags']]
new_df['tags'] = new_df['tags'].apply(lambda x: " ".join(x))
new_df['tags'] = new_df['tags'].apply(lambda x: x.lower())
```

### Stemming with Porter Stemmer
Words often appear in various inflected forms (e.g., `'dance'`, `'dancing'`, `'danced'`, `'dancer'`). Without normalization, these variations create separate vector dimensions.

#### Why Stemming?
Stemming reduces words to their root/stem by stripping suffixes:
- `'action'`, `'actions'` $\rightarrow$ `'action'`
- `'loved'`, `'loving'`, `'love'` $\rightarrow$ `'love'`

```python
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()

def stem(text):
    L = []
    for i in text.split():
        L.append(ps.stem(i))
    return " ".join(L)

new_df['tags'] = new_df['tags'].apply(stem)
```

#### Stemming vs. Lemmatization (Interview Comparison)

| Dimension | Stemming (Porter Stemmer) | Lemmatization (WordNet Lemmatizer) |
| :--- | :--- | :--- |
| **Mechanism** | Rule-based string truncation (heuristic algorithm). | Vocabulary lookup + morphological analysis based on POS tag. |
| **Output** | May produce non-real words (e.g., `'comput'`, `'danc'`). | Always produces valid dictionary words (e.g., `'compute'`, `'dance'`). |
| **Speed / Overhead**| Fast, light computational footprint ($O(N)$ string operations). | Slower, requires dictionary lookup & Part-of-Speech context. |
| **Choice Rationale**| Ideal for Bag-of-Words recommendation where token index matching matters more than human readability. | Ideal for downstream tasks like QA, Machine Translation, or Summarization. |

---

## 3. Vector Space Model: Bag of Words (CountVectorizer)

To compute numerical similarities, text tags must be converted into dense or sparse numerical vectors.

### Vectorizer Configuration
```python
from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer(max_features=5000, stop_words='english')
vectors = cv.fit_transform(new_df['tags']).toarray()
```

### Key Parameters Explained:
1. `max_features=5000`: Extracts the top 5,000 most frequent unique words across the entire corpus of 4,806 movies. Limits vocabulary size to control memory footprint and remove ultra-rare noise.
2. `stop_words='english'`: Automatically filters out uninformative English stop words (`'is'`, `'the'`, `'in'`, `'at'`, `'and'`).
3. Output Shape: Matrix of dimension **$4,806 \times 5,000$** where $M_{i,j}$ represents the term frequency of word $j$ in movie $i$.

---

## 4. Mathematical Deep Dive: Cosine Similarity vs. Euclidean Distance

After constructing the feature matrix $V \in \mathbb{R}^{4806 \times 5000}$, we compute pair-wise item similarities.

```
       Movie A Vector (5000-D) ──┐
                                 ├──> Cosine Similarity Function ──> Sim Score ∈ [0, 1]
       Movie B Vector (5000-D) ──┘
```

### Mathematical Definitions

#### 1. Euclidean Distance ($d_E$)
The straight-line distance between two points in $D$-dimensional space:
$$d_E(\mathbf{A}, \mathbf{B}) = \sqrt{\sum_{i=1}^{D} (A_i - B_i)^2} = \|\mathbf{A} - \mathbf{B}\|_2$$

#### 2. Cosine Similarity ($\text{cos}(\theta)$)
The cosine of the angle between two vectors in $D$-dimensional space:
$$\text{Sim}(\mathbf{A}, \mathbf{B}) = \cos(\theta) = \frac{\mathbf{A} \cdot \mathbf{B}}{\|\mathbf{A}\|_2 \|\mathbf{B}\|_2} = \frac{\sum_{i=1}^{D} A_i B_i}{\sqrt{\sum_{i=1}^{D} A_i^2} \sqrt{\sum_{i=1}^{D} B_i^2}}$$

### Crucial Interview Defense: Why Cosine Similarity over Euclidean Distance?

1. **Curse of Dimensionality**:
   - In high-dimensional spaces ($D = 5000$), Euclidean distance concentrates (distances between almost all pairs of points converge to similar values), making distance comparisons unstable.
2. **Magnitude / Document Length Sensitivity**:
   - Suppose Movie A has a long, detailed plot overview (200 words) and Movie B has a short summary (20 words) describing the *exact same* plot.
   - Movie A's term vector will have higher frequencies (larger magnitude $\|\mathbf{A}\|$), while Movie B's term vector will have smaller magnitude ($\|\mathbf{B}\|$).
   - **Euclidean distance** will measure a large distance $d_E(\mathbf{A}, \mathbf{B})$ purely due to vector length differences.
   - **Cosine similarity** normalizes by vector magnitude ($\|\mathbf{A}\| \|\mathbf{B}\|$), measuring purely direction/orientation. It returns $\cos(\theta) \approx 1.0$, correctly identifying high similarity regardless of summary length.

```python
from sklearn.metrics.pairwise import cosine_similarity

similarity = cosine_similarity(vectors) # Output shape: (4806, 4806)
```

---

## 5. Alternative Modeling Approaches & Evaluation Trade-offs

During interviews, candidate selection of Bag-of-Words is often challenged with: *"Why not TF-IDF, Word2Vec, BERT, or Matrix Factorization?"*

### Vectorization Strategy Comparison

| Approach | Mechanics | Advantages | Disadvantages |
| :--- | :--- | :--- | :--- |
| **CountVectorizer (BoW)** *(Used)* | Term frequency counts across top 5,000 vocabulary words. | Extremely fast, simple, low memory overhead, highly interpretable. | Ignores word order and term importance across corpus. |
| **TF-IDF Vectorizer** | Multiplies Term Frequency (TF) by Inverse Document Frequency (IDF). | Penalizes overly common words that slip past stop word lists. | Still ignores semantic context and word order. |
| **Word Embeddings (Word2Vec / GloVe)** | Dense continuous vector spaces ($D=300$) trained on co-occurrence. | Captures semantic similarity (e.g., `king - man + woman = queen`). | Averaging word vectors across a document loses fine-grained keyword signal. |
| **Transformer Embeddings (SBERT)** | Pretrained Transformer embeddings (e.g., `all-MiniLM-L6-v2`). | Captures full sentence context and deep semantic nuance. | Higher compute/latency for inference; requires GPU or vector search index for scale. |

### Recommendation Paradigm Comparison

| Model Family | Data Required | Pros | Cons |
| :--- | :--- | :--- | :--- |
| **Content-Based Filtering** *(Used)* | Item metadata only. | Solves Cold Start; no user privacy concerns; instant for new items. | Overspecialization (filter bubble); cannot discover serendipitous tastes outside metadata. |
| **Collaborative Filtering (SVD/ALS)** | User-Item interaction matrix (ratings/clicks). | Learns complex implicit patterns; provides serendipitous recommendations. | Severe Cold Start for new users/items; sparse matrix issue. |
| **Hybrid Systems** | Item metadata + User interactions. | Combines strengths of both; industry standard at Netflix/YouTube. | Increased system complexity and infrastructure costs. |

---

## 6. Recommendation Logic Implementation

The recommendation function finds top 5 closest items while preserving index identity using Python's `enumerate`:

```python
def recommend(movie):
    # 1. Fetch index of input movie title
    movie_index = new_df[new_df['title'] == movie].index[0]
    
    # 2. Get similarity vector for the movie and pair with original indices
    distances = similarity[movie_index]
    movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]
    
    # 3. Print top 5 recommendations
    for i in movies_list:
        print(new_df.iloc[i[0]].title)
```
*Note*: Slice `[1:6]` skips index `0` because index `0` is the movie itself ($\cos(\theta) = 1.0$).
