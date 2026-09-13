# Comprehensive EDA & Data Preprocessing Challenges

This document details all technical challenges, engineering trade-offs, root-cause analyses, and solutions encountered during the **Exploratory Data Analysis (EDA) and Preprocessing** phase of the **CineRecommend** Movie Recommender System.

---

## 1. Executive Summary of EDA Pipeline

```
Raw CSV Datasets (movies.csv: 4803x20, credits.csv: 4803x4)
                    │
                    ▼
[Challenge 5] Merge on 'title' (4,809 records post-merge)
                    │
                    ▼
[Challenge 6 & 7] Feature Selection: Drop 16 Non-Content Columns & Drop 3 Null Overviews
                    │ (4,806 Clean Movie Records)
                    ▼
[Challenge 1] Parse Stringified JSON Strings (ast.literal_eval)
                    │
   ┌────────────────┼────────────────┬────────────────┐
   ▼                ▼                ▼                ▼
Genres           Keywords         [Challenge 2]    [Challenge 3]
Extract Names    Extract Names    Top 3 Cast       Director Only
   │                │                │                │
   └────────────────┴────────────────┴────────────────┘
                    │
                    ▼
[Challenge 4] Space Stripping & Entity Collapsing (e.g., "Sam Worthington" -> "SamWorthington")
                    │
                    ▼
Concatenate [overview + genres + keywords + cast + crew] -> 'tags' vector
```

---

## 2. Detailed Breakdown of EDA Challenges & Technical Solutions

### Challenge 1: Parsing Stringified JSON Data Literals
* **Context & Symptom:** Columns such as `genres`, `keywords`, `cast`, and `crew` were stored as stringified arrays of dictionary objects in the raw CSV (e.g., `'[{"id": 28, "name": "Action"}, {"id": 12, "name": "Adventure"}]'`).
* **Why Standard Parsing Failed:**
  * Standard string splitting (`str.split()`) left JSON syntax noise (`[`, `{`, `"id":`, `}`).
  * `json.loads()` threw parsing errors because TMDB CSVs used single quotes (`'`) for string keys instead of standard JSON double quotes (`"`).
* **Technical Solution:** Used Python's Abstract Syntax Tree module (`ast.literal_eval`), which safely parses Python literal strings without risk of code injection.

```python
import ast

def extract_names(obj_string):
    """Safely evaluates stringified list of dicts and extracts 'name' values."""
    names = []
    for item in ast.literal_eval(obj_string):
        names.append(item['name'])
    return names

# Applied to genres and keywords
movies['genres'] = movies['genres'].apply(extract_names)
movies['keywords'] = movies['keywords'].apply(extract_names)
```

---

### Challenge 2: Cast Noise Reduction vs. Feature Information Loss
* **Context & Symptom:** The raw `cast` JSON contained up to 30+ actors per movie, including background extras and minor single-scene roles.
* **Impact on Recommendation:** Retaining all cast members introduced heavy vocabulary noise into Bag-of-Words / TF-IDF matrices. Unrelated movies sharing minor background extras would register false similarity.
* **Technical Solution:** Implemented a strict slice extraction rule to retain only the top 3 lead actors, capturing the primary cast signal while suppressing tail noise.

```python
def extract_top3_cast(obj_string):
    """Extracts top 3 cast members from cast array."""
    top3 = []
    counter = 0
    for item in ast.literal_eval(obj_string):
        if counter < 3:
            top3.append(item['name'])
            counter += 1
        else:
            break
    return top3

movies['cast'] = movies['cast'].apply(extract_top3_cast)
```

---

### Challenge 3: Filtering Target Crew Roles from Dense Metadata
* **Context & Symptom:** The `crew` column contained hundreds of production entries (producers, cinematographers, sound designers, gaffers, editors).
* **Impact on Recommendation:** Including all crew members flooded feature vectors with production staff names that users do not associate with movie themes.
* **Technical Solution:** Filtered crew objects specifically for `job == 'Director'`. Directors possess distinct artistic signatures, genre preferences, and storytelling styles, making them a high-potency recommendation signal.

```python
def extract_director(obj_string):
    """Filters crew list exclusively for the Director role."""
    directors = []
    for item in ast.literal_eval(obj_string):
        if item.get('job') == 'Director':
            directors.append(item['name'])
            break # Retain primary director
    return directors

movies['crew'] = movies['crew'].apply(extract_director)
```

---

### Challenge 4: Multi-Word Entity Ambiguity & Token Collisions (The "Space Removal" Problem)
* **Context & Symptom:** Full names and multi-word descriptors contain internal spaces (e.g., actor `"Sam Worthington"`, director `"Sam Raimi"`, genre `"Science Fiction"`).
* **Root Cause of Error:** When text vectorizers (e.g., `CountVectorizer`) tokenize text by whitespace, `"Sam Worthington"` becomes `['Sam', 'Worthington']` and `"Sam Raimi"` becomes `['Sam', 'Raimi']`.
* **Impact on Model:** The token `'Sam'` becomes a shared feature between an Action/Sci-Fi movie (*Avatar*) and a Superhero movie (*Spider-Man*). The model calculates artificially high cosine similarity between completely different movie genres based solely on a common first name.
* **Technical Solution:** Transformed all elements in `genres`, `keywords`, `cast`, and `crew` by stripping internal spaces, turning multi-word names into atomic, unique entity tokens.

```python
# Before Space Stripping:
# 'Sam Worthington' -> ['Sam', 'Worthington']
# 'Sam Raimi'       -> ['Sam', 'Raimi']

# Space Stripping Transformation:
movies['genres']   = movies['genres'].apply(lambda x: [i.replace(" ", "") for i in x])
movies['keywords'] = movies['keywords'].apply(lambda x: [i.replace(" ", "") for i in x])
movies['cast']     = movies['cast'].apply(lambda x: [i.replace(" ", "") for i in x])
movies['crew']     = movies['crew'].apply(lambda x: [i.replace(" ", "") for i in x])

# After Space Stripping:
# 'Sam Worthington' -> 'SamWorthington'
# 'Sam Raimi'       -> 'SamRaimi'
# 'Science Fiction' -> 'ScienceFiction'
```

---

### Challenge 5: Dataset Merging & Index Discrepancies
* **Context & Symptom:** Combining `tmdb_5000_movies.csv` (4,803 rows) and `tmdb_5000_credits.csv` (4,803 rows).
* **Merging Strategy Analysis:** Merging on `id` vs `title`:
  * `movies.csv` uses `id` as the TMDB movie identifier.
  * `credits.csv` uses `movie_id` as foreign key.
  * Merging on `title` resulted in 4,809 rows due to duplicate title entries across dataset versions.
* **Technical Solution:** Verified schema key alignment post-merge and dropped residual metadata duplicate rows, ensuring clean 1:1 row index mapping for vector similarity matrices.

```python
movies = movies.merge(credits, on='title')
```

---

### Challenge 6: Feature Selection Rationale: Content Signal vs. Numeric Scale Distortion
* **Context & Symptom:** Deciding whether to incorporate numerical metrics (`budget`, `revenue`, `popularity`, `vote_average`, `vote_count`, `runtime`).
* **Why Numeric Features Were Dropped:**
  1. **Production Budget / Revenue Distortion:** A \$200M Sci-Fi movie (*Interstellar*) and a \$200M Animated Comedy (*Toy Story 3*) have high budget similarity but zero plot/genre overlap.
  2. **Popularity/Rating Skew:** Content-based filtering seeks to match *thematic similarity*, not acclaim. Mixing popularity metrics with word count vectors skews cosine distance calculation away from topic matches.
* **Action Taken:** Dropped 16 non-content columns, retaining exclusively 7 core metadata columns (`movie_id`, `title`, `overview`, `genres`, `keywords`, `cast`, `crew`).

---

### Challenge 7: Low-Entropy Categorical Features & Class Imbalance
* **Context & Symptom:** Analyzing `original_language` distribution across the dataset:
  * Over **94% (4,510+ out of 4,806)** of movies had `original_language == 'en'`.
* **Impact on Model:** Low-entropy features provide virtually no discriminative power for vector distance metrics while increasing vector dimensionality.
* **Action Taken:** Dropped `original_language` from the feature set.

---

### Challenge 8: Missing Text Data in Core Plot Feature
* **Context & Symptom:** Checking null counts (`movies.isnull().sum()`) revealed 3 missing values in the `overview` column.
* **Impact:** `overview` serves as the core semantic backbone for plot keywords. Text vectorization functions throw exceptions when encountering `float` (`NaN`) in string pipelines.
* **Technical Solution:** Imputed vs Dropped: Since 3 records represent less than 0.06% of the dataset, imputing with empty strings would distort plot vectors. Dropped null rows (`movies.dropna(inplace=True)`), yielding **4,806 clean, valid rows**.

---

## 3. Summary Matrix of EDA Decisions

| Feature / Problem | Initial State | EDA Finding / Challenge | Resolution |
| :--- | :--- | :--- | :--- |
| **`genres` & `keywords`** | Stringified JSON | Cannot split directly; JSON errors on `'` quotes | Parsed with `ast.literal_eval()` |
| **`cast`** | Array of 30+ actors | Extra actors add vector noise | Extracted top 3 lead actors only |
| **`crew`** | 100+ crew roles | Technical staff diluted movie topic signal | Filtered exclusively for `job == 'Director'` |
| **Entity Names** | Multi-word strings | White-space tokenization caused name collisions | Stripped spaces (e.g. `Sam Worthington` $\rightarrow$ `SamWorthington`) |
| **Numerical Scale** | `budget`, `revenue`, `runtime` | Scale similarity $\neq$ thematic similarity | Dropped 16 numeric & non-content columns |
| **`original_language`** | Categorical | >94% English (near-zero entropy) | Dropped column to prevent dimensionality bloat |
| **Missing `overview`** | 3 `NaN` records | Vectorizer crashes on `float` type | Applied `dropna()` (4,806 remaining rows) |
