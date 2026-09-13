# Module 1: Project Overview, Business Problem & Exploratory Data Analysis (EDA)

## 1. Project Objective & Problem Statement
### Objective
**CineRecommend** is a personalized **Content-Based Movie Recommendation System** designed to suggest top-5 similar movies based on textual and metadata similarity (plot summary, genres, key plot tags, top actors, and directors).

### Problem Solved
In real-world recommendation applications, platforms face two major recommendation paradigms:
1. **Collaborative Filtering**: Relies on user interaction matrices (ratings, clicks, watch history, purchase logs).
2. **Content-Based Filtering**: Relies purely on item attributes and features.

#### Why Content-Based Filtering for CineRecommend?
- **User Cold Start Problem**: New users have zero interaction logs/ratings. Collaborative filtering completely fails for new users. Content-based filtering requires zero user history—it recommends items similar to a selected item.
- **Data Availability**: The TMDB 5000 dataset contains rich item metadata (plot summaries, genres, cast, crew, keywords) but no user interaction history. Content-based filtering is mathematically and architecturally the optimal choice.
- **Interpretability**: Recommendations can be easily explained to the user (e.g., *"Because you watched Avatar, you might like movies with similar sci-fi themes, directed by James Cameron, or featuring Sam Worthington"*).

---

## 2. Dataset Structure & Schema Breakdown

The project uses two primary datasets sourced from The Movie Database (TMDB):
- [`tmdb_5000_movies.csv`](file:///Users/utarshpareek/MovieRecommendarSystem/tmdb_5000_movies.csv) (4,803 rows × 20 columns)
- [`tmdb_5000_credits.csv`](file:///Users/utarshpareek/MovieRecommendarSystem/tmdb_5000_credits.csv) (4,803 rows × 4 columns)

### Primary Datasets & Columns

| Dataset | Column Name | Data Type | Description |
| :--- | :--- | :--- | :--- |
| **movies** | `budget` | Int64 | Production budget in USD |
| **movies** | `genres` | String (JSON) | List of genre dicts (`[{"id": 28, "name": "Action"}, ...]`) |
| **movies** | `homepage` | String | Official movie website URL |
| **movies** | `id` | Int64 | TMDB Movie ID (unique key) |
| **movies** | `keywords` | String (JSON) | Plot keywords (`[{"id": 1463, "name": "culture clash"}, ...]`) |
| **movies** | `original_language` | String | ISO 639-1 language code (e.g., `'en'`, `'fr'`) |
| **movies** | `original_title` | String | Movie title in original language |
| **movies** | `overview` | String | Full text plot synopsis |
| **movies** | `popularity` | Float64 | TMDB popularity metric score |
| **movies** | `production_companies` | String (JSON) | Production companies info |
| **movies** | `production_countries` | String (JSON) | Production countries info |
| **movies** | `release_date` | String | Release date (`YYYY-MM-DD`) |
| **movies** | `revenue` | Int64 | Box office revenue in USD |
| **movies** | `runtime` | Float64 | Movie duration in minutes |
| **movies** | `spoken_languages` | String (JSON) | Languages spoken in the movie |
| **movies** | `status` | String | Release status (e.g., `'Released'`) |
| **movies** | `tagline` | String | Catchphrase or tagline |
| **movies** | `title` | String | Standardized English movie title |
| **movies** | `vote_average` | Float64 | Average rating (0 to 10) |
| **movies** | `vote_count` | Int64 | Total vote count |
| **credits** | `movie_id` | Int64 | Foreign key mapping to `movies.id` |
| **credits** | `title` | String | Movie title |
| **credits** | `cast` | String (JSON) | Full cast array (`[{"cast_id":..., "character":..., "name":...}]`) |
| **credits** | `crew` | String (JSON) | Full crew array including Director, Writer, Producer |

---

## 3. Dataset Merging & Feature Selection Rationale

### Dataset Merging
The two datasets are merged on the common key `title`:
```python
movies = movies.merge(credits, on='title')
```
*Note*: Merging on `title` produces 4,809 rows (handling minor duplicate titles across credits/movies).

### Retained vs. Rejected Features (Interview Justification Matrix)

During data science interviews, interviewers frequently ask: *"Why did you keep feature X and drop feature Y?"* Here is the explicit engineering rationale:

```
Merged Dataset (23 Columns)
 ├── Retained (7 Columns): [movie_id, title, overview, genres, keywords, cast, crew]
 └── Dropped (16 Columns): [budget, revenue, popularity, vote_average, vote_count, 
                           original_language, homepage, release_date, runtime, 
                           spoken_languages, production_companies, production_countries, 
                           status, tagline, original_title, id]
```

#### Detailed Feature Decision Table

| Feature Name | Action | Detailed Technical Rationale for Interview |
| :--- | :--- | :--- |
| `movie_id` | **Retained** | Required for fetching movie poster URLs via TMDB API endpoint in the Streamlit UI. |
| `title` | **Retained** | Serves as the primary user search identifier and key for similarity lookup. |
| `overview` | **Retained** | Core text feature. Contains plot summary. Essential for capturing semantic theme. |
| `genres` | **Retained** | Highest impact content descriptor (e.g., Sci-Fi, Action, Drama). |
| `keywords` | **Retained** | Fine-grained thematic tags (e.g., "space opera", "time travel", "dystopia"). |
| `cast` | **Retained** | Top 3 lead actors heavily influence user preference and movie similarity. |
| `crew` | **Retained** | Extracted Director name. Directors have distinct artistic styles and genre signatures. |
| `budget` & `revenue` | **Dropped** | A \$200M sci-fi movie and a \$200M animated comedy have zero thematic similarity. Budget reflects production scale, not content similarity. |
| `original_language` | **Dropped** | Extreme class imbalance: 94%+ (4,510+ out of 4,806) of movies are in English (`'en'`). High feature entropy imbalance makes it uninformative for vector similarity. |
| `popularity` & `vote_average` | **Dropped** | Popularity and rating measure quality/acclaim, not content similarity. Mixing quality metrics with content vectors distorts pure genre/theme matching. |
| `production_companies` | **Dropped** | Warner Bros or Universal produces movies across all genres; introduces heavy noise without strong thematic signal. |
| `release_date` & `runtime` | **Dropped** | Movie runtime does not dictate plot/genre similarity. Release year could add era context, but text keywords already capture retro/futuristic themes more effectively. |

---

## 4. Exploratory Data Analysis & Data Preprocessing Pipeline

### Missing Value Analysis & Mitigation
Checking missing values across the 7 selected features:
```python
movies.isnull().sum()
```
- `overview`: 3 missing values.
- Remedy: Dropped missing rows using `movies.dropna(inplace=True)`. Remaining dataset size: 4,806 clean rows.

### Extracting Structured Information from JSON Strings

Columns like `genres`, `keywords`, `cast`, and `crew` are stored as JSON-formatted string literals (e.g., `'[{"id": 28, "name": "Action"}]'`). Standard Python string operations cannot parse them directly.

#### 1. Parsing Genres & Keywords
Using Python's `ast.literal_eval` (Abstract Syntax Tree) to safely evaluate string representations of Python structures:
```python
import ast

def convert(obj):
    L = []
    for i in ast.literal_eval(obj):
        L.append(i['name'])
    return L

movies['genres'] = movies['genres'].apply(convert)
movies['keywords'] = movies['keywords'].apply(convert)
```

#### 2. Extracting Top 3 Cast Members
To focus on lead actors without diluting the tag vector with minor background extra actors:
```python
def convert3(obj):
    counter = 0
    L = []
    for i in ast.literal_eval(obj):
        if counter != 3:
            L.append(i['name'])
            counter += 1
        else:
            break
    return L

movies['cast'] = movies['cast'].apply(convert3)
```

#### 3. Extracting Director from Crew
Crew contains hundreds of roles (Cinematographer, Sound Editor, Costume Design, etc.). We filter exclusively for `job == 'Director'`:
```python
def fetchDirector(obj):
    L = []
    for i in ast.literal_eval(obj):
        if i['job'] == 'Director':
            L.append(i['name'])
            break
    return L

movies['crew'] = movies['crew'].apply(fetchDirector)
```

#### 4. Splitting Overview into Token Lists
`overview` is originally a single paragraph string. We split it into a list of word tokens:
```python
movies['overview'] = movies['overview'].apply(lambda x: x.split())
```

---

## 5. Summary Flow of EDA to Feature Extraction

```
[Raw CSVs] ──> Merge on 'title' ──> Drop 16 Non-Content Columns ──> Drop Null Overviews
   │
   ├── ast.literal_eval() ──> Extract Genre Names
   ├── ast.literal_eval() ──> Extract Keyword Names
   ├── ast.literal_eval() ──> Extract Top 3 Cast Names
   ├── ast.literal_eval() ──> Extract Director Name
   └── lambda x.split()   ──> Convert Overview Paragraph to Token List
```

This completes the data cleaning and extraction phase, leaving us with clean tokenized lists for every movie ready for entity collapsing and tag concatenation.
