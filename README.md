# 🎬 CineRecommend - Personalised Movie Recommendation System

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://movierecommendarsystem-3wiq5yjnan2xqjszeiabof.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **🚀 Live Web App:** [https://movierecommendarsystem-3wiq5yjnan2xqjszeiabof.streamlit.app/](https://movierecommendarsystem-3wiq5yjnan2xqjszeiabof.streamlit.app/)

---

## 📌 Project Overview

**CineRecommend** is an end-to-end Content-Based Movie Recommendation System that recommends movies similar to a user's selection using Natural Language Processing (NLP) techniques and Vector Space Cosine Similarity.

The application processes plot keywords, cast, crew, metadata, and genres from the TMDB 5000 Movie Dataset to calculate pair-wise movie similarity scores. It features a reactive **Streamlit** front-end that dynamically fetches high-resolution movie posters via **The Movie Database (TMDB) API**.

---

## ✨ Key Features

- **Personalised Recommendations:** Select or search any movie from the dataset to discover top 5 similar recommendations.
- **Dynamic Poster Fetching:** Retrieves real-time poster artwork directly using the TMDB REST API.
- **Cloud-Optimized Artifact Management:** Large model similarity matrix (`similarity.pkl`) is auto-downloaded on cold start via `gdown` to keep repository footprints lightweight.
- **Responsive 5-Column Display:** Renders recommendations seamlessly across a modern grid layout.

---

## 🛠️ Tech Stack & Dependencies

- **Language:** Python
- **Frontend / Framework:** Streamlit
- **Machine Learning & NLP:** `scikit-learn` (`CountVectorizer`), `nltk` (`PorterStemmer`), `pandas`, `numpy`
- **Data Persistence:** `pickle`, `gdown`
- **API & HTTP:** `requests` (TMDB API v3)
- **Deployment Platform:** Streamlit Community Cloud

---

## 📐 System Architecture

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
                                                               v Uploaded to Cloud / Drive
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

## 🚀 Quickstart & Local Setup

### 1. Clone the Repository
```bash
git clone https://github.com/pareekutkarsh004/Movie_Recommendar_System.git
cd Movie_Recommendar_System
```

### 2. Set Up Virtual Environment & Install Dependencies
```bash
python -m venv .venv
source .venv/bin/activate   # On Windows use: .venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Run the Streamlit Application
```bash
streamlit run app.py
```
*Note: On initial startup, `app.py` will automatically download `movies.pkl` and `similarity.pkl` model weights if they are not already present in the directory.*

---

## 📚 Technical Documentation & Interview Guides

This repository includes comprehensive modular documentation covering the complete lifecycle:

- [📄 01: Project Overview & Exploratory Data Analysis](01_project_overview_and_eda.md)
- [📄 02: NLP Feature Engineering & Vectorization](02_nlp_vectorization_modeling.md)
- [📄 03: System Architecture & Deployment Engineering](03_system_architecture_deployment.md)
- [📄 04: Revision & Architectural Cheat Sheet](04_quick_revision_cheat_sheet.md)
- [📄 05: Master Interview Q&A Bank](05_interview_qna_masterbank.md)
- [📄 EDA & Dataset Challenges Breakdown](eda_challenges.md)

---

## 🌐 Live Deployment Link

- **Live Application:** [https://movierecommendarsystem-3wiq5yjnan2xqjszeiabof.streamlit.app/](https://movierecommendarsystem-3wiq5yjnan2xqjszeiabof.streamlit.app/)
- **GitHub Repository:** [https://github.com/pareekutkarsh004/Movie_Recommendar_System](https://github.com/pareekutkarsh004/Movie_Recommendar_System)
