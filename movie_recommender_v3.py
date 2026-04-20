# Generated from: movie_recommender_v3.ipynb
# Converted at: 2026-04-13T19:14:05.879Z
# Next step (optional): refactor into modules & generate tests with RunCell
# Quick start: pip install runcell

# # 🎬 Hybrid Movie Recommender System
# ### Using SBERT + TF-IDF + KMeans Clustering vs TMDB API Recommendations
# 
# **Subject:** Machine Learning  
# **Dataset:** TMDB Movies (Cleaned) — ~46,000 movies  
# **Approach:** Content-based filtering using semantic embeddings (SBERT) + TF-IDF, KMeans clustering for search optimization, and comparison against TMDB's built-in recommendation engine.
# 
# ---
# 
# ## 📋 Table of Contents
# 1. [Setup & Library Installation](#1-setup)
# 2. [Data Loading & Preprocessing](#2-data)
# 3. [Feature Engineering](#3-features)
#    - 3.1 SBERT Embeddings (title + overview)
#    - 3.2 TF-IDF Vectors (genres, keywords, cast, director)
# 4. [KMeans Clustering](#4-clustering)
# 5. [Hybrid Similarity & Recommendation Engine](#5-recommender)
# 6. [TMDB API Comparison](#6-tmdb-api)
# 7. [Evaluation & Results Analysis](#7-evaluation)
# 8. [Conclusion](#8-conclusion)


# ---
# ## 1. Setup & Library Installation <a id='1-setup'></a>


# Install required libraries (run once)
# !pip install sentence-transformers scikit-learn pandas numpy requests tqdm matplotlib seaborn plotly -q

import pandas as pd
import numpy as np
import ast
import os
import requests
import warnings
import time
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from tqdm.auto import tqdm

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import silhouette_score

from sentence_transformers import SentenceTransformer

warnings.filterwarnings('ignore')
tqdm.pandas()

# ─── CONFIGURATION ───────────────────────────────────────────────────────────
TMDB_API_KEY   = "d40c03000a8ca51da2011456878dbd25"   # <-- Replace with your key
TMDB_BASE_URL  = "https://api.themoviedb.org/3"
DATA_PATH      = "dataset/tmdb_movies_cleaned.csv"   # adjust path if needed
SBERT_MODEL    = "all-MiniLM-L6-v2"          # fast & accurate
N_CLUSTERS     = 40                           # tune as needed
EMBEDDING_FILE = "sbert_embeddings.npy"       # cache SBERT embeddings
TOP_N          = 10                           # recommendations to show

print("✅ Libraries loaded successfully")
print(f"   SBERT Model  : {SBERT_MODEL}")
print(f"   Clusters     : {N_CLUSTERS}")
print(f"   Top-N Recs   : {TOP_N}")
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import NearestNeighbors


# ---
# ## 2. Data Loading & Preprocessing <a id='2-data'></a>


df = pd.read_csv(DATA_PATH)
print(f"Dataset shape: {df.shape}")
print(f"Columns      : {list(df.columns)}")
df.head(3)

# ── Null analysis ──────────────────────────────────────────────────────────────
null_pct = (df.isnull().sum() / len(df) * 100).round(2)
print("Missing values (%):\n", null_pct[null_pct > 0])

def safe_parse_list(val):
    """Parse stringified Python lists safely."""
    if pd.isna(val) or val == '':
        return []
    try:
        parsed = ast.literal_eval(val)
        return parsed if isinstance(parsed, list) else []
    except:
        return []

# Parse list columns
for col in ['genres', 'cast', 'keywords']:
    df[col] = df[col].apply(safe_parse_list)

# Fill text nulls
df['overview']  = df['overview'].fillna('')
df['title']     = df['title'].fillna('Unknown')
df['director']  = df['director'].fillna('')

# Drop duplicates
df = df.drop_duplicates(subset='movie_id').reset_index(drop=True)

# Ensure numeric columns are valid
df['popularity']   = pd.to_numeric(df['popularity'],   errors='coerce').fillna(0)
df['vote_average'] = pd.to_numeric(df['vote_average'], errors='coerce').fillna(0)
df['vote_count']   = pd.to_numeric(df['vote_count'],   errors='coerce').fillna(0)
df['release_year'] = pd.to_numeric(df['release_year'], errors='coerce').fillna(0)

# Only keep movies with some overview text
df = df[df['overview'].str.len() > 20].reset_index(drop=True)

# Build a quick lookup by title (lower-case)
df['title_lower'] = df['title'].str.lower().str.strip()

print(f"Clean dataset size: {len(df):,} movies")
df[['movie_id','title','release_year','genres','vote_average','popularity']].head(5)

# ── EDA ────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 4))

axes[0].hist(df['vote_average'], bins=30, color='steelblue', edgecolor='white')
axes[0].set_title('Vote Average Distribution')
axes[0].set_xlabel('Vote Average')

axes[1].hist(np.log1p(df['popularity']), bins=30, color='coral', edgecolor='white')
axes[1].set_title('Log(Popularity) Distribution')
axes[1].set_xlabel('log(popularity)')

year_counts = df[df['release_year'] > 1950]['release_year'].value_counts().sort_index()
axes[2].plot(year_counts.index, year_counts.values, color='green')
axes[2].set_title('Movies per Year')
axes[2].set_xlabel('Year')

plt.tight_layout()
plt.savefig('eda_distributions.png', dpi=120, bbox_inches='tight')
plt.show()
print("EDA figure saved.")

# ---
# ## 3. Feature Engineering <a id='3-features'></a>
# 
# We build **two complementary feature spaces**:
# 
# | Feature space | Captures | Method |
# |---|---|---|
# | SBERT embeddings | Semantic meaning of title + overview | `sentence-transformers` |
# | TF-IDF vectors | Genres, keywords, cast, director | `sklearn TfidfVectorizer` |


# ### 3.1 SBERT Embeddings


# Combine title + overview into a rich text field for SBERT
df['sbert_text'] = df['title'] + ". " + df['overview']

if os.path.exists(EMBEDDING_FILE):
    print(f"Loading cached SBERT embeddings from '{EMBEDDING_FILE}'...")
    sbert_embeddings = np.load(EMBEDDING_FILE)
    print(f"Embeddings shape: {sbert_embeddings.shape}")
else:
    print(f"Encoding {len(df):,} movies with {SBERT_MODEL}...")
    sbert_model = SentenceTransformer(SBERT_MODEL)
    sbert_embeddings = sbert_model.encode(
        df['sbert_text'].tolist(),
        batch_size=256,
        show_progress_bar=True,
        normalize_embeddings=True   # unit vectors → cosine = dot product
    )
    np.save(EMBEDDING_FILE, sbert_embeddings)
    print(f"Saved to '{EMBEDDING_FILE}'. Shape: {sbert_embeddings.shape}")

# ### 3.1.5 Zero-Shot Semantic Probe Framework
# We inject explicit semantic axes covering Narrative, Sci-Fi, Tone, Structure, and Genre.


probes = {
    'philosophical_depth': 'A film exploring deep philosophical ideas such as existence, time, reality, consciousness, identity, morality, and the meaning of life.',
    'emotional_depth': 'A film focused on strong emotional experiences such as love, loss, sacrifice, relationships, and human connection.',
    'psychological_complexity': 'A film involving complex mental states, perception, dreams, memory, mind-bending narratives, or psychological tension.',
    'intellectual_complexity': 'A film requiring high cognitive engagement, non-linear storytelling, abstract ideas, or complex reasoning.',
    'space_exploration': 'A science fiction film involving space travel, astronauts, planets, galaxies, missions, and exploration beyond Earth.',
    'ai_and_consciousness': 'A film about artificial intelligence, robots, consciousness, human-machine relationships, or synthetic life.',
    'time_manipulation': 'A film involving time travel, relativity, time loops, temporal paradoxes, or nonlinear time.',
    'dystopian_future': 'A film set in a dystopian or futuristic society with societal collapse, control systems, or bleak futures.',
    'dark_serious_tone': 'A serious, grounded, and mature film with dark themes, realism, and minimal humor.',
    'light_comedic_tone': 'A light, humorous, or comedic film with fun, playful, or entertaining elements.',
    'action_intensity': 'A film focused on action, fast pacing, combat, explosions, and physical conflict.',
    'horror_intensity': 'A film involving fear, suspense, terror, supernatural elements, or disturbing content.',
    'non_linear_narrative': 'A film with non-linear storytelling, multiple timelines, fragmented narrative, or layered structure.',
    'mystery_investigation': 'A film centered around uncovering secrets, solving mysteries, or investigative storytelling.',
    'character_driven': 'A film focused primarily on character development, personal journeys, and internal conflict.',
    'spectacle_blockbuster': 'A large-scale film focused on visual spectacle, effects, and grand cinematic experience.',
    'superhero': 'A superhero film involving comic-book characters, powers, and heroic narratives.',
    'crime_realism': 'A grounded crime film involving law enforcement, criminals, moral ambiguity, and realism.',
    'adventure_family': 'A light adventure film suitable for general audiences, often with family-friendly themes and simple storytelling.'
}

def build_probe_prompt(text):
    return f'This description represents a specific movie characteristic:\n\n{text}\n\nThe embedding should capture the semantic meaning of this characteristic.'

print('Encoding Semantic Probes (19 axes)...')
if 'sbert_model' not in locals():
    sbert_model = SentenceTransformer(SBERT_MODEL)

probe_embeddings = {}
for name, text in probes.items():
    probe_embeddings[name] = sbert_model.encode(build_probe_prompt(text), normalize_embeddings=True)
    
print('Computing Matrix Profile for 46,000 movies...')
for name, vec in probe_embeddings.items():
    df[name] = np.dot(sbert_embeddings, vec)

print(f"Database now profiled across {len(probes)} semantic dimensions!")


# ### 3.2 TF-IDF Vectors


def build_metadata_soup(row):
    """Combine genres, keywords, top-3 cast, and director into a single string."""
    genres   = ' '.join(row['genres'][:5])          if row['genres']   else ''
    keywords = ' '.join(row['keywords'][:8])         if row['keywords'] else ''
    cast     = ' '.join(row['cast'][:3])             if row['cast']     else ''
    director = row['director'].replace(' ', '_')     if row['director'] else ''
    # Repeat director & genres to up-weight them
    return f"{genres} {genres} {director} {director} {cast} {keywords}"

df['metadata_soup'] = df.apply(build_metadata_soup, axis=1)

print("Sample metadata soup:")
print(df['metadata_soup'].iloc[0])

tfidf_vectorizer = TfidfVectorizer(
    max_features=10_000,
    ngram_range=(1, 2),
    min_df=2,
    sublinear_tf=True
)
tfidf_matrix = tfidf_vectorizer.fit_transform(df['metadata_soup'])

print(f"TF-IDF matrix: {tfidf_matrix.shape}")
print(f"Vocabulary size: {len(tfidf_vectorizer.vocabulary_):,}")

print(f"Applying TruncatedSVD to compress TF-IDF dimensions...")
svd = TruncatedSVD(n_components=300, random_state=42)
tfidf_reduced = svd.fit_transform(tfidf_matrix)

print(f"Reduced TF-IDF matrix: {tfidf_reduced.shape}")

# Plot Explained Variance
plt.figure(figsize=(8, 4))
plt.plot(np.cumsum(svd.explained_variance_ratio_), color='coral', linewidth=2)
plt.title("Cumulative Explained Variance of TF-IDF Components")
plt.xlabel("Number of Components")
plt.ylabel("Explained Variance")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('tfidf_pca_variance.png', dpi=120)
plt.show()


# ---
# ## 4. KMeans Clustering <a id='4-clustering'></a>
# 
# Clustering the **SBERT embedding space** divides the 46k movies into thematic groups.  
# At recommendation time we **search only within the query movie's cluster**, reducing complexity from O(N) → O(N/K).


print("Running Elbow Method for KMeans (k=10 to 60)...")
k_values = [10, 20, 30, 40, 50, 60]
inertias = []
for k in k_values:
    km = MiniBatchKMeans(n_clusters=k, random_state=42, batch_size=2048, n_init=3)
    km.fit(sbert_embeddings)
    inertias.append(km.inertia_)
    
plt.figure(figsize=(8, 4))
plt.plot(k_values, inertias, marker='o', color='steelblue', linewidth=2)
plt.title("Elbow Method For Optimal k")
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Inertia")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('kmeans_elbow_plot.png', dpi=120)
plt.show()

print(f"\nFitting final KMeans with {N_CLUSTERS} clusters on SBERT embeddings...")
kmeans = MiniBatchKMeans(
    n_clusters=N_CLUSTERS,
    random_state=42,
    batch_size=2048,
    n_init=10
)
df['cluster'] = kmeans.fit_predict(sbert_embeddings)

cluster_sizes = df['cluster'].value_counts().sort_index()
print(f"Cluster sizes — min: {cluster_sizes.min()}, max: {cluster_sizes.max()}, mean: {cluster_sizes.mean():.0f}")


# ── Silhouette score on a sample (full dataset would be slow) ─────────────────
sample_idx = np.random.choice(len(df), min(5000, len(df)), replace=False)
sil_score  = silhouette_score(sbert_embeddings[sample_idx], df['cluster'].iloc[sample_idx])
print(f"Silhouette Score (sample of 5000): {sil_score:.4f}  (higher is better, max=1)")

# ── 2D PCA visualization of clusters ─────────────────────────────────────────
pca = PCA(n_components=2, random_state=42)
emb_2d = pca.fit_transform(sbert_embeddings[sample_idx])

plt.figure(figsize=(12, 8))
scatter = plt.scatter(
    emb_2d[:, 0], emb_2d[:, 1],
    c=df['cluster'].iloc[sample_idx], cmap='tab20',
    alpha=0.4, s=8
)
plt.colorbar(scatter, label='Cluster')
plt.title(f'SBERT Embeddings — PCA 2D ({N_CLUSTERS} KMeans Clusters, n=5000 sample)')
plt.xlabel('PC1'); plt.ylabel('PC2')
plt.tight_layout()
plt.savefig('cluster_pca.png', dpi=120, bbox_inches='tight')
plt.show()

# ── Show dominant genres per cluster ─────────────────────────────────────────
from collections import Counter

cluster_genre_summary = {}
for c in range(N_CLUSTERS):
    genres_in_cluster = [g for genres in df[df['cluster'] == c]['genres'] for g in genres]
    top_genres = [g for g, _ in Counter(genres_in_cluster).most_common(3)]
    cluster_genre_summary[c] = top_genres

print("Top-3 genres per cluster (first 10 clusters):")
for c, genres in list(cluster_genre_summary.items())[:10]:
    size = (df['cluster'] == c).sum()
    print(f"  Cluster {c:2d} ({size:4d} movies): {genres}")

# ---
# ## 5. Hybrid Similarity & Recommendation Engine <a id='5-recommender'></a>
# 
# **Final Score formula:**
# 
# $$\text{score}(q, m) = \alpha \cdot \text{cos}_{\text{SBERT}}(q, m) + \beta \cdot \text{cos}_{\text{TF-IDF}}(q, m) + \gamma \cdot \text{norm\_popularity}(m)$$
# 
# Default weights: α = 0.60, β = 0.30, γ = 0.10


def get_movie_index(title_query: str):
    title_q = title_query.lower().strip()
    exact = df[df['title_lower'] == title_q]
    if not exact.empty:
        return exact.index[0]
    partial = df[df['title_lower'].str.contains(title_q, na=False)]
    if not partial.empty:
        return partial.index[0]
    return None


def recommend_movies(
    title: str,
    top_n: int = TOP_N,
    hybrid_weights: tuple = (0.65, 0.20, 0.10, 0.05)
):
    title_lower = title.strip().lower()
    match = df[df['title_lower'].str.contains(title_lower, na=False)]
    
    if match.empty:
        return f"Movie '{title}' not found in database."
        
    idx = match.index[0]
    seed_movie = df.iloc[idx]
    
    cluster_label = seed_movie['cluster']
    cluster_idx = df[df['cluster'] == cluster_label].index
    
    if len(cluster_idx) < top_n + 1:
        cluster_idx = df.index
        
    w_sbert, w_tfidf, w_vote, w_pop = hybrid_weights
    
    q_sbert = sbert_embeddings[idx].reshape(1, -1)
    c_sbert = sbert_embeddings[cluster_idx]
    sbert_sims = cosine_similarity(q_sbert, c_sbert).flatten()
    
    q_tfidf = tfidf_reduced[idx].reshape(1, -1)
    c_tfidf = tfidf_reduced[cluster_idx]
    tfidf_sims = cosine_similarity(q_tfidf, c_tfidf).flatten()
    
    candidates = df.iloc[cluster_idx].copy()
    
    scaler = MinMaxScaler()
    candidates['pop_norm'] = scaler.fit_transform(candidates[['popularity_log']])
    candidates['vote_norm'] = scaler.fit_transform(candidates[['vote_average_norm']])
    
    candidates['base_hybrid_score'] = (
        (w_sbert * sbert_sims) + 
        (w_tfidf * tfidf_sims) + 
        (w_vote * candidates['vote_norm']) + 
        (w_pop * candidates['pop_norm'])
    )
    
    # --- SEMANTIC PROBE FRAMEWORK ---
    probe_keys = list(probes.keys())
    seed_profile = {name: seed_movie[name] for name in probe_keys}
    
    # Normalize Weights (Cap sub-zero matches to avoid inverting axes)
    total_val = sum(max(v, 0.001) for v in seed_profile.values())
    weights = {k: max(v, 0.001) / total_val for k, v in seed_profile.items()}
    
    candidates['dynamic_probe_score'] = 0.0
    for k in probe_keys:
        candidates['dynamic_probe_score'] += weights[k] * candidates[k]
        
    candidates['final_score'] = candidates['base_hybrid_score'] + candidates['dynamic_probe_score']
    
    # Exact Output Penalties
    if weights['light_comedic_tone'] < 0.05:
        candidates['final_score'] -= 0.3 * candidates['light_comedic_tone']
    if weights['horror_intensity'] < 0.05:
        candidates['final_score'] -= 0.2 * candidates['horror_intensity']
        
    # QUALITY FLOOR 
    candidates = candidates[
        (candidates['vote_count'] >= 100) &
        (candidates['vote_average'] >= 6.0)
    ]
        
    candidates = candidates.drop(idx, errors='ignore')
    top_recs = candidates.sort_values(by='final_score', ascending=False).head(top_n)
    
    print("="*80)
    print(f"\033[1mComparing recommendations for: {seed_movie['title']} ({seed_movie['release_year']})\033[0m")
    if pd.notna(seed_movie.get('movie_id')):
        print(f"  TMDB ID: {int(seed_movie['movie_id'])}")
    print("="*80 + "\n")
    
    print(f"🎬 \033[1mQuery:\033[0m '{seed_movie['title']}' ({seed_movie['release_year']})")
    print(f"   Genres : {seed_movie['genres']}")
    print(f"   Cluster: {cluster_label}")
    
    # Top 3 weighted dimensions
    top_weights = sorted(weights.items(), key=lambda x: x[1], reverse=True)[:4]
    top_w_str = ", ".join([f"{k} ({v:.2f})" for k, v in top_weights])
    print(f"   Top Profile Axes: {top_w_str}")
    
    print("\n\033[1m🏆 Semantic Probed Recommendations:\033[0m\n")
    
    for i, (_, row) in enumerate(top_recs.iterrows(), 1):
        score_val = row['final_score']
        print(f"{i:2d}. {row['title']} ({row['release_year']})")
        print(f"    ⭐ {row['vote_average']} | 🎯 Score: {score_val:.4f} | 🎭 {row['genres']}")
        
    top_recs.insert(0, 'rank', range(1, len(top_recs) + 1))
    return top_recs

# ── Demo 1: Action movie ──────────────────────────────────────────────────────
recs_1 = recommend_movies("Inception", top_n=10)
display(recs_1[['rank','title','release_year','genres','vote_average','final_score']].style
        .background_gradient(subset='final_score', cmap='Greens')
        .format({'final_score': '{:.4f}', 'vote_average': '{:.1f}'}))

# ── Demo 2: With genre filter ─────────────────────────────────────────────────
recs_2 = recommend_movies("The Dark Knight", top_n=10)
display(recs_2[['rank','title','release_year','genres','vote_average','final_score']].style
        .background_gradient(subset='final_score', cmap='Blues')
        .format({'final_score': '{:.4f}', 'vote_average': '{:.1f}'}))

# ── Demo 3: Year-filtered ─────────────────────────────────────────────────────
recs_3 = recommend_movies("Interstellar", top_n=10)
display(recs_3[['rank','title','release_year','genres','vote_average','final_score']].style
        .background_gradient(subset='final_score', cmap='Oranges')
        .format({'final_score': '{:.4f}', 'vote_average': '{:.1f}'}))

# ---
# ## 6. TMDB API Comparison <a id='6-tmdb-api'></a>
# 
# We call TMDB's `/movie/{id}/recommendations` endpoint and compare the results side-by-side with our hybrid recommender.


def tmdb_get_recommendations(movie_id: int, top_n: int = TOP_N) -> pd.DataFrame:
    """
    Fetch TMDB's official recommendations for a given movie_id.
    Returns a DataFrame of top-N recommended movies.
    """
    url = f"{TMDB_BASE_URL}/movie/{movie_id}/recommendations"
    params = {"api_key": TMDB_API_KEY, "language": "en-US", "page": 1}
    
    # 🌟 Added User-Agent header to prevent API from dropping the connection
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }

    try:
        # 🌟 Passed headers into the request here
        resp = requests.get(url, params=params, headers=headers, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        results = data.get('results', [])[:top_n]
        if not results:
            print("   TMDB returned 0 recommendations for this movie.")
            return pd.DataFrame()

        records = []
        for i, m in enumerate(results, 1):
            records.append({
                'rank'         : i,
                'movie_id'     : m.get('id'),
                'title'        : m.get('title', 'N/A'),
                'release_year' : m.get('release_date', '0000')[:4],
                'vote_average' : m.get('vote_average', 0),
                'popularity'   : m.get('popularity', 0),
                'overview'     : m.get('overview', '')[:120] + '...'
            })
        return pd.DataFrame(records)

    except requests.exceptions.HTTPError as e:
        print(f"   TMDB API error: {e}")
        return pd.DataFrame()
    except Exception as e:
        print(f"   Request error: {e}")
        return pd.DataFrame()


def compare_recommendations(title: str, top_n: int = TOP_N, **rec_kwargs):
    """
    Side-by-side comparison of our hybrid recommender vs TMDB's API.
    """
    idx = get_movie_index(title)
    if idx is None:
        print(f"Movie '{title}' not found.")
        return

    movie_id   = int(df.loc[idx, 'movie_id'])
    movie_year = int(df.loc[idx, 'release_year'])

    print(f"\n{'='*65}")
    print(f"  Comparing recommendations for: {df.loc[idx,'title']} ({movie_year})")
    print(f"  TMDB ID: {movie_id}")
    print(f"{'='*65}")

    # Our recommendations
    our_recs  = recommend_movies(title, top_n=top_n, **rec_kwargs)

    # TMDB recommendations
    print("\nFetching TMDB API recommendations...")
    tmdb_recs = tmdb_get_recommendations(movie_id, top_n=top_n)

    # ── Print side by side ────────────────────────────────────────────────────
    print(f"\n{'─'*30}  OUR HYBRID  {'─'*30}")
    if not our_recs.empty:
        display(our_recs[['rank','title','release_year','vote_average','final_score']]
                .style.background_gradient(subset='final_score', cmap='Greens')
                .format({'final_score': '{:.4f}', 'vote_average': '{:.1f}'}))

    print(f"\n{'─'*30}  TMDB API    {'─'*30}")
    if not tmdb_recs.empty:
        display(tmdb_recs[['rank','title','release_year','vote_average','popularity']]
                .style.background_gradient(subset='popularity', cmap='Blues')
                .format({'vote_average': '{:.1f}', 'popularity': '{:.1f}'}))

    # ── Overlap analysis ──────────────────────────────────────────────────────
    if not our_recs.empty and not tmdb_recs.empty:
        our_titles  = set(our_recs['title'].str.lower())
        tmdb_titles = set(tmdb_recs['title'].str.lower())
        overlap = our_titles & tmdb_titles
        print(f"\n📊 Overlap: {len(overlap)}/{top_n} movies appear in both lists")
        if overlap:
            print(f"   Common: {', '.join(overlap)}")
        return our_recs, tmdb_recs

    return our_recs, tmdb_recs


print("✅ TMDB comparison functions defined.")


# ── Comparison 1: Inception ───────────────────────────────────────────────────
# NOTE: Replace TMDB_API_KEY above with your actual key from https://www.themoviedb.org/settings/api
our_1, tmdb_1 = compare_recommendations("Inception", top_n=10)

# ── Comparison 2: The Dark Knight ────────────────────────────────────────────
our_2, tmdb_2 = compare_recommendations("The Dark Knight", top_n=10)

# ── Comparison 3: Interstellar ───────────────────────────────────────────────
our_3, tmdb_3 = compare_recommendations("Interstellar", top_n=10)

# ── Popularity comparison chart ───────────────────────────────────────────────
def plot_popularity_comparison(our_recs, tmdb_recs, query_title):
    if our_recs is None or tmdb_recs is None:
        return
    if our_recs.empty or tmdb_recs.empty:
        return

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    # Vote averages
    axes[0].barh(our_recs['title'].str[:30], our_recs['vote_average'],
                 color='steelblue', label='Our Recs')
    axes[0].barh(tmdb_recs['title'].str[:30], tmdb_recs['vote_average'],
                 color='coral', alpha=0.7, label='TMDB')
    axes[0].set_title(f'Vote Averages — "{query_title}"')
    axes[0].set_xlabel('Vote Average')
    axes[0].legend()

    # Popularity
    axes[1].barh(our_recs['title'].str[:30],
                 np.log1p(our_recs['popularity']),
                 color='seagreen', label='Our Recs')
    if 'popularity' in tmdb_recs.columns:
        axes[1].barh(tmdb_recs['title'].str[:30],
                     np.log1p(tmdb_recs['popularity']),
                     color='gold', alpha=0.7, label='TMDB')
    axes[1].set_title(f'Log Popularity — "{query_title}"')
    axes[1].set_xlabel('log(popularity)')
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(f'comparison_{query_title[:15].replace(" ","_")}.png', dpi=120, bbox_inches='tight')
    plt.show()

plot_popularity_comparison(our_1, tmdb_1, "Inception")

# ---
# ## 7. Evaluation & Results Analysis <a id='7-evaluation'></a>


# ── Weight sensitivity analysis ───────────────────────────────────────────────
# Test different alpha/beta/gamma combinations for 'Inception'
weight_configs = [
    (0.70, 0.20, 0.10, 'SBERT-heavy'),
    (0.60, 0.30, 0.10, 'Balanced (default)'),
    (0.40, 0.50, 0.10, 'TF-IDF-heavy'),
    (0.50, 0.30, 0.20, 'Popularity-boost'),
]

print("Weight Sensitivity Analysis — 'Inception'\n")
print(f"{'Config':<25} | {'Top-1':<35} | {'Top-2':<35} | {'Top-3':<35}")
print('-' * 130)

for alpha, beta, gamma, label in weight_configs:
    idx_q = get_movie_index('Inception')
    if idx_q is None: continue
    r = recommend_movies('Inception', top_n=3)
    if not r.empty:
        titles = r['title'].tolist()
        while len(titles) < 3: titles.append('N/A')
        print(f"{label:<25} | {titles[0]:<35} | {titles[1]:<35} | {titles[2]:<35}")

# ── Genre coverage analysis ───────────────────────────────────────────────────
test_movies = ['Inception', 'The Dark Knight', 'Interstellar',
               'The Godfather', 'Toy Story']

print("Genre coverage in recommendations:\n")
for tm in test_movies:
    idx_q = get_movie_index(tm)
    if idx_q is None: continue
    query_genres = set(df.loc[idx_q, 'genres'])
    recs = recommend_movies(tm, top_n=10)
    if recs.empty: continue
    rec_genres_all = [g for genres_list in recs['genres'] for g in genres_list]
    genre_overlap = sum(1 for g in rec_genres_all if g in query_genres)
    coverage = genre_overlap / max(len(rec_genres_all), 1) * 100
    print(f"  {tm:<22}: query genres={list(query_genres)}, genre overlap={coverage:.1f}%")

# ── Cluster efficiency gain ───────────────────────────────────────────────────
cluster_sizes = df['cluster'].value_counts()
avg_cluster_size = cluster_sizes.mean()
total_movies = len(df)

print("Clustering Efficiency:")
print(f"  Total movies         : {total_movies:,}")
print(f"  Avg cluster size     : {avg_cluster_size:.0f}")
print(f"  Search space reduced by: {(1 - avg_cluster_size/total_movies)*100:.1f}%")

# Bar chart of cluster sizes
plt.figure(figsize=(14, 4))
plt.bar(range(N_CLUSTERS), cluster_sizes.sort_index(), color='steelblue')
plt.axhline(avg_cluster_size, color='red', linestyle='--', label=f'Mean = {avg_cluster_size:.0f}')
plt.title('KMeans Cluster Sizes')
plt.xlabel('Cluster ID'); plt.ylabel('Movie Count')
plt.legend()
plt.tight_layout()
plt.savefig('cluster_sizes.png', dpi=120, bbox_inches='tight')
plt.show()

# ---
# ## 8. Conclusion <a id='8-conclusion'></a>
# 
# ### Summary of Findings
# 
# | Component | Result |
# |---|---|
# | Dataset | ~46,000 TMDB movies after cleaning |
# | SBERT model | `all-MiniLM-L6-v2` — 384-dim embeddings |
# | TF-IDF vocab | ~10,000 features from genres, cast, keywords, director |
# | KMeans clusters | 40 clusters, silhouette score reported above |
# | Cluster speedup | ~97% reduction in search space per query |
# | Hybrid formula | 60% SBERT + 30% TF-IDF + 10% popularity |
# 
# ### Achievements
# - ✅ **SBERT** captures rich semantic similarity beyond keyword matching
# - ✅ **TF-IDF** enforces genre/cast/keyword alignment between query and results  
# - ✅ **KMeans clustering** on SBERT space groups thematically similar movies, enabling fast intra-cluster search  
# - ✅ **TMDB API comparison** provides a qualitative benchmark against a production recommendation engine  
# - ✅ **Flexible filters** (year range, genre) allow user-driven refinement  
# 
# ### Limitations
# - Content-based only — no collaborative filtering (user history not available)
# - Cold-start: new movies with short overviews get weaker SBERT embeddings  
# - Cluster boundary effects: a movie near a cluster edge may miss some relevant films  
# 
# ### Possible Improvements
# - Add collaborative filtering layer (ALS / NCF) when user interaction data is available  
# - Expand to multilingual SBERT for non-English films  
# - Use HNSW approximate nearest neighbour (FAISS) for even faster retrieval  
# - Cross-cluster fallback when intra-cluster results are sparse  


# ── Interactive search ─────────────────────────────────────────────────────────
# Uncomment and run interactively in Jupyter

# query = input("Enter a movie title: ")
# genre = input("Genre filter (leave blank for none): ").strip() or None
# our_r, tmdb_r = compare_recommendations(query, top_n=10, genre_filter=genre)

print("\n✅ Notebook execution complete.")
print("To use the recommender: call recommend_movies('Your Movie Title')")
print("To compare with TMDB:   call compare_recommendations('Your Movie Title')")