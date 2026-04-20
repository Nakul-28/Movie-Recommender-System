"""
main.py — FastAPI REST API for the Hybrid Movie Recommendation System
=====================================================================
Endpoints:
  GET  /health                       — liveness + readiness check
  GET  /search?query=...             — fuzzy movie title search
  GET  /recommendations?title=...    — hybrid SBERT + TF-IDF recommendations
  GET  /compare?title=...            — side-by-side vs TMDB API
  POST /recommendations/semantic     — probe-weight-driven vibe search
  GET  /movie/{movie_id}             — single movie metadata lookup
  GET  /clusters                     — cluster overview (sizes + top genres)

Run:
  uvicorn main:app --reload
Docs:
  http://localhost:8000/docs
"""

# ---------------------------------------------------------------------------
# 0. Stdlib / third-party imports (nothing from the notebook yet)
# ---------------------------------------------------------------------------
import os
import ast
import time
import warnings
from collections import Counter
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests as req_lib

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
from sentence_transformers import SentenceTransformer

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# 1. Configuration  (mirrors movie_recommender_v3.py)
# ---------------------------------------------------------------------------
TMDB_API_KEY   = os.getenv("TMDB_API_KEY", "d40c03000a8ca51da2011456878dbd25")
TMDB_BASE_URL  = "https://api.themoviedb.org/3"
DATA_PATH      = os.getenv("DATA_PATH", "dataset/tmdb_movies_cleaned.csv")
SBERT_MODEL    = "all-MiniLM-L6-v2"
N_CLUSTERS     = 40
EMBEDDING_FILE = "sbert_embeddings.npy"
TOP_N_DEFAULT  = 10

# ---------------------------------------------------------------------------
# 2. Semantic Probe definitions (identical to the notebook)
# ---------------------------------------------------------------------------
PROBES: Dict[str, str] = {
    "philosophical_depth":     "A film exploring deep philosophical ideas such as existence, time, reality, consciousness, identity, morality, and the meaning of life.",
    "emotional_depth":         "A film focused on strong emotional experiences such as love, loss, sacrifice, relationships, and human connection.",
    "psychological_complexity":"A film involving complex mental states, perception, dreams, memory, mind-bending narratives, or psychological tension.",
    "intellectual_complexity": "A film requiring high cognitive engagement, non-linear storytelling, abstract ideas, or complex reasoning.",
    "space_exploration":       "A science fiction film involving space travel, astronauts, planets, galaxies, missions, and exploration beyond Earth.",
    "ai_and_consciousness":    "A film about artificial intelligence, robots, consciousness, human-machine relationships, or synthetic life.",
    "time_manipulation":       "A film involving time travel, relativity, time loops, temporal paradoxes, or nonlinear time.",
    "dystopian_future":        "A film set in a dystopian or futuristic society with societal collapse, control systems, or bleak futures.",
    "dark_serious_tone":       "A serious, grounded, and mature film with dark themes, realism, and minimal humor.",
    "light_comedic_tone":      "A light, humorous, or comedic film with fun, playful, or entertaining elements.",
    "action_intensity":        "A film focused on action, fast pacing, combat, explosions, and physical conflict.",
    "horror_intensity":        "A film involving fear, suspense, terror, supernatural elements, or disturbing content.",
    "non_linear_narrative":    "A film with non-linear storytelling, multiple timelines, fragmented narrative, or layered structure.",
    "mystery_investigation":   "A film centered around uncovering secrets, solving mysteries, or investigative storytelling.",
    "character_driven":        "A film focused primarily on character development, personal journeys, and internal conflict.",
    "spectacle_blockbuster":   "A large-scale film focused on visual spectacle, effects, and grand cinematic experience.",
    "superhero":               "A superhero film involving comic-book characters, powers, and heroic narratives.",
    "crime_realism":           "A grounded crime film involving law enforcement, criminals, moral ambiguity, and realism.",
    "adventure_family":        "A light adventure film suitable for general audiences, often with family-friendly themes and simple storytelling.",
}

# ---------------------------------------------------------------------------
# 3. Engine state (populated during startup)
# ---------------------------------------------------------------------------
_engine: Dict[str, Any] = {}   # df, sbert_embeddings, tfidf_reduced, kmeans, …


def _safe_parse_list(val) -> list:
    if pd.isna(val) or val == "":
        return []
    try:
        parsed = ast.literal_eval(val)
        return parsed if isinstance(parsed, list) else []
    except Exception:
        return []


def load_engine() -> None:
    """Load dataset, compute / restore embeddings, build TF-IDF & KMeans."""
    t0 = time.time()
    print("⏳ Loading dataset …")
    df = pd.read_csv(DATA_PATH)

    # ── Preprocessing ──────────────────────────────────────────────────────
    for col in ["genres", "cast", "keywords"]:
        df[col] = df[col].apply(_safe_parse_list)
    df["overview"]  = df["overview"].fillna("")
    df["title"]     = df["title"].fillna("Unknown")
    df["director"]  = df["director"].fillna("")
    df = df.drop_duplicates(subset="movie_id").reset_index(drop=True)

    for col in ["popularity", "vote_average", "vote_count", "release_year"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    df = df[df["overview"].str.len() > 20].reset_index(drop=True)
    df["title_lower"] = df["title"].str.lower().str.strip()

    # ── SBERT embeddings ───────────────────────────────────────────────────
    print("⏳ Loading SBERT embeddings …")
    df["sbert_text"] = df["title"] + ". " + df["overview"]

    if os.path.exists(EMBEDDING_FILE):
        sbert_embeddings = np.load(EMBEDDING_FILE)
    else:
        print("  Encoding with SBERT (this may take a while) …")
        sbert_model = SentenceTransformer(SBERT_MODEL)
        sbert_embeddings = sbert_model.encode(
            df["sbert_text"].tolist(),
            batch_size=256,
            show_progress_bar=True,
            normalize_embeddings=True,
        )
        np.save(EMBEDDING_FILE, sbert_embeddings)

    # ── Zero-shot Semantic Probes ──────────────────────────────────────────
    print("⏳ Computing semantic probe scores …")
    if "sbert_model" not in dir():
        sbert_model = SentenceTransformer(SBERT_MODEL)

    probe_embeddings: Dict[str, np.ndarray] = {}
    for name, text in PROBES.items():
        prompt = f"This description represents a specific movie characteristic:\n\n{text}\n\nThe embedding should capture the semantic meaning of this characteristic."
        probe_embeddings[name] = sbert_model.encode(prompt, normalize_embeddings=True)

    for name, vec in probe_embeddings.items():
        df[name] = np.dot(sbert_embeddings, vec)

    # ── TF-IDF ────────────────────────────────────────────────────────────
    print("⏳ Building TF-IDF matrix …")

    def _metadata_soup(row) -> str:
        genres   = " ".join(row["genres"][:5])           if row["genres"]   else ""
        keywords = " ".join(row["keywords"][:8])         if row["keywords"] else ""
        cast     = " ".join(row["cast"][:3])             if row["cast"]     else ""
        director = row["director"].replace(" ", "_")     if row["director"] else ""
        return f"{genres} {genres} {director} {director} {cast} {keywords}"

    df["metadata_soup"] = df.apply(_metadata_soup, axis=1)

    tfidf_vectorizer = TfidfVectorizer(
        max_features=10_000, ngram_range=(1, 2), min_df=2, sublinear_tf=True
    )
    tfidf_matrix = tfidf_vectorizer.fit_transform(df["metadata_soup"])

    svd = TruncatedSVD(n_components=300, random_state=42)
    tfidf_reduced = svd.fit_transform(tfidf_matrix)

    # ── Log / normalised popularity & vote columns ─────────────────────────
    df["popularity_log"]    = np.log1p(df["popularity"])
    df["vote_average_norm"] = df["vote_average"]

    # ── KMeans clustering ─────────────────────────────────────────────────
    print("⏳ Fitting KMeans clustering …")
    kmeans = MiniBatchKMeans(
        n_clusters=N_CLUSTERS, random_state=42, batch_size=2048, n_init=10
    )
    df["cluster"] = kmeans.fit_predict(sbert_embeddings)

    # ── Cluster genre summary (for /clusters endpoint) ─────────────────────
    cluster_genre_summary: Dict[int, list] = {}
    cluster_sizes: Dict[int, int] = {}
    for c in range(N_CLUSTERS):
        mask = df["cluster"] == c
        cluster_sizes[c] = int(mask.sum())
        genres_flat = [g for gl in df.loc[mask, "genres"] for g in gl]
        top_genres  = [g for g, _ in Counter(genres_flat).most_common(5)]
        cluster_genre_summary[c] = top_genres

    elapsed = time.time() - t0
    print(f"✅ Engine ready in {elapsed:.1f}s  ({len(df):,} movies, {N_CLUSTERS} clusters)")

    # ── Store everything globally ─────────────────────────────────────────
    _engine.update(
        df=df,
        sbert_embeddings=sbert_embeddings,
        tfidf_reduced=tfidf_reduced,
        tfidf_vectorizer=tfidf_vectorizer,
        sbert_model=sbert_model,
        probe_embeddings=probe_embeddings,
        cluster_genre_summary=cluster_genre_summary,
        cluster_sizes=cluster_sizes,
        ready=True,
        startup_time=elapsed,
    )


# ---------------------------------------------------------------------------
# 4. FastAPI app with lifespan (replaces on_event deprecated pattern)
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    load_engine()
    yield
    _engine.clear()


app = FastAPI(
    title="🎬 Hybrid Movie Recommender API",
    description=(
        "REST API for the SBERT + TF-IDF + KMeans hybrid movie recommendation system. "
        "Built from `movie_recommender_v3.py`."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# 5. Pydantic models
# ---------------------------------------------------------------------------
class MovieResult(BaseModel):
    rank:         int
    movie_id:     Optional[int]
    title:        str
    release_year: int
    genres:       List[str]
    vote_average: float
    popularity:   float
    final_score:  Optional[float] = None


class RecommendationResponse(BaseModel):
    query_title:     str
    query_movie_id:  Optional[int]
    cluster:         int
    top_probe_axes:  List[Tuple[str, float]]
    recommendations: List[MovieResult]


class CompareResponse(BaseModel):
    query_title:        str
    query_movie_id:     int
    overlap_count:      int
    overlap_titles:     List[str]
    hybrid_recs:        List[MovieResult]
    tmdb_recs:          List[Dict[str, Any]]


class SemanticWeights(BaseModel):
    """Custom probe weights for /recommendations/semantic.
    Any probe key not supplied defaults to 0.0.
    """
    weights: Dict[str, float] = Field(
        default_factory=dict,
        example={
            "dark_serious_tone":       0.9,
            "philosophical_depth":     0.8,
            "psychological_complexity": 0.7,
            "space_exploration":       0.5,
        },
    )
    top_n: int = Field(default=10, ge=1, le=50)
    min_votes:        int   = Field(default=100,  ge=0)
    min_vote_average: float = Field(default=6.0,  ge=0.0)


class SemanticResponse(BaseModel):
    applied_weights:  Dict[str, float]
    recommendations:  List[MovieResult]


class ClusterInfo(BaseModel):
    cluster_id:  int
    size:        int
    top_genres:  List[str]


# ---------------------------------------------------------------------------
# 6. Helper functions
# ---------------------------------------------------------------------------
def _require_engine():
    if not _engine.get("ready"):
        raise HTTPException(status_code=503, detail="Engine not ready yet. Retry shortly.")


def _get_movie_index(title_query: str) -> Optional[int]:
    df = _engine["df"]
    tq = title_query.lower().strip()
    exact = df[df["title_lower"] == tq]
    if not exact.empty:
        return int(exact.index[0])
    partial = df[df["title_lower"].str.contains(tq, na=False, regex=False)]
    if not partial.empty:
        return int(partial.index[0])
    return None


def _row_to_movie_result(rank: int, row: pd.Series, final_score: Optional[float] = None) -> MovieResult:
    return MovieResult(
        rank=rank,
        movie_id=int(row["movie_id"]) if pd.notna(row.get("movie_id")) else None,
        title=row["title"],
        release_year=int(row["release_year"]),
        genres=list(row["genres"]) if isinstance(row["genres"], list) else [],
        vote_average=float(row["vote_average"]),
        popularity=float(row["popularity"]),
        final_score=round(float(final_score), 6) if final_score is not None else None,
    )


def _compute_recommendations(
    idx: int,
    top_n: int = TOP_N_DEFAULT,
    hybrid_weights: Tuple[float, float, float, float] = (0.65, 0.20, 0.10, 0.05),
    min_votes: int = 100,
    min_vote_average: float = 6.0,
) -> pd.DataFrame:
    """Core recommendation logic — mirrors recommend_movies() from the notebook."""
    df              = _engine["df"]
    sbert_embeddings = _engine["sbert_embeddings"]
    tfidf_reduced   = _engine["tfidf_reduced"]

    seed_movie    = df.iloc[idx]
    cluster_label = seed_movie["cluster"]
    cluster_idx   = df[df["cluster"] == cluster_label].index

    if len(cluster_idx) < top_n + 1:
        cluster_idx = df.index

    w_sbert, w_tfidf, w_vote, w_pop = hybrid_weights

    q_sbert   = sbert_embeddings[idx].reshape(1, -1)
    c_sbert   = sbert_embeddings[cluster_idx]
    sbert_sims = cosine_similarity(q_sbert, c_sbert).flatten()

    q_tfidf   = tfidf_reduced[idx].reshape(1, -1)
    c_tfidf   = tfidf_reduced[cluster_idx]
    tfidf_sims = cosine_similarity(q_tfidf, c_tfidf).flatten()

    candidates = df.iloc[cluster_idx].copy()
    scaler = MinMaxScaler()
    candidates["pop_norm"]  = scaler.fit_transform(candidates[["popularity_log"]])
    candidates["vote_norm"] = scaler.fit_transform(candidates[["vote_average_norm"]])

    candidates["base_hybrid_score"] = (
        w_sbert * sbert_sims
        + w_tfidf * tfidf_sims
        + w_vote  * candidates["vote_norm"].values
        + w_pop   * candidates["pop_norm"].values
    )

    # Semantic probe weighting
    probe_keys   = list(PROBES.keys())
    seed_profile = {name: float(seed_movie[name]) for name in probe_keys}
    total_val    = sum(max(v, 0.001) for v in seed_profile.values())
    weights      = {k: max(v, 0.001) / total_val for k, v in seed_profile.items()}

    candidates["dynamic_probe_score"] = 0.0
    for k in probe_keys:
        candidates["dynamic_probe_score"] += weights[k] * candidates[k]

    candidates["final_score"] = candidates["base_hybrid_score"] + candidates["dynamic_probe_score"]

    if weights.get("light_comedic_tone", 0) < 0.05:
        candidates["final_score"] -= 0.3 * candidates["light_comedic_tone"]
    if weights.get("horror_intensity", 0) < 0.05:
        candidates["final_score"] -= 0.2 * candidates["horror_intensity"]

    # Quality floor + drop seed
    candidates = candidates[
        (candidates["vote_count"]   >= min_votes) &
        (candidates["vote_average"] >= min_vote_average)
    ]
    candidates = candidates.drop(idx, errors="ignore")
    top_recs   = candidates.sort_values("final_score", ascending=False).head(top_n)
    top_recs   = top_recs.copy()
    top_recs.insert(0, "rank", range(1, len(top_recs) + 1))
    return top_recs, weights


def _fetch_tmdb_recs(movie_id: int, top_n: int = TOP_N_DEFAULT) -> List[Dict]:
    """Call TMDB /movie/{id}/recommendations and return raw list of dicts."""
    url     = f"{TMDB_BASE_URL}/movie/{movie_id}/recommendations"
    params  = {"api_key": TMDB_API_KEY, "language": "en-US", "page": 1}
    headers = {"User-Agent": "MovieRecommenderAPI/1.0"}
    try:
        resp = req_lib.get(url, params=params, headers=headers, timeout=10)
        resp.raise_for_status()
        results = resp.json().get("results", [])[:top_n]
        return [
            {
                "rank":         i + 1,
                "movie_id":     m.get("id"),
                "title":        m.get("title", "N/A"),
                "release_year": int(m.get("release_date", "0")[:4] or 0),
                "vote_average": m.get("vote_average", 0),
                "popularity":   m.get("popularity", 0),
                "overview":     (m.get("overview", "")[:150] + "…")
                                if len(m.get("overview", "")) > 150 else m.get("overview", ""),
            }
            for i, m in enumerate(results)
        ]
    except Exception as exc:
        return [{"error": str(exc)}]


# ---------------------------------------------------------------------------
# 7. Endpoints
# ---------------------------------------------------------------------------

# ── 7.1  Health ──────────────────────────────────────────────────────────────
@app.get("/health", summary="Health & readiness check", tags=["System"])
def health():
    """Returns 200 once the ML engine has finished loading."""
    if not _engine.get("ready"):
        raise HTTPException(status_code=503, detail="Engine is still loading.")
    df = _engine["df"]
    return {
        "status":        "ok",
        "movies_loaded": len(df),
        "clusters":      N_CLUSTERS,
        "sbert_model":   SBERT_MODEL,
        "startup_time_s": round(_engine.get("startup_time", 0), 2),
    }


# ── 7.2  Search ──────────────────────────────────────────────────────────────
@app.get("/search", summary="Search for a movie by title", tags=["Discovery"])
def search(
    query: str = Query(..., description="Partial or full movie title"),
    limit: int = Query(default=10, ge=1, le=50),
):
    """
    Returns up to `limit` movies whose title contains `query` (case-insensitive).
    Use this to confirm the exact title / movie_id before calling `/recommendations`.
    """
    _require_engine()
    df = _engine["df"]
    q  = query.lower().strip()
    matches = df[df["title_lower"].str.contains(q, na=False, regex=False)].head(limit)
    if matches.empty:
        raise HTTPException(status_code=404, detail=f"No movies found matching '{query}'.")

    return {
        "query":   query,
        "count":   len(matches),
        "results": [
            {
                "movie_id":     int(r["movie_id"]) if pd.notna(r.get("movie_id")) else None,
                "title":        r["title"],
                "release_year": int(r["release_year"]),
                "genres":       r["genres"],
                "vote_average": float(r["vote_average"]),
                "popularity":   float(r["popularity"]),
                "cluster":      int(r["cluster"]),
            }
            for _, r in matches.iterrows()
        ],
    }


# ── 7.3  Movie detail ─────────────────────────────────────────────────────────
@app.get("/movie/{movie_id}", summary="Get metadata for a single movie", tags=["Discovery"])
def get_movie(movie_id: int):
    """Fetch full metadata for a movie by its TMDB movie_id."""
    _require_engine()
    df  = _engine["df"]
    row = df[df["movie_id"] == movie_id]
    if row.empty:
        raise HTTPException(status_code=404, detail=f"movie_id {movie_id} not found.")
    r = row.iloc[0]
    probe_scores = {k: round(float(r[k]), 4) for k in PROBES}
    return {
        "movie_id":     int(r["movie_id"]),
        "title":        r["title"],
        "release_year": int(r["release_year"]),
        "overview":     r["overview"],
        "genres":       r["genres"],
        "cast":         r["cast"],
        "director":     r["director"],
        "vote_average": float(r["vote_average"]),
        "vote_count":   int(r["vote_count"]),
        "popularity":   float(r["popularity"]),
        "cluster":      int(r["cluster"]),
        "probe_scores": probe_scores,
    }


# ── 7.4  Recommendations ─────────────────────────────────────────────────────
@app.get(
    "/recommendations",
    summary="Get hybrid recommendations for a movie title",
    response_model=RecommendationResponse,
    tags=["Recommendations"],
)
def recommendations(
    title:           str   = Query(..., description="Seed movie title"),
    top_n:           int   = Query(default=10, ge=1,   le=50),
    w_sbert:         float = Query(default=0.65, ge=0.0, le=1.0, description="SBERT similarity weight"),
    w_tfidf:         float = Query(default=0.20, ge=0.0, le=1.0, description="TF-IDF similarity weight"),
    w_vote:          float = Query(default=0.10, ge=0.0, le=1.0, description="Vote average weight"),
    w_pop:           float = Query(default=0.05, ge=0.0, le=1.0, description="Popularity weight"),
    min_votes:       int   = Query(default=100, ge=0,   description="Minimum vote count quality floor"),
    min_vote_average:float = Query(default=6.0, ge=0.0, description="Minimum vote average quality floor"),
):
    """
    Returns the top-N recommendations for a given movie title using the
    SBERT + TF-IDF + KMeans hybrid model with dynamic semantic probing.

    Weights are automatically normalised to sum to 1; pass custom values to
    shift the balance between semantic similarity, metadata matching, and quality signals.
    """
    _require_engine()
    df  = _engine["df"]
    idx = _get_movie_index(title)
    if idx is None:
        raise HTTPException(status_code=404, detail=f"Movie '{title}' not found. Try /search first.")

    # Normalise weights
    total_w = w_sbert + w_tfidf + w_vote + w_pop
    if total_w == 0:
        raise HTTPException(status_code=422, detail="At least one weight must be > 0.")
    hw = (w_sbert / total_w, w_tfidf / total_w, w_vote / total_w, w_pop / total_w)

    top_recs, weights = _compute_recommendations(
        idx, top_n=top_n, hybrid_weights=hw,
        min_votes=min_votes, min_vote_average=min_vote_average,
    )

    seed = df.iloc[idx]
    top_probe_axes = sorted(weights.items(), key=lambda x: x[1], reverse=True)[:4]

    return RecommendationResponse(
        query_title    = seed["title"],
        query_movie_id = int(seed["movie_id"]) if pd.notna(seed.get("movie_id")) else None,
        cluster        = int(seed["cluster"]),
        top_probe_axes = [(k, round(v, 4)) for k, v in top_probe_axes],
        recommendations= [
            _row_to_movie_result(int(row["rank"]), row, float(row["final_score"]))
            for _, row in top_recs.iterrows()
        ],
    )


# ── 7.5  Compare ─────────────────────────────────────────────────────────────
@app.get(
    "/compare",
    summary="Compare hybrid model vs TMDB API recommendations",
    response_model=CompareResponse,
    tags=["Recommendations"],
)
def compare(
    title: str = Query(..., description="Seed movie title"),
    top_n: int = Query(default=10, ge=1, le=50),
):
    """
    Runs our hybrid recommender **and** TMDB's `/movie/{id}/recommendations` for the
    same seed movie, then returns both lists along with overlap statistics.
    """
    _require_engine()
    df  = _engine["df"]
    idx = _get_movie_index(title)
    if idx is None:
        raise HTTPException(status_code=404, detail=f"Movie '{title}' not found. Try /search first.")

    seed     = df.iloc[idx]
    movie_id = int(seed["movie_id"]) if pd.notna(seed.get("movie_id")) else None

    top_recs, _ = _compute_recommendations(idx, top_n=top_n)
    tmdb_recs   = _fetch_tmdb_recs(movie_id, top_n=top_n) if movie_id else []

    our_titles   = {r["title"].lower() for _, r in top_recs.iterrows()} if not top_recs.empty else set()
    tmdb_titles  = {r["title"].lower() for r in tmdb_recs if "title" in r}
    overlap      = our_titles & tmdb_titles

    return CompareResponse(
        query_title    = seed["title"],
        query_movie_id = movie_id or -1,
        overlap_count  = len(overlap),
        overlap_titles = sorted(overlap),
        hybrid_recs    = [
            _row_to_movie_result(int(row["rank"]), row, float(row["final_score"]))
            for _, row in top_recs.iterrows()
        ],
        tmdb_recs = tmdb_recs,
    )


# ── 7.6  Semantic / vibe search ──────────────────────────────────────────────
@app.post(
    "/recommendations/semantic",
    summary="Probe-weight-driven 'vibe' recommendations (no seed movie needed)",
    response_model=SemanticResponse,
    tags=["Recommendations"],
)
def semantic_recommendations(body: SemanticWeights):
    """
    Supply custom weights for any of the 19 semantic probe axes to describe a
    *vibe* (e.g. `dark_serious_tone: 0.9, space_exploration: 0.8`) and receive
    recommendations that best match that profile — **no seed movie required**.

    Valid probe keys:
    `philosophical_depth`, `emotional_depth`, `psychological_complexity`,
    `intellectual_complexity`, `space_exploration`, `ai_and_consciousness`,
    `time_manipulation`, `dystopian_future`, `dark_serious_tone`,
    `light_comedic_tone`, `action_intensity`, `horror_intensity`,
    `non_linear_narrative`, `mystery_investigation`, `character_driven`,
    `spectacle_blockbuster`, `superhero`, `crime_realism`, `adventure_family`
    """
    _require_engine()
    df  = _engine["df"]

    # Validate probe keys
    invalid = [k for k in body.weights if k not in PROBES]
    if invalid:
        raise HTTPException(
            status_code=422,
            detail=f"Unknown probe key(s): {invalid}. Valid keys: {list(PROBES.keys())}",
        )

    # Build normalised weight dict (fill missing probes with 0)
    raw_weights   = {k: body.weights.get(k, 0.0) for k in PROBES}
    total_val     = sum(max(v, 0.001) for v in raw_weights.values())
    norm_weights  = {k: max(v, 0.001) / total_val for k, v in raw_weights.items()}

    # Compute vibe score for every movie
    candidates = df[
        (df["vote_count"]   >= body.min_votes) &
        (df["vote_average"] >= body.min_vote_average)
    ].copy()

    candidates["vibe_score"] = 0.0
    for k, w in norm_weights.items():
        candidates["vibe_score"] += w * candidates[k]

    # Penalty axes (same as main engine)
    if norm_weights.get("light_comedic_tone", 0) < 0.05:
        candidates["vibe_score"] -= 0.3 * candidates["light_comedic_tone"]
    if norm_weights.get("horror_intensity", 0) < 0.05:
        candidates["vibe_score"] -= 0.2 * candidates["horror_intensity"]

    top_recs = candidates.sort_values("vibe_score", ascending=False).head(body.top_n)

    return SemanticResponse(
        applied_weights={k: round(v, 4) for k, v in norm_weights.items() if body.weights.get(k, 0) > 0},
        recommendations=[
            MovieResult(
                rank=i + 1,
                movie_id=int(row["movie_id"]) if pd.notna(row.get("movie_id")) else None,
                title=row["title"],
                release_year=int(row["release_year"]),
                genres=list(row["genres"]),
                vote_average=float(row["vote_average"]),
                popularity=float(row["popularity"]),
                final_score=round(float(row["vibe_score"]), 6),
            )
            for i, (_, row) in enumerate(top_recs.iterrows())
        ],
    )


# ── 7.7  Cluster overview ─────────────────────────────────────────────────────
@app.get(
    "/clusters",
    summary="Overview of all KMeans clusters",
    response_model=List[ClusterInfo],
    tags=["Discovery"],
)
def clusters():
    """Returns the size and dominant genres for every cluster in the model."""
    _require_engine()
    return [
        ClusterInfo(
            cluster_id=c,
            size=_engine["cluster_sizes"][c],
            top_genres=_engine["cluster_genre_summary"][c],
        )
        for c in sorted(_engine["cluster_sizes"].keys())
    ]


# ---------------------------------------------------------------------------
# 8. Entry point (for running directly with `python main.py`)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
