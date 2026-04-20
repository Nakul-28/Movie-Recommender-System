/**
 * TypeScript types for the Movie Recommender System
 */

export interface Movie {
  movie_id: number;
  title: string;
  release_year: number;
  genres: string[];
  vote_average: number;
  popularity: number;
  overview?: string;
  cast?: string[];
  keywords?: string[];
  director?: string;
  cluster?: number;
  final_score?: number;
}

export interface SearchResult {
  movie_id: number;
  title: string;
  release_year: number;
  genres: string[];
  vote_average: number;
  popularity: number;
  cluster: number;
  poster_url?: string | null;
}

export interface SemanticProbe {
  name: string;
  score: number;
}

export interface MovieExplain {
  movie_id: number;
  title: string;
  release_year: number;
  overview: string;
  genres: string[];
  cast: string[];
  keywords: string[];
  director: string;
  vote_average: number;
  popularity: number;
  cluster_id: number;
  cluster_label: string;
  top_probes: SemanticProbe[];
  poster_url?: string | null;
}

export interface ClusterLabel {
  cluster_id: number;
  label: string;
  keywords: string[];
  top_probe: string;
  size: number;
}

export interface SimilarityBreakdown {
  sbert_similarity: number;
  tfidf_similarity: number;
  overall_score: number;
}

export interface SharedMetadata {
  keywords: string[];
  genres: string[];
  cast: string[];
}

export interface ProbeAlignment {
  name: string;
  difference: number;
}

export interface RecommendationExplain {
  rank: number;
  movie_id: number;
  title: string;
  release_year: number;
  genres: string[];
  vote_average: number;
  popularity: number;
  final_score: number;
  poster_url?: string | null;
  similarity_breakdown: SimilarityBreakdown;
  shared_metadata: SharedMetadata;
  probe_alignment: ProbeAlignment[];
}

export interface RecommendationsResponse {
  query_title: string;
  query_movie_id: number;
  cluster: number;
  cluster_label: string;
  top_probe_axes: Array<[string, number]>;
  recommendations: RecommendationExplain[];
}

export interface ClustersInfoResponse {
  total_clusters: number;
  clusters: ClusterLabel[];
}
