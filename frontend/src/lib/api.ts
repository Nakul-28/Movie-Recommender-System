/**
 * API client for the Movie Recommender System
 */

import axios, { AxiosInstance } from 'axios';
import {
  Movie,
  SearchResult,
  MovieExplain,
  RecommendationsResponse,
  ClustersInfoResponse,
} from './types';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

class ApiClient {
  private client: AxiosInstance;

  constructor() {
    this.client = axios.create({
      baseURL: API_BASE_URL,
      timeout: 15000,
      headers: {
        'Content-Type': 'application/json',
      },
    });
  }

  /**
   * Check if API is healthy
   */
  async health() {
    return this.client.get('/health');
  }

  /**
   * Search for movies by title
   */
  async searchMovies(query: string, limit: number = 10): Promise<SearchResult[]> {
    const response = await this.client.get('/search', {
      params: { query, limit },
    });
    return response.data.results || [];
  }

  /**
   * Get movie details with explainability
   */
  async getMovieExplain(movieId: number): Promise<MovieExplain> {
    const response = await this.client.get(`/movie/${movieId}/explain`);
    return response.data;
  }

  /**
   * Get recommendations for a movie with explainability
   */
  async getRecommendationsExplain(
    title: string,
    topN: number = 10
  ): Promise<RecommendationsResponse> {
    const response = await this.client.get('/recommendations/explain', {
      params: {
        title,
        top_n: topN,
      },
    });
    return response.data;
  }

  /**
   * Get all cluster information and labels
   */
  async getClustersInfo(): Promise<ClustersInfoResponse> {
    const response = await this.client.get('/clusters/info');
    return response.data;
  }

  /**
   * Get featured/popular movies for the landing page hero
   */
  async getFeaturedMovies(limit: number = 24): Promise<Array<{
    movie_id: number;
    title: string;
    poster_url: string;
    vote_average: number;
    release_year: number;
  }>> {
    const response = await this.client.get('/featured', { params: { limit } });
    return response.data.movies || [];
  }
}

// Export singleton instance
export const apiClient = new ApiClient();
export default apiClient;
