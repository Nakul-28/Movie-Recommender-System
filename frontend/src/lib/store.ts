/**
 * Zustand store for application state
 */

import { create } from 'zustand';
import { MovieExplain, RecommendationsResponse } from '@/lib/types';

export interface AppState {
  // Selected movie
  selectedMovie: MovieExplain | null;
  setSelectedMovie: (movie: MovieExplain | null) => void;

  // Recommendations
  recommendations: RecommendationsResponse | null;
  setRecommendations: (recs: RecommendationsResponse | null) => void;

  // Loading states
  isLoadingSearch: boolean;
  setIsLoadingSearch: (loading: boolean) => void;

  isLoadingRecommendations: boolean;
  setIsLoadingRecommendations: (loading: boolean) => void;

  // Search history
  searchHistory: string[];
  addToSearchHistory: (title: string) => void;
  clearSearchHistory: () => void;

  // Expanded recommendation card
  expandedRecommendationId: number | null;
  setExpandedRecommendationId: (id: number | null) => void;
}

export const useAppStore = create<AppState>((set) => ({
  // Initial state
  selectedMovie: null,
  setSelectedMovie: (movie) => set({ selectedMovie: movie }),

  recommendations: null,
  setRecommendations: (recs) => set({ recommendations: recs }),

  isLoadingSearch: false,
  setIsLoadingSearch: (loading) => set({ isLoadingSearch: loading }),

  isLoadingRecommendations: false,
  setIsLoadingRecommendations: (loading) => set({ isLoadingRecommendations: loading }),

  searchHistory: [],
  addToSearchHistory: (title) =>
    set((state) => {
      const updated = [title, ...state.searchHistory.filter((h) => h !== title)].slice(0, 5);
      return { searchHistory: updated };
    }),
  clearSearchHistory: () => set({ searchHistory: [] }),

  expandedRecommendationId: null,
  setExpandedRecommendationId: (id) => set({ expandedRecommendationId: id }),
}));
