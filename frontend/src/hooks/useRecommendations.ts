/**
 * Hook for fetching recommendations
 * Results are written to the global Zustand store so RecommendationCarousel can read them.
 */

import { useState, useCallback } from 'react';
import { apiClient } from '@/lib/api';
import { useAppStore } from '@/lib/store';
import { RecommendationsResponse } from '@/lib/types';

export const useRecommendations = () => {
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const { setRecommendations, recommendations } = useAppStore();

  const fetch = useCallback(async (movieTitle: string, topN: number = 10) => {
    setIsLoading(true);
    setError(null);

    try {
      const recs = await apiClient.getRecommendationsExplain(movieTitle, topN);
      // Write into global store so RecommendationCarousel (which reads from store) can render
      setRecommendations(recs);
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to fetch recommendations';
      setError(message);
      setRecommendations(null);
    } finally {
      setIsLoading(false);
    }
  }, [setRecommendations]);

  return { recommendations, isLoading, error, fetch };
};
