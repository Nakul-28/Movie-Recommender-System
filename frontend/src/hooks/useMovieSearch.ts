/**
 * Hook for movie search
 */

import { useState } from 'react';
import { apiClient } from '@/lib/api';
import { SearchResult } from '@/lib/types';

export const useMovieSearch = () => {
  const [results, setResults] = useState<SearchResult[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const search = async (query: string) => {
    if (!query.trim()) {
      setResults([]);
      return;
    }

    setIsLoading(true);
    setError(null);

    try {
      const searchResults = await apiClient.searchMovies(query, 20);
      setResults(searchResults);
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Search failed';
      setError(message);
      setResults([]);
    } finally {
      setIsLoading(false);
    }
  };

  return { results, isLoading, error, search };
};
