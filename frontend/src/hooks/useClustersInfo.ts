/**
 * Hook for fetching cluster information
 */

import { useState, useEffect } from 'react';
import { apiClient } from '@/lib/api';
import { ClustersInfoResponse } from '@/lib/types';

export const useClustersInfo = () => {
  const [clustersInfo, setClustersInfo] = useState<ClustersInfoResponse | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchClusters = async () => {
      setIsLoading(true);
      setError(null);

      try {
        const info = await apiClient.getClustersInfo();
        setClustersInfo(info);
      } catch (err) {
        const message = err instanceof Error ? err.message : 'Failed to fetch clusters';
        setError(message);
      } finally {
        setIsLoading(false);
      }
    };

    fetchClusters();
  }, []);

  return { clustersInfo, isLoading, error };
};
