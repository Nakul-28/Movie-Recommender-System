import React, { useEffect } from 'react';
import { useParams, useSearchParams, useNavigate } from 'react-router-dom';
import { motion } from 'framer-motion';
import { useAppStore } from '@/lib/store';
import { apiClient } from '@/lib/api';
import { useRecommendations } from '@/hooks/useRecommendations';
import type { SearchResult } from '@/lib/types';
import SearchBar from '@/components/SearchBar';
import MovieInputPanel from '@/components/MovieInputPanel';
import RecommendationCarousel from '@/components/RecommendationCarousel';

const LoadingScreen = () => (
  <motion.div
    initial={{ opacity: 0 }}
    animate={{ opacity: 1 }}
    exit={{ opacity: 0 }}
    className="fixed inset-0 z-50 flex flex-col items-center justify-center bg-[#0a0a0f] text-white"
  >
    <div className="flex flex-col items-center gap-6">
      <span className="text-6xl animate-bounce">🎬</span>
      <div className="w-16 h-16 border-4 border-violet-500 border-t-transparent rounded-full animate-spin" />
      <h2 className="text-2xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-white to-violet-400">
        Loading Cinema Data...
      </h2>
      <p className="text-gray-400 text-sm">Fetching posters and generating recommendations</p>
    </div>
  </motion.div>
);

const MoviePage: React.FC = () => {
  const { id } = useParams<{ id: string }>();
  const [searchParams] = useSearchParams();
  const title = searchParams.get('title') || '';
  const navigate = useNavigate();

  const { selectedMovie, recommendations } = useAppStore();
  const { fetch: fetchRecommendations, isLoading: isLoadingRecs, error: recsError } = useRecommendations();

  const handleSelectMovie = (movie: SearchResult) => {
    navigate(`/movies/${movie.movie_id}?title=${encodeURIComponent(movie.title)}`);
  };

  useEffect(() => {
    const loadMovieData = async () => {
      if (!id) return;
      
      const movieIdNum = parseInt(id, 10);
      try {
        useAppStore.setState({ selectedMovie: null, recommendations: null }); // Clear previous
        const movieExplain = await apiClient.getMovieExplain(movieIdNum);
        if (movieExplain) {
          useAppStore.setState({ selectedMovie: movieExplain });
          if (title) {
            await fetchRecommendations(title);
          }
        }
      } catch (error) {
        console.error('Error loading movie:', error);
      }
    };

    loadMovieData();
  }, [id, title, fetchRecommendations]);

  const isLoading = !selectedMovie || isLoadingRecs;

  if (isLoading) {
    return <LoadingScreen />;
  }

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, scale: 0.98 }}
      transition={{ duration: 0.5 }}
      className="page-transition-wrapper min-h-screen pb-20"
    >
      {/* Compact sticky search bar */}
      <div className="sticky top-0 z-30 border-b border-white/5 bg-[#0d0d14]/90 backdrop-blur-md py-4 px-6 shadow-md shadow-black/50">
        <div className="max-w-7xl mx-auto flex flex-col gap-3">
          <button
            onClick={() => navigate('/home')}
            className="text-gray-400 hover:text-white transition text-sm hover:underline self-start"
          >
            ← Back to Home
          </button>
          <div className="w-full max-w-2xl mx-auto">
            <SearchBar onSelectMovie={handleSelectMovie} />
          </div>
        </div>
      </div>

      {/* Main content — movie detail + recommendations */}
      <main className="max-w-7xl mx-auto px-6 py-10 space-y-10">
        {/* Movie detail panel */}
        <section id="movie-panel">
          <MovieInputPanel />
        </section>

        {/* Error */}
        {recsError && (
          <section className="text-center text-red-400 py-4">
            <p>⚠️ {recsError}</p>
          </section>
        )}

        {/* Recommendations grid */}
        {recommendations && recommendations.recommendations.length > 0 && (
          <section>
            <div className="flex items-center justify-between mb-6">
              <h3 className="text-2xl font-bold">🎯 Top Recommendations</h3>
              <span className="text-gray-400 text-sm">
                {recommendations.recommendations.length} films · based on "{recommendations.query_title}"
              </span>
            </div>
            <RecommendationCarousel />
          </section>
        )}
      </main>
    </motion.div>
  );
};

export default MoviePage;
