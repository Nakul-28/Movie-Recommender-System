import React from 'react';
import { motion } from 'framer-motion';
import { useAppStore } from '@/lib/store';
import MovieCard from './MovieCard';

const RecommendationCarousel: React.FC = () => {
  const { recommendations } = useAppStore();

  if (!recommendations || recommendations.recommendations.length === 0) {
    return null;
  }

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
      className="space-y-4"
    >
      <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-5 gap-4">
        {recommendations.recommendations.map((movie, idx) => (
          <div key={movie.movie_id}>
            <MovieCard movie={{ ...movie, rank: idx + 1 }} />
          </div>
        ))}
      </div>

      <div className="text-center text-secondary text-sm mt-8">
        <p>Showing {recommendations.recommendations.length} recommendations from AI analysis</p>
      </div>
    </motion.div>
  );
};

export default RecommendationCarousel;
