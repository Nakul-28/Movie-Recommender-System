import React from 'react';
import { motion } from 'framer-motion';
import type { RecommendationExplain } from '@/lib/types';

interface MovieCardProps {
  movie: RecommendationExplain;
}

const FALLBACK_POSTER = 'https://via.placeholder.com/500x750/1a1f2e/6366f1?text=🎬+No+Poster';

const MovieCard: React.FC<MovieCardProps> = ({ movie }) => {
  const posterUrl = movie.poster_url || FALLBACK_POSTER;
  const matchPercentage = (movie.similarity_breakdown.overall_score * 100).toFixed(0);

  return (
    <motion.div
      className="relative group rounded-xl overflow-hidden aspect-[2/3] bg-dark-bg shadow-lg hover:shadow-2xl hover:shadow-accent/40 transition-all duration-300"
      initial={{ opacity: 0, scale: 0.95 }}
      animate={{ opacity: 1, scale: 1 }}
      transition={{ duration: 0.4 }}
    >
      {/* Poster Image */}
      <img
        src={posterUrl}
        alt={`${movie.title} poster`}
        className="w-full h-full object-cover transition-transform duration-700 group-hover:scale-110"
        onError={(e) => { (e.currentTarget as HTMLImageElement).src = FALLBACK_POSTER; }}
      />

      {/* Floating Badges */}
      <div className="absolute top-2 left-2 bg-accent/90 text-white font-bold rounded-full w-8 h-8 flex items-center justify-center text-xs shadow-md z-10">
        #{movie.rank}
      </div>
      <div className="absolute top-2 right-2 bg-success/90 text-white font-bold px-2 py-1 rounded text-xs shadow-md z-10">
        {matchPercentage}% Match
      </div>

      {/* Hover Overlay */}
      <div className="absolute inset-0 bg-gradient-to-t from-deep-blue/95 via-deep-blue/40 to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-300 flex flex-col justify-end p-4">
        <h3 className="text-white font-bold text-lg leading-tight drop-shadow-md mb-1">{movie.title}</h3>
        <p className="text-gray-300 text-xs shadow-black drop-shadow-md">{movie.release_year}</p>
        {movie.genres && movie.genres.length > 0 && (
          <div className="flex flex-wrap gap-1 mt-2">
            {movie.genres.slice(0, 2).map((g) => (
              <span key={g} className="text-[10px] uppercase font-semibold tracking-wider bg-white/20 px-2 py-0.5 rounded text-white backdrop-blur-sm">
                {g}
              </span>
            ))}
          </div>
        )}
      </div>
    </motion.div>
  );
};

export default MovieCard;
