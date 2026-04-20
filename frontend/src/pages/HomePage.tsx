import React from 'react';
import { useNavigate } from 'react-router-dom';
import HeroSection from '@/components/HeroSection';
import type { SearchResult } from '@/lib/types';
import { motion } from 'framer-motion';

const HomePage: React.FC = () => {
  const navigate = useNavigate();

  const handleSelectMovie = (movie: SearchResult) => {
    navigate(`/movies/${movie.movie_id}?title=${encodeURIComponent(movie.title)}`);
  };

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0, scale: 0.95 }}
      transition={{ duration: 0.5 }}
      className="page-transition-wrapper"
    >
      <HeroSection onSelectMovie={handleSelectMovie} />
    </motion.div>
  );
};

export default HomePage;
