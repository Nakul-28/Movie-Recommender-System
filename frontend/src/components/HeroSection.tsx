import React, { useEffect, useState, useCallback } from 'react';
import { apiClient } from '@/lib/api';
import type { SearchResult } from '@/lib/types';
import SearchBar from './SearchBar';

interface FeaturedMovie {
  movie_id: number;
  title: string;
  poster_url: string;
  vote_average: number;
  release_year: number;
}

interface HeroSectionProps {
  onSelectMovie: (movie: SearchResult) => void;
}

const FALLBACK_POSTER = 'https://via.placeholder.com/200x300/1a1f2e/6366f1?text=🎬';

const PosterImage: React.FC<{ movie: FeaturedMovie }> = ({ movie }) => {
  const [src, setSrc] = useState(movie.poster_url || FALLBACK_POSTER);
  return (
    <div
      className="flex-shrink-0 rounded-xl overflow-hidden shadow-lg"
      style={{ width: '130px', height: '195px' }}
      title={movie.title}
    >
      <img
        src={src}
        alt={movie.title}
        className="w-full h-full object-cover"
        loading="lazy"
        onError={() => setSrc(FALLBACK_POSTER)}
      />
    </div>
  );
};

const HeroSection: React.FC<HeroSectionProps> = ({ onSelectMovie }) => {
  const [movies, setMovies] = useState<FeaturedMovie[]>([]);

  useEffect(() => {
    apiClient.getFeaturedMovies(32).then(setMovies).catch(() => setMovies([]));
  }, []);

  // Split movies into 3 rows
  const chunkSize = Math.ceil(movies.length / 3);
  const row1 = movies.slice(0, chunkSize);
  const row2 = movies.slice(chunkSize, chunkSize * 2);
  const row3 = movies.slice(chunkSize * 2);

  // Duplicate each row for seamless looping
  const doubled = (arr: FeaturedMovie[]) => [...arr, ...arr];

  return (
    <div className="relative min-h-screen flex flex-col items-center justify-center overflow-hidden bg-[#0a0a0f]">

      {/* ── Scrolling poster rows ── */}
      <div className="absolute inset-0 flex flex-col gap-4 py-6 opacity-40 select-none pointer-events-none"
           style={{ maskImage: 'linear-gradient(to bottom, transparent 0%, black 20%, black 80%, transparent 100%)' }}>

        {[
          { data: doubled(row1), cls: 'poster-row-left' },
          { data: doubled(row2), cls: 'poster-row-right' },
          { data: doubled(row3), cls: 'poster-row-left', delay: '-10s' },
        ].map(({ data, cls, delay }, rowIdx) => (
          <div key={rowIdx} className="overflow-hidden w-full flex-1">
            <div
              className={`flex gap-4 ${cls}`}
              style={delay ? { animationDelay: delay } : {}}
            >
              {data.map((movie, i) => (
                <PosterImage key={`${movie.movie_id}-${i}`} movie={movie} />
              ))}
            </div>
          </div>
        ))}
      </div>

      {/* ── Gradient overlays ── */}
      <div className="absolute inset-0 bg-gradient-to-b from-[#0a0a0f]/80 via-[#0a0a0f]/30 to-[#0a0a0f]/80 pointer-events-none" />
      <div className="absolute inset-0 bg-gradient-radial from-transparent via-transparent to-[#0a0a0f] pointer-events-none" />

      {/* ── Hero content ── */}
      <div className="relative z-10 flex flex-col items-center text-center px-6 space-y-8 max-w-3xl">

        {/* Branding */}
        <div className="space-y-3">
          <h1 className="text-5xl md:text-7xl font-extrabold tracking-tight"
              style={{ background: 'linear-gradient(135deg, #fff 30%, #a78bfa 70%)', WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent' }}>
            Movie Recommender
          </h1>
          <p className="text-lg md:text-xl text-gray-400 font-light max-w-xl mx-auto leading-relaxed">
            Powered by <span className="text-violet-400 font-medium">SBERT</span>, <span className="text-violet-400 font-medium">TF-IDF</span> &amp; <span className="text-violet-400 font-medium">K-Means</span> — discover films that truly match your vibe.
          </p>
        </div>

        {/* Search bar */}
        <div className="w-full max-w-2xl">
          <SearchBar onSelectMovie={onSelectMovie} />
        </div>

        {/* Feature pills */}
        <div className="flex flex-wrap justify-center gap-3 text-sm">
          {['Semantic Analysis', 'Hybrid Scoring', 'Cluster Explainability', '19 Semantic Probes'].map((tag) => (
            <span key={tag} className="px-5 py-2 rounded-full border border-white/10 bg-white/5 backdrop-blur-md text-gray-300 font-medium tracking-wide shadow-[0_0_15px_rgba(255,255,255,0.03)] hover:bg-white/10 hover:border-white/20 hover:text-white transition-all duration-300 cursor-default">
              {tag}
            </span>
          ))}
        </div>

      </div>
    </div>
  );
};

export default HeroSection;
