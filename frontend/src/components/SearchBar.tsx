import React, { useState } from 'react';
import { useAppStore } from '@/lib/store';
import { useMovieSearch } from '@/hooks/useMovieSearch';
import type { SearchResult } from '@/lib/types';

interface SearchBarProps {
  onSelectMovie: (movie: SearchResult) => void;
}

const SearchBar: React.FC<SearchBarProps> = ({ onSelectMovie }) => {
  const [query, setQuery] = useState('');
  const [isOpen, setIsOpen] = useState(false);
  // Single hook instance — SearchBar is the sole owner of search state
  const { results, search, isLoading } = useMovieSearch();
  const { searchHistory, addToSearchHistory } = useAppStore();

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const value = e.target.value;
    setQuery(value);
    search(value);
    if (!isOpen) setIsOpen(true);
  };

  const handleSelectResult = (result: SearchResult) => {
    setQuery('');
    setIsOpen(false);
    addToSearchHistory(result.title);
    onSelectMovie(result);
  };

  const displayedResults = query.trim() ? results.slice(0, 15) : [];
  // Recent searches removed for now

  return (
    <div className="relative w-full max-w-2xl mx-auto">
      <div className="relative">
        <span className="absolute left-4 top-1/2 -translate-y-1/2 text-secondary text-lg">🎬</span>
        <input
          type="text"
          placeholder="Search for a movie..."
          value={query}
          onChange={handleInputChange}
          onFocus={() => setIsOpen(true)}
          onBlur={() => setTimeout(() => setIsOpen(false), 200)}
          className="input-field pl-12"
          autoComplete="off"
        />
        {isLoading && (
          <span className="absolute right-4 top-1/2 -translate-y-1/2 text-secondary animate-pulse text-sm">
            Searching…
          </span>
        )}
      </div>

      {isOpen && displayedResults.length > 0 && (
        <div className="absolute top-full left-0 right-0 mt-2 bg-card-bg border border-accent/20 rounded-xl shadow-2xl z-50 max-h-[420px] overflow-y-auto">

          {displayedResults.length > 0 && (
            <>
              <div className="px-4 py-2 text-xs font-semibold text-secondary uppercase tracking-wider bg-dark-bg/50 sticky top-0">
                Results ({displayedResults.length})
              </div>
              {displayedResults.map((result) => (
                <button
                  key={result.movie_id}
                  onClick={() => handleSelectResult(result)}
                  className="w-full text-left px-3 py-3 hover:bg-dark-bg border-b border-accent/10 last:border-b-0 transition"
                >
                  <div className="flex items-center gap-3">
                    {/* Poster thumbnail */}
                    <div className="w-10 h-14 flex-shrink-0 rounded overflow-hidden bg-dark-bg">
                      {result.poster_url ? (
                        <img
                          src={result.poster_url}
                          alt={result.title}
                          className="w-full h-full object-cover"
                          onError={(e) => { (e.currentTarget as HTMLImageElement).style.display = 'none'; }}
                        />
                      ) : (
                        <div className="w-full h-full flex items-center justify-center text-secondary text-lg">🎬</div>
                      )}
                    </div>

                    {/* Info */}
                    <div className="flex-1 min-w-0">
                      <p className="text-primary font-medium truncate">{result.title}</p>
                      <p className="text-secondary text-xs mt-0.5">
                        {result.release_year} &nbsp;•&nbsp; ⭐ {result.vote_average?.toFixed(1)}
                      </p>
                    </div>

                    {/* Genre badges */}
                    <div className="flex gap-1 flex-shrink-0">
                      {result.genres?.slice(0, 2).map((genre) => (
                        <span
                          key={genre}
                          className="px-2 py-0.5 bg-accent/20 text-accent text-xs rounded-full"
                        >
                          {genre}
                        </span>
                      ))}
                    </div>
                  </div>
                </button>
              ))}
            </>
          )}
        </div>
      )}
    </div>
  );
};

export default SearchBar;