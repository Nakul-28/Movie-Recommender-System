import { useAppStore } from '@/lib/store';
const FALLBACK_POSTER = 'https://via.placeholder.com/500x750/1a1f2e/6366f1?text=🎬';

const MovieInputPanel: React.FC = () => {
  const { selectedMovie } = useAppStore();

  if (!selectedMovie) return null;

  const posterUrl = selectedMovie.poster_url || FALLBACK_POSTER;

  return (
    <div className="card overflow-hidden">
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">

        {/* Poster */}
        <div className="md:col-span-1">
          <div className="relative rounded-xl overflow-hidden shadow-2xl shadow-accent/20 aspect-[2/3]">
            <img
              src={posterUrl}
              alt={`${selectedMovie.title} poster`}
              className="w-full h-full object-cover"
              onError={(e) => { (e.currentTarget as HTMLImageElement).src = FALLBACK_POSTER; }}
            />
            <div className="absolute inset-0 bg-gradient-to-t from-deep-blue/80 via-transparent to-transparent" />
          </div>
        </div>

        {/* Info */}
        <div className="md:col-span-2 space-y-6">
          <div>
            <h2 className="text-3xl font-bold text-primary mb-2">{selectedMovie.title}</h2>
            <div className="flex flex-wrap items-center gap-3 mb-4">
              <span className="badge badge-primary">{selectedMovie.release_year}</span>
              <span className="badge badge-success">⭐ {selectedMovie.vote_average?.toFixed(1)}</span>
              {selectedMovie.cluster_label && (
                <span className="badge badge-accent text-xs truncate max-w-xs">{selectedMovie.cluster_label}</span>
              )}
            </div>
          </div>

          {/* Overview */}
          <div>
            <p className="text-secondary leading-relaxed">{selectedMovie.overview}</p>
          </div>

          {/* Director + Cast */}
          {selectedMovie.director && (
            <div className="space-y-1 text-sm">
              <span className="text-secondary font-medium">Director: </span>
              <span className="text-primary">{selectedMovie.director}</span>
            </div>
          )}
          {selectedMovie.cast && selectedMovie.cast.length > 0 && (
            <div className="text-sm">
              <span className="text-secondary font-medium">Cast: </span>
              <span className="text-primary">{selectedMovie.cast.slice(0, 4).join(', ')}</span>
            </div>
          )}

          {/* Genre badges */}
          {selectedMovie.genres && selectedMovie.genres.length > 0 && (
            <div className="flex flex-wrap gap-2">
              {selectedMovie.genres.map((g) => (
                <span key={g} className="badge badge-primary text-xs">{g}</span>
              ))}
            </div>
          )}

        </div>
      </div>
    </div>
  );
};

export default MovieInputPanel;
