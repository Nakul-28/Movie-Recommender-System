import React from 'react';
import { getConfidenceLevel } from '@/lib/utils';

interface SimilarityMeterProps {
  score: number;
  size?: 'sm' | 'md' | 'lg';
  showLabel?: boolean;
}

const SimilarityMeter: React.FC<SimilarityMeterProps> = ({ score, size = 'md', showLabel = true }) => {
  const confidence = getConfidenceLevel(score);
  const scorePercent = Math.round(score * 100);

  const toneClass = score < 30 ? 'text-danger' : score < 70 ? 'text-amber-400' : 'text-success';
  const progressToneClass = score < 30 ? 'tone-danger' : score < 70 ? 'tone-warning' : 'tone-success';

  const sizeClasses = {
    sm: 'h-1',
    md: 'h-2',
    lg: 'h-3',
  };

  return (
    <div className="space-y-2">
      {showLabel && (
        <div className="flex justify-between items-center text-sm">
          <span className="text-secondary">Similarity</span>
          <div>
            <span className={`font-bold ${toneClass}`}>
              {scorePercent}%
            </span>
            <span className="text-secondary ml-2">({confidence})</span>
          </div>
        </div>
      )}
      <progress
        value={scorePercent}
        max={100}
        className={`similarity-progress ${size} ${progressToneClass}`}
      />
    </div>
  );
};

export default SimilarityMeter;
