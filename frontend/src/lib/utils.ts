/**
 * Utility functions for the frontend
 */

/**
 * Format a number as a percentage with 1 decimal place
 */
export const formatPercentage = (value: number): string => {
  return `${(value * 100).toFixed(1)}%`;
};

/**
 * Format a similarity score as a nice display
 */
export const formatSimilarity = (score: number): string => {
  return `${(score * 100).toFixed(0)}%`;
};

/**
 * Get color based on similarity score
 * 0-30%: Red, 30-70%: Yellow, 70-100%: Green
 */
export const getSimilarityColor = (
  score: number
): 'text-danger-red' | 'text-warning-yellow' | 'text-success-green' => {
  if (score < 0.3) return 'text-danger-red';
  if (score < 0.7) return 'text-warning-yellow';
  return 'text-success-green';
};

/**
 * Get background color for similarity meter
 */
export const getSimilarityBgColor = (
  score: number
): 'bg-red-500' | 'bg-yellow-500' | 'bg-green-500' => {
  if (score < 0.3) return 'bg-red-500';
  if (score < 0.7) return 'bg-yellow-500';
  return 'bg-green-500';
};

/**
 * Truncate text to a maximum length
 */
export const truncateText = (text: string, maxLength: number): string => {
  if (text.length <= maxLength) return text;
  return text.substring(0, maxLength) + '...';
};

/**
 * Format probe name to a human-readable format
 * "philosophical_depth" -> "Philosophical Depth"
 */
export const formatProbeName = (probeName: string): string => {
  return probeName
    .split('_')
    .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
    .join(' ');
};

/**
 * Get confidence level based on similarity score
 */
export const getConfidenceLevel = (score: number): string => {
  if (score >= 0.9) return 'Very High';
  if (score >= 0.8) return 'High';
  if (score >= 0.7) return 'Good';
  if (score >= 0.5) return 'Moderate';
  return 'Low';
};
