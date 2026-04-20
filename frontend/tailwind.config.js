/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        'deep-blue': '#0B0F19',
        'dark-bg': '#111827',
        'card-bg': '#1A1F2E',
        'primary': '#E2E8F0',
        'secondary': '#94A3B8',
        'accent': '#6366F1',
        'accent-hover': '#4F46E5',
        'success': '#10B981',
        'danger': '#EF4444',
      }
    },
  },
  plugins: [],
}