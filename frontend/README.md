# Explainable Cinema Frontend

A Vite + React frontend for the explainable movie recommendation system.

## Prerequisites

- Node.js 18+
- npm
- Backend API running at [http://localhost:8000](http://localhost:8000)

## Setup

1. Install dependencies:

   npm install

2. Configure environment in .env.local:

   `VITE_API_BASE_URL=http://localhost:8000`

3. Start development server:

   npm run dev

4. Open the app:

   [http://localhost:3000](http://localhost:3000)

## Available Scripts

- npm run dev: start local dev server
- npm run build: type-check and production build
- npm run preview: preview production build locally

## Project Structure

src/

- components/: UI components
- hooks/: data-fetching and interaction hooks
- lib/: API client, state store, shared types, utilities
- styles/: global Tailwind styles

## API Endpoints Used

- /health
- /search
- /movie/{id}/explain
- /recommendations/explain
- /clusters/info

## Notes

- This app is configured for localhost development.
- Frontend runs on port 3000.
- Backend CORS must allow [http://localhost:3000](http://localhost:3000).
