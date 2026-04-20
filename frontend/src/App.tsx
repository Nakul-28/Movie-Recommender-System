import React, { useEffect, useState } from 'react';
import { Routes, Route, Navigate, useLocation } from 'react-router-dom';
import { AnimatePresence } from 'framer-motion';
import { apiClient } from '@/lib/api';
import HomePage from '@/pages/HomePage';
import MoviePage from '@/pages/MoviePage';

function App() {
  const location = useLocation();

  return (
    <div className="min-h-screen bg-[#0a0a0f] text-white flex flex-col">
      {/* AnimatePresence for smooth route transitions */}
      <div className="flex-1">
        <AnimatePresence mode="wait">
          <Routes location={location} key={location.pathname}>
            <Route path="/home" element={<HomePage />} />
            <Route path="/movies/:id" element={<MoviePage />} />
            <Route path="*" element={<Navigate to="/home" replace />} />
          </Routes>
        </AnimatePresence>
      </div>
    </div>
  );
}

export default App;
