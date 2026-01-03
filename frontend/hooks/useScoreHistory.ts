"use client";

import { useState, useCallback, useRef } from "react";

export interface ScoreHistoryPoint {
  time: string;
  score: number;
}

export const useScoreHistory = () => {
  const [scoreHistory, setScoreHistory] = useState<ScoreHistoryPoint[]>([]);
  const historyRef = useRef<ScoreHistoryPoint[]>([]);
  const lastUpdateRef = useRef<number>(0);

  const addScore = useCallback((score: number) => {
    const now = Date.now();
    // Throttle updates to max once per 500ms to reduce re-renders
    if (now - lastUpdateRef.current < 500) return;
    
    const timeStr = new Date(now).toLocaleTimeString("en-US", { hour12: false });
    const point: ScoreHistoryPoint = { time: timeStr, score };

    historyRef.current.push(point);
    if (historyRef.current.length > 60) {
      historyRef.current = historyRef.current.slice(-60);
    }

    lastUpdateRef.current = now;
    setScoreHistory([...historyRef.current]);
  }, []);

  return { scoreHistory, addScore };
};
