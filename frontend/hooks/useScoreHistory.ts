"use client";

import { useState, useEffect, useRef } from "react";

export interface ScoreHistoryPoint {
  time: string;
  score: number;
}

export const useScoreHistory = () => {
  const [scoreHistory, setScoreHistory] = useState<ScoreHistoryPoint[]>([]);
  const historyRef = useRef<ScoreHistoryPoint[]>([]);

  const addScore = (score: number) => {
    const now = new Date();
    const timeStr = now.toLocaleTimeString("en-US", { hour12: false });
    const point: ScoreHistoryPoint = { time: timeStr, score };

    historyRef.current = [...historyRef.current, point];
    if (historyRef.current.length > 60) {
      historyRef.current = historyRef.current.slice(historyRef.current.length - 60);
    }

    setScoreHistory([...historyRef.current]);
  };

  return { scoreHistory, addScore };
};
