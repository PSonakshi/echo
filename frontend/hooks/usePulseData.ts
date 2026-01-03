"use client";

import { useState, useEffect, useRef } from "react";

export interface PulseDataPoint {
  time: string;
  price: number;
  sentiment: number;
}

export interface PulseMetrics {
  score: number;
  phrases: string[];
  consensus: "Bullish" | "Bearish" | "Neutral";
}

export const usePulseData = () => {
  const [data, setData] = useState<PulseDataPoint[]>([]);
  const [metrics, setMetrics] = useState<PulseMetrics>({
    score: 5.0,
    phrases: ["waiting for signal..."],
    consensus: "Neutral",
  });

  useEffect(() => {
    // Use a ref to accumulate all generated points at the emission rate,
    // but update React state at a lower/higher frequency to avoid too many re-renders.
    const dataRef = { current: [] as PulseDataPoint[] };

    const emitter = setInterval(() => {
      const now = new Date();
      const timeStr = now.toLocaleTimeString("en-US", { hour12: false });

      const lastPrice = dataRef.current.length > 0 ? dataRef.current[dataRef.current.length - 1].price : 100;
      const volatility = (Math.random() - 0.5) * 5;
      const newPrice = Math.max(50, lastPrice + volatility);

      let sentiment = (Math.random() - 0.5) * 2; // -1 to 1
      if (newPrice > lastPrice) sentiment += 0.3;

      const newPoint = {
        time: timeStr,
        price: parseFloat(newPrice.toFixed(2)),
        sentiment: parseFloat(sentiment.toFixed(2)),
      };

      dataRef.current = [...dataRef.current, newPoint];
      if (dataRef.current.length > 300) {
        dataRef.current = dataRef.current.slice(dataRef.current.length - 300);
      }

      // Update metrics less frequently (every emitter tick is fine here)
      setMetrics((prev) => {
        const newScore = Math.max(1, Math.min(10, prev.score + (Math.random() - 0.5)));

        const allPhrases = [
          "moon soon",
          "dev based",
          "chart cooking",
          "fud detected",
          "accumulation",
          "breakout",
          "liquidity grab",
          "wagmi",
          "ngmi",
          "send it",
        ];
        const shuffled = allPhrases.sort(() => 0.5 - Math.random());

        return {
          score: parseFloat(newScore.toFixed(1)),
          phrases: shuffled.slice(0, 5),
          consensus: newScore > 6 ? "Bullish" : newScore < 4 ? "Bearish" : "Neutral",
        };
      });
    }, 500);

    // UI update interval — control how often React state updates (smoothness vs freshness)
    const uiInterval = setInterval(() => {
      // shallow copy to trigger state update only at this cadence
      setData((_) => [...dataRef.current]);
    }, 200);

    return () => {
      clearInterval(emitter);
      clearInterval(uiInterval);
    };
  }, []);

  return { data, metrics };
};