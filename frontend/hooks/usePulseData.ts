"use client";

import { useState, useEffect, useRef, useCallback } from "react";

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

  const dataRef = useRef<PulseDataPoint[]>([]);
  const lastUpdateRef = useRef<number>(0);
  const metricsUpdateRef = useRef<number>(0);

  const allPhrases = useRef([
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
  ]);

  useEffect(() => {
    const emitter = setInterval(() => {
      const now = Date.now();
      const timeStr = new Date(now).toLocaleTimeString("en-US", { hour12: false });

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

      dataRef.current.push(newPoint);
      if (dataRef.current.length > 300) {
        dataRef.current = dataRef.current.slice(-300);
      }

      // Update metrics less frequently (every 1000ms instead of 500ms)
      if (now - metricsUpdateRef.current > 1000) {
        metricsUpdateRef.current = now;
        setMetrics((prev) => {
          const newScore = Math.max(1, Math.min(10, prev.score + (Math.random() - 0.5)));
          const shuffled = [...allPhrases.current].sort(() => 0.5 - Math.random());

          return {
            score: parseFloat(newScore.toFixed(1)),
            phrases: shuffled.slice(0, 5),
            consensus: newScore > 6 ? "Bullish" : newScore < 4 ? "Bearish" : "Neutral",
          };
        });
      }
    }, 500);

    // UI update interval — reduced frequency for better performance
    const uiInterval = setInterval(() => {
      const now = Date.now();
      // Throttle UI updates to max once per 300ms
      if (now - lastUpdateRef.current > 300) {
        lastUpdateRef.current = now;
        setData([...dataRef.current]);
      }
    }, 300);

    return () => {
      clearInterval(emitter);
      clearInterval(uiInterval);
    };
  }, []);

  return { data, metrics };
};