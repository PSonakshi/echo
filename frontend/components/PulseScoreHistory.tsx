"use client";

import React, { useMemo } from "react";
import type { ScoreHistoryPoint } from "@/hooks/useScoreHistory";

type Props = { history: ScoreHistoryPoint[] };

const PulseScoreHistory = React.memo(function PulseScoreHistory({ history }: Props) {
  const recent = useMemo(() => {
    if (!history.length) return [];
    // Show last 15 entries in reverse (newest first)
    return [...history].reverse().slice(0, 15);
  }, [history]);

  if (!recent.length) return null;

  return (
    <div className="w-full space-y-1">
      {recent.map((point, idx) => (
        <div
          key={`${point.time}-${point.score}`}
          className="flex items-center justify-between text-xs p-2 rounded hover:bg-white/5 transition-colors"
        >
          <span className="text-foreground/60 font-mono">{point.time}</span>
          <span className={`font-semibold ${
            point.score > 6? 'text-emerald-400' :
            point.score > 3 ? 'text-yellow-400' :
            'text-red-400'
          }`}>
            {point.score.toFixed(1)}
          </span>
        </div>
      ))}
    </div>
  );
});

export default PulseScoreHistory;
