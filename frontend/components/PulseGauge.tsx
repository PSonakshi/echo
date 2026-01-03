"use client";

import { Activity } from "lucide-react";
import { cn } from "@/lib/utils";

interface PulseGaugeProps {
  score: number;
}

export default function PulseGauge({ score }: PulseGaugeProps) {
  let colorClass = "text-paleYellow border-paleYellow";
  let bgClass = "bg-paleYellow/10";
  let status = "NEUTRAL";

  if (score >= 7) {
    colorClass = "text-mintGreen border-mintGreen";
    bgClass = "bg-mintGreen/10";
    status = "HYPE";
  } else if (score <= 3) {
    colorClass = "text-softPeach border-softPeach";
    bgClass = "bg-softPeach/10";
    status = "FUD";
  }

  return (
    <div className="glass-panel rounded-2xl p-6 flex flex-col items-center justify-center relative overflow-hidden h-full min-h-[240px] transition-all duration-500 hover:border-opacity-30 hover:border-white">
      <div
        className={cn(
          "absolute inset-0 opacity-20 blur-2xl transition-colors duration-1000",
          bgClass
        )}
      />

      <div className="flex items-center gap-2 mb-4 z-10">
        <span className="text-lavender">
          <Activity className="w-5 h-5" />
        </span>
        <h3 className="text-sm font-semibold text-warmBeige/70 uppercase tracking-wider font-mono">
          Pulse Score
        </h3>
      </div>

      <div className="relative z-10 flex flex-col items-center">
        <div
          className={cn(
            "text-7xl font-bold font-mono tracking-tighter transition-all duration-500 drop-shadow-lg",
            colorClass
          )}
        >
          {score}
        </div>

        <div
          className={cn(
            "mt-4 px-4 py-1.5 rounded-full text-xs font-bold border backdrop-blur-md uppercase tracking-widest transition-colors duration-500",
            colorClass,
            bgClass
          )}
        >
          {status}
        </div>
      </div>

      <div
        className={cn(
          "absolute -bottom-6 -right-6 opacity-10 rotate-12 transition-colors duration-500",
          colorClass
        )}
      >
        <svg width="120" height="120" viewBox="0 0 24 24" fill="currentColor">
          <path d="M13 2L3 14h9l-1 8 10-12h-9l1-8z" />
        </svg>
      </div>
    </div>
  );
}