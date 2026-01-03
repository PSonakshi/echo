"use client";

type Props = { score: number };

export default function PulseScore({ score }: Props) {
  const scoreColor =
    score > 7 ? "text-emerald-400" : score > 4 ? "text-yellow-400" : "text-red-400";

  return (
    <div className="glass-panel rounded-2xl p-6 h-full flex flex-col items-center justify-center text-center">
      <div className="text-xs font-bold text-foreground/60 uppercase tracking-widest mb-4">Pulse Score</div>
      <div className={`text-6xl md:text-7xl font-extrabold ${scoreColor} transition-colors duration-300 mb-2`}>{score.toFixed(1)}</div>
      <div className="text-xs md:text-sm text-foreground/50">
        {score > 7 ? "🚀 Bullish" : score > 4 ? "⚖️ Neutral" : "🔴 Bearish"}
      </div>
    </div>
  );
}
