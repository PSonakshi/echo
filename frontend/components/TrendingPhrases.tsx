"use client";

type Props = { phrases: string[] };

export default function TrendingPhrases({ phrases }: Props) {
  return (
    <div className="glass-panel rounded-2xl p-6 h-full flex flex-col">
      <h3 className="text-xs font-bold text-foreground/60 uppercase tracking-widest mb-4">Top Trending</h3>
      <div className="space-y-2 flex-1 overflow-y-auto">
        {phrases.slice(0, 5).map((phrase, idx) => (
          <div key={idx} className="flex items-center gap-2 p-2 rounded-lg bg-white/3 hover:bg-white/5 transition-colors">
            <span className="text-xs font-bold text-emerald-400 min-w-fit">#{idx + 1}</span>
            <span className="text-xs text-foreground/80 truncate">{phrase}</span>
          </div>
        ))}
      </div>
    </div>
  );
}
