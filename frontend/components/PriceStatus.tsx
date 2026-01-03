"use client";

type Props = { 
  symbol?: string;
  price: number; 
  divergence?: "bullish" | "bearish" | "neutral"; 
};

export default function PriceStatus({ symbol = "BTC", price, divergence = "neutral" }: Props) {
  const divergenceColor =
    divergence === "bullish" ? "text-emerald-400" : divergence === "bearish" ? "text-red-400" : "text-yellow-400";

  const divergenceLabel =
    divergence === "bullish" ? "Bullish Divergence" : divergence === "bearish" ? "Bearish Divergence" : "Neutral";

  return (
    <div className="glass-panel rounded-2xl p-6 h-full flex flex-col justify-center">
      <div className="text-xs font-bold text-foreground/60 uppercase tracking-widest mb-6">Price & Status</div>
      <div className="space-y-5">
        <div>
          <div className="text-xs text-foreground/60 mb-2 uppercase tracking-widest">{symbol}</div>
          <div className="text-4xl md:text-5xl font-bold text-foreground">${price.toFixed(2)}</div>
        </div>
        <div className={`text-xs font-semibold uppercase tracking-wider ${divergenceColor}`}>{divergenceLabel}</div>
      </div>
    </div>
  );
}
