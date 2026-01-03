"use client";

import { useEffect, useRef, useState } from "react";
import {
  TrendingUp,
  TrendingDown,
  ChevronDown,
  Maximize2,
} from "lucide-react";

import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  ReferenceArea,
} from "recharts";

import PulseGauge from "@/components/PulseGauge";
import TrendChart from "@/components/TrendChart";
import NarrativeCloud from "@/components/NarrativeCloud";
import ChatInterface from "@/components/ChatInterface";
import PulseScoreHistory from "@/components/PulseScoreHistory";
import TrendingPhrases from "@/components/TrendingPhrases";
import PriceStatus from "@/components/PriceStatus";

import { usePulseData } from "@/hooks/usePulseData";
import { useScoreHistory } from "@/hooks/useScoreHistory";

/* ---------------- Coins ---------------- */
const COINS = [
  { symbol: "BTC", name: "Bitcoin", base: 98420 },
  { symbol: "ETH", name: "Ethereum", base: 4200 },
  { symbol: "SOL", name: "Solana", base: 185 },
  { symbol: "AVAX", name: "Avalanche", base: 42 },
];

export default function Dashboard() {
  const { data, metrics } = usePulseData();
  const { scoreHistory, addScore } = useScoreHistory();

  const [expanded, setExpanded] =
    useState<"market" | "chat" | "score" | null>(null);

  const [coin, setCoin] = useState(COINS[0]);
  const [price, setPrice] = useState(coin.base);
  const [divergence, setDivergence] =
    useState<"bullish" | "bearish" | "neutral">("neutral");

  const [coinMenuOpen, setCoinMenuOpen] = useState(false);
  const lastScoreRef = useRef<number | null>(null);

  /* ---------------- Pulse Loop ---------------- */
  useEffect(() => {
    const interval = setInterval(() => {
      if (lastScoreRef.current !== metrics.score) {
        addScore(metrics.score);
        lastScoreRef.current = metrics.score;
      }

      setPrice((p) => {
        const delta = (Math.random() - 0.5) * coin.base * 0.002;
        return parseFloat((p + delta).toFixed(2));
      });

      if (metrics.score > 6.5) setDivergence("bullish");
      else if (metrics.score < 3.5) setDivergence("bearish");
      else setDivergence("neutral");
    }, 1000);

    return () => clearInterval(interval);
  }, [metrics.score, addScore, coin.base]);

  /* ---------------- Score Data ---------------- */
  const scores = scoreHistory.map((s, i) => ({
    t: i,
    score: s.score,
  }));

  const last = scores.at(-1)?.score ?? 0;
  const prev = scores.at(-2)?.score ?? last;
  const trendUp = last >= prev;

  return (
    <main className="min-h-screen px-4 md:px-8 space-y-8 max-w-[1600px] mx-auto relative">
      <div className="fixed inset-0 -z-10 bg-[radial-gradient(ellipse_at_top,_var(--tw-gradient-stops))] from-softBlue/10 via-background to-background" />

      {/* ================= HEADER ================= */}
      <header className="mb-10 flex flex-col md:flex-row justify-between gap-4">
        <div>
          <h1 className="text-4xl md:text-6xl font-bold bg-gradient-to-r from-softBlue via-lavender to-lightSky bg-clip-text text-transparent">
            NARRATIVE PULSE
          </h1>
          <p className="text-warmBeige/60 mt-2 tracking-widest uppercase text-sm">
            Live Sentiment Stream Active
          </p>
        </div>
      </header>

      {/* ================= TOP GRID ================= */}
<div className="grid grid-cols-1 md:grid-cols-4 gap-5 items-stretch">
  {/* Pulse */}
  <div onClick={() => setExpanded("score")} className="cursor-pointer group relative">
    <PulseGauge score={metrics.score} />
    <div className="absolute top-4 right-4 opacity-0 group-hover:opacity-100 transition-opacity">
      <Maximize2 size={16} className="text-emerald-400" />
    </div>
  </div>

  {/* Trending phrases */}
  <TrendingPhrases phrases={metrics.phrases} />

  {/* Coin selector + price */}
  <div className="glass-panel rounded-2xl p-4 flex flex-col justify-between relative">
    <button
      onClick={() => setCoinMenuOpen(!coinMenuOpen)}
      className="text-left hover:opacity-80 transition-opacity"
    >
      <div className="text-sm font-mono text-muted-foreground">
        {coin.name}
      </div>
      <div className="text-2xl font-bold mt-2">
        {coin.symbol}
      </div>
    </button>

    {coinMenuOpen && (
      <div className="absolute top-full left-0 right-0 mt-2 glass-panel rounded-xl overflow-hidden z-30">
        {COINS.map((c) => (
          <button
            key={c.symbol}
            onClick={() => {
              setCoin(c);
              setPrice(c.base);
              setCoinMenuOpen(false);
            }}
            className={`w-full px-4 py-3 text-left text-sm hover:bg-white/10 transition-colors ${
              c.symbol === coin.symbol ? "bg-emerald-400/20" : ""
            }`}
          >
            <div className="font-semibold">{c.symbol}</div>
            <div className="text-xs text-muted-foreground">{c.name}</div>
          </button>
        ))}
      </div>
    )}

    <div className="text-xl font-bold mt-1">
      ${price.toLocaleString()}
    </div>
  </div>

  {/* Price status */}
  <PriceStatus
    symbol={coin.symbol}
    price={price}
    divergence={divergence}
  />
</div>


      {/* ================= SCORE HISTORY (STATIC & GRADUAL) ================= */}
      <div className="glass-panel rounded-2xl p-6 h-80 flex flex-col gap-4">
        <div className="flex items-center justify-between">
          <h3 className="text-xs font-bold uppercase tracking-widest text-foreground/60">
            Score History
          </h3>

          <div className="flex items-center gap-1 text-sm text-mintGreen">
            {trendUp ? (
              <TrendingUp size={16} />
            ) : (
              <TrendingDown size={16} />
            )}
            {last.toFixed(2)}
          </div>
        </div>

        {/* Chart */}
        <div className="h-40">
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart data={scores}>
              <defs>
                <linearGradient id="staticScoreGradient" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor="#34d399" stopOpacity={0.35} />
                  <stop offset="100%" stopColor="#34d399" stopOpacity={0.05} />
                </linearGradient>
              </defs>

              {/* Subtle context bands */}
              <ReferenceArea y1={0} y2={3.5} fill="#ffffff" fillOpacity={0.03} />
              <ReferenceArea y1={3.5} y2={6.5} fill="#ffffff" fillOpacity={0.04} />
              <ReferenceArea y1={6.5} y2={10} fill="#34d399" fillOpacity={0.05} />

              <XAxis hide />
              <YAxis domain={[0, 10]} hide />

              <Tooltip
                contentStyle={{
                  background: "#0b1220",
                  border: "none",
                  fontSize: 12,
                }}
              />

              <Area
                type="monotone"
                dataKey="score"
                stroke="#34d399"
                fill="url(#staticScoreGradient)"
                strokeWidth={2}
                dot={false}
                isAnimationActive={false}
              />
            </AreaChart>
          </ResponsiveContainer>
        </div>

        {/* Scrollable log */}
        <div className="flex-1 overflow-y-auto pr-2">
          <PulseScoreHistory history={scoreHistory} />
        </div>
      </div>

      {/* ================= MAIN GRID ================= */}
      <div className="flex flex-col lg:flex-row gap-8">
        <div className="flex-1 flex flex-col gap-7">
          <div className="glass-panel rounded-2xl min-h-[180px] p-8 relative">
            <TrendingUp className="absolute top-6 right-6 opacity-10" />
            <h3 className="text-xs uppercase tracking-widest text-muted-foreground">
              Influencer Consensus
            </h3>
            <div className="text-3xl font-bold mt-2">
              {metrics.consensus}
            </div>
          </div>

          <div className="glass-panel rounded-2xl p-6 h-[420px] relative group flex flex-col">
            <button
              onClick={() => setExpanded("market")}
              className="absolute top-4 right-4 opacity-0 group-hover:opacity-100 transition-opacity z-10"
            >
              <Maximize2 size={16} />
            </button>

            <h3 className="text-xs uppercase tracking-widest mb-3 text-muted-foreground flex-shrink-0">
              Market Analysis
            </h3>

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 flex-1 min-h-0">
              <div className="min-h-0">
                <TrendChart data={data} expanded={false} />
              </div>
              <div className="min-h-0">
                <NarrativeCloud phrases={metrics.phrases} />
              </div>
            </div>
          </div>
        </div>

        <div className="w-full lg:w-[420px]">
          <div
            onClick={() => setExpanded("chat")}
            className="glass-panel rounded-2xl h-full cursor-pointer hover:scale-[1.01]"
          >
            <ChatInterface expanded={false} />
          </div>
        </div>
      </div>
      {/* ================= EXPANDED MODALS ================= */}
      {expanded && (
        <div className="fixed inset-0 z-50 flex items-center justify-center">
          <div
            className="absolute inset-0 bg-black/60 backdrop-blur-sm"
            onClick={() => setExpanded(null)}
          />

          <div className="glass-panel z-60 w-[90vw] h-[90vh] rounded-2xl p-8 overflow-auto relative">
            <button
              onClick={() => setExpanded(null)}
              className="absolute top-6 right-6 bg-white/10 hover:bg-white/20 text-foreground p-3 rounded-lg transition-colors text-sm font-semibold"
            >
              Close
            </button>

            {/* PULSE SCORE EXPANDED */}
            {expanded === "score" && (
              <div className="flex flex-col gap-8">
                <div>
                  <h2 className="text-5xl font-bold mb-2">Pulse Score Analysis</h2>
                  <p className="text-foreground/60">Real-time sentiment and market pulse metrics</p>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
                  {/* Main Score */}
                  <div className="glass-panel rounded-2xl p-8 flex flex-col items-center justify-center">
                    <div className="text-sm text-foreground/60 mb-4">Current Score</div>
                    <div className={`text-7xl font-extrabold ${
                      metrics.score > 7 ? 'text-emerald-400' : metrics.score > 4 ? 'text-yellow-400' : 'text-red-400'
                    }`}>
                      {metrics.score.toFixed(1)}
                    </div>
                    <div className="text-sm mt-4 text-center">
                      {metrics.score > 7 ? '🚀 Bullish' : metrics.score > 4 ? '⚖️ Neutral' : '🔴 Bearish'}
                    </div>
                  </div>

                  {/* FUD */}
                  <div className="glass-panel rounded-2xl p-8 flex flex-col justify-center">
                    <div className="text-sm text-red-400/60 mb-4 uppercase tracking-widest font-bold">FUD Level</div>
                    <div className="text-xs text-foreground/50">Negative sentiment and fear indicators</div>
                  </div>

                  {/* NEUTRAL */}
                  <div className="glass-panel rounded-2xl p-8 flex flex-col justify-center">
                    <div className="text-sm text-yellow-400/60 mb-4 uppercase tracking-widest font-bold">Neutral</div>
                    <div className="text-xs text-foreground/50">Balanced market sentiment</div>
                  </div>
                </div>

                {/* HYPE */}
                <div className="glass-panel rounded-2xl p-8 flex flex-col justify-center">
                  <div className="text-sm text-emerald-400/60 mb-4 uppercase tracking-widest font-bold">Hype & FOMO</div>
                  <div className="text-sm text-foreground/50">Bullish sentiment, positive indicators, and hype metrics</div>
                </div>
              </div>
            )}

            {/* MARKET ANALYSIS EXPANDED */}
            {expanded === "market" && (
              <div className="flex flex-col gap-8 h-full">
                <div className="flex-shrink-0">
                  <h2 className="text-5xl font-bold mb-2">Market Analysis</h2>
                  <p className="text-foreground/60">Comprehensive {coin.symbol} market insights and trend analysis</p>
                </div>

                <div className="flex-1 grid grid-cols-1 lg:grid-cols-2 gap-6 min-h-0">
                  <div className="glass-panel rounded-2xl p-6 flex flex-col min-h-0">
                    <h3 className="text-sm font-bold uppercase tracking-widest mb-4 text-foreground/60 flex-shrink-0">Price Action</h3>
                    <div className="flex-1 min-h-0">
                      <TrendChart data={data} expanded={true} />
                    </div>
                  </div>
                  <div className="glass-panel rounded-2xl p-6 flex flex-col min-h-0">
                    <h3 className="text-sm font-bold uppercase tracking-widest mb-4 text-foreground/60 flex-shrink-0">Narrative Cloud</h3>
                    <div className="flex-1 min-h-0">
                      <NarrativeCloud phrases={metrics.phrases} />
                    </div>
                  </div>
                </div>
              </div>
            )}

            {/* CHAT EXPANDED */}
            {expanded === "chat" && (
              <div className="flex flex-col h-full">
                <div className="mb-6">
                  <h2 className="text-5xl font-bold mb-2">AI Assistant</h2>
                  <p className="text-foreground/60">Real-time market insights and sentiment analysis</p>
                </div>
                <div className="flex-1 min-h-0">
                  <ChatInterface expanded={true} />
                </div>
              </div>
            )}
          </div>
        </div>
      )}
      
    </main>
  );
}
