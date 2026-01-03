"use client";

import { useEffect, useState } from "react";
import { TrendingUp } from "lucide-react";
import PulseGauge from "@/components/PulseGauge";
import TrendChart from "@/components/TrendChart";
import NarrativeCloud from "@/components/NarrativeCloud";
import ChatInterface from "@/components/ChatInterface";
import { usePulseData } from "@/hooks/usePulseData";

export default function Dashboard() {
  const { data, metrics } = usePulseData();
  const [expanded, setExpanded] = useState<"chart" | "chat" | null>(null);

  return (
    <main className="min-h-screen px-4 md:px-8 space-y-8 max-w-[1600px] mx-auto relative">
      <div className="fixed inset-0 pointer-events-none -z-10 bg-[radial-gradient(ellipse_at_top,_var(--tw-gradient-stops))] from-softBlue/10 via-background to-background"></div>

      {/* Header */}
      <header className="mb-10 flex flex-col md:flex-row md:items-end justify-between gap-4">
        <div>
          <h1 className="text-4xl md:text-6xl font-bold bg-gradient-to-r from-softBlue via-lavender to-lightSky bg-clip-text text-transparent inline-block tracking-tight">
            NARRATIVE PULSE
          </h1>
          <p className="text-warmBeige/60 mt-2 font-light tracking-widest uppercase text-sm">
            Live Sentiment Stream Active
          </p>
        </div>
        <div className="glass-panel px-4 py-2 rounded-lg flex gap-4 text-xs font-mono text-muted-foreground">
          <span>BTC: $98,420</span>
          <span className="text-mintGreen">ETH: $4,200</span>
        </div>
      </header>
{/* Grid Layout */}
<div className="flex gap-8 min-h-0">
  
  {/* Left Column */}
  <div className="flex-1 flex flex-col gap-7 min-h-0">
    <PulseGauge score={metrics.score} />

    <div className="glass-panel rounded-2xl bg-gradient-to-br from-lavender/5 to-transparent relative overflow-hidden group min-h-[180px] flex flex-col justify-center">
      <div className="p-8">
        <div className="absolute top-0 right-10 left-10 p-4 opacity-10 group-hover:opacity-20 transition-opacity">
          <TrendingUp className="w-12 h-12" />
        </div>

        <h3 className="text-xs font-bold text-muted-foreground mb-1 uppercase tracking-widest">
          Influencer Consensus
        </h3>

        <div
          className={`text-3xl font-bold ${
            metrics.consensus === "Bullish"
              ? "text-mintGreen"
              : metrics.consensus === "Bearish"
              ? "text-softPeach"
              : "text-foreground"
          }`}
        >
          {metrics.consensus}
        </div>
      </div>
    </div>

    {/* Narrative Cloud MUST have height */}
    <div className="flex-1 min-h-[220px] min-h-0">
      <NarrativeCloud phrases={metrics.phrases} />
    </div>
  </div>

  {/* Right Column (50%) */}
  <div className="w-1/2 flex flex-col gap-6 min-h-0">

    {/* Trend Chart wrapper with fixed height */}
    <div
      className="glass-panel rounded-2xl p-6 h-[420px] min-h-[300px] flex flex-col cursor-pointer hover:scale-[1.01] transition-transform"
      onClick={() => setExpanded("chart")}
    >
      <TrendChart data={data} expanded={false} />
    </div>

    {/* Chat grows naturally */}
    <div className="flex-1 min-h-0 cursor-pointer" onClick={() => setExpanded("chat")}>
      <ChatInterface expanded={false} />
    </div>
  </div>
</div>

      {expanded && (
        <div className="fixed inset-0 z-50 flex items-center justify-center">
          <div
            className="absolute inset-0 bg-black/50 backdrop-blur-sm"
            onClick={() => setExpanded(null)}
          />

          <div className="glass-panel z-60 w-[70%] h-[70%] rounded-2xl p-6 overflow-auto relative">
            <button
              onClick={() => setExpanded(null)}
              className="absolute top-4 right-4 bg-white/5 text-foreground p-2 rounded-md hover:bg-white/10"
            >
              Close
            </button>

            <div className="h-full w-full">
              {expanded === "chart" ? (
                <div className="w-full h-full flex flex-col">
                  <TrendChart data={data} expanded={true} />
                </div>
              ) : (
                <div className="w-full h-full flex flex-col">
                  <ChatInterface expanded={true} />
                </div>
              )}
            </div>
          </div>
        </div>
      )}
    </main>
  );
}
