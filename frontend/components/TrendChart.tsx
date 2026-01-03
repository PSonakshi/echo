"use client";

import React, { useMemo } from "react";
import { TrendingUp } from "lucide-react";
import {
  Line,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ComposedChart,
} from "recharts";
import type { PulseDataPoint } from "@/hooks/usePulseData";

interface TrendChartProps {
  data: PulseDataPoint[];
  expanded?: boolean;
}

const TrendChart = React.memo(function TrendChart({ data, expanded }: TrendChartProps) {
  // Downsample if dataset is large to improve render performance
  const displayData = useMemo(() => {
    if (!data || data.length === 0) return [];
    
    const maxPoints = expanded ? 1000 : 400; // Reduced for better performance
    if (data.length <= maxPoints) return data;
    
    const step = Math.ceil(data.length / maxPoints);
    return data.filter((_, i) => i % step === 0);
  }, [data, expanded]);

  if (!displayData.length) {
    return (
      <div className={`${expanded ? "h-full" : "h-[320px]"} flex items-center justify-center text-muted-foreground`}>
        Initializing stream...
      </div>
    );
  }

  return (
    <div className={`${expanded ? "h-full" : "h-[320px]"} flex flex-col min-h-0`}>
      {/* Header */}
      <div className="mb-4 flex items-center justify-between flex-shrink-0">
        <h3 className="text-lg font-semibold text-foreground flex items-center gap-2">
          <span className="text-lightSky">
            <TrendingUp className="w-5 h-5" />
          </span>
          Market Velocity
          <span className="text-muted-foreground text-xs font-mono ml-2 opacity-60">
            /// PRICE vs SENTIMENT
          </span>
        </h3>

        <div className="flex gap-4 text-xs font-mono">
          <span className="flex items-center gap-2">
            <span className="w-2 h-2 rounded-full bg-price" />
            Price
          </span>
          <span className="flex items-center gap-2">
            <span className="w-2 h-2 rounded-full bg-sentiment opacity-80" />
            Sentiment
          </span>
        </div>
      </div>

      {/* Chart Container */}
      <div className="flex-1 min-h-0">
        <ResponsiveContainer width="100%" height="100%">
          <ComposedChart data={displayData}>
            <defs>
              <linearGradient id="sentimentGradient" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#5B0779" stopOpacity={0.22} />
                <stop offset="95%" stopColor="#5B0779" stopOpacity={0} />
              </linearGradient>
            </defs>

            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.03)" vertical={false} />

            <XAxis
              dataKey="time"
              stroke="rgba(234,241,229,0.45)"
              tick={{ fill: "rgba(234,241,229,0.45)", fontSize: 10 }}
              tickLine={false}
              axisLine={false}
              tickFormatter={(t: string) => t.slice(0, 5)}
            />

            <YAxis
              yAxisId="left"
              stroke="#525ECD"
              tick={{ fill: "#525ECD", fontSize: 10 }}
              tickLine={false}
              axisLine={false}
              width={40}
            />

            <YAxis
              yAxisId="right"
              orientation="right"
              domain={[-1.5, 1.5]}
              stroke="#5B0779"
              tick={{ fill: "#5B0779", fontSize: 10 }}
              tickLine={false}
              axisLine={false}
              width={40}
            />

            <Tooltip
              content={({ active, payload, label }: any) => {
                if (!active || !payload || payload.length === 0) return null;
                
                const priceEntry = payload.find((p: any) => p.dataKey === "price");
                const sentimentEntry = payload.find((p: any) => p.dataKey === "sentiment");
                const priceVal = priceEntry ? priceEntry.value : null;
                const sentimentVal = sentimentEntry ? sentimentEntry.value : null;

                // Calculate percent change
                const idx = displayData.findIndex((d) => d.time === label);
                let pctText = "N/A";
                if (idx > 0 && priceVal != null) {
                  const prev = displayData[idx - 1];
                  if (prev?.price) {
                    const pct = ((priceVal - prev.price) / prev.price) * 100;
                    pctText = `${pct >= 0 ? "+" : ""}${pct.toFixed(2)}%`;
                  }
                }

                return (
                  <div className="bg-background/95 backdrop-blur-sm border border-white/10 rounded-lg p-3 shadow-lg">
                    <p className="text-xs text-foreground/60 mb-1">{label}</p>
                    {priceVal && (
                      <div className="flex justify-between mb-1">
                        <span className="text-sm text-price">Price</span>
                        <span className="text-sm text-foreground font-mono">${priceVal.toFixed(2)}</span>
                      </div>
                    )}
                    {sentimentVal && (
                      <div className="flex justify-between mb-1">
                        <span className="text-sm text-sentiment">Sentiment</span>
                        <span className="text-sm text-foreground font-mono">{sentimentVal.toFixed(2)}</span>
                      </div>
                    )}
                    <div className="flex justify-between mt-2 pt-2 border-t border-white/10">
                      <span className="text-xs text-foreground/60">Change</span>
                      <span className="text-xs text-price font-mono">{pctText}</span>
                    </div>
                  </div>
                );
              }}
            />

            <Area
              yAxisId="right"
              type="monotone"
              dataKey="sentiment"
              stroke="#5B0779"
              fill="url(#sentimentGradient)"
              strokeWidth={1.5}
              dot={false}
              isAnimationActive={false}
            />

            <Line
              yAxisId="left"
              type="monotone"
              dataKey="price"
              stroke="#525ECD"
              strokeWidth={expanded ? 3 : 2}
              dot={false}
              activeDot={{ r: expanded ? 6 : 4, fill: "#525ECD" }}
              isAnimationActive={false}
            />
          </ComposedChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
});

export default TrendChart;