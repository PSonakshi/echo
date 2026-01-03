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

function TrendChart({ data, expanded }: TrendChartProps) {
  if (!data || data.length === 0) {
    return (
      <div className={`glass-panel ${expanded ? "h-full" : "h-[400px]"} flex items-center justify-center text-muted-foreground rounded-2xl`}>
        Initializing stream...
      </div>
    );
  }

  // Downsample if dataset is large to improve render performance
  const maxPoints = expanded ? 2000 : 800;
  const displayData = useMemo(() => {
    if (!data || data.length <= maxPoints) return data;
    const step = Math.ceil(data.length / maxPoints);
    return data.filter((_, i) => i % step === 0);
  }, [data, maxPoints]);

  return (
    <div className={`glass-panel p-6 rounded-2xl ${expanded ? "h-full" : "h-[400px]"} flex flex-col overflow-hidden`}>
      {/* Header */}
      <div className="mb-4 flex items-center justify-between">
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
            <span className="w-2 h-2 rounded-full" style={{ backgroundColor: "var(--price-color)" }} />
            Price
          </span>
          <span className="flex items-center gap-2">
            <span className="w-2 h-2 rounded-full opacity-80" style={{ backgroundColor: "var(--sentiment-color)" }} />
            Sentiment
          </span>
        </div>
      </div>

      {/* Chart Container — responsive to expanded */}
      <div className={`relative w-full ${expanded ? "h-full" : "h-[260px]"}`}>
        <ResponsiveContainer width="100%" height="100%">
          <ComposedChart data={displayData}>
            <defs>
              <linearGradient id="sentimentGradient" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="var(--sentiment-color)" stopOpacity={0.22} />
                <stop offset="95%" stopColor="var(--sentiment-color)" stopOpacity={0} />
              </linearGradient>
            </defs>

            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.03)" vertical={false} />

            <XAxis
              dataKey="time"
              stroke="rgba(234,241,229,0.45)"
              tick={{ fill: "rgba(234,241,229,0.45)", fontSize: 10, fontFamily: "Open Sauce One" }}
              tickLine={false}
              axisLine={false}
              tickFormatter={(t: string) => t.slice(0, 5)}
            />

            <YAxis
              yAxisId="left"
              stroke="var(--accent-color)"
              tick={{ fill: "var(--accent-color)", fontSize: 10, fontFamily: "Open Sauce One" }}
              tickLine={false}
              axisLine={false}
              width={40}
            />

            <YAxis
              yAxisId="right"
              orientation="right"
              domain={[-1.5, 1.5]}
              stroke="var(--highlight-color)"
              tick={{ fill: "var(--highlight-color)", fontSize: 10, fontFamily: "Open Sauce One" }}
              tickLine={false}
              axisLine={false}
              width={40}
            />

            {
              /* Custom tooltip to show time and percent change for price */
            }
            <Tooltip
              content={({ active, payload, label }: any) => {
                if (!active || !payload || payload.length === 0) return null;
                // find price and sentiment values from payload
                const priceEntry = payload.find((p: any) => p.dataKey === "price");
                const sentimentEntry = payload.find((p: any) => p.dataKey === "sentiment");
                const priceVal = priceEntry ? priceEntry.value : null;
                const sentimentVal = sentimentEntry ? sentimentEntry.value : null;

                // compute percent change using displayData accessible via closure
                const idx = displayData.findIndex((d) => d.time === label);
                let pctText = "N/A";
                if (idx > 0) {
                  const prev = displayData[idx - 1];
                  if (prev && prev.price && priceVal != null) {
                    const pct = ((priceVal - prev.price) / prev.price) * 100;
                    pctText = `${pct >= 0 ? "+" : ""}${pct.toFixed(2)}%`;
                  }
                }

                return (
                  <div style={{ backgroundColor: "rgba(12,12,16,0.92)", border: "1px solid rgba(234,241,229,0.06)", borderRadius: 10, padding: 8, minWidth: 160 }}>
                    <div style={{ color: "var(--accent-color)", fontFamily: "Open Sauce One", fontSize: 12, marginBottom: 6 }}>{label}</div>
                    {priceVal != null && (
                      <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 4 }}>
                        <div style={{ color: "var(--price-color)", fontFamily: "Open Sauce One", fontSize: 13 }}>Price</div>
                        <div style={{ color: "var(--text-color)", fontFamily: "Open Sauce One", fontSize: 13 }}>${priceVal.toFixed(2)}</div>
                      </div>
                    )}

                    {sentimentVal != null && (
                      <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 4 }}>
                        <div style={{ color: "var(--sentiment-color)", fontFamily: "Open Sauce One", fontSize: 13 }}>Sentiment</div>
                        <div style={{ color: "var(--text-color)", fontFamily: "Open Sauce One", fontSize: 13 }}>{sentimentVal.toFixed(2)}</div>
                      </div>
                    )}

                    <div style={{ display: "flex", justifyContent: "space-between", marginTop: 6 }}>
                      <div style={{ color: "rgba(234,241,229,0.6)", fontFamily: "Open Sauce One", fontSize: 12 }}>Change</div>
                      <div style={{ color: "var(--price-color)", fontFamily: "Open Sauce One", fontSize: 12 }}>{pctText}</div>
                    </div>
                  </div>
                );
              }}
            />


            <Area
              yAxisId="right"
              dataKey="sentiment"
              type="basis"
              fill="url(#sentimentGradient)"
              stroke="var(--sentiment-color)"
              strokeWidth={expanded ? 1.5 : 1}
              isAnimationActive={false}
              strokeLinejoin="round"
              strokeLinecap="round"
            />

            <Line
              yAxisId="left"
              dataKey="price"
              type="basis"
              stroke="var(--price-color)"
              strokeWidth={expanded ? 3.5 : 2.5}
              dot={false}
              activeDot={{ r: expanded ? 8 : 6, fill: "var(--price-color)" }}
              isAnimationActive={false}
              strokeLinejoin="round"
              strokeLinecap="round"
            />
          </ComposedChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}

export default React.memo(TrendChart);
