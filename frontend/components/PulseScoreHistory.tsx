"use client";

import { LineChart, Line, XAxis, YAxis, CartesianGrid, ResponsiveContainer, Tooltip } from "recharts";
import type { ScoreHistoryPoint } from "@/hooks/useScoreHistory";

type Props = { history: ScoreHistoryPoint[] };

export default function PulseScoreHistory({ history }: Props) {
  if (!history.length) return null;

  return (
    <div className="w-full h-full">
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={history} margin={{ top: 5, right: 10, left: -20, bottom: 5 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="rgba(234,241,229,0.08)" />
          <XAxis dataKey="time" tick={{ fontSize: 11, fill: "rgba(234,241,229,0.5)" }} />
          <YAxis domain={[0, 10]} tick={{ fontSize: 11, fill: "rgba(234,241,229,0.5)" }} />
          <Tooltip contentStyle={{ backgroundColor: "#2A2F4A", border: "1px solid rgba(234,241,229,0.06)", borderRadius: "8px" }} />
          <Line type="monotone" dataKey="score" stroke="#22C55E" strokeWidth={2} dot={false} isAnimationActive={false} />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
