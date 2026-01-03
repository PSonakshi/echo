"use client";

import { MessageSquare } from "lucide-react";

interface NarrativeCloudProps {
  phrases: string[];
}

export default function NarrativeCloud({ phrases }: NarrativeCloudProps) {
  return (
    <div className="glass-panel p-6 rounded-2xl h-full flex flex-col">
      <div className="flex items-center gap-2 mb-6">
        <span className="text-lightSky">
          <MessageSquare className="w-5 h-5" />
        </span>
        <h3 className="text-lg font-semibold">Narrative Cloud</h3>
      </div>

      <div className="flex flex-wrap gap-3 content-start">
        {phrases.map((phrase, i) => (
          <div
            key={`${phrase}-${i}`}
            className="px-4 py-2 rounded-xl text-sm font-medium bg-white/5 border border-white/5 text-foreground hover:bg-white/10 hover:border-lightSky/30 transition-all duration-300 cursor-default animate-pulse-slow"
            style={{ animationDelay: `${i * 0.2}s` }}
          >
            <span className="text-lightSky mr-1">#</span>
            {phrase}
          </div>
        ))}
      </div>
    </div>
  );
}