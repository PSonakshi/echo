"use client";

import React, { useMemo } from "react";
import { MessageSquare } from "lucide-react";

interface NarrativeCloudProps {
  phrases: string[];
}

const NarrativeCloud = React.memo(function NarrativeCloud({ phrases }: NarrativeCloudProps) {
  const displayPhrases = useMemo(() => {
    // Limit to 12 phrases for better performance and layout
    return phrases.slice(0, 12);
  }, [phrases]);

  return (
    <div className="glass-panel p-6 rounded-2xl h-full flex flex-col min-h-0">
      <div className="flex items-center gap-2 mb-6 flex-shrink-0">
        <span className="text-lightSky">
          <MessageSquare className="w-5 h-5" />
        </span>
        <h3 className="text-lg font-semibold">Narrative Cloud</h3>
      </div>

      <div className="flex-1 overflow-y-auto">
        <div className="grid grid-cols-2 sm:grid-cols-3 gap-3 auto-rows-min">
          {displayPhrases.map((phrase, i) => (
            <div
              key={`${phrase}-${i}`}
              className="px-3 py-2 rounded-xl text-sm font-medium bg-white/5 border border-white/5 text-foreground hover:bg-white/10 hover:border-lightSky/30 transition-all duration-300 cursor-default animate-pulse-slow text-center"
              style={{ animationDelay: `${i * 0.1}s` }}
            >
              <span className="text-lightSky mr-1">#</span>
              <span className="break-words">{phrase}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
});

export default NarrativeCloud;