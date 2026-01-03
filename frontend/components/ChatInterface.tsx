"use client";

import { useState, useRef, useEffect } from "react";
import { Bot, Send } from "lucide-react";

interface Message {
  role: "ai" | "user";
  text: string;
}

interface ChatInterfaceProps {
  expanded?: boolean;
}

export default function ChatInterface({ expanded }: ChatInterfaceProps) {
  const [input, setInput] = useState("");
  const [messages, setMessages] = useState<Message[]>([
    {
      role: "ai",
      text: "Systems online. Monitoring social sentiment stream. Ask me about the current narrative.",
    },
  ]);
  const [isTyping, setIsTyping] = useState(false);
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [messages, isTyping]);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim()) return;

    const userMsg = input;
    setInput("");
    setMessages((prev) => [...prev, { role: "user", text: userMsg }]);
    setIsTyping(true);

    // Simulate AI Latency
    setTimeout(() => {
      const responses = [
        "I'm detecting a divergence. Price is dipping but sentiment remains bullish. Possible accumulation.",
        "Influencer activity is spiking on the phrase 'moon soon'.",
        "Pulse score is stable at 7.2. Volatility is expected to increase.",
        "Analysis complete: Bearish sentiment dominates the last 15 minutes.",
      ];
      const randomResponse =
        responses[Math.floor(Math.random() * responses.length)];
      setMessages((prev) => [...prev, { role: "ai", text: randomResponse }]);
      setIsTyping(false);
    }, 1500);
  };

  return (
    <div className={`glass-panel rounded-2xl flex flex-col ${expanded ? "h-full p-6" : "h-[300px]"} overflow-hidden relative`}>
      <div className="p-4 border-b border-white/5 bg-white/5 flex items-center gap-2 backdrop-blur-md z-20">
        <span className="text-mintGreen">
          <Bot className="w-5 h-5" />
        </span>
        <h3 className="font-semibold text-sm tracking-wide">Analyst AI</h3>
        <span className="ml-auto w-2 h-2 rounded-full bg-mintGreen animate-pulse"></span>
      </div>

      <div
        className={`flex-1 overflow-y-auto ${expanded ? "p-6 space-y-6" : "p-4 space-y-4"} relative z-10`}
        ref={scrollRef}
      >
        {messages.map((m, i) => (
          <div
            key={i}
            className={`max-w-[85%] rounded-2xl p-4 text-sm leading-relaxed ${
              m.role === "user"
                ? "bg-softBlue/20 text-softBlue ml-auto rounded-tr-none border border-softBlue/20"
                : "bg-white/5 text-foreground mr-auto rounded-tl-none border border-white/5"
            }`}
          >
            {m.text}
          </div>
        ))}
        {isTyping && (
          <div className={`text-xs text-muted-foreground flex gap-1 ml-2 items-center h-8 ${expanded ? "text-sm" : ""}`}>
            <span
              className="w-1.5 h-1.5 rounded-full bg-mutedPurple animate-bounce"
              style={{ animationDelay: "0s" }}
            ></span>
            <span
              className="w-1.5 h-1.5 rounded-full bg-mutedPurple animate-bounce"
              style={{ animationDelay: "0.1s" }}
            ></span>
            <span
              className="w-1.5 h-1.5 rounded-full bg-mutedPurple animate-bounce"
              style={{ animationDelay: "0.2s" }}
            ></span>
          </div>
        )}
      </div>

      <form
        onSubmit={handleSubmit}
        className={`border-t border-white/5 bg-black/20 flex gap-2 z-20 backdrop-blur-md ${expanded ? "p-4" : "p-3"}`}
      >
        <input
          className={`flex-1 bg-white/5 border border-white/5 rounded-xl px-4 py-2 ${expanded ? "text-base" : "text-sm"} text-foreground placeholder:text-muted-foreground/50 focus:outline-none focus:border-softBlue/50 focus:bg-white/10 transition-all`}
          placeholder="Ask about market conditions..."
          value={input}
          onChange={(e) => setInput(e.target.value)}
        />
        <button
          type="submit"
          className="p-2.5 rounded-xl bg-softBlue text-background hover:bg-lightSky hover:scale-105 transition-all duration-200 active:scale-95"
        >
          <Send className="w-4 h-4" />
        </button>
      </form>
    </div>
  );
}