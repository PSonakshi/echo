"use client";

import { ChevronDown } from "lucide-react";
import { useState } from "react";

export type Coin = {
  symbol: string;
  name: string;
  base: number;
};

type Props = {
  coins: Coin[];
  value: Coin;
  onChange: (coin: Coin) => void;
};

export default function CoinDropdown({ coins, value, onChange }: Props) {
  const [open, setOpen] = useState(false);

  return (
    <div className="relative">
      {/* Trigger */}
      <button
        onClick={() => setOpen((p) => !p)}
        className="glass-panel px-6 py-4 rounded-xl flex items-center gap-5 font-mono hover:bg-white/10 transition"
      >
        <div className="text-left">
          <div className="text-lg font-bold leading-tight">
            {value.symbol}
          </div>
          <div className="text-xs text-muted-foreground">
            {value.name}
          </div>
        </div>

        <ChevronDown
          size={26}
          strokeWidth={2.5}
          className={`transition-transform duration-200 ${
            open ? "rotate-180" : ""
          }`}
        />
      </button>

      {/* Menu */}
      {open && (
        <div className="absolute right-0 mt-3 w-64 glass-panel rounded-xl overflow-hidden z-20">
          {coins.map((coin) => (
            <button
              key={coin.symbol}
              onClick={() => {
                onChange(coin);
                setOpen(false);
              }}
              className={`w-full px-5 py-4 text-left hover:bg-white/10 transition ${
                coin.symbol === value.symbol
                  ? "bg-white/5 text-softBlue"
                  : ""
              }`}
            >
              <div className="text-base font-semibold">
                {coin.symbol}
              </div>
              <div className="text-xs text-muted-foreground">
                {coin.name}
              </div>
            </button>
          ))}
        </div>
      )}
    </div>
  );
}
