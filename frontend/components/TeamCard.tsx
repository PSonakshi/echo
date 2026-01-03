"use client";

import RevealOnScroll from "@/components/RevealOnScroll";
import Image from "next/image";

type Props = {
  name: string;
  role: string;
  img?: string;
  bio?: string;
};

export default function TeamCard({ name, role, img, bio }: Props) {
  return (
    <RevealOnScroll>
      <div className="group h-full">
        <div className="glass-panel rounded-2xl p-6 h-full flex flex-col items-center justify-between border border-white/10 hover:border-emerald-400/30 transition-all duration-300 hover:shadow-lg hover:shadow-emerald-400/10 hover:scale-105">
          {/* Front - Image and Name */}
          <div className="flex flex-col items-center gap-4 w-full">
            {img ? (
              <img 
                src={img} 
                alt={name} 
                className="w-28 h-28 rounded-xl object-cover border border-white/10 group-hover:border-emerald-400/30 transition-all" 
              />
            ) : (
              <div className="w-28 h-28 rounded-xl bg-gradient-to-br from-emerald-400/30 to-green-500/20 flex items-center justify-center text-lg font-bold text-emerald-300 border border-white/10">
                {name.split(" ")[0][0]}
              </div>
            )}
            <div className="text-center">
              <h3 className="font-bold text-base text-foreground mb-1">{name}</h3>
              <p className="text-xs font-semibold text-emerald-400 uppercase tracking-widest">{role}</p>
            </div>
          </div>

          {/* Bio - shown on hover */}
          <p className="text-xs text-foreground/60 text-center mt-4 opacity-0 group-hover:opacity-100 transition-opacity duration-300">
            {bio ?? "Contributes to product and roadmap."}
          </p>
        </div>
      </div>
    </RevealOnScroll>
  );
}
