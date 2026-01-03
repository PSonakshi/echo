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
    <RevealOnScroll className="w-40 h-56">
      <div className="flip-card w-40 h-56">
        <div className="flip-inner">
          <div className="flip-front rounded-xl glass-panel flex flex-col items-center justify-center p-4">
            {img ? (
              <img src={img} alt={name} className="w-20 h-20 rounded-full mb-3 object-cover" />
            ) : (
              <div className="w-20 h-20 rounded-full mb-3 bg-emerald-400/30 flex items-center justify-center text-foreground">{name.split(" ")[0][0]}</div>
            )}
            <div className="font-semibold text-sm text-center">{name}</div>
          </div>

          <div className="flip-back rounded-xl glass-panel flex flex-col items-center justify-center p-4">
            <div className="font-semibold text-sm">{role}</div>
            <p className="text-xs text-foreground/70 mt-2 text-center">{bio ?? "Contributes to product and roadmap."}</p>
          </div>
        </div>
      </div>
    </RevealOnScroll>
  );
}
