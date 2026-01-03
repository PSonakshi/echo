"use client";

import Link from "next/link";

export default function Page() {
  return (
    <main className="bg-black text-white overflow-hidden">

      {/* ================= HERO ================= */}
      <section className="relative min-h-screen flex items-center">
        {/* COLOR BANDS */}
        <div className="absolute inset-0 -z-10">
          <div
            className="absolute right-0 top-0 h-full w-[60%]"
            style={{
              background:
                "linear-gradient(90deg, #16002d, #3b0f70, #7a1cff, #4361ee, #4cc9f0, #80ffdb, #ffd166, #ff7a00, #ff0054)",
              filter: "blur(70px)",
              opacity: 1,
            }}
          />
          <div
            className="absolute inset-0"
            style={{
              background:
                "repeating-linear-gradient(90deg, rgba(255,255,255,0.25) 0px, rgba(255,255,255,0.25) 3px, transparent 3px, transparent 12px)",
              mixBlendMode: "overlay",
              opacity: 0.4,
            }}
          />
          <div className="absolute inset-0 bg-gradient-to-l from-black via-black/40 to-black" />
        </div>

        {/* TEXT */}
        <div className="max-w-7xl mx-auto px-16">
          <p className="text-xs tracking-widest text-white/40 mb-8">
            · ECHO
          </p>

          <h1 className="text-[3.8rem] leading-tight font-medium text-white/80">
            Echo is an
            <br />
            AI-powered narrative 
            <br/>
            intelligence engine
          
            <br />
            designed for the high-velocity world 
            <br />
           of cryptocurrency.
          
          </h1>

      </div>
      </section>

      {/* ================= ANNOUNCEMENT ================= */}
      <section className="relative py-48 flex justify-center">
        <div
          className="absolute inset-0"
          style={{
            background:
              "linear-gradient(90deg, #0a1a3a, #0d3b66, #1b9aaa, #ffd166, #ef476f)",
            filter: "blur(120px)",
            opacity: 0.8,
          }}
        />

      <div className="relative z-10 bg-black rounded-[36px] border border-white/10 px-24 py-28 text-center max-w-4xl">
  {/* HERO LINE */}
  <p className="text-3xl md:text-4xl font-semibold tracking-tight text-white mb-8">
    Listen to the Market <span className="text-white/60">Before It Moves.</span>
  </p>

  {/* BODY COPY */}
  <p className="text-white/75 text-base md:text-lg leading-relaxed mb-10">
    While others look at charts, Echo listens to the conversation.
    By processing thousands of social signals per second through our proprietary
    Pathway streaming pipeline, we decode the chaos of market sentiment into a
    single, actionable metric: the <span className="text-white font-medium">Pulse Score</span>.
    We don’t just track what people are saying—we correlate it with price action
    in real time to spot divergences, detect manipulation, and identify the next
    breakout narrative before it hits the candles.
    <br />
    <br />
    <span className="text-white font-medium">
      Stop doom-scrolling. Start listening.
    </span>
  </p>


         
        </div>
      </section>

      {/* ================= TEAM ================= */}
      <section className="relative py-40">
        <div
          className="absolute left-0 top-0 h-full w-[35%]"
          style={{
            background:
              "linear-gradient(180deg, #0d3b66, #1b9aaa)",
            filter: "blur(80px)",
            opacity: 0.45,
          }}
        />

        <div className="max-w-7xl mx-auto px-16 grid grid-cols-[1fr_2fr] gap-24 items-center">
          <div>
            <h2 className="text-6xl font-medium mb-6">
              Created by
              <br />
              Developers
              <br />
            </h2>

            <p className="text-white/50 max-w-sm">
            
            </p>

            
          </div>

          <div className="grid grid-cols-3 gap-12">
            {[
              {
                src: "/photos/sona.jpeg",
                name: "Sonakshi Pradhan",
                role: "Frontend & UI Designer",
              },
              {
                src: "/photos/srinu.jpeg",
                name: "Srinibas",
                role: "System Architect & Domain Expert",
              },
              {
                src: "/photos/anurag.jpeg",
                name: "Anurag",
                role: "Backend & Data Architect",
              },
            ].map((member) => (
              <div
                key={member.name}
                className="flip-card w-full h-[420px] group"
              >
                <div className="flip-inner h-full w-full">
                  {/* FRONT - Image */}
                  <div className="flip-front rounded-md overflow-hidden h-full w-full">
                    <div
                      className="p-[3px] rounded-md h-full flex items-center justify-center"
                      style={{
                        background:
                          "linear-gradient(180deg, #4cc9f0, #ffd166, #ef476f)",
                      }}
                    >
                      <img
                        src={member.src}
                        className="rounded-md object-cover w-full h-full"
                        alt={member.name}
                      />
                    </div>
                  </div>

                  {/* BACK - Name & Role */}
                  <div className="flip-back rounded-md h-full w-full flex flex-col items-center justify-center p-6 text-center bg-gradient-to-br from-emerald-400/10 to-cyan-400/10 border border-emerald-400/20">
                    <h3 className="text-2xl font-bold text-white mb-3">
                      {member.name}
                    </h3>
                    <p className="text-emerald-400 font-semibold text-sm uppercase tracking-widest">
                      {member.role}
                    </p>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* ================= FOOTER ================= */}
      <footer className="relative border-t border-white/10 py-12 px-16">
        <div className="absolute inset-0 -z-10 bg-gradient-to-b from-transparent via-emerald-400/5 to-transparent" />
        
        <div className="max-w-7xl mx-auto flex flex-col md:flex-row justify-between items-center gap-8">
          {/* LEFT - CGAPI Logo */}
          <div className="flex items-center gap-3">
            <img
              src="/photos/CGAPI-Lockup@2x-1.png"
              alt="CGAPI"
              className="h-8 object-contain"
            />
          </div>

          {/* CENTER - Pathway Logo & Text */}
          <div className="flex items-center gap-3">
            <span className="text-xs tracking-widest text-white/40">
              POWERED BY
            </span>
            <img
              src="/photos/pathway-logo-black.png"
              alt="Pathway"
              className="h-6 object-contain invert"
            />
          </div>

          {/* RIGHT - Language Selector */}
          <div className="flex items-center gap-2">
            <span className="text-xs tracking-widest text-white/40">
              LANGUAGE
            </span>
            <select className="bg-white/5 border border-white/10 rounded-md px-3 py-2 text-xs text-white cursor-pointer hover:bg-white/10 transition-colors">
              <option value="en">English</option>
              <option value="es">Español</option>
              <option value="fr">Français</option>
              <option value="de">Deutsch</option>
              <option value="ja">日本語</option>
            </select>
          </div>
        </div>

        {/* BOTTOM - Copyright */}
        <div className="max-w-7xl mx-auto mt-8 pt-8 border-t border-white/5 text-center text-xs text-white/30">
          © 2026 Echo. All rights reserved.
        </div>

      </footer>
    </main>
  );
}
