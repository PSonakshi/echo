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
            · pathway
          </p>

          <h1 className="text-[3.8rem] leading-tight font-medium text-white/80">
            We wanted to
            <br />
            understand the{" "}
            <span className="text-white">why</span>.
            <br />
            We&apos;re fundamentally
            <br />
            changing the way
            <br />
            models think.
          </h1>

          <div className="mt-12">
            <div className="bg-white rounded-md w-[320px] p-4 text-black">
              <p className="text-sm font-medium mb-2">
                Join the waitlist
              </p>
              <div className="flex items-center gap-2 border-b border-black/30 pb-1">
                <input
                  placeholder="example@email.com"
                  className="flex-1 outline-none text-sm"
                />
                <span>→</span>
              </div>
            </div>
          </div>
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
          <p className="text-sm text-white/60 mb-6">
            The massively parallel post-Transformer reasoning architecture
            which opens the door to <span className="text-white">generalization</span> over time.
          </p>

          <h2 className="text-6xl font-medium mb-8">
            Announcing <span className="text-white">BDH</span>
          </h2>

          <p className="text-white/50 mb-12">
            “The Missing Link Between the Transformer
            <br />
            and Models of the Brain”
          </p>

          <Link
            href="#"
            className="inline-block bg-white text-black px-10 py-4 rounded-md font-medium"
          >
            Read the paper →
          </Link>
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
            opacity: 0.8,
          }}
        />

        <div className="max-w-7xl mx-auto px-16 grid grid-cols-[1fr_2fr] gap-24 items-center">
          <div>
            <h2 className="text-6xl font-medium mb-6">
              Created by
              <br />
              scientists &
              <br />
              researchers
            </h2>

            <p className="text-white/50 max-w-sm">
              Headed by co-founder & CEO, Zuzanna Stamirowska,
              CTO Jan Chorowski, and CSO Adrian Kosowski.
              The team has already built AI tooling,
              amassing 108k stars on Github.
            </p>

            <p className="mt-4 text-white underline">
              Access our tooling here →
            </p>
          </div>

          <div className="grid grid-cols-3 gap-12">
            {[
              "/team/1.jpg",
              "/team/2.jpg",
              "/team/3.jpg",
            ].map((src) => (
              <div
                key={src}
                className="p-[3px] rounded-md"
                style={{
                  background:
                    "linear-gradient(180deg, #4cc9f0, #ffd166, #ef476f)",
                }}
              >
                <img
                  src={src}
                  className="rounded-md object-cover w-full h-[420px]"
                />
              </div>
            ))}
          </div>
        </div>
      </section>
    </main>
  );
}
