"use client";

import { useEffect, useRef, useState, PropsWithChildren } from "react";

export default function RevealOnScroll({ children, className = "", rootMargin = "0px 0px -10% 0px" }: PropsWithChildren<{ className?: string; rootMargin?: string }>) {
  const ref = useRef<HTMLDivElement | null>(null);
  const [inView, setInView] = useState(false);

  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const obs = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) {
            setInView(true);
            obs.unobserve(entry.target);
          }
        });
      },
      { threshold: 0.15, rootMargin }
    );

    obs.observe(el);
    return () => obs.disconnect();
  }, [rootMargin]);

  return (
    <div
      ref={ref}
      className={`${className} transition-all duration-700 ease-out ${inView ? "opacity-100 translate-y-0" : "opacity-0 translate-y-6"}`}
    >
      {children}
    </div>
  );
}
