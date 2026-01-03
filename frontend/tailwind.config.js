/** @type {import('tailwindcss').Config} */
module.exports = {
  darkMode: ["class"],
  content: [
    "./app/**/*.{js,ts,jsx,tsx}",
    "./components/**/*.{js,ts,jsx,tsx}",
    "./pages/**/*.{js,ts,jsx,tsx}",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    container: {
      center: true,
      padding: "2rem",
      screens: {
        "2xl": "1400px",
      },
    },
    extend: {
      colors: {
        // Primary palette
        background: "#1C1A26",
        foreground: "#EAF1E5",
        border: "rgba(234, 241, 229, 0.08)",

        // Panels and UI
        panel: "#2A2F4A",

        // Colorful palette
        richViolet: "#5B0779",
        electricIndigo: "#525ECD",
        softOffWhite: "#EAF1E5",
        paleWarmYellow: "#EDE6AA",
        pastelBlue: "#B2BFE2",
        mutedPeach: "#BC9985",
        earthyBrown: "#8B645D",

        // Dedicated semantics
        price: "#525ECD",
        sentiment: "#5B0779",

        // compatibility aliases
        softBlue: "#525ECD",
        lightSky: "#7F8FE0",
        lavender: "#7D78C9",
      },
      fontFamily: {
        sans: ["Open Sauce One", "sans-serif"],
        mono: ["JetBrains Mono", "monospace"],
      },
      keyframes: {
        blob: {
          "0%": { transform: "translate(-50%, -10%) scale(1)" },
          "50%": { transform: "translate(-48%, -12%) scale(1.03)" },
          "100%": { transform: "translate(-50%, -10%) scale(1)" },
        },
        "neon-pulse": {
          "0%": { filter: "drop-shadow(0 0 6px rgba(34,197,94,0.35))" },
          "50%": { filter: "drop-shadow(0 0 18px rgba(34,197,94,0.55))" },
          "100%": { filter: "drop-shadow(0 0 6px rgba(34,197,94,0.35))" },
        },
        "accordion-down": {
          from: { height: "0" },
          to: { height: "var(--radix-accordion-content-height)" },
        },
        "accordion-up": {
          from: { height: "var(--radix-accordion-content-height)" },
          to: { height: "0" },
        },
        "fade-in": {
          "0%": { opacity: "0" },
          "100%": { opacity: "1" },
        },
        "zoom-in": {
          "0%": { transform: "scale(0.95)", opacity: "0" },
          "100%": { transform: "scale(1)", opacity: "1" },
        },
        "slide-in-from-bottom": {
          "0%": { transform: "translateY(100%)", opacity: "0" },
          "100%": { transform: "translateY(0)", opacity: "1" },
        },
        "spin-slow": {
          from: { transform: "rotate(0deg)" },
          to: { transform: "rotate(360deg)" },
        },
        scan: {
          "0%": { left: "-100%" },
          "100%": { left: "200%" },
        },
        float: {
          "0%, 100%": { transform: "translateY(0)" },
          "50%": { transform: "translateY(-6px)" },
        },
        "grid-scroll": {
          "0%": { backgroundPosition: "0 0" },
          "100%": { backgroundPosition: "40px 40px" },
        },
        "crt-flicker": {
          "0%": { opacity: "1" },
          "50%": { opacity: "0.97" },
          "100%": { opacity: "1" },
        },
        glitch: {
          "0%": { transform: "translate(0)" },
          "20%": { transform: "translate(-2px, 2px)" },
          "40%": { transform: "translate(-2px, -2px)" },
          "60%": { transform: "translate(2px, 2px)" },
          "80%": { transform: "translate(2px, -2px)" },
          "100%": { transform: "translate(0)" },
        },
        shimmer: {
          "0%": { backgroundPosition: "200% 0" },
          "100%": { backgroundPosition: "-200% 0" },
        },
        "pulse-slow": {
          "0%, 100%": { opacity: "1" },
          "50%": { opacity: "0.7" },
        },
      },
      animation: {
        "accordion-down": "accordion-down 0.2s ease-out",
        "accordion-up": "accordion-up 0.2s ease-out",
        "fade-in": "fade-in 0.5s ease-out",
        "zoom-in": "zoom-in 0.3s ease-out",
        "slide-in-from-bottom":
          "slide-in-from-bottom 0.4s cubic-bezier(0.16, 1, 0.3, 1)",
        "spin-slow": "spin-slow 12s linear infinite",
        "scan": "scan 3s linear infinite",
        "float": "float 4s ease-in-out infinite",
        "blob": "blob 8s ease-in-out infinite",
        "neon-pulse": "neon-pulse 2.6s ease-in-out infinite",
        "grid-scroll": "grid-scroll 3s linear infinite",
        "crt-flicker": "crt-flicker 0.15s infinite",
        "glitch":
          "glitch 0.3s cubic-bezier(.25, .46, .45, .94) both infinite",
        "pulse-slow": "pulse-slow 3s ease-in-out infinite",
      },
    },
  },
  plugins: [require("tailwindcss-animate")],
};
