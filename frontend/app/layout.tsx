import "./globals.css";
import type { Metadata } from "next";
import Navbar from "@/components/navbar";

export const metadata: Metadata = {
  title: "Crypto Narrative Pulse Tracker",
  description: "Live sentiment analysis and market velocity tracker",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className="dark">
      <head>
        {/* Open Sauce One Font */}
        <link
          href="https://fonts.cdnfonts.com/css/open-sauce-one"
          rel="stylesheet"
        />
        {/* JetBrains Mono Font */}
        <link
          href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&display=swap"
          rel="stylesheet"
        />
      </head>
      <body className="min-h-screen selection:bg-[#7DA6D9] selection:text-[#1F1F1F] font-sans overflow-x-hidden">
        <Navbar />
        {children}
      </body>
    </html>
  );
}