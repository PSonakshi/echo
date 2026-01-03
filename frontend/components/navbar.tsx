"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

export default function Navbar() {
  const pathname = usePathname();

  const links = [
    { href: "/dashboard", label: "Dashboard" },
    { href: "/login", label: "Login" },
    { href: "/signup", label: "Signup" },
  ];

  const isActive = (href: string) => pathname === href;

  return (
    <nav className="sticky top-0 z-40 w-full">
      {/* Glass morphism background */}
      <div className="glass-panel border-b border-white/10 backdrop-blur-md">
        <div className="max-w-[1600px] mx-auto px-8 py-4 flex items-center justify-between">
          {/* Logo / Branding */}
          <Link href="/" className="text-2xl font-bold bg-gradient-to-r from-price via-sentiment to-pastelBlue bg-clip-text text-purple">
            ECHO
          </Link>

          {/* Navigation Links */}
          <div className="flex items-center gap-8">
            {links.map((link) => (
              <Link
                key={link.href}
                href={link.href}
                className={`font-medium text-sm tracking-wide transition-colors ${
                  isActive(link.href)
                    ? "text-price border-b-2 border-price pb-1"
                    : "text-foreground/70 hover:text-foreground"
                }`}
              >
                {link.label}
              </Link>
            ))}
          </div>
        </div>
      </div>
    </nav>
  );
}
