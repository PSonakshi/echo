"use client";

import { useState, useEffect } from "react";
import Link from "next/link";
import { Mail, Lock, LogIn } from "lucide-react";
import { useRouter } from "next/navigation";

export default function LoginPage() {
  const router = useRouter();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (typeof window !== "undefined" && localStorage.getItem("auth") === "true") {
      router.replace("/dashboard");
    }
  }, [router]);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!email || !password) return;
    setLoading(true);
    // mock auth: persist a flag and redirect
    setTimeout(() => {
      localStorage.setItem("auth", "true");
      setLoading(false);
      router.push("/dashboard");
    }, 600);
  };

  return (
    <main className="min-h-screen flex items-center justify-center px-4 py-0 bg-gradient-to-br from-background via-background to-richViolet/10">
      <div className="w-full max-w-4xl px-4">
        <div className="glass-panel rounded-3xl p-6 md:p-10 grid md:grid-cols-2 gap-6 items-center border border-white/6">
          <div className="px-2">
            <h2 className="text-3xl font-extrabold mb-2 neon-text">Echo</h2>
            <p className="text-foreground/60 mb-4">Sign in to access your dashboard and live sentiment feeds.</p>
            <div className="hidden md:block mt-6 w-full h-40 rounded-xl bg-gradient-to-br from-emerald-300 to-green-500 opacity-20 blur-xl" />
          </div>

          <div>
            <form onSubmit={handleSubmit} className="space-y-5">
              <div className="space-y-2">
                <label className="text-sm font-medium text-foreground block">Email Address</label>
                <div className="relative group">
                  <Mail className="absolute left-4 top-1/2 -translate-y-1/2 w-5 h-5 text-sentiment opacity-40 group-focus-within:opacity-100 transition-opacity" />
                  <input
                    type="email"
                    placeholder="you@example.com"
                    value={email}
                    onChange={(e) => setEmail(e.target.value)}
                    required
                    className="w-full bg-white/5 border border-white/10 rounded-lg pl-12 pr-4 py-3 text-foreground placeholder:text-foreground/30 focus:outline-none focus:border-sentiment focus:bg-white/10 focus:ring-2 focus:ring-sentiment/20 transition-all"
                  />
                </div>
              </div>

              <div className="space-y-2">
                <div className="flex justify-between items-center">
                  <label className="text-sm font-medium text-foreground block">Password</label>
                </div>
                <div className="relative group">
                  <Lock className="absolute left-4 top-1/2 -translate-y-1/2 w-5 h-5 text-sentiment opacity-40 group-focus-within:opacity-100 transition-opacity" />
                  <input
                    type="password"
                    placeholder="••••••••"
                    value={password}
                    onChange={(e) => setPassword(e.target.value)}
                    required
                    className="w-full bg-white/5 border border-white/10 rounded-lg pl-12 pr-4 py-3 text-foreground placeholder:text-foreground/30 focus:outline-none focus:border-sentiment focus:bg-white/10 focus:ring-2 focus:ring-sentiment/20 transition-all"
                  />
                </div>
              </div>

              <button
                type="submit"
                disabled={loading}
                className="w-full bg-gradient-to-r from-emerald-400 to-green-600 py-3 rounded-lg font-semibold text-background hover:shadow-lg hover:scale-[1.02] transition-all disabled:opacity-70 disabled:cursor-not-allowed flex items-center justify-center gap-2"
              >
                {loading ? (
                  <>
                    <div className="w-4 h-4 border-2 border-background/30 border-t-background rounded-full animate-spin" />
                    Signing in...
                  </>
                ) : (
                  <>
                    <LogIn className="w-4 h-4" />
                    Sign In
                  </>
                )}
              </button>
            </form>

            <div className="mt-4">
              <div className="relative">
                <div className="absolute inset-0 flex items-center">
                  <div className="w-full border-t border-white/8" />
                </div>
                <div className="relative flex justify-center text-xs">
                  <span className="px-2 bg-[#071018] text-foreground/60">or continue with</span>
                </div>
              </div>

              <div className="mt-3 flex gap-3">
                <button className="flex-1 border border-white/8 rounded-lg py-2 text-sm">Continue with X</button>
                <button className="flex-1 border border-white/8 rounded-lg py-2 text-sm">Continue with Google</button>
              </div>
            </div>
          </div>
        </div>

        <div className="text-center mt-6 text-sm text-foreground/60">
          Don't have an account? <Link href="/signup" className="text-emerald-300 font-semibold">Create one</Link>
        </div>
      </div>
    </main>
  );
}
