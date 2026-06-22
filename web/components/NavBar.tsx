"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

const LINKS = [
  { href: "/", label: "LIVE" },
  { href: "/people", label: "PEOPLE" },
  { href: "/cameras", label: "CAMERAS" },
  { href: "/search", label: "SEARCH" },
];

function isActive(pathname: string, href: string): boolean {
  if (href === "/") return pathname === "/";
  return pathname === href || pathname.startsWith(`${href}/`);
}

export function NavBar() {
  const pathname = usePathname();
  return (
    <header className="sticky top-0 z-10 flex flex-wrap items-center justify-between gap-4 border-b border-edge bg-void/90 px-6 py-4 backdrop-blur sm:px-10">
      <Link href="/" className="flex items-center gap-2">
        <span className="h-2 w-2 rounded-full bg-green shadow-[0_0_6px_var(--green)] rec-dot" />
        <span className="font-display text-lg font-bold tracking-[0.2em] text-ink">
          SENTINEL
        </span>
        <span className="text-[11px] tracking-[0.3em] text-faint">// SURVEILLANCE</span>
      </Link>
      <nav className="flex items-center gap-1">
        {LINKS.map((l) => {
          const active = isActive(pathname, l.href);
          return (
            <Link
              key={l.href}
              href={l.href}
              className={`px-3 py-1.5 text-xs uppercase tracking-widest transition-colors ${
                active
                  ? "border border-green/60 text-green"
                  : "border border-transparent text-faint hover:text-ink"
              }`}
            >
              {l.label}
            </Link>
          );
        })}
      </nav>
    </header>
  );
}
