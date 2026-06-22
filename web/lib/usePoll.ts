"use client";

import { useEffect, useRef, useState } from "react";

/** Re-run an async fetcher on an interval; returns the latest data/error/loading. */
export function usePoll<T>(fetcher: () => Promise<T>, intervalMs = 1500) {
  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState<Error | null>(null);
  const [loading, setLoading] = useState(true);
  const fnRef = useRef(fetcher);
  fnRef.current = fetcher;

  useEffect(() => {
    let alive = true;
    const tick = async () => {
      try {
        const d = await fnRef.current();
        if (alive) {
          setData(d);
          setError(null);
        }
      } catch (e) {
        if (alive) setError(e as Error);
      } finally {
        if (alive) setLoading(false);
      }
    };
    tick();
    const id = setInterval(tick, intervalMs);
    return () => {
      alive = false;
      clearInterval(id);
    };
  }, [intervalMs]);

  return { data, error, loading };
}
