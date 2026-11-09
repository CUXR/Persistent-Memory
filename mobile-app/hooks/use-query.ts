import { DependencyList, useCallback, useEffect, useState } from "react";

export type QueryState<T> =
  | { status: "loading" }
  | { status: "error"; error: Error }
  | { status: "success"; data: T };

export function toError(e: unknown): Error {
  return e instanceof Error ? e : new Error(String(e));
}

export function useQuery<T>(fetcher: () => Promise<T>, deps: DependencyList) {
  const [state, setState] = useState<QueryState<T>>({ status: "loading" });
  const [attempt, setAttempt] = useState(0);

  useEffect(() => {
    let cancelled = false;
    setState({ status: "loading" });
    fetcher().then(
      (data) => !cancelled && setState({ status: "success", data }),
      (e) => !cancelled && setState({ status: "error", error: toError(e) }),
    );
    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [...deps, attempt]);

  const retry = useCallback(() => setAttempt((a) => a + 1), []);
  const setData = useCallback((data: T) => setState({ status: "success", data }), []);

  return { state, retry, setData };
}
