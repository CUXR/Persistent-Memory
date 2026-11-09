import { DependencyList, useCallback, useEffect, useRef, useState } from "react";

import type { Page } from "@/data/types";
import { toError } from "@/hooks/use-query";

type PagedListState<T> = {
  status: "loading" | "error" | "success";
  items: T[];
  nextCursor?: string;
  error?: Error;
  loadingMore: boolean;
  loadMoreError?: Error;
  refreshing: boolean;
};

function initialState<T>(): PagedListState<T> {
  return { status: "loading", items: [], loadingMore: false, refreshing: false };
}

export function usePagedList<T>(fetchPage: (cursor?: string) => Promise<Page<T>>, deps: DependencyList) {
  const [state, setState] = useState<PagedListState<T>>(initialState);
  const [attempt, setAttempt] = useState(0);
  // Each first-page load gets a new generation so responses from stale loads are dropped.
  const generation = useRef(0);
  const stateRef = useRef(state);
  const fetchRef = useRef(fetchPage);

  useEffect(() => {
    stateRef.current = state;
    fetchRef.current = fetchPage;
  });

  const loadFirstPage = useCallback((mode: "initial" | "refresh") => {
    const gen = ++generation.current;
    setState((s) => (mode === "initial" ? initialState() : { ...s, refreshing: true }));
    fetchRef.current(undefined).then(
      (page) => {
        if (gen !== generation.current) return;
        setState({ ...initialState(), status: "success", items: page.items, nextCursor: page.nextCursor });
      },
      (e) => {
        if (gen !== generation.current) return;
        setState((s) =>
          mode === "initial"
            ? { ...initialState(), status: "error", error: toError(e) }
            : { ...s, refreshing: false, loadMoreError: toError(e) },
        );
      },
    );
  }, []);

  useEffect(() => {
    loadFirstPage("initial");
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [...deps, attempt]);

  const loadMore = useCallback(() => {
    const s = stateRef.current;
    if (s.status !== "success" || !s.nextCursor || s.loadingMore || s.refreshing) return;
    // Guard against onEndReached firing again before the next render.
    stateRef.current = { ...s, loadingMore: true };
    setState((cur) => ({ ...cur, loadingMore: true, loadMoreError: undefined }));
    const gen = generation.current;
    fetchRef.current(s.nextCursor).then(
      (page) => {
        if (gen !== generation.current) return;
        setState((cur) => ({
          ...cur,
          items: [...cur.items, ...page.items],
          nextCursor: page.nextCursor,
          loadingMore: false,
        }));
      },
      (e) => {
        if (gen !== generation.current) return;
        setState((cur) => ({ ...cur, loadingMore: false, loadMoreError: toError(e) }));
      },
    );
  }, []);

  const refresh = useCallback(() => loadFirstPage("refresh"), [loadFirstPage]);
  const retry = useCallback(() => setAttempt((a) => a + 1), []);

  return { ...state, hasMore: Boolean(state.nextCursor), loadMore, refresh, retry };
}
