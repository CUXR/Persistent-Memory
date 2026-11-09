import React, { createContext, ReactNode, useContext, useEffect, useState } from "react";

import { useDataSource } from "@/data/DataSourceContext";
import type { ConnectionStatus } from "@/data/types";

export type ConnectionState =
  | { kind: "checking" }
  | { kind: "connected"; status: ConnectionStatus }
  | { kind: "glassesOffline"; status: ConnectionStatus }
  | { kind: "serverUnreachable" };

const POLL_INTERVAL_MS = 15_000;

const ConnectionStatusContext = createContext<ConnectionState>({ kind: "checking" });

export function ConnectionStatusProvider({ children }: { children: ReactNode }) {
  const { source, revision } = useDataSource();
  const [state, setState] = useState<ConnectionState>({ kind: "checking" });

  useEffect(() => {
    let cancelled = false;
    const check = () =>
      source.getConnectionStatus().then(
        (status) =>
          !cancelled &&
          setState({ kind: status.glasses === "connected" ? "connected" : "glassesOffline", status }),
        () => !cancelled && setState({ kind: "serverUnreachable" }),
      );
    check();
    const timer = setInterval(check, POLL_INTERVAL_MS);
    return () => {
      cancelled = true;
      clearInterval(timer);
    };
  }, [source, revision]);

  return <ConnectionStatusContext.Provider value={state}>{children}</ConnectionStatusContext.Provider>;
}

export function useConnectionStatus(): ConnectionState {
  return useContext(ConnectionStatusContext);
}
