import React, { createContext, ReactNode, useContext, useMemo, useRef, useState } from "react";

import { createFixtureDataSource, FixtureScenario } from "./fixtures";
import type { DataSource } from "./types";

type DataSourceContextValue = {
  source: DataSource;
  // Bumped after mutations so list screens refetch.
  revision: number;
  invalidate: () => void;
  // Only present while running on fixtures.
  fixtures?: {
    scenario: FixtureScenario;
    setScenario: (scenario: FixtureScenario) => void;
  };
};

const DataSourceContext = createContext<DataSourceContextValue | undefined>(undefined);

export function FixtureDataSourceProvider({ children }: { children: ReactNode }) {
  const [scenario, setScenarioState] = useState<FixtureScenario>("normal");
  const [revision, setRevision] = useState(0);
  const scenarioRef = useRef(scenario);

  const [source] = useState(() => createFixtureDataSource(() => scenarioRef.current));

  const value = useMemo<DataSourceContextValue>(
    () => ({
      source,
      revision,
      invalidate: () => setRevision((r) => r + 1),
      fixtures: {
        scenario,
        setScenario: (next) => {
          scenarioRef.current = next;
          setScenarioState(next);
          setRevision((r) => r + 1);
        },
      },
    }),
    [source, revision, scenario],
  );

  return <DataSourceContext.Provider value={value}>{children}</DataSourceContext.Provider>;
}

export function useDataSource() {
  const ctx = useContext(DataSourceContext);
  if (!ctx) throw new Error("useDataSource must be used within a DataSource provider");
  return ctx;
}
