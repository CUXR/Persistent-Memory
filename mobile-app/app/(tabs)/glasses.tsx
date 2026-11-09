import Ionicons from "@expo/vector-icons/Ionicons";
import React, { ComponentProps, useState } from "react";
import { ActivityIndicator, Pressable, ScrollView, StyleSheet, View } from "react-native";

import { AppText } from "@/components/app-text";
import { Button } from "@/components/button";
import { describeConnection } from "@/components/connection-pill";
import { GlassesPreview } from "@/components/glasses-preview";
import { OptionCard } from "@/components/option-card";
import { ErrorState } from "@/components/state-views";
import { Radius, Spacing } from "@/constants/theme";
import { useDataSource } from "@/data/DataSourceContext";
import { FIXTURE_SCENARIOS } from "@/data/fixtures";
import type { DisplayMode, DisplaySettings } from "@/data/types";
import { useConnectionStatus } from "@/hooks/use-connection-status";
import { toError, useQuery } from "@/hooks/use-query";
import { useTheme } from "@/hooks/use-theme";
import { formatRelative } from "@/lib/format";

const MODES: {
  value: DisplayMode;
  icon: ComponentProps<typeof Ionicons>["name"];
  title: string;
  description: string;
}[] = [
  { value: "off", icon: "eye-off-outline", title: "Nothing", description: "Keep the display blank when someone is recognized." },
  { value: "name", icon: "person-outline", title: "Name only", description: "Show the person's name." },
  {
    value: "name_and_facts",
    icon: "list-outline",
    title: "Name + facts",
    description: "Show the name plus a few facts the server picks as most relevant.",
  },
];

type Sync = { kind: "idle" } | { kind: "saving"; mode: DisplayMode } | { kind: "error"; mode: DisplayMode; message: string };

export default function GlassesScreen() {
  const colors = useTheme();
  const { source, revision, fixtures } = useDataSource();
  const { state, retry, setData } = useQuery(() => source.getDisplaySettings(), [source, revision]);
  const [sync, setSync] = useState<Sync>({ kind: "idle" });
  // Preview with the brief of the most recently seen named person, as the glasses would show it.
  const { state: briefState } = useQuery(async () => {
    const page = await source.listPeople({});
    const person = page.items.find((p) => !p.isUnnamed);
    return person ? source.getBrief(person.id) : undefined;
  }, [source, revision]);
  const brief = briefState.status === "success" ? briefState.data : undefined;

  const choose = async (mode: DisplayMode) => {
    setSync({ kind: "saving", mode });
    try {
      const saved: DisplaySettings = await source.setDisplayMode(mode);
      setData(saved);
      setSync({ kind: "idle" });
    } catch (e) {
      setSync({ kind: "error", mode, message: toError(e).message });
    }
  };

  let settingsSection: React.ReactNode;
  if (state.status === "loading") {
    settingsSection = <ActivityIndicator style={styles.loading} accessibilityLabel="Loading display settings" />;
  } else if (state.status === "error") {
    settingsSection = <ErrorState message={state.error.message} onRetry={retry} />;
  } else {
    const settings = state.data;
    // Show the pending choice right away; fall back to the saved one if the save fails.
    const shownMode = sync.kind === "saving" ? sync.mode : settings.mode;
    settingsSection = (
      <>
        <View accessibilityRole="radiogroup" style={styles.options}>
          {MODES.map((m) => (
            <OptionCard
              key={m.value}
              icon={m.icon}
              title={m.title}
              description={m.description}
              selected={shownMode === m.value}
              disabled={sync.kind === "saving"}
              onPress={() => m.value !== shownMode && choose(m.value)}
            />
          ))}
        </View>

        <SyncStatus settings={settings} sync={sync} onRetry={(mode) => choose(mode)} />

        <AppText variant="label" tone="muted" style={styles.sectionLabel}>
          Preview
        </AppText>
        <GlassesPreview mode={shownMode} brief={brief} />
        <AppText variant="caption" tone="subtle">
          {brief?.displayName
            ? `Shown for ${brief.displayName}, the person you saw most recently. The server picks the facts: top hobbies and affiliations.`
            : "Example only. The server picks the facts for each person: top hobbies and affiliations."}
        </AppText>
      </>
    );
  }

  return (
    <ScrollView style={{ backgroundColor: colors.background }} contentContainerStyle={styles.content}>
      <ConnectionCard />

      <AppText variant="label" tone="muted" style={styles.sectionLabel}>
        When someone is recognized, show
      </AppText>
      {settingsSection}

      {__DEV__ && fixtures ? (
        <View style={[styles.devCard, { borderColor: colors.border }]}>
          <AppText variant="label" tone="muted">
            Dev · fixture scenario
          </AppText>
          <AppText variant="caption" tone="subtle">
            Switch to preview loading, empty and error states across every screen.
          </AppText>
          <View style={styles.chips}>
            {FIXTURE_SCENARIOS.map((s) => {
              const active = fixtures.scenario === s.value;
              return (
                <Pressable
                  key={s.value}
                  accessibilityRole="radio"
                  accessibilityState={{ checked: active }}
                  onPress={() => {
                    setSync({ kind: "idle" });
                    fixtures.setScenario(s.value);
                  }}
                  style={[
                    styles.chip,
                    { backgroundColor: active ? colors.tint : colors.surface },
                  ]}
                >
                  <AppText variant="caption" style={{ color: active ? colors.onTint : colors.text, fontWeight: "600" }}>
                    {s.label}
                  </AppText>
                </Pressable>
              );
            })}
          </View>
        </View>
      ) : null}
    </ScrollView>
  );
}

function ConnectionCard() {
  const colors = useTheme();
  const connection = useConnectionStatus();
  const { label, color, soft } = describeConnection(connection, colors);

  let detail = "Checking the server…";
  if (connection.kind === "connected") detail = "Your glasses are online and receiving settings.";
  if (connection.kind === "glassesOffline")
    detail = `Last seen ${
      connection.status.glassesLastSeenAt ? formatRelative(connection.status.glassesLastSeenAt) : "a while ago"
    }. Changes will apply when they reconnect.`;
  if (connection.kind === "serverUnreachable") detail = "The app can't reach the server, so changes can't be saved.";

  return (
    <View style={[styles.connection, { backgroundColor: soft }]}>
      <Ionicons name="glasses-outline" size={28} color={color} />
      <View style={{ flex: 1 }}>
        <AppText variant="bodyStrong" style={{ color }}>
          {label}
        </AppText>
        <AppText variant="caption" tone="muted">
          {detail}
        </AppText>
      </View>
    </View>
  );
}

function SyncStatus({
  settings,
  sync,
  onRetry,
}: {
  settings: DisplaySettings;
  sync: Sync;
  onRetry: (mode: DisplayMode) => void;
}) {
  const colors = useTheme();

  if (sync.kind === "saving") {
    return (
      <View style={styles.syncRow}>
        <ActivityIndicator size="small" color={colors.textMuted} />
        <AppText variant="caption" tone="muted">
          Saving to server…
        </AppText>
      </View>
    );
  }
  if (sync.kind === "error") {
    return (
      <View style={[styles.syncError, { backgroundColor: colors.dangerSoft }]}>
        <Ionicons name="alert-circle" size={16} color={colors.danger} />
        <AppText variant="caption" tone="danger" style={{ flex: 1 }}>
          {sync.message} Your glasses are still using the previous setting.
        </AppText>
        <Button title="Retry" variant="ghost" onPress={() => onRetry(sync.mode)} />
      </View>
    );
  }
  return (
    <View style={styles.syncRow}>
      <Ionicons
        name={settings.appliedToGlasses ? "checkmark-circle" : "time-outline"}
        size={16}
        color={settings.appliedToGlasses ? colors.success : colors.warning}
      />
      <AppText variant="caption" tone="muted">
        {settings.appliedToGlasses
          ? `Saved · on your glasses since ${formatRelative(settings.updatedAt)}`
          : "Saved to server · waiting for glasses to reconnect"}
      </AppText>
    </View>
  );
}

const styles = StyleSheet.create({
  content: { padding: Spacing.lg, gap: Spacing.md, paddingBottom: Spacing.xxl },
  loading: { paddingVertical: Spacing.xl },
  connection: {
    flexDirection: "row",
    alignItems: "center",
    gap: Spacing.md,
    padding: Spacing.lg,
    borderRadius: Radius.lg,
  },
  sectionLabel: { marginTop: Spacing.md },
  options: { gap: Spacing.sm },
  syncRow: { flexDirection: "row", alignItems: "center", gap: Spacing.sm },
  syncError: {
    flexDirection: "row",
    alignItems: "center",
    gap: Spacing.sm,
    paddingLeft: Spacing.md,
    borderRadius: Radius.md,
  },
  devCard: {
    marginTop: Spacing.xl,
    padding: Spacing.lg,
    gap: Spacing.sm,
    borderRadius: Radius.lg,
    borderWidth: 1,
    borderStyle: "dashed",
  },
  chips: { flexDirection: "row", flexWrap: "wrap", gap: Spacing.sm },
  chip: { paddingHorizontal: Spacing.md, paddingVertical: 6, borderRadius: Radius.pill },
});
