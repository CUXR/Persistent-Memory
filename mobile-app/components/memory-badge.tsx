import Ionicons from "@expo/vector-icons/Ionicons";
import React, { ComponentProps } from "react";
import { StyleSheet, View } from "react-native";

import { AppText } from "@/components/app-text";
import { Radius, Spacing, ThemeColors } from "@/constants/theme";
import type { RecordingStatus } from "@/data/types";
import { useTheme } from "@/hooks/use-theme";

export type MemoryState = "saved" | RecordingStatus;

const CONFIG: Record<
  MemoryState,
  { label: string; icon: ComponentProps<typeof Ionicons>["name"]; tone: "success" | "warning" | "danger" }
> = {
  saved: { label: "Saved memory", icon: "bookmark", tone: "success" },
  pending: { label: "Queued", icon: "time-outline", tone: "warning" },
  processing: { label: "Processing", icon: "hourglass-outline", tone: "warning" },
  needs_review: { label: "Needs review", icon: "alert-circle-outline", tone: "warning" },
  failed: { label: "Failed", icon: "close-circle-outline", tone: "danger" },
};

function toneColors(tone: "success" | "warning" | "danger", colors: ThemeColors) {
  if (tone === "success") return { fg: colors.success, bg: colors.successSoft };
  if (tone === "warning") return { fg: colors.warning, bg: colors.warningSoft };
  return { fg: colors.danger, bg: colors.dangerSoft };
}

export function MemoryBadge({ state }: { state: MemoryState }) {
  const colors = useTheme();
  const { label, icon, tone } = CONFIG[state];
  const { fg, bg } = toneColors(tone, colors);
  const provisional = state !== "saved";
  return (
    <View
      style={[
        styles.badge,
        { backgroundColor: bg },
        provisional ? { borderColor: fg, borderStyle: "dashed" } : { borderColor: "transparent" },
      ]}
    >
      <Ionicons name={icon} size={11} color={fg} />
      <AppText variant="caption" style={[styles.text, { color: fg }]}>
        {label}
      </AppText>
    </View>
  );
}

export function UnnamedBadge() {
  const colors = useTheme();
  return (
    <View style={[styles.badge, { backgroundColor: colors.tintSoft, borderColor: "transparent" }]}>
      <Ionicons name="person-add-outline" size={11} color={colors.tint} />
      <AppText variant="caption" style={[styles.text, { color: colors.tint }]}>
        Needs a name
      </AppText>
    </View>
  );
}

const styles = StyleSheet.create({
  badge: {
    flexDirection: "row",
    alignItems: "center",
    alignSelf: "flex-start",
    gap: 4,
    paddingHorizontal: Spacing.sm,
    paddingVertical: 2,
    borderRadius: Radius.pill,
    borderWidth: 1,
  },
  text: { fontSize: 11, lineHeight: 14, fontWeight: "600" },
});
