import React from "react";
import { StyleSheet, View } from "react-native";

import { AppText } from "@/components/app-text";
import type { ThemeColors } from "@/constants/theme";
import { Radius, Spacing } from "@/constants/theme";
import { ConnectionState, useConnectionStatus } from "@/hooks/use-connection-status";
import { useTheme } from "@/hooks/use-theme";

export function describeConnection(state: ConnectionState, colors: ThemeColors) {
  switch (state.kind) {
    case "checking":
      return { label: "Checking…", color: colors.textSubtle, soft: colors.surface };
    case "connected":
      return { label: "Glasses connected", color: colors.success, soft: colors.successSoft };
    case "glassesOffline":
      return { label: "Glasses offline", color: colors.warning, soft: colors.warningSoft };
    case "serverUnreachable":
      return { label: "Server unreachable", color: colors.danger, soft: colors.dangerSoft };
  }
}

export function ConnectionPill() {
  const colors = useTheme();
  const { label, color, soft } = describeConnection(useConnectionStatus(), colors);
  return (
    <View
      accessibilityRole="text"
      accessibilityLabel={`Status: ${label}`}
      style={[styles.pill, { backgroundColor: soft }]}
    >
      <View style={[styles.dot, { backgroundColor: color }]} />
      <AppText variant="caption" style={[styles.text, { color }]}>
        {label}
      </AppText>
    </View>
  );
}

const styles = StyleSheet.create({
  pill: {
    flexDirection: "row",
    alignItems: "center",
    gap: 6,
    paddingHorizontal: Spacing.sm + 2,
    paddingVertical: 4,
    borderRadius: Radius.pill,
    marginRight: Spacing.lg,
  },
  dot: { width: 8, height: 8, borderRadius: 4 },
  text: { fontSize: 12, lineHeight: 16, fontWeight: "600" },
});
