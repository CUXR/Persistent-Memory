import React, { ReactNode } from "react";
import { StyleSheet, View } from "react-native";

import { AppText } from "@/components/app-text";
import { Spacing } from "@/constants/theme";

export function SectionHeader({ title, right }: { title: string; right?: ReactNode }) {
  return (
    <View style={styles.header}>
      <AppText variant="label" tone="muted" accessibilityRole="header">
        {title}
      </AppText>
      {right}
    </View>
  );
}

const styles = StyleSheet.create({
  header: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "space-between",
    paddingHorizontal: Spacing.lg,
    paddingTop: Spacing.xl,
    paddingBottom: Spacing.sm,
  },
});
