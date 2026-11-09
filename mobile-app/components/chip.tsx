import React from "react";
import { Pressable, ScrollView, StyleSheet } from "react-native";

import { AppText } from "@/components/app-text";
import { Radius, Spacing } from "@/constants/theme";
import { useTheme } from "@/hooks/use-theme";

export function Chip({ label, selected, onPress }: { label: string; selected: boolean; onPress: () => void }) {
  const colors = useTheme();
  return (
    <Pressable
      accessibilityRole="radio"
      accessibilityState={{ checked: selected }}
      onPress={onPress}
      style={({ pressed }) => [
        styles.chip,
        { backgroundColor: selected ? colors.tint : pressed ? colors.border : colors.surface },
      ]}
    >
      <AppText variant="caption" style={[styles.label, { color: selected ? colors.onTint : colors.text }]}>
        {label}
      </AppText>
    </Pressable>
  );
}

export function ChipRow<T extends string>({
  options,
  value,
  onChange,
  inset = true,
}: {
  options: { value: T; label: string }[];
  value: T;
  onChange: (value: T) => void;
  inset?: boolean;
}) {
  return (
    <ScrollView
      horizontal
      showsHorizontalScrollIndicator={false}
      accessibilityRole="radiogroup"
      contentContainerStyle={[styles.row, inset && styles.inset]}
    >
      {options.map((o) => (
        <Chip key={o.value} label={o.label} selected={o.value === value} onPress={() => onChange(o.value)} />
      ))}
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  chip: { paddingHorizontal: Spacing.md, paddingVertical: 6, borderRadius: Radius.pill },
  label: { fontWeight: "600" },
  row: { gap: Spacing.sm, paddingVertical: Spacing.sm },
  inset: { paddingHorizontal: Spacing.lg },
});
