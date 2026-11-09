import Ionicons from "@expo/vector-icons/Ionicons";
import React, { ComponentProps } from "react";
import { Pressable, StyleSheet, View } from "react-native";

import { AppText } from "@/components/app-text";
import { Radius, Spacing } from "@/constants/theme";
import { useTheme } from "@/hooks/use-theme";

type Props = {
  icon: ComponentProps<typeof Ionicons>["name"];
  title: string;
  description: string;
  selected: boolean;
  disabled?: boolean;
  onPress: () => void;
};

export function OptionCard({ icon, title, description, selected, disabled, onPress }: Props) {
  const colors = useTheme();
  return (
    <Pressable
      accessibilityRole="radio"
      accessibilityState={{ checked: selected, disabled }}
      accessibilityLabel={`${title}. ${description}`}
      onPress={onPress}
      disabled={disabled}
      style={({ pressed }) => [
        styles.card,
        {
          borderColor: selected ? colors.tint : colors.border,
          backgroundColor: selected ? colors.tintSoft : pressed ? colors.surface : colors.background,
          opacity: disabled && !selected ? 0.6 : 1,
        },
      ]}
    >
      <Ionicons name={icon} size={22} color={selected ? colors.tint : colors.icon} />
      <View style={styles.body}>
        <AppText variant="bodyStrong">{title}</AppText>
        <AppText variant="caption" tone="muted">
          {description}
        </AppText>
      </View>
      <View style={[styles.radio, { borderColor: selected ? colors.tint : colors.textSubtle }]}>
        {selected ? <View style={[styles.radioDot, { backgroundColor: colors.tint }]} /> : null}
      </View>
    </Pressable>
  );
}

const styles = StyleSheet.create({
  card: {
    flexDirection: "row",
    alignItems: "center",
    gap: Spacing.md,
    padding: Spacing.lg,
    borderRadius: Radius.lg,
    borderWidth: 1.5,
  },
  body: { flex: 1, gap: 2 },
  radio: { width: 20, height: 20, borderRadius: 10, borderWidth: 2, alignItems: "center", justifyContent: "center" },
  radioDot: { width: 10, height: 10, borderRadius: 5 },
});
