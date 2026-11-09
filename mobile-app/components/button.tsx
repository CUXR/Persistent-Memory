import React from "react";
import { ActivityIndicator, Pressable, StyleProp, StyleSheet, ViewStyle } from "react-native";

import { AppText } from "@/components/app-text";
import { Radius, Spacing } from "@/constants/theme";
import { useTheme } from "@/hooks/use-theme";

type Props = {
  title: string;
  onPress: () => void;
  variant?: "primary" | "secondary" | "ghost";
  disabled?: boolean;
  loading?: boolean;
  style?: StyleProp<ViewStyle>;
};

export function Button({ title, onPress, variant = "primary", disabled, loading, style }: Props) {
  const colors = useTheme();
  const inactive = disabled || loading;
  const background =
    variant === "primary" ? colors.tint : variant === "secondary" ? colors.surface : "transparent";
  const textColor = variant === "primary" ? colors.onTint : colors.tint;

  return (
    <Pressable
      accessibilityRole="button"
      accessibilityState={{ disabled: inactive, busy: loading }}
      onPress={onPress}
      disabled={inactive}
      style={({ pressed }) => [
        styles.base,
        { backgroundColor: background, opacity: inactive ? 0.5 : pressed ? 0.75 : 1 },
        style,
      ]}
    >
      {loading ? (
        <ActivityIndicator size="small" color={textColor} />
      ) : (
        <AppText variant="bodyStrong" style={{ color: textColor }}>
          {title}
        </AppText>
      )}
    </Pressable>
  );
}

const styles = StyleSheet.create({
  base: {
    minHeight: 40,
    paddingHorizontal: Spacing.lg,
    borderRadius: Radius.md,
    alignItems: "center",
    justifyContent: "center",
  },
});
