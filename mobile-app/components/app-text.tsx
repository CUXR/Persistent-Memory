import React from "react";
import { StyleSheet, Text, TextProps } from "react-native";

import type { ThemeColors } from "@/constants/theme";
import { useTheme } from "@/hooks/use-theme";

type Variant = "title" | "heading" | "body" | "bodyStrong" | "caption" | "label";
type Tone = "default" | "muted" | "subtle" | "tint" | "danger" | "warning" | "success";

const toneColor: Record<Tone, keyof ThemeColors> = {
  default: "text",
  muted: "textMuted",
  subtle: "textSubtle",
  tint: "tint",
  danger: "danger",
  warning: "warning",
  success: "success",
};

export type AppTextProps = TextProps & { variant?: Variant; tone?: Tone };

export function AppText({ variant = "body", tone = "default", style, ...rest }: AppTextProps) {
  const colors = useTheme();
  return <Text style={[styles[variant], { color: colors[toneColor[tone]] }, style]} {...rest} />;
}

const styles = StyleSheet.create({
  title: { fontSize: 24, lineHeight: 30, fontWeight: "700" },
  heading: { fontSize: 17, lineHeight: 22, fontWeight: "600" },
  body: { fontSize: 15, lineHeight: 21 },
  bodyStrong: { fontSize: 15, lineHeight: 21, fontWeight: "600" },
  caption: { fontSize: 13, lineHeight: 18 },
  label: { fontSize: 12, lineHeight: 16, fontWeight: "600", letterSpacing: 0.4, textTransform: "uppercase" },
});
