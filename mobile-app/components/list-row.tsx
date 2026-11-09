import Ionicons from "@expo/vector-icons/Ionicons";
import React, { ReactNode } from "react";
import { Pressable, StyleProp, StyleSheet, View, ViewStyle } from "react-native";

import { AppText } from "@/components/app-text";
import { Spacing } from "@/constants/theme";
import { useTheme } from "@/hooks/use-theme";

type Props = {
  title: string;
  subtitle?: string;
  meta?: string;
  leading?: ReactNode;
  badge?: ReactNode;
  onPress?: () => void;
  subtitleLines?: number;
  // Muted, italic subtitle for content that isn't final yet.
  tentative?: boolean;
  style?: StyleProp<ViewStyle>;
};

export function ListRow({ title, subtitle, meta, leading, badge, onPress, subtitleLines = 2, tentative, style }: Props) {
  const colors = useTheme();
  return (
    <Pressable
      accessibilityRole={onPress ? "button" : undefined}
      onPress={onPress}
      disabled={!onPress}
      style={({ pressed }) => [
        styles.row,
        { backgroundColor: pressed ? colors.surface : colors.background, borderBottomColor: colors.border },
        style,
      ]}
    >
      {leading}
      <View style={styles.body}>
        <View style={styles.titleLine}>
          <AppText variant="bodyStrong" numberOfLines={1} style={styles.title}>
            {title}
          </AppText>
          {meta ? (
            <AppText variant="caption" tone="subtle">
              {meta}
            </AppText>
          ) : null}
        </View>
        {subtitle ? (
          <AppText
            variant="caption"
            tone="muted"
            numberOfLines={subtitleLines}
            style={tentative ? styles.tentative : undefined}
          >
            {subtitle}
          </AppText>
        ) : null}
        {badge ? <View style={styles.badge}>{badge}</View> : null}
      </View>
      {onPress ? <Ionicons name="chevron-forward" size={18} color={colors.textSubtle} /> : null}
    </Pressable>
  );
}

const styles = StyleSheet.create({
  row: {
    flexDirection: "row",
    alignItems: "center",
    gap: Spacing.md,
    paddingHorizontal: Spacing.lg,
    paddingVertical: Spacing.md,
    borderBottomWidth: StyleSheet.hairlineWidth,
  },
  body: { flex: 1, gap: 2 },
  titleLine: { flexDirection: "row", alignItems: "baseline", gap: Spacing.sm },
  title: { flex: 1 },
  tentative: { fontStyle: "italic" },
  badge: { flexDirection: "row", marginTop: Spacing.xs },
});
