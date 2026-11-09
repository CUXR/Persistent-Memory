import Ionicons from "@expo/vector-icons/Ionicons";
import React, { ComponentProps, useEffect, useRef } from "react";
import { ActivityIndicator, Animated, StyleSheet, View } from "react-native";

import { AppText } from "@/components/app-text";
import { Button } from "@/components/button";
import { Radius, Spacing } from "@/constants/theme";
import { useTheme } from "@/hooks/use-theme";

type IconName = ComponentProps<typeof Ionicons>["name"];

export function EmptyState({ icon, title, message }: { icon: IconName; title: string; message?: string }) {
  const colors = useTheme();
  return (
    <View style={styles.center}>
      <View style={[styles.iconCircle, { backgroundColor: colors.surface }]}>
        <Ionicons name={icon} size={28} color={colors.textMuted} />
      </View>
      <AppText variant="heading" style={styles.centerText}>
        {title}
      </AppText>
      {message ? (
        <AppText tone="muted" style={styles.centerText}>
          {message}
        </AppText>
      ) : null}
    </View>
  );
}

export function ErrorState({ message, onRetry }: { message: string; onRetry: () => void }) {
  const colors = useTheme();
  return (
    <View style={styles.center}>
      <View style={[styles.iconCircle, { backgroundColor: colors.dangerSoft }]}>
        <Ionicons name="cloud-offline-outline" size={28} color={colors.danger} />
      </View>
      <AppText variant="heading" style={styles.centerText}>
        Something went wrong
      </AppText>
      <AppText tone="muted" style={styles.centerText}>
        {message}
      </AppText>
      <Button title="Try again" onPress={onRetry} style={{ marginTop: Spacing.sm }} />
    </View>
  );
}

export function InlineError({ message, onRetry }: { message: string; onRetry?: () => void }) {
  const colors = useTheme();
  return (
    <View style={[styles.inlineError, { backgroundColor: colors.dangerSoft }]}>
      <Ionicons name="alert-circle" size={16} color={colors.danger} />
      <AppText variant="caption" tone="danger" style={{ flex: 1 }}>
        {message}
      </AppText>
      {onRetry ? <Button title="Retry" variant="ghost" onPress={onRetry} /> : null}
    </View>
  );
}

export function LoadMoreFooter({
  loading,
  error,
  onRetry,
}: {
  loading: boolean;
  error?: Error;
  onRetry: () => void;
}) {
  if (error) {
    return (
      <View style={styles.footer}>
        <InlineError message={error.message} onRetry={onRetry} />
      </View>
    );
  }
  if (loading) {
    return (
      <View style={styles.footer}>
        <ActivityIndicator accessibilityLabel="Loading more" />
      </View>
    );
  }
  return null;
}

function SkeletonBlock({ width, height, round }: { width: number | `${number}%`; height: number; round?: boolean }) {
  const colors = useTheme();
  return (
    <View
      style={{
        width,
        height,
        borderRadius: round ? height / 2 : Radius.sm,
        backgroundColor: colors.skeleton,
      }}
    />
  );
}

export function SkeletonRows({ count = 6, avatar = true }: { count?: number; avatar?: boolean }) {
  const colors = useTheme();
  const opacity = useRef(new Animated.Value(1)).current;

  useEffect(() => {
    const loop = Animated.loop(
      Animated.sequence([
        Animated.timing(opacity, { toValue: 0.45, duration: 700, useNativeDriver: true }),
        Animated.timing(opacity, { toValue: 1, duration: 700, useNativeDriver: true }),
      ]),
    );
    loop.start();
    return () => loop.stop();
  }, [opacity]);

  return (
    <Animated.View accessibilityLabel="Loading" accessibilityRole="progressbar" style={{ opacity }}>
      {Array.from({ length: count }, (_, i) => (
        <View key={i} style={[styles.skeletonRow, { borderBottomColor: colors.border }]}>
          {avatar ? <SkeletonBlock width={40} height={40} round /> : null}
          <View style={{ flex: 1, gap: Spacing.sm }}>
            <SkeletonBlock width="55%" height={14} />
            <SkeletonBlock width="85%" height={12} />
          </View>
        </View>
      ))}
    </Animated.View>
  );
}

const styles = StyleSheet.create({
  center: {
    flexGrow: 1,
    alignItems: "center",
    justifyContent: "center",
    gap: Spacing.sm,
    paddingHorizontal: Spacing.xxl,
    paddingVertical: Spacing.xxl,
  },
  centerText: { textAlign: "center" },
  iconCircle: {
    width: 56,
    height: 56,
    borderRadius: 28,
    alignItems: "center",
    justifyContent: "center",
    marginBottom: Spacing.sm,
  },
  inlineError: {
    flexDirection: "row",
    alignItems: "center",
    gap: Spacing.sm,
    paddingLeft: Spacing.md,
    borderRadius: Radius.md,
  },
  footer: { padding: Spacing.lg },
  skeletonRow: {
    flexDirection: "row",
    alignItems: "center",
    gap: Spacing.md,
    paddingHorizontal: Spacing.lg,
    paddingVertical: Spacing.md,
    borderBottomWidth: StyleSheet.hairlineWidth,
  },
});
