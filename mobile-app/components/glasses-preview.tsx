import React from "react";
import { StyleSheet, View } from "react-native";

import { AppText } from "@/components/app-text";
import { Fonts, Radius, Spacing } from "@/constants/theme";
import type { Brief, DisplayMode } from "@/data/types";
import { useTheme } from "@/hooks/use-theme";

const FALLBACK: Brief = {
  personId: "sample",
  displayName: "Maya Chen",
  isUnnamed: false,
  hobbies: ["Runs with a Saturday morning running club"],
  affiliations: ["Works at Stripe as a product designer"],
};

// The glasses show the name plus the server-selected facts from the person's brief.
export function GlassesPreview({ mode, brief }: { mode: DisplayMode; brief?: Brief }) {
  const colors = useTheme();
  const b = brief ?? FALLBACK;
  const name = b.isUnnamed || !b.displayName ? "Unknown person" : b.displayName;
  const facts = [...b.affiliations, ...b.hobbies];
  const lensText = { color: colors.lensText, fontFamily: Fonts?.mono };

  const label =
    mode === "off"
      ? "Preview: the glasses display stays blank"
      : mode === "name"
        ? `Preview: the glasses show ${name}`
        : `Preview: the glasses show ${name}, ${facts.join(", ") || "and no facts yet"}`;

  return (
    <View accessible accessibilityLabel={label} style={[styles.lens, { backgroundColor: colors.lens }]}>
      {mode === "off" ? (
        <AppText variant="caption" style={{ color: colors.lensTextMuted, fontFamily: Fonts?.mono }}>
          display off
        </AppText>
      ) : (
        <View style={styles.content}>
          <AppText variant="heading" style={lensText}>
            {name}
          </AppText>
          {mode === "name_and_facts" ? (
            facts.length > 0 ? (
              facts.map((fact) => (
                <AppText key={fact} variant="caption" numberOfLines={1} style={lensText}>
                  · {fact}
                </AppText>
              ))
            ) : (
              <AppText variant="caption" style={{ color: colors.lensTextMuted, fontFamily: Fonts?.mono }}>
                no facts learned yet
              </AppText>
            )
          ) : null}
        </View>
      )}
    </View>
  );
}

const styles = StyleSheet.create({
  lens: {
    minHeight: 120,
    borderRadius: Radius.lg,
    alignItems: "center",
    justifyContent: "center",
    padding: Spacing.lg,
  },
  content: { alignSelf: "stretch", gap: 2 },
});
