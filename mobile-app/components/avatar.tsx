import React from "react";
import { StyleSheet, View } from "react-native";

import { AppText } from "@/components/app-text";

const PALETTE = ["#4C8DAE", "#8E6CC0", "#C0765A", "#4E9A72", "#B5832E", "#5A6FC0", "#B05A84", "#3E8F8F"];

function hash(text: string) {
  let h = 0;
  for (let i = 0; i < text.length; i++) h = (h * 31 + text.charCodeAt(i)) | 0;
  return Math.abs(h);
}

function initials(name: string) {
  const parts = name.trim().split(/\s+/).filter(Boolean);
  if (parts.length === 0) return "?";
  if (parts[0] === "Unknown" || parts[0] === "Unnamed") return "?";
  return (parts[0][0] + (parts.length > 1 ? parts[parts.length - 1][0] : "")).toUpperCase();
}

export function Avatar({ id, name, size = 40 }: { id: string; name: string; size?: number }) {
  return (
    <View
      accessibilityElementsHidden
      importantForAccessibility="no-hide-descendants"
      style={[
        styles.circle,
        { width: size, height: size, borderRadius: size / 2, backgroundColor: PALETTE[hash(id) % PALETTE.length] },
      ]}
    >
      <AppText style={{ color: "#FFFFFF", fontSize: size * 0.38, fontWeight: "600" }}>{initials(name)}</AppText>
    </View>
  );
}

const styles = StyleSheet.create({
  circle: { alignItems: "center", justifyContent: "center" },
});
