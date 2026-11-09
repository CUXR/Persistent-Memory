import { Platform } from "react-native";

const light = {
  text: "#11181C",
  textMuted: "#5F676D",
  textSubtle: "#8A9196",
  background: "#FFFFFF",
  surface: "#F4F5F6",
  border: "#E3E6E8",
  tint: "#0A7EA4",
  tintSoft: "#E1F1F7",
  onTint: "#FFFFFF",
  success: "#1E8E5A",
  successSoft: "#E3F4EB",
  warning: "#9A6200",
  warningSoft: "#FFF4DB",
  danger: "#C93838",
  dangerSoft: "#FCE8E8",
  icon: "#687076",
  tabIconDefault: "#687076",
  tabIconSelected: "#0A7EA4",
  skeleton: "#E8EBED",
  lens: "#0B0F12",
  lensText: "#7CF5C4",
  lensTextMuted: "#4FA88A",
};

const dark: typeof light = {
  text: "#ECEDEE",
  textMuted: "#A1A7AC",
  textSubtle: "#7A8086",
  background: "#151718",
  surface: "#1E2124",
  border: "#2C3034",
  tint: "#4CB6DB",
  tintSoft: "#173341",
  onTint: "#0B1215",
  success: "#4ACB8B",
  successSoft: "#173326",
  warning: "#F0B44C",
  warningSoft: "#3A2C12",
  danger: "#F07171",
  dangerSoft: "#3D1C1C",
  icon: "#9BA1A6",
  tabIconDefault: "#9BA1A6",
  tabIconSelected: "#4CB6DB",
  skeleton: "#26292D",
  lens: "#000000",
  lensText: "#7CF5C4",
  lensTextMuted: "#4FA88A",
};

export const Colors = { light, dark };
export type ThemeColors = typeof light;

export const Spacing = { xs: 4, sm: 8, md: 12, lg: 16, xl: 24, xxl: 32 } as const;
export const Radius = { sm: 6, md: 10, lg: 14, pill: 999 } as const;

export const Fonts = Platform.select({
  ios: { sans: "system-ui", mono: "ui-monospace" },
  default: { sans: "normal", mono: "monospace" },
});
