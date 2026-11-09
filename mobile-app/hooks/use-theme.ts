import { Colors, ThemeColors } from "@/constants/theme";
import { useColorScheme } from "@/hooks/use-color-scheme";

export function useTheme(): ThemeColors {
  return useColorScheme() === "dark" ? Colors.dark : Colors.light;
}
