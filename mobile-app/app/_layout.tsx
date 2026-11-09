import { DarkTheme, DefaultTheme, Theme, ThemeProvider } from "@react-navigation/native";
import { Stack } from "expo-router";
import { StatusBar } from "expo-status-bar";
import React from "react";

import { FixtureDataSourceProvider } from "@/data/DataSourceContext";
import { useColorScheme } from "@/hooks/use-color-scheme";
import { ConnectionStatusProvider } from "@/hooks/use-connection-status";
import { useTheme } from "@/hooks/use-theme";

export default function RootLayout() {
  const scheme = useColorScheme() === "dark" ? "dark" : "light";
  const colors = useTheme();
  const base = scheme === "dark" ? DarkTheme : DefaultTheme;
  const navTheme: Theme = {
    ...base,
    colors: {
      ...base.colors,
      primary: colors.tint,
      background: colors.background,
      card: colors.background,
      text: colors.text,
      border: colors.border,
    },
  };

  return (
    <FixtureDataSourceProvider>
      <ConnectionStatusProvider>
        <ThemeProvider value={navTheme}>
          <StatusBar style={scheme === "dark" ? "light" : "dark"} />
          <Stack>
            <Stack.Screen name="(tabs)" options={{ headerShown: false }} />
            <Stack.Screen name="history/[id]" options={{ title: "Conversation" }} />
            <Stack.Screen name="people/[id]" options={{ title: "Person" }} />
          </Stack>
        </ThemeProvider>
      </ConnectionStatusProvider>
    </FixtureDataSourceProvider>
  );
}
