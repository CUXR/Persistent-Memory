import Ionicons from "@expo/vector-icons/Ionicons";
import { Tabs } from "expo-router";
import React, { ComponentProps } from "react";

import { ConnectionPill } from "@/components/connection-pill";
import { HapticTab } from "@/components/haptic-tab";
import { useTheme } from "@/hooks/use-theme";

type IconName = ComponentProps<typeof Ionicons>["name"];

function tabIcon(name: IconName) {
  return function TabIcon({ color, size }: { color: string; size: number }) {
    return <Ionicons name={name} size={size} color={color} />;
  };
}

export default function TabLayout() {
  const colors = useTheme();

  return (
    <Tabs
      screenOptions={{
        tabBarActiveTintColor: colors.tabIconSelected,
        tabBarInactiveTintColor: colors.tabIconDefault,
        tabBarButton: HapticTab,
        headerRight: () => <ConnectionPill />,
      }}
    >
      <Tabs.Screen name="index" options={{ title: "History", tabBarIcon: tabIcon("time-outline") }} />
      <Tabs.Screen name="people" options={{ title: "People", tabBarIcon: tabIcon("people-outline") }} />
      <Tabs.Screen name="glasses" options={{ title: "Glasses", tabBarIcon: tabIcon("glasses-outline") }} />
    </Tabs>
  );
}
