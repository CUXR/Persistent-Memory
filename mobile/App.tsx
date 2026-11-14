import { useState } from 'react';
import { StatusBar } from 'expo-status-bar';
import { SafeAreaView, StyleSheet } from 'react-native';

import { FixtureAdapter } from './src/data';
import SmokeScreen from './src/screens/SmokeScreen';

export default function App() {
  // Visible artificial latency so the smoke screen's "Loading…" state is reachable.
  const [adapter] = useState(() => new FixtureAdapter({ delayMs: 400 }));

  return (
    <SafeAreaView style={styles.container}>
      <SmokeScreen adapter={adapter} />
      <StatusBar style="auto" />
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#fff',
  },
});
