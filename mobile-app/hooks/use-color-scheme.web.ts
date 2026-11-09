import { useSyncExternalStore } from 'react';
import { Appearance, ColorSchemeName } from 'react-native';

function subscribe(onChange: () => void) {
  const subscription = Appearance.addChangeListener(onChange);
  return () => subscription.remove();
}

function getSnapshot(): ColorSchemeName {
  return Appearance.getColorScheme();
}

// The static HTML is rendered without knowing the viewer's theme, so the server
// snapshot is 'light'. useSyncExternalStore then switches every subscriber to the
// real scheme in the same render after hydration, so no component is left on the
// old scheme (a per-component "hasHydrated" flag let some rows stay light while
// others turned dark).
function getServerSnapshot(): ColorSchemeName {
  return 'light';
}

export function useColorScheme(): ColorSchemeName {
  return useSyncExternalStore(subscribe, getSnapshot, getServerSnapshot);
}
