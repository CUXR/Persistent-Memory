import { useCallback, useEffect, useState } from 'react';
import { Button, ScrollView, StyleSheet, Text, View } from 'react-native';

import type { ConversationListItem, Page, PersonListItem, RequestState } from '../domain';
import type { DataKey, FixtureAdapter, Scenario } from '../data';

/**
 * Not a real screen — #48 owns real screens and will delete/replace this.
 * Proves the scaffold + adapter run end-to-end on iOS/Android: renders
 * loading/empty/error/populated for two panes independently, and exercises
 * rename + display-mode actions against the same FixtureAdapter instance.
 */
export default function SmokeScreen({ adapter }: { adapter: FixtureAdapter }) {
  return (
    <ScrollView contentContainerStyle={styles.container}>
      <Text style={styles.title}>Mobile data-layer smoke screen</Text>
      <ConversationsSection adapter={adapter} />
      <PeopleSection adapter={adapter} />
      <DisplayModeSection adapter={adapter} />
    </ScrollView>
  );
}

function ScenarioButtons({
  dataKey,
  adapter,
  onChange,
}: {
  dataKey: DataKey;
  adapter: FixtureAdapter;
  onChange: () => void;
}) {
  const pick = (scenario: Scenario) => {
    adapter.setScenario(dataKey, scenario);
    onChange();
  };
  return (
    <View style={styles.row}>
      <Button title="Populated" onPress={() => pick('populated')} />
      <Button title="Empty" onPress={() => pick('empty')} />
      <Button title="Error" onPress={() => pick('error')} />
    </View>
  );
}

function ConversationsSection({ adapter }: { adapter: FixtureAdapter }) {
  const [state, setState] = useState<RequestState<Page<ConversationListItem>>>({ status: 'loading' });

  const fetchData = useCallback(() => {
    adapter
      .listConversations({ limit: 5 })
      .then((data) => setState({ status: 'success', data }))
      .catch((error) => setState({ status: 'error', error }));
  }, [adapter]);

  const load = useCallback(() => {
    setState((previous) => ({ status: 'loading', previous: previous.status === 'success' ? previous.data : undefined }));
    fetchData();
  }, [fetchData]);

  useEffect(() => {
    fetchData();
  }, [fetchData]);

  return (
    <View style={styles.section}>
      <Text style={styles.heading}>Recent conversations</Text>
      <ScenarioButtons dataKey="conversations" adapter={adapter} onChange={load} />
      {state.status === 'loading' && <Text>Loading…</Text>}
      {state.status === 'error' && <Text style={styles.error}>Error: {state.error.message}</Text>}
      {state.status === 'success' && state.data.items.length === 0 && <Text>No conversations yet.</Text>}
      {state.status === 'success' &&
        state.data.items.map((item) => (
          <Text key={item.id}>
            • {item.summary} [{item.provisionalStatus}]
          </Text>
        ))}
      <Button title="Reload" onPress={load} />
    </View>
  );
}

function PeopleSection({ adapter }: { adapter: FixtureAdapter }) {
  const [state, setState] = useState<RequestState<Page<PersonListItem>>>({ status: 'loading' });

  const fetchData = useCallback(() => {
    adapter
      .listPeople({})
      .then((data) => setState({ status: 'success', data }))
      .catch((error) => setState({ status: 'error', error }));
  }, [adapter]);

  const load = useCallback(() => {
    setState((previous) => ({ status: 'loading', previous: previous.status === 'success' ? previous.data : undefined }));
    fetchData();
  }, [fetchData]);

  useEffect(() => {
    fetchData();
  }, [fetchData]);

  const renameFirst = () => {
    const first = state.status === 'success' ? state.data.items[0] : undefined;
    if (!first) return;
    adapter.renamePerson({ personId: first.id, displayName: `${first.name} (renamed)` }).then(load);
  };

  return (
    <View style={styles.section}>
      <Text style={styles.heading}>People</Text>
      <ScenarioButtons dataKey="people" adapter={adapter} onChange={load} />
      {state.status === 'loading' && <Text>Loading…</Text>}
      {state.status === 'error' && <Text style={styles.error}>Error: {state.error.message}</Text>}
      {state.status === 'success' && state.data.items.length === 0 && <Text>No people yet.</Text>}
      {state.status === 'success' &&
        state.data.items.map((item) => <Text key={item.id}>• {item.name}</Text>)}
      <Button title="Rename first person" onPress={renameFirst} />
      <Button title="Reload" onPress={load} />
    </View>
  );
}

function DisplayModeSection({ adapter }: { adapter: FixtureAdapter }) {
  const [status, setStatus] = useState('idle');
  const [modeLabel, setModeLabel] = useState('(unloaded)');

  const load = useCallback(() => {
    adapter.getDisplayMode().then((setting) => {
      setStatus(setting.syncStatus);
      setModeLabel(setting.mode);
    });
  }, [adapter]);

  useEffect(load, [load]);

  const set = (mode: 'none' | 'name' | 'name_and_facts') => {
    setStatus('pending');
    adapter
      .setDisplayMode(mode)
      .then((setting) => {
        setStatus(setting.syncStatus);
        setModeLabel(setting.mode);
      })
      .catch(() => setStatus('error'));
  };

  return (
    <View style={styles.section}>
      <Text style={styles.heading}>Glasses display mode</Text>
      <Text>
        Current: {modeLabel} ({status})
      </Text>
      <View style={styles.row}>
        <Button title="None" onPress={() => set('none')} />
        <Button title="Name" onPress={() => set('name')} />
        <Button title="Name + facts" onPress={() => set('name_and_facts')} />
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: { padding: 16, gap: 16 },
  title: { fontSize: 18, fontWeight: 'bold' },
  section: { gap: 4, borderTopWidth: 1, borderColor: '#ddd', paddingTop: 12 },
  heading: { fontSize: 16, fontWeight: '600' },
  row: { flexDirection: 'row', gap: 8 },
  error: { color: 'crimson' },
});
