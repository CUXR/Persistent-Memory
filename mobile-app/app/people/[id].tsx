import Ionicons from "@expo/vector-icons/Ionicons";
import { Stack, useLocalSearchParams } from "expo-router";
import React, { useState } from "react";
import { FlatList, KeyboardAvoidingView, Platform, Pressable, StyleSheet, TextInput, View } from "react-native";

import { AppText } from "@/components/app-text";
import { Avatar } from "@/components/avatar";
import { Button } from "@/components/button";
import { ChipRow } from "@/components/chip";
import { ListRow } from "@/components/list-row";
import { SectionHeader } from "@/components/section-header";
import { EmptyState, ErrorState, InlineError, LoadMoreFooter, SkeletonRows } from "@/components/state-views";
import { Radius, Spacing } from "@/constants/theme";
import { useDataSource } from "@/data/DataSourceContext";
import { Fact, FactCategory, NAME_PART_MAX_LENGTH, Person, PersonDetail } from "@/data/types";
import { usePagedList } from "@/hooks/use-paged-list";
import { toError, useQuery } from "@/hooks/use-query";
import { useTheme } from "@/hooks/use-theme";
import {
  FACT_CATEGORY_LABELS,
  formatConfidence,
  formatDate,
  formatDateTime,
  formatRelation,
  formatRelative,
  personLabel,
} from "@/lib/format";
import { openEpisode, openPerson } from "@/lib/routes";

type CategoryFilter = "all" | FactCategory;

const CATEGORY_FILTERS: { value: CategoryFilter; label: string }[] = [
  { value: "all", label: "All" },
  { value: "hobby", label: FACT_CATEGORY_LABELS.hobby },
  { value: "affiliation", label: FACT_CATEGORY_LABELS.affiliation },
  { value: "visual_descriptor", label: FACT_CATEGORY_LABELS.visual_descriptor },
];

const CATEGORY_ICONS: Record<FactCategory, React.ComponentProps<typeof Ionicons>["name"]> = {
  hobby: "bicycle-outline",
  affiliation: "business-outline",
  visual_descriptor: "eye-outline",
};

export default function PersonScreen() {
  const colors = useTheme();
  const { id } = useLocalSearchParams<{ id: string }>();
  const { source } = useDataSource();
  const [category, setCategory] = useState<CategoryFilter>("all");
  const { state, retry, setData } = useQuery(() => source.getPerson(id), [source, id]);
  const facts = usePagedList(
    (cursor) => source.listFacts(id, { category: category === "all" ? undefined : category, cursor }),
    [source, id, category],
  );

  if (state.status === "loading") {
    return (
      <View style={[styles.screen, { backgroundColor: colors.background }]}>
        <SkeletonRows count={6} />
      </View>
    );
  }
  if (state.status === "error") {
    return (
      <View style={[styles.screen, { backgroundColor: colors.background }]}>
        <ErrorState message={state.error.message} onRetry={retry} />
      </View>
    );
  }

  const person = state.data;

  let factsEmpty: React.ReactNode;
  if (facts.status === "loading") factsEmpty = <SkeletonRows count={3} avatar={false} />;
  else if (facts.status === "error")
    factsEmpty = (
      <View style={styles.inset}>
        <InlineError message={facts.error?.message ?? "Couldn't load facts."} onRetry={facts.retry} />
      </View>
    );
  else
    factsEmpty = (
      <EmptyState
        icon="bulb-outline"
        title={category === "all" ? "Nothing learned yet" : `No ${FACT_CATEGORY_LABELS[category].toLowerCase()} yet`}
        message="Facts the server learns about this person from conversations will show up here."
      />
    );

  return (
    <KeyboardAvoidingView style={styles.screen} behavior={Platform.OS === "ios" ? "padding" : undefined}>
      <Stack.Screen options={{ title: personLabel(person) }} />
      <FlatList
        style={{ backgroundColor: colors.background }}
        contentContainerStyle={styles.content}
        keyboardShouldPersistTaps="handled"
        data={facts.status === "success" ? facts.items : []}
        keyExtractor={(fact) => fact.id}
        renderItem={({ item }) => <FactRow fact={item} />}
        onEndReached={facts.loadMore}
        onEndReachedThreshold={0.5}
        ListHeaderComponent={
          <>
            <PersonHeader person={person} onRenamed={(updated) => setData({ ...person, ...updated })} />
            <ChipRow options={CATEGORY_FILTERS} value={category} onChange={setCategory} />
          </>
        }
        ListEmptyComponent={factsEmpty}
        ListFooterComponent={
          <LoadMoreFooter loading={facts.loadingMore} error={facts.loadMoreError} onRetry={facts.loadMore} />
        }
      />
    </KeyboardAvoidingView>
  );
}

function PersonHeader({ person, onRenamed }: { person: PersonDetail; onRenamed: (updated: Person) => void }) {
  const colors = useTheme();
  const seen = person.lastSeenAt ? `Last seen ${formatRelative(person.lastSeenAt)}` : "Not seen yet";

  return (
    <View>
      <View style={styles.hero}>
        <Avatar id={person.id} name={person.displayName} size={72} />
        <NameEditor person={person} onRenamed={onRenamed} />
        <AppText variant="caption" tone="muted">
          {seen} · {person.episodeCount} conversation{person.episodeCount === 1 ? "" : "s"}
        </AppText>
      </View>

      {person.latestSummary ? (
        <>
          <SectionHeader title="Latest summary" />
          <View style={[styles.card, { backgroundColor: colors.surface }]}>
            <AppText>{person.latestSummary.text}</AppText>
            {person.latestSummary.episodeTimeStart ? (
              <AppText variant="caption" tone="subtle">
                From {formatDateTime(person.latestSummary.episodeTimeStart)}
              </AppText>
            ) : null}
          </View>
        </>
      ) : null}

      {person.relationships.length > 0 ? (
        <>
          <SectionHeader title="Connections" />
          <View style={styles.relationships}>
            {person.relationships.map((r) => (
              <Pressable
                key={r.id}
                accessibilityRole="link"
                accessibilityLabel={`${formatRelation(r.relation)} ${personLabel(r.target)}`}
                onPress={() => openPerson(r.target.id)}
                style={({ pressed }) => [
                  styles.relationship,
                  { backgroundColor: pressed ? colors.border : colors.surface },
                ]}
              >
                <AppText variant="caption" tone="muted">
                  {formatRelation(r.relation)}
                </AppText>
                <AppText variant="caption" style={styles.bold}>
                  {personLabel(r.target)}
                </AppText>
              </Pressable>
            ))}
          </View>
        </>
      ) : null}

      {person.recentEpisodes.length > 0 ? (
        <>
          <SectionHeader title="Recent conversations" />
          {person.recentEpisodes.map((e) => (
            <ListRow
              key={e.id}
              title={formatDateTime(e.startTime)}
              subtitle={e.summary}
              subtitleLines={2}
              onPress={() => openEpisode(e.id)}
            />
          ))}
        </>
      ) : null}

      <SectionHeader title="What you've learned" />
    </View>
  );
}

type SaveState = { kind: "idle" } | { kind: "saving" } | { kind: "saved" } | { kind: "error"; message: string };

function NameEditor({ person, onRenamed }: { person: PersonDetail; onRenamed: (updated: Person) => void }) {
  const colors = useTheme();
  const { source, invalidate } = useDataSource();
  const [editing, setEditing] = useState(false);
  const [first, setFirst] = useState(person.firstName);
  const [last, setLast] = useState(person.lastName);
  const [save, setSave] = useState<SaveState>({ kind: "idle" });

  const firstTrimmed = first.trim();
  const lastTrimmed = last.trim();
  const firstError = !firstTrimmed
    ? "First name is required."
    : firstTrimmed.length > NAME_PART_MAX_LENGTH
      ? `First name must be ${NAME_PART_MAX_LENGTH} characters or fewer.`
      : null;
  const lastError =
    lastTrimmed.length > NAME_PART_MAX_LENGTH ? `Last name must be ${NAME_PART_MAX_LENGTH} characters or fewer.` : null;
  const unchanged = !person.isUnnamed && firstTrimmed === person.firstName && lastTrimmed === person.lastName;
  const touched = first !== person.firstName || last !== person.lastName;

  const startEditing = () => {
    setFirst(person.firstName);
    setLast(person.lastName);
    setSave({ kind: "idle" });
    setEditing(true);
  };

  const submit = async () => {
    if (firstError || lastError || unchanged) return;
    setSave({ kind: "saving" });
    try {
      const updated = await source.renamePerson(person.id, { firstName: firstTrimmed, lastName: lastTrimmed });
      onRenamed(updated);
      invalidate();
      setEditing(false);
      setSave({ kind: "saved" });
    } catch (e) {
      setSave({ kind: "error", message: toError(e).message });
    }
  };

  if (!editing) {
    if (person.isUnnamed) {
      return (
        <View style={[styles.unnamedCallout, { backgroundColor: colors.tintSoft }]}>
          <AppText variant="heading" style={styles.centerText}>
            Unnamed person
          </AppText>
          <AppText variant="caption" tone="muted" style={styles.centerText}>
            Your glasses saved this face but don&apos;t know who it is yet. Add a name so the glasses can show it.
          </AppText>
          <Button title="Add name" onPress={startEditing} style={styles.calloutButton} />
        </View>
      );
    }
    return (
      <View style={styles.nameBlock}>
        <View style={styles.nameRow}>
          <AppText variant="title" style={styles.centerText}>
            {person.displayName}
          </AppText>
          <Pressable
            accessibilityRole="button"
            accessibilityLabel={`Rename ${person.displayName}`}
            hitSlop={10}
            onPress={startEditing}
            style={[styles.editButton, { backgroundColor: colors.surface }]}
          >
            <Ionicons name="pencil" size={14} color={colors.tint} />
          </Pressable>
        </View>
        {save.kind === "saved" ? (
          <AppText variant="caption" tone="success">
            Name saved
          </AppText>
        ) : null}
      </View>
    );
  }

  const saving = save.kind === "saving";
  const inputStyle = (hasError: boolean) => [
    styles.input,
    { color: colors.text, backgroundColor: colors.surface, borderColor: hasError ? colors.danger : colors.border },
  ];

  return (
    <View style={styles.form}>
      <AppText variant="label" tone="muted">
        First name
      </AppText>
      <TextInput
        value={first}
        onChangeText={(text) => {
          setFirst(text);
          if (save.kind === "error") setSave({ kind: "idle" });
        }}
        autoFocus
        autoCapitalize="words"
        textContentType="givenName"
        returnKeyType="next"
        editable={!saving}
        accessibilityLabel="First name"
        style={inputStyle(touched && Boolean(firstError))}
      />
      <AppText variant="label" tone="muted">
        Last name (optional)
      </AppText>
      <TextInput
        value={last}
        onChangeText={(text) => {
          setLast(text);
          if (save.kind === "error") setSave({ kind: "idle" });
        }}
        autoCapitalize="words"
        textContentType="familyName"
        returnKeyType="done"
        onSubmitEditing={submit}
        editable={!saving}
        accessibilityLabel="Last name"
        style={inputStyle(Boolean(lastError))}
      />
      <AppText variant="caption" tone={(touched && firstError) || lastError ? "danger" : "subtle"}>
        {(touched && firstError) || lastError || "The glasses will use this name from now on."}
      </AppText>
      {save.kind === "error" ? <InlineError message={save.message} /> : null}
      <View style={styles.formActions}>
        <Button title="Cancel" variant="secondary" onPress={() => setEditing(false)} disabled={saving} style={styles.flex} />
        <Button
          title="Save"
          onPress={submit}
          loading={saving}
          disabled={Boolean(firstError || lastError) || unchanged}
          style={styles.flex}
        />
      </View>
    </View>
  );
}

function FactRow({ fact }: { fact: Fact }) {
  const colors = useTheme();
  const sourceId = fact.sourceEpisodeId;
  const details = [
    fact.category ? FACT_CATEGORY_LABELS[fact.category] : undefined,
    `Learned ${formatDate(fact.createdAt)}`,
    formatConfidence(fact.confidence),
  ].filter(Boolean);

  return (
    <View style={[styles.fact, { borderBottomColor: colors.border }]}>
      <Ionicons
        name={fact.category ? CATEGORY_ICONS[fact.category] : "bulb-outline"}
        size={16}
        color={colors.tint}
        style={styles.factIcon}
      />
      <View style={styles.factBody}>
        <AppText>{fact.text}</AppText>
        <AppText variant="caption" tone="subtle">
          {details.join(" · ")}
        </AppText>
        {sourceId ? (
          <Pressable accessibilityRole="link" hitSlop={8} onPress={() => openEpisode(sourceId)}>
            <AppText variant="caption" tone="tint">
              View conversation
            </AppText>
          </Pressable>
        ) : null}
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  screen: { flex: 1 },
  flex: { flex: 1 },
  content: { paddingBottom: Spacing.xxl },
  inset: { paddingHorizontal: Spacing.lg },
  hero: { alignItems: "center", gap: Spacing.sm, paddingHorizontal: Spacing.lg, paddingTop: Spacing.xl },
  nameBlock: { alignItems: "center", gap: 2 },
  nameRow: { flexDirection: "row", alignItems: "center", gap: Spacing.sm },
  centerText: { textAlign: "center", flexShrink: 1 },
  bold: { fontWeight: "600" },
  editButton: { width: 28, height: 28, borderRadius: 14, alignItems: "center", justifyContent: "center" },
  unnamedCallout: { alignSelf: "stretch", alignItems: "center", gap: Spacing.xs, padding: Spacing.lg, borderRadius: Radius.lg },
  calloutButton: { marginTop: Spacing.sm, alignSelf: "stretch" },
  form: { alignSelf: "stretch", gap: Spacing.sm },
  input: {
    fontSize: 17,
    borderWidth: 1,
    borderRadius: Radius.md,
    paddingHorizontal: Spacing.md,
    paddingVertical: Spacing.sm + 2,
  },
  formActions: { flexDirection: "row", gap: Spacing.sm },
  card: { marginHorizontal: Spacing.lg, padding: Spacing.md, borderRadius: Radius.md, gap: Spacing.xs },
  relationships: { flexDirection: "row", flexWrap: "wrap", gap: Spacing.sm, paddingHorizontal: Spacing.lg },
  relationship: { paddingHorizontal: Spacing.md, paddingVertical: Spacing.sm, borderRadius: Radius.md },
  fact: {
    flexDirection: "row",
    gap: Spacing.md,
    paddingHorizontal: Spacing.lg,
    paddingVertical: Spacing.md,
    borderBottomWidth: StyleSheet.hairlineWidth,
  },
  factIcon: { marginTop: 3 },
  factBody: { flex: 1, gap: 2 },
});
