import Ionicons from "@expo/vector-icons/Ionicons";
import { Stack, useLocalSearchParams } from "expo-router";
import React, { ReactNode } from "react";
import { FlatList, Pressable, StyleSheet, View } from "react-native";

import { AppText } from "@/components/app-text";
import { Avatar } from "@/components/avatar";
import { MemoryBadge } from "@/components/memory-badge";
import { EmptyState, ErrorState, SkeletonRows } from "@/components/state-views";
import { Radius, Spacing } from "@/constants/theme";
import { useDataSource } from "@/data/DataSourceContext";
import type { Episode, Fact, PersonRef, Recording, TranscriptTurn } from "@/data/types";
import { useQuery } from "@/hooks/use-query";
import { useTheme } from "@/hooks/use-theme";
import {
  FACT_CATEGORY_LABELS,
  formatDateTime,
  formatDuration,
  formatOffset,
  formatSeconds,
  personLabel,
} from "@/lib/format";
import { openPerson } from "@/lib/routes";

const LOW_CONFIDENCE = 0.6;

export default function HistoryDetailScreen() {
  const { id, kind } = useLocalSearchParams<{ id: string; kind?: string }>();
  return kind === "recording" ? <RecordingView id={id} /> : <EpisodeView id={id} />;
}

function Loading() {
  const colors = useTheme();
  return (
    <View style={[styles.screen, { backgroundColor: colors.background }]}>
      <SkeletonRows count={8} avatar={false} />
    </View>
  );
}

function Failed({ message, onRetry }: { message: string; onRetry: () => void }) {
  const colors = useTheme();
  return (
    <View style={[styles.screen, { backgroundColor: colors.background }]}>
      <ErrorState message={message} onRetry={onRetry} />
    </View>
  );
}

function EpisodeView({ id }: { id: string }) {
  const colors = useTheme();
  const { source } = useDataSource();
  const { state, retry } = useQuery(() => source.getEpisode(id), [source, id]);

  if (state.status === "loading") return <Loading />;
  if (state.status === "error") return <Failed message={state.error.message} onRetry={retry} />;

  const episode = state.data;
  return (
    <FlatList
      style={{ backgroundColor: colors.background }}
      contentContainerStyle={styles.content}
      data={episode.transcript}
      keyExtractor={(turn) => turn.id}
      renderItem={({ item }) => <Turn turn={item} provisional={false} />}
      ListHeaderComponent={<EpisodeHeader episode={episode} />}
      ListEmptyComponent={<EmptyState icon="mic-outline" title="No transcript" message="Nothing was transcribed." />}
    />
  );
}

function EpisodeHeader({ episode }: { episode: Episode }) {
  const colors = useTheme();
  return (
    <View style={styles.header}>
      <Stack.Screen options={{ title: "Conversation" }} />
      <View style={styles.headerTop}>
        <AppText variant="caption" tone="muted" style={{ flex: 1 }}>
          {formatDateTime(episode.startTime)} · {formatDuration(episode.startTime, episode.endTime)}
        </AppText>
        <MemoryBadge state="saved" />
      </View>
      <PeopleChips people={episode.people} />

      {episode.summary ? (
        <View style={[styles.card, { backgroundColor: colors.surface }]}>
          <AppText variant="label" tone="muted">
            Summary
          </AppText>
          <AppText>{episode.summary}</AppText>
        </View>
      ) : null}

      {episode.learnedFacts.length > 0 ? <LearnedFacts facts={episode.learnedFacts} /> : null}

      <AppText variant="label" tone="muted" style={styles.transcriptLabel}>
        Transcript
      </AppText>
    </View>
  );
}

function LearnedFacts({ facts }: { facts: Fact[] }) {
  const colors = useTheme();
  return (
    <View style={[styles.card, { backgroundColor: colors.surface }]}>
      <AppText variant="label" tone="muted">
        Learned in this conversation
      </AppText>
      {facts.map((fact) => (
        <View key={fact.id} style={styles.learnedFact}>
          <Ionicons name="bulb-outline" size={14} color={colors.tint} style={{ marginTop: 3 }} />
          <View style={{ flex: 1 }}>
            <AppText>{fact.text}</AppText>
            {fact.category ? (
              <AppText variant="caption" tone="subtle">
                {FACT_CATEGORY_LABELS[fact.category]}
              </AppText>
            ) : null}
          </View>
        </View>
      ))}
    </View>
  );
}

function RecordingView({ id }: { id: string }) {
  const colors = useTheme();
  const { source } = useDataSource();
  const { state, retry } = useQuery(() => source.getRecording(id), [source, id]);

  if (state.status === "loading") return <Loading />;
  if (state.status === "error") return <Failed message={state.error.message} onRetry={retry} />;

  const recording = state.data;
  return (
    <FlatList
      style={{ backgroundColor: colors.background }}
      contentContainerStyle={styles.content}
      data={recording.transcript}
      keyExtractor={(turn) => turn.id}
      renderItem={({ item }) => <Turn turn={item} provisional />}
      ListHeaderComponent={<RecordingHeader recording={recording} />}
      ListEmptyComponent={
        <EmptyState
          icon="mic-outline"
          title="No transcript yet"
          message={
            recording.status === "failed"
              ? "Processing stopped before any speech was transcribed."
              : "The transcript will appear here once the server transcribes the audio."
          }
        />
      }
    />
  );
}

function recordingBanner(recording: Recording): { title: string; body: string } {
  switch (recording.status) {
    case "pending":
      return {
        title: "Queued for processing",
        body: "This recording is saved and waiting for the server to transcribe it. It isn't a memory yet.",
      };
    case "processing":
      return {
        title: "Provisional transcript",
        body: "The server is still processing this recording. Words and speakers may change, and it isn't saved as a memory yet.",
      };
    case "needs_review":
      return {
        title: "Needs review",
        body: `${recording.error ?? "The server couldn't confirm who was speaking."} The transcript is kept, but no memories were created from it.`,
      };
    case "failed":
      return {
        title: `Processing failed (${recording.stage})`,
        body: `${recording.error ?? "Something went wrong while processing."} No memories were created.`,
      };
  }
}

function RecordingHeader({ recording }: { recording: Recording }) {
  const colors = useTheme();
  const failed = recording.status === "failed";
  const fg = failed ? colors.danger : colors.warning;
  const bg = failed ? colors.dangerSoft : colors.warningSoft;
  const banner = recordingBanner(recording);

  return (
    <View style={styles.header}>
      <Stack.Screen options={{ title: "Recording" }} />
      <View style={styles.headerTop}>
        <AppText variant="caption" tone="muted" style={{ flex: 1 }}>
          {formatDateTime(recording.startTime)} · {formatSeconds(recording.durationSeconds)}
        </AppText>
        <MemoryBadge state={recording.status} />
      </View>
      <PeopleChips people={[recording.person]} />

      <View style={[styles.banner, { backgroundColor: bg, borderColor: fg }]}>
        <Ionicons name={failed ? "close-circle-outline" : "hourglass-outline"} size={18} color={fg} />
        <View style={{ flex: 1 }}>
          <AppText variant="bodyStrong" style={{ color: fg }}>
            {banner.title}
          </AppText>
          <AppText variant="caption" style={{ color: fg }}>
            {banner.body}
          </AppText>
        </View>
      </View>

      {recording.failedSegmentCount > 0 ? (
        <AppText variant="caption" tone="danger">
          {recording.failedSegmentCount} audio segment{recording.failedSegmentCount === 1 ? "" : "s"} couldn&apos;t be
          transcribed.
        </AppText>
      ) : null}

      <AppText variant="label" tone="muted" style={styles.transcriptLabel}>
        Transcript
      </AppText>
    </View>
  );
}

function PeopleChips({ people }: { people: PersonRef[] }) {
  const colors = useTheme();
  return (
    <View style={styles.people}>
      {people.map((person) => (
        <Pressable
          key={person.id}
          accessibilityRole="link"
          accessibilityLabel={`Open ${personLabel(person)}`}
          onPress={() => openPerson(person.id)}
          style={({ pressed }) => [styles.personChip, { backgroundColor: pressed ? colors.border : colors.surface }]}
        >
          <Avatar id={person.id} name={person.displayName} size={22} />
          <AppText variant="caption" style={styles.chipText}>
            {personLabel(person)}
          </AppText>
        </Pressable>
      ))}
    </View>
  );
}

function Turn({ turn, provisional }: { turn: TranscriptTurn; provisional: boolean }) {
  const colors = useTheme();
  const isWearer = turn.speaker === "user";
  const unknown = turn.speaker === "unknown";
  const lowConfidence = turn.asrConfidence !== undefined && turn.asrConfidence < LOW_CONFIDENCE;

  let meta: ReactNode = (
    <AppText variant="caption" tone="subtle">
      {formatOffset(turn.startSeconds)}
    </AppText>
  );
  if (lowConfidence) {
    meta = (
      <AppText variant="caption" tone="warning">
        {formatOffset(turn.startSeconds)} · unclear audio
      </AppText>
    );
  }

  return (
    <View style={[styles.turn, isWearer && styles.turnWearer]}>
      <View
        style={[
          styles.bubble,
          { backgroundColor: isWearer ? colors.tintSoft : colors.surface },
          provisional && { borderWidth: 1, borderStyle: "dashed", borderColor: colors.warning },
        ]}
      >
        <View style={styles.turnMeta}>
          <AppText
            variant="caption"
            style={styles.speaker}
            tone={unknown ? "warning" : isWearer ? "tint" : "default"}
          >
            {turn.speakerLabel}
          </AppText>
          {meta}
        </View>
        <AppText
          tone={provisional || lowConfidence ? "muted" : "default"}
          style={provisional || lowConfidence ? styles.tentative : undefined}
        >
          {turn.text}
        </AppText>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  screen: { flex: 1 },
  content: { paddingBottom: Spacing.xxl },
  header: { padding: Spacing.lg, gap: Spacing.md },
  headerTop: { flexDirection: "row", alignItems: "center", gap: Spacing.sm },
  people: { flexDirection: "row", flexWrap: "wrap", gap: Spacing.sm },
  personChip: {
    flexDirection: "row",
    alignItems: "center",
    gap: 6,
    paddingLeft: 4,
    paddingRight: Spacing.md,
    paddingVertical: 4,
    borderRadius: Radius.pill,
  },
  chipText: { fontWeight: "600" },
  card: { padding: Spacing.md, borderRadius: Radius.md, gap: Spacing.xs },
  learnedFact: { flexDirection: "row", gap: Spacing.sm, marginTop: Spacing.xs },
  banner: {
    flexDirection: "row",
    gap: Spacing.sm,
    padding: Spacing.md,
    borderRadius: Radius.md,
    borderWidth: 1,
    borderStyle: "dashed",
  },
  transcriptLabel: { marginTop: Spacing.sm },
  turn: { paddingHorizontal: Spacing.lg, paddingVertical: Spacing.xs, alignItems: "flex-start" },
  turnWearer: { alignItems: "flex-end" },
  bubble: { maxWidth: "85%", paddingHorizontal: Spacing.md, paddingVertical: Spacing.sm, borderRadius: Radius.lg },
  turnMeta: { flexDirection: "row", gap: Spacing.sm, marginBottom: 2 },
  speaker: { fontWeight: "600" },
  tentative: { fontStyle: "italic" },
});
