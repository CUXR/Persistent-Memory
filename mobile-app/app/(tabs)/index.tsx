import React, { useState } from "react";
import { FlatList, RefreshControl, StyleSheet, View } from "react-native";

import { Avatar } from "@/components/avatar";
import { ListRow } from "@/components/list-row";
import { MemoryBadge } from "@/components/memory-badge";
import { SearchBar } from "@/components/search-bar";
import { EmptyState, ErrorState, LoadMoreFooter, SkeletonRows } from "@/components/state-views";
import { useDataSource } from "@/data/DataSourceContext";
import type { HistoryItem, RecordingSummary } from "@/data/types";
import { useDebouncedValue } from "@/hooks/use-debounced-value";
import { usePagedList } from "@/hooks/use-paged-list";
import { useTheme } from "@/hooks/use-theme";
import { formatDateTime, joinPeople } from "@/lib/format";
import { openHistoryItem } from "@/lib/routes";

function recordingSubtitle(r: RecordingSummary) {
  if (r.status === "pending") return "Waiting to be transcribed…";
  if (r.status === "failed") return r.transcriptPreview || "Processing failed before a transcript was made.";
  return r.transcriptPreview || "Transcribing…";
}

export default function HistoryScreen() {
  const colors = useTheme();
  const { source, revision } = useDataSource();
  const [query, setQuery] = useState("");
  const debouncedQuery = useDebouncedValue(query.trim());

  const list = usePagedList(
    (cursor) => source.listHistory({ query: debouncedQuery || undefined, cursor }),
    [source, debouncedQuery, revision],
  );

  const renderItem = ({ item }: { item: HistoryItem }) => {
    if (item.kind === "episode") {
      const lead = item.people[0];
      return (
        <ListRow
          title={joinPeople(item.people)}
          meta={formatDateTime(item.startTime)}
          subtitle={item.summary}
          leading={lead ? <Avatar id={lead.id} name={lead.displayName} /> : undefined}
          onPress={() => openHistoryItem(item)}
        />
      );
    }
    const accent = item.status === "failed" ? colors.danger : colors.warning;
    return (
      <ListRow
        title={joinPeople([item.person])}
        meta={formatDateTime(item.startTime)}
        subtitle={recordingSubtitle(item)}
        tentative
        badge={<MemoryBadge state={item.status} />}
        leading={<Avatar id={item.person.id} name={item.person.displayName} />}
        onPress={() => openHistoryItem(item)}
        style={[styles.provisionalRow, { borderLeftColor: accent, backgroundColor: colors.surface }]}
      />
    );
  };

  let body: React.ReactNode;
  if (list.status === "loading") {
    body = <SkeletonRows />;
  } else if (list.status === "error") {
    body = <ErrorState message={list.error?.message ?? "Couldn't load conversations."} onRetry={list.retry} />;
  } else {
    body = (
      <FlatList
        data={list.items}
        keyExtractor={(item) => `${item.kind}-${item.id}`}
        renderItem={renderItem}
        onEndReached={list.loadMore}
        onEndReachedThreshold={0.5}
        keyboardDismissMode="on-drag"
        contentContainerStyle={list.items.length === 0 ? styles.emptyContainer : undefined}
        refreshControl={<RefreshControl refreshing={list.refreshing} onRefresh={list.refresh} />}
        ListEmptyComponent={
          debouncedQuery ? (
            <EmptyState
              icon="search-outline"
              title="No matches"
              message={`No conversations mention "${debouncedQuery}".`}
            />
          ) : (
            <EmptyState
              icon="chatbubbles-outline"
              title="No conversations yet"
              message="Conversations captured by your glasses will show up here."
            />
          )
        }
        ListFooterComponent={
          <LoadMoreFooter loading={list.loadingMore} error={list.loadMoreError} onRetry={list.loadMore} />
        }
      />
    );
  }

  return (
    <View style={[styles.screen, { backgroundColor: colors.background }]}>
      <SearchBar value={query} onChangeText={setQuery} placeholder="Search people, summaries, transcripts" />
      {body}
    </View>
  );
}

const styles = StyleSheet.create({
  screen: { flex: 1 },
  emptyContainer: { flexGrow: 1 },
  provisionalRow: { borderLeftWidth: 3 },
});
