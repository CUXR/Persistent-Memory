import React, { useState } from "react";
import { FlatList, RefreshControl, StyleSheet, View } from "react-native";

import { Avatar } from "@/components/avatar";
import { ChipRow } from "@/components/chip";
import { ListRow } from "@/components/list-row";
import { UnnamedBadge } from "@/components/memory-badge";
import { SearchBar } from "@/components/search-bar";
import { EmptyState, ErrorState, LoadMoreFooter, SkeletonRows } from "@/components/state-views";
import { useDataSource } from "@/data/DataSourceContext";
import type { Person } from "@/data/types";
import { useDebouncedValue } from "@/hooks/use-debounced-value";
import { usePagedList } from "@/hooks/use-paged-list";
import { useTheme } from "@/hooks/use-theme";
import { formatRelative, personLabel } from "@/lib/format";
import { openPerson } from "@/lib/routes";

type Filter = "all" | "unnamed";

const FILTERS: { value: Filter; label: string }[] = [
  { value: "all", label: "Everyone" },
  { value: "unnamed", label: "Needs a name" },
];

export default function PeopleScreen() {
  const colors = useTheme();
  const { source, revision } = useDataSource();
  const [query, setQuery] = useState("");
  const [filter, setFilter] = useState<Filter>("all");
  const debouncedQuery = useDebouncedValue(query.trim());

  const list = usePagedList(
    (cursor) =>
      source.listPeople({ query: debouncedQuery || undefined, unnamedOnly: filter === "unnamed", cursor }),
    [source, debouncedQuery, filter, revision],
  );

  const renderItem = ({ item }: { item: Person }) => {
    const seen = item.lastSeenAt ? `Last seen ${formatRelative(item.lastSeenAt)}` : "Not seen yet";
    const conversations = `${item.episodeCount} conversation${item.episodeCount === 1 ? "" : "s"}`;
    return (
      <ListRow
        title={personLabel(item)}
        subtitle={`${seen} · ${conversations}`}
        subtitleLines={1}
        badge={item.isUnnamed ? <UnnamedBadge /> : undefined}
        leading={<Avatar id={item.id} name={item.displayName} />}
        onPress={() => openPerson(item.id)}
      />
    );
  };

  let empty: React.ReactNode;
  if (debouncedQuery) {
    empty = <EmptyState icon="search-outline" title="No matches" message={`No one named "${debouncedQuery}".`} />;
  } else if (filter === "unnamed") {
    empty = (
      <EmptyState
        icon="checkmark-circle-outline"
        title="Everyone has a name"
        message="New faces your glasses see will show up here until you name them."
      />
    );
  } else {
    empty = (
      <EmptyState icon="people-outline" title="No people yet" message="People your glasses recognize will appear here." />
    );
  }

  let body: React.ReactNode;
  if (list.status === "loading") {
    body = <SkeletonRows />;
  } else if (list.status === "error") {
    body = <ErrorState message={list.error?.message ?? "Couldn't load people."} onRetry={list.retry} />;
  } else {
    body = (
      <FlatList
        data={list.items}
        keyExtractor={(item) => item.id}
        renderItem={renderItem}
        onEndReached={list.loadMore}
        onEndReachedThreshold={0.5}
        keyboardDismissMode="on-drag"
        contentContainerStyle={list.items.length === 0 ? styles.emptyContainer : undefined}
        refreshControl={<RefreshControl refreshing={list.refreshing} onRefresh={list.refresh} />}
        ListEmptyComponent={empty}
        ListFooterComponent={
          <LoadMoreFooter loading={list.loadingMore} error={list.loadMoreError} onRetry={list.loadMore} />
        }
      />
    );
  }

  return (
    <View style={[styles.screen, { backgroundColor: colors.background }]}>
      <SearchBar value={query} onChangeText={setQuery} placeholder="Search people" />
      <View style={[styles.filters, { borderBottomColor: colors.border }]}>
        <ChipRow options={FILTERS} value={filter} onChange={setFilter} />
      </View>
      {body}
    </View>
  );
}

const styles = StyleSheet.create({
  screen: { flex: 1 },
  filters: { borderBottomWidth: StyleSheet.hairlineWidth, flexGrow: 0 },
  emptyContainer: { flexGrow: 1 },
});
