import { isRecordId } from "./router.js";

export function getErrorMessage(error) {
  if (error instanceof Error && error.message) {
    return error.message;
  }
  return "Something unexpected happened while loading this page.";
}

function settle(promise) {
  return promise.then(
    (value) => ({ value, error: null }),
    (error) => ({ value: null, error: getErrorMessage(error) }),
  );
}

const NOTHING = { value: null, error: null };

function invalidId(label) {
  return { value: null, error: `${label} id in the address is not valid.` };
}

/** Whether the masthead search box should be overwritten with the route's query. */
export function shouldSyncSearch(searchQuery, lastSyncedSearchQuery) {
  return searchQuery !== lastSyncedSearchQuery;
}

/**
 * Route loaders: the list request drives the page state, while every detail
 * request is settled independently so a failure only affects its own pane.
 */
export function createRouteLoaders(apiClient) {
  return {
    async conversations({ episodeId }) {
      const listPromise = apiClient.listRecentConversations();
      let detailPromise = Promise.resolve(NOTHING);
      if (episodeId) {
        detailPromise = isRecordId(episodeId)
          ? settle(apiClient.getConversation(episodeId))
          : Promise.resolve(invalidId("The conversation"));
      }
      const [items, detail] = await Promise.all([listPromise, detailPromise]);
      return { items, selectedConversation: detail.value, detailError: detail.error };
    },

    async people({ personId, searchQuery, contextQuery }) {
      const directory = await apiClient.listPeople(searchQuery);
      const base = {
        directory,
        items: directory.items,
        selectedProfile: null,
        detailError: null,
        selectedContext: null,
        contextError: null,
        contextQuery,
        searchQuery,
        impliedSelection: false,
      };

      if (personId && !isRecordId(personId)) {
        return { ...base, detailError: invalidId("The person").error };
      }

      const resolvedId = directory.resolution?.person_id || null;
      const impliedPersonId = !personId && resolvedId ? resolvedId : null;
      const selectedPersonId = personId || impliedPersonId;
      // Closing an explicit selection that the current search would re-resolve
      // must clear the search instead, or the profile never goes away.
      const impliedSelection = Boolean(impliedPersonId) || (Boolean(personId) && resolvedId === personId);

      const [profile, context] = await Promise.all([
        selectedPersonId ? settle(apiClient.getPersonProfile(selectedPersonId)) : NOTHING,
        selectedPersonId && contextQuery ? settle(apiClient.getPersonContext(selectedPersonId, contextQuery)) : NOTHING,
      ]);

      return {
        ...base,
        selectedProfile: profile.value,
        detailError: profile.error,
        selectedContext: context.value,
        contextError: context.error,
        impliedSelection,
      };
    },
  };
}
