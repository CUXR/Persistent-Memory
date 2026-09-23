# Database Structure

## Table Structure

### Person
- `id` - UUID primary key
- `user_id` - owner of this person record; links to `User`
- `name` - (`first_name`, `last_name`, `display_name`)
- `facts` - stored in `PersonFact`
- `dialogues` - linked through `Episodes`
- `face` - vector representation of facial features for recognition (`face_embedding`)
- `voice` - vector representation of vocal features for recognition (`voice_embedding`)
- `face_embedding_model` - model/version used to generate the face embedding
- `voice_embedding_model` - model/version used to generate the voice embedding
- `last_seen_at` - timestamp of the most recent known interaction
- `created_at` - timestamp when the person record was created
- `updated_at` - timestamp when the person record was last updated

### PersonFact
- `id` - UUID primary key
- `person_id` - links to `Person`
- `fact_text` - remembered fact about the person
- `fact_category` - `"visual_descriptor", "affiliation", "hobby"`
- `source_episode_id` - link to the `Episode` where the fact was learned or most recently updated
- `confidence` - 0-1 confidence used as the relevance signal in ranked retrieval
- `embedding` - optional vector representation of `fact_text`, used for similarity ranking when a query embedding is supplied

### Summary
- `id` - UUID primary key
- `person_id` - links to `Person`
- `summary_text` - narrative summary slice for the person
- `episode_id` - link to the source `Episode`; its `importance_score` is used as the relevance signal in ranked retrieval
- `embedding` - optional vector representation of `summary_text`, used for similarity ranking when a query embedding is supplied

### Episodes
- `id` - UUID primary key
- `start_time` - timestamp of conversation start
- `end_time` - timestamp of conversation end; nullable for in-progress conversations
- `dialogue_summary` - summary of conversation
- `importance_score` - optional score for ranking memorable conversations
- `user` - person wearing the glasses; links to `User`
- `person` - person talking to; links to `Person`
- `created_at` - timestamp when the episode record was created
- `updated_at` - timestamp when the episode record was last updated

### User
- `id` - UUID primary key
- `name` - (`first_name`, `last_name`, `display_name`)
- `username` - unique application username
- `facts` - stored in `UserFact`
- `dialogues` - linked through `Episodes`
- `created_at` - timestamp when the user record was created
- `updated_at` - timestamp when the user record was last updated

### UserFact
- `id` - UUID primary key
- `user_id` - links to `User`
- `fact_text` - remembered fact about the user
- `created_at` - timestamp when the fact was created
- `updated_at` - timestamp when the fact was last updated

## Ranked Memory Retrieval

`MemoryStore.get_relevant_memories(person_id, limit, query_embedding=None)` returns a
bounded, ranked slice of a person's facts and summaries for surfacing on re-encounter,
plus a short `interaction_context` string (e.g. "Met 2 weeks ago: ..."). Each item is
scored by a blend of:

- **recency** - exponential decay from the item's anchor timestamp (half-life configurable, default 14 days)
- **relevance** - `PersonFact.confidence`, or the linked `Episode.importance_score` for summaries
- **similarity** - cosine similarity against an optional `query_embedding`, when both it and the item's `embedding` are present

See `app/services/memory_ranking.py` for the scoring functions and
`POST /people/{person_id}/memories/relevant` for the API endpoint.
