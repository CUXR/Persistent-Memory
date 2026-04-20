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
- `fact_category` - normalized category for the fact (for example `visual_descriptor`, `affiliation`, `hobby`)
- `fact_text` - remembered fact about the person
- `source` - optional link to the `Episode` where the fact was learned
- `embedding` - optional retrieval embedding for RAG-style fact lookup
- `confidence` - optional confidence score for the fact
- `created_at` - timestamp when the fact was created
- `updated_at` - timestamp when the fact was last updated

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
