# Person briefings HTTP contract (#34)

For the glasses (#39), the phone app (#48, #51), and device intake (#49). Each
`*.json` file here is one request and the exact response the server returns for
it. `backend/tests/test_briefing_api.py` replays every fixture against the app,
so they stay accurate. Build client models and offline fixtures from them. The
full schema is also served at `/openapi.json`.

## Owner scoping

Every route needs an `X-Owner-Id` header naming the user whose memories to
read. Each user only sees their own people. A missing or malformed header, or
one that names no user, returns `401`.

This header is a development stand-in shared with the read APIs (#35), not
authentication. When real user accounts arrive, the server will identify the
user from their login instead, and the routes and responses stay the same.

## Routes

| Route | Use |
| --- | --- |
| `GET /people/{person_id}/briefing?max_facts=5` | Everything to remember about a known person, e.g. one that face recognition identified. |
| `GET /people/briefing?name=Sam&max_facts=5` | Look someone up by name. Returns `found` (with the same brief), `ambiguous`, or `not_found`. |
| `GET /people/{person_id}/facts?category=hobby&cursor=…&limit=5` | One page of a category. Without `cursor`, the first page. |

`category` is `hobby`, `affiliation`, or `visual_descriptor`. `max_facts` and
`limit` set the page size. Both run from 1 to 50 and default to 5. A brief has
no limit on its total number of facts. Follow `next_cursor` until it is `null`.

## Looking someone up by name

1. Call `GET /people/briefing?name=…`.
2. `found`: the brief is in the response, identical to `/people/{id}/briefing`.
3. `ambiguous`: show the candidates. Each one has a `person_id` and `hints`,
   such as where they work or what they look like. After the wearer picks one,
   call `GET /people/{person_id}/briefing`. This works even when two people have
   the same name.
4. `not_found`: show `message`.

## Reading a brief

- Every brief has the same shape: `display_name`, `is_unnamed`, the first page
  of `hobbies`, `affiliations`, and `visual_descriptors`, and `last_interaction`.
- The client decides what to display. The glasses' display setting (nothing,
  name only, or name and facts) lives on the glasses, so changes apply at once.
  For example, the glasses can hide `visual_descriptors` while the person is in
  view.
- `display_name` is `null` when the person is unnamed (`is_unnamed: true`).
  Never show the `Unknown person <uuid>` placeholder.
- A category with no facts is `{"items": [], "next_cursor": null}`.
- `last_interaction` is `null` if the wearer has never spoken with the person.

## Fact order and cursors

Facts come best first. Until fact ranking exists, "best" means newest first.
The fact ID breaks ties, so paging never skips or repeats a fact, even when new
facts arrive between requests. Treat `next_cursor` as opaque: pass it back
unchanged, with the same person and category it came from. When the ordering
changes, cursors issued under the old ordering return `400`. Fetch the brief
again to get new ones. Fact `id`s are stable, so you can use them to
de-duplicate retries.

## Errors

| Status | When | Fixture |
| --- | --- | --- |
| 401 | Missing or malformed `X-Owner-Id`, or no such user | `error_unknown_owner.json` |
| 404 | Unknown person ID, or another owner's person (same response for both) | `error_person_not_found.json` |
| 400 | Malformed or outdated cursor, or one from a different person or category | `error_invalid_cursor.json` |
| 422 | Page size out of range, unknown category, blank name | FastAPI validation body |

## Updating fixtures

After an intended contract change:

```sh
UPDATE_CONTRACT_FIXTURES=1 python -m pytest backend/tests/test_briefing_api.py
```

Review the JSON diff before you commit it, because client teams build against it.
