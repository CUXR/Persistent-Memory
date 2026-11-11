# Person briefings HTTP contract (#34)

For the glasses (#39), the phone app (#48, #51), and device intake (#49). Each
`*.json` file here is one request and the exact response the server returns for
it. `backend/tests/test_briefing_api.py` replays every fixture against the app,
so they stay accurate. Build client models and offline fixtures from them. The
full schema is also served at `/openapi.json`.

## Authentication

Every route needs `Authorization: Bearer <token>`. The server finds the owner
from the token. Clients never send an owner ID. A missing, unknown, or revoked
token returns `401` with `WWW-Authenticate: Bearer`.

Issue one token per client and revoke it if the device is lost:

```sh
python -m app.cli.tokens create --owner-id USER_UUID --label "RayNeo glasses"
python -m app.cli.tokens list --owner-id USER_UUID
python -m app.cli.tokens revoke TOKEN_ID
```

The token is printed once. Only its SHA-256 is stored. Pairing and onboarding
flows are not part of this contract yet. They can issue the same tokens.

## Routes

| Route | Use |
|---|---|
| `GET /briefings/people/{person_id}?max_facts=5` | Face path: the person in view. Follows the owner's display mode. |
| `GET /briefings/by-name?name=Sam&max_facts=5` | Name path: someone the wearer asked about. Returns `found`, `ambiguous`, or `not_found`. |
| `GET /briefings/facts?cursor=…&limit=5` | The next page of one fact category. |
| `GET /briefings/display-mode` | The owner's glasses display mode. |
| `PUT /briefings/display-mode` | Set it: `{"display_mode": "nothing" \| "name" \| "name_and_facts"}`. |

`max_facts` and `limit` set the page size for each category. Both run from 1 to
50 and default to 5. A brief has no limit on its total number of facts. Follow
`next_cursor` until it is `null`.

## Reading a brief

- `display_mode` tells you what the brief contains:
  - `nothing`: only `person_id`, `display_mode`, and `is_unnamed`.
  - `name`: adds `display_name`.
  - `name_and_facts`: adds the fact pages and `last_interaction`.
- `display_name` is `null` when the person is unnamed (`is_unnamed: true`), or
  when the mode is `nothing`. Never show the `Unknown person <uuid>` placeholder.
- A fact section is `null` when the brief does not include it. It is
  `{"items": [], "next_cursor": null}` when the brief includes it but the person
  has no such facts.
- `visual_descriptors` is always `null` on the face path. The person is already
  in view. The name path includes it.
- The display mode applies only to the face path. A name lookup is an explicit
  request, so it always returns the full brief.
- `last_interaction` is `null` if the wearer has never spoken with the person.

## Fact order and cursors

Facts come best first. Until fact ranking exists, "best" means newest first.
The fact ID breaks ties, so paging never skips or repeats a fact. Treat
`next_cursor` as opaque: pass it back unchanged. Each cursor belongs to one
person and one category. When the ordering changes, cursors issued under the
old ordering return `400`. Fetch the brief again to get new ones. Fact `id`s
are stable, so you can use them to de-duplicate retries.

## Errors

| Status | When | Fixture |
|---|---|---|
| 401 | Missing, unknown, or revoked token | `error_unauthorized.json` |
| 404 | Unknown person ID, or another owner's person (same response for both) | `error_person_not_found.json` |
| 400 | Malformed or outdated cursor | `error_invalid_cursor.json` |
| 422 | Page size out of range, blank name, unknown display mode | FastAPI validation body |

## Updating fixtures

After an intended contract change:

```sh
UPDATE_CONTRACT_FIXTURES=1 python -m pytest backend/tests/test_briefing_api.py
```

Review the JSON diff before you commit it, because client teams build against it.
