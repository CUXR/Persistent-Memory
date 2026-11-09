# Persistent Memory mobile app

Companion app for the glasses. It lets the wearer browse saved conversations
(episodes) and recordings the server is still processing, review and name the
people the glasses have recognized, and choose what the glasses display when
someone is recognized: nothing, the name, or the name plus server-selected facts.
The app talks only to the backend and never to the glasses directly. The glasses
app itself is separate (Kotlin, #39).

Built with Expo (SDK 54) and expo-router. One codebase runs on Android and iOS.

## Run it

Requires Node 20+.

```sh
cd mobile-app
npm install
npx expo start
```

**On a phone (Android or iOS):** install **Expo Go** from the Play Store or App
Store, connect the phone to the same Wi-Fi as your computer, then scan the QR
code shown by `expo start` (Android: from Expo Go; iOS: with the Camera app). If
your network blocks device-to-computer traffic, run `npx expo start --tunnel`.

**Android emulator:** install Android Studio, create a device in Device Manager,
start it, then press `a` in the `expo start` terminal.

**iOS simulator (macOS only):** install Xcode, then press `i` in the
`expo start` terminal.

Checks:

```sh
npx tsc --noEmit
npm run lint
```

## Structure

| Path | Contents |
| --- | --- |
| `app/(tabs)/index.tsx` | History: search, saved episodes and provisional recordings |
| `app/history/[id].tsx` | Episode or recording detail and transcript (`kind` param) |
| `app/(tabs)/people.tsx` | People list, with a "Needs a name" filter |
| `app/people/[id].tsx` | Person detail: naming/renaming, summary, connections, facts |
| `app/(tabs)/glasses.tsx` | Display selector, glasses preview, connection status |
| `components/` | Shared UI: rows, badges, chips, loading/empty/error states |
| `data/types.ts` | Data contract shared with #51 |
| `data/fixtures.ts` | Fake `DataSource` used until the live API exists |
| `data/DataSourceContext.tsx` | Provides the `DataSource` to every screen |

## Data contract (#48 ↔ #51)

Screens only call the `DataSource` interface in `data/types.ts`. Its shapes
mirror `backend/app/models` and `backend/app/schema`:

- **Saved memories** are episodes (`Episode`), with the summary, transcript and
  facts learned in that conversation.
- **Provisional transcripts** are audio jobs that have not produced an episode yet
  (`Recording`, status `pending`, `processing`, `needs_review` or `failed`).
  `complete` jobs appear as episodes; `no_speech` jobs are hidden.
- **People** have a first and last name. Faces saved automatically keep the
  `Unknown person <UUID>` placeholder (`isUnnamed`) until the wearer names them.
- **Facts** carry a `fact_category` (hobby, affiliation, visual descriptor) and
  a confidence. **Relationships** come from `person_edges`.
- **Name + facts** on the glasses shows the person's `Brief` (`schema/briefing.py`):
  top hobbies and affiliations.

To connect the backend, implement `DataSource` against the API and provide it in
place of `FixtureDataSourceProvider`; screens need no changes. Display settings
and connection status have no backend counterpart yet.

## Reviewing states with fixtures

In development builds, the bottom of the Glasses tab has a **fixture scenario**
switch: Normal, Empty, Server error, Slow network, Saves fail, and Glasses
offline. Every screen's loading, empty and error states can be reviewed with it.
Renames and the display mode persist until the app restarts.
