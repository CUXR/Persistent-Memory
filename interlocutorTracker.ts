// interlocutorTracker.ts
//
// Mock interlocutor tracker for EgoMem.
//
// - Loads wearer_person_id from wearer_state on startup
// - Polls every 2s for a (mock) interlocutor identity
// - Debounces identity switches (requires N consecutive polls; default 2)
// - On confirmed change, refreshes active Level-1 interlocutor context via get_profile_context(person_id)
// - Emits `interlocutor_changed` event
//
// Notes:
// - Wearer identity is fixed; only the interlocutor is tracked.
// - Mock identity comes from an in-memory timeline (replace later with AV recognition).

import { EventEmitter } from "events";

/** Shape of whatever your Level-1 context function returns. Keep it loose for now. */
export type Level1InterlocutorContext = unknown;

export type GetProfileContextFn = (personId: string) => Promise<Level1InterlocutorContext>;

/** Minimal wearer_state contract. */
/** TODO: Need to implement the wearer state so the polling can actually grab this info */
export type WearerState = { wearer_person_id: string };

export type WearerStateLoader = () => Promise<WearerState> | WearerState;

export type InterlocutorChangedEvent = {
  previous_person_id: string | null;
  new_person_id: string | null;
  context: Level1InterlocutorContext | null;
};

export type InterlocutorTrackerEvents = {
  interlocutor_changed: (evt: InterlocutorChangedEvent) => void;
  error: (err: Error) => void;
};

/** A single mock identity change point. `atMs` is time since tracker start. */
export type MockIdentityPoint = { atMs: number; personId: string | null };

/**
 * In-memory identity timeline:
 * - The "current" personId is the last point whose atMs <= elapsedMs.
 */
export class MockInterlocutorStream {
  private readonly timeline: MockIdentityPoint[];
  private readonly defaultPersonId: string | null;

  constructor(timeline: MockIdentityPoint[], defaultPersonId: string | null = null) {
    this.timeline = [...timeline].sort((a, b) => a.atMs - b.atMs);
    this.defaultPersonId = defaultPersonId;
  }

  getCurrentPersonId(elapsedMs: number): string | null {
    if (this.timeline.length === 0) return this.defaultPersonId;

    let current: string | null = this.defaultPersonId;
    for (const point of this.timeline) {
      if (point.atMs <= elapsedMs) current = point.personId;
      else break;
    }
    return current;
  }
}

export type InterlocutorTrackerOptions = {
  pollIntervalMs?: number; // default 2000
  debounceConsecutivePolls?: number; // default 2
  mockStream?: MockInterlocutorStream;

  // Dependency injection (recommended for tests)
  // TODO: Need to implement the getProfileContext function so that the tracker can actually fetch the profile context for the interlocutor
  getProfileContext?: GetProfileContextFn;
  loadWearerState?: WearerStateLoader;
};

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

/**
 * Attempts to load wearer_state + get_profile_context from local modules if not provided.
 * This avoids hard-coding your repo layout, while still “just working” in many setups.
 */
async function defaultLoadWearerState(): Promise<WearerState> {
  // Try common patterns:
  // 1) exported const wearer_state
  // 2) exported function getWearerState()
  // 3) JSON file wearer_state.json
  try {
    // eslint-disable-next-line @typescript-eslint/no-var-requires
    const mod = require("./wearer_state");
    if (mod?.wearer_state?.wearer_person_id) return mod.wearer_state as WearerState;
    if (typeof mod?.getWearerState === "function") return await mod.getWearerState();
  } catch {
    // ignore
  }

  try {
    // eslint-disable-next-line @typescript-eslint/no-var-requires
    const json = require("./wearer_state.json");
    if (json?.wearer_person_id) return json as WearerState;
  } catch {
    // ignore
  }

  throw new Error(
    "Failed to load wearer_state. Provide options.loadWearerState or ensure ./wearer_state(.ts) or ./wearer_state.json exists."
  );
}

function defaultGetProfileContext(): GetProfileContextFn {
  try {
    // eslint-disable-next-line @typescript-eslint/no-var-requires
    const mod = require("./get_profile_context");
    if (typeof mod?.get_profile_context === "function") return mod.get_profile_context as GetProfileContextFn;
    if (typeof mod?.getProfileContext === "function") return mod.getProfileContext as GetProfileContextFn;
  } catch {
    // ignore
  }

  throw new Error(
    "Failed to load get_profile_context. Provide options.getProfileContext or ensure ./get_profile_context(.ts) exports get_profile_context()."
  );
}

export class InterlocutorTracker extends EventEmitter {
  public wearer_person_id: string | null = null;

  /** The interlocutor we consider currently active (debounced + applied). */
  public activeInterlocutorPersonId: string | null = null;

  /** The always-ready Level-1 memory context for the active interlocutor. */
  public activeLevel1InterlocutorContext: Level1InterlocutorContext | null = null;

  private readonly pollIntervalMs: number;
  private readonly debounceConsecutivePolls: number;
  private readonly mockStream: MockInterlocutorStream;

  private readonly getProfileContext: GetProfileContextFn;
  private readonly loadWearerState: WearerStateLoader;

  private startedAtMs: number = 0;
  private running = false;

  // Debounce state
  private candidateId: string | null = null;
  private candidateCount = 0;

  // Prevent overlapping refresh calls if polling interval is shorter than fetch time
  private refreshInFlight: Promise<void> | null = null;

  constructor(options: InterlocutorTrackerOptions = {}) {
    super();

    this.pollIntervalMs = options.pollIntervalMs ?? 2000;
    this.debounceConsecutivePolls = options.debounceConsecutivePolls ?? 2;

    this.mockStream =
      options.mockStream ??
      new MockInterlocutorStream(
        [
          // Default mock timeline (safe, deterministic).
          // Replace/override in tests or app bootstrap.
          { atMs: 0, personId: null },
          { atMs: 5_000, personId: "person_alice" },
          { atMs: 25_000, personId: "person_bob" },
          { atMs: 45_000, personId: "person_alice" },
          { atMs: 65_000, personId: null },
        ],
        null
      );

    this.getProfileContext = options.getProfileContext ?? defaultGetProfileContext();
    this.loadWearerState = options.loadWearerState ?? defaultLoadWearerState;
  }

  /**
   * Startup:
   * - load wearer_person_id once
   * - begin polling loop (continuous)
   */
  async start(): Promise<void> {
    if (this.running) return;

    const wearerState = await Promise.resolve(this.loadWearerState());
    if (!wearerState?.wearer_person_id) {
      throw new Error("wearer_state missing wearer_person_id");
    }
    this.wearer_person_id = wearerState.wearer_person_id;

    this.running = true;
    this.startedAtMs = Date.now();

    // Kick off loop without awaiting it (but ensure it can't throw unhandled).
    void this.pollLoop().catch((err) => {
      // No console errors per acceptance criteria: re-emit so caller can handle/log as desired.
      this.running = false;
      this.emit("error", err);
    });
  }

  async stop(): Promise<void> {
    this.running = false;
    // Wait for any in-flight refresh to finish so tests can be deterministic.
    if (this.refreshInFlight) await this.refreshInFlight;
    this.refreshInFlight = null;
  }

  /** Single poll tick: read mock identity + debounce + maybe apply switch. */
  private async pollOnce(): Promise<void> {
    const elapsed = Date.now() - this.startedAtMs;
    const observedId = this.mockStream.getCurrentPersonId(elapsed);

    // If observed matches current active, clear debounce state.
    if (observedId === this.activeInterlocutorPersonId) {
      this.candidateId = null;
      this.candidateCount = 0;
      return;
    }

    // Debounce: need N consecutive polls of the same *new* id.
    if (observedId !== this.candidateId) {
      this.candidateId = observedId;
      this.candidateCount = 1;
      return;
    }

    this.candidateCount += 1;

    if (this.candidateCount < this.debounceConsecutivePolls) return;

    // Confirmed switch: apply it.
    await this.applyInterlocutorSwitch(observedId);

    // Reset debounce state after applying.
    this.candidateId = null;
    this.candidateCount = 0;
  }

  private async applyInterlocutorSwitch(newPersonId: string | null): Promise<void> {
    const prev = this.activeInterlocutorPersonId;

    // No-op safety (should already be filtered out)
    if (newPersonId === prev) return;

    // Ensure we don't overlap profile refreshes (can happen if get_profile_context is slow).
    if (this.refreshInFlight) {
      await this.refreshInFlight;
      // Re-check in case another switch happened while we waited.
      if (newPersonId === this.activeInterlocutorPersonId) return;
    }

    this.refreshInFlight = (async () => {
      // Update active id first (so readers stop using old context ASAP),
      // then update context. If fetch fails, we revert id+context.
      this.activeInterlocutorPersonId = newPersonId;

      if (newPersonId === null) {
        this.activeLevel1InterlocutorContext = null;
        this.emit("interlocutor_changed", {
          previous_person_id: prev,
          new_person_id: null,
          context: null,
        } satisfies InterlocutorChangedEvent);
        return;
      }

      let ctx: Level1InterlocutorContext;
      try {
        ctx = await this.getProfileContext(newPersonId);
      } catch (e) {
        // Revert on failure (prevents “active id changed but context missing”)
        this.activeInterlocutorPersonId = prev;
        throw e;
      }

      this.activeLevel1InterlocutorContext = ctx;

      this.emit("interlocutor_changed", {
        previous_person_id: prev,
        new_person_id: newPersonId,
        context: ctx,
      } satisfies InterlocutorChangedEvent);
    })();

    try {
      await this.refreshInFlight;
    } finally {
      this.refreshInFlight = null;
    }
  }

  /** Continuous polling loop: runs until stop() sets running=false. */
  private async pollLoop(): Promise<void> {
    while (this.running) {
      await this.pollOnce();
      await sleep(this.pollIntervalMs);
    }
  }

  // Typed event helpers (optional convenience)
  override on<U extends keyof InterlocutorTrackerEvents>(event: U, listener: InterlocutorTrackerEvents[U]): this {
    return super.on(event, listener as any);
  }
  override emit<U extends keyof InterlocutorTrackerEvents>(event: U, ...args: Parameters<InterlocutorTrackerEvents[U]>): boolean {
    return super.emit(event, ...(args as any));
  }
}