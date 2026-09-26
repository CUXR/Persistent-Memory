// interlocutorTracker.ts
//
// Interlocutor tracker driven by persistent IDs from face recognition.
//
// - Loads wearer_person_id from wearer_state on startup
// - Polls every 2s for the current face-recognition person ID
// - Debounces identity switches (requires N consecutive polls; default 2)
// - On confirmed change, refreshes active Level-1 interlocutor context via get_profile_context(person_id)
// - Emits `interlocutor_changed` event
//
// Notes:
// - Wearer identity is fixed; only the interlocutor is tracked.
// - The recognition source must return a persisted person ID, including for unknown faces.

import { EventEmitter } from "events";

/** Shape of whatever your Level-1 context function returns. Keep it loose for now. */
export type Level1InterlocutorContext = unknown;

export type GetProfileContextFn = (personId: string) => Promise<Level1InterlocutorContext>;

/** Minimal wearer_state contract. */
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

export type InterlocutorTrackerOptions = {
  pollIntervalMs?: number; // default 2000
  debounceConsecutivePolls?: number; // default 2
  // The face recognizer persists unknown faces before returning their IDs.
  getCurrentPersonId: () => Promise<string | null> | string | null;

  // Application adapters supply the owner and backend memory context.
  getProfileContext: GetProfileContextFn;
  loadWearerState: WearerStateLoader;
};

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

export class InterlocutorTracker extends EventEmitter {
  public wearer_person_id: string | null = null;

  /** The interlocutor we consider currently active (debounced + applied). */
  public activeInterlocutorPersonId: string | null = null;

  /** The always-ready Level-1 memory context for the active interlocutor. */
  public activeLevel1InterlocutorContext: Level1InterlocutorContext | null = null;

  private readonly pollIntervalMs: number;
  private readonly debounceConsecutivePolls: number;
  private readonly getCurrentPersonId: InterlocutorTrackerOptions["getCurrentPersonId"];

  private readonly getProfileContext: GetProfileContextFn;
  private readonly loadWearerState: WearerStateLoader;

  private running = false;

  // Debounce state
  private candidateId: string | null = null;
  private candidateCount = 0;

  // Prevent overlapping refresh calls if polling interval is shorter than fetch time
  private refreshInFlight: Promise<void> | null = null;

  constructor(options: InterlocutorTrackerOptions) {
    super();

    this.pollIntervalMs = options.pollIntervalMs ?? 2000;
    this.debounceConsecutivePolls = options.debounceConsecutivePolls ?? 2;

    if (!Number.isFinite(this.pollIntervalMs) || this.pollIntervalMs <= 0 ||
        !Number.isInteger(this.debounceConsecutivePolls) || this.debounceConsecutivePolls < 1) {
      throw new Error("Invalid polling or debounce configuration");
    }
    this.getCurrentPersonId = options.getCurrentPersonId;
    this.getProfileContext = options.getProfileContext;
    this.loadWearerState = options.loadWearerState;
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
    this.candidateId = null;
    this.candidateCount = 0;

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

  /** Single poll tick: read persisted identity + debounce + maybe apply switch. */
  private async pollOnce(): Promise<void> {
    const observedId = await this.getCurrentPersonId();
    if (!this.running) return;

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
    } else {
      this.candidateCount += 1;
    }

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
      const previousContext = this.activeLevel1InterlocutorContext;
      this.activeInterlocutorPersonId = newPersonId;
      this.activeLevel1InterlocutorContext = null;

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
        this.activeLevel1InterlocutorContext = previousContext;
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