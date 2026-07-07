import type { StateCreator } from 'zustand';
import type {
  AgentToolCallState,
  AutoRun,
  ProviderConfig,
  ModelSelection,
  ModelCapability,
  ContextWatcherStatus,
  EngineStatus,
  WorkerFailureContext,
  WorkerResult,
  WorkerSpec,
} from '../types/agent';
import { AUTO_SENTINEL_MODEL, AUTO_SENTINEL_PROVIDER_ID } from '../types/agent';
import type { PendingApproval, TodoItem } from '../types/agentContract';

// ── Types ──────────────────────────────────────────────────────────────────

export interface ActiveTool {
  toolCallId: string;
  toolName: string;
  argsSummary: string;
}

export interface AgentStreamingState {
  isStreaming: boolean;
  textBuffer: string;
  activeTool: ActiveTool | null;
  toolCalls: AgentToolCallState[];
  startedAt: number | null;
  error: string | null;
  totalTokens: number;
  contextLimit: number;
}

/** One tool call observed inside the currently-running worker. Mirrors the
 * existing AgentToolCallState shape but with `status` rather than success
 * bool so an in-flight call is representable. */
export interface AutoToolActivity {
  toolCallId: string;
  toolName: string;
  argsSummary: string;
  status: 'running' | 'ok' | 'error';
  /** Populated when the tool finishes; the running-worker card renders a
   * short suffix on the matching row. */
  summary?: string;
}

/** Live Auto-mode run state for one Auto turn. Keyed by `runId` in the slice
 *  so multiple Auto turns within the same session each have their own record
 *  (one inline panel per turn). The chat-message renderer looks up state by
 *  the `auto_run_id` carried on the persisted assistant row; AutoRunPanel
 *  hydrates from `agent_get_auto_run` on mount when the key is missing. */
export interface AutoRunState {
  runId: string;
  /** Session this run belongs to. Used by event handlers that need to act
   *  on session-scoped state (Thinking placeholder, approvals). */
  sessionId: string;
  status:
    | 'planning'
    | 'awaiting_approval'
    | 'running'
    | 'replanning'
    | 'awaiting_failure_decision'
    | 'done'
    | 'failed';
  /** Set when status='awaiting_failure_decision' — the failed worker context
   *  for the Retry / Replan / Cancel banner. */
  pendingFailure: WorkerFailureContext | null;
  /** Latest plan version emitted by the orchestrator. */
  plan: { version: number; reasoning: string; workers: WorkerSpec[] } | null;
  /** Completed worker results so far (ordered). */
  workerResults: WorkerResult[];
  /** Worker that's currently running, if any. */
  currentWorker: { workerId: string; model: string; prompt: string } | null;
  /** Live tool calls for `currentWorker` — appended on AutoWorkerToolStart,
   * patched on AutoWorkerToolEnd, cleared on worker end / new worker start. */
  currentWorkerTools: AutoToolActivity[];
  /** Why the orchestrator was re-invoked (set when status='replanning'). */
  replanReason: string | null;
  /** Terminal failure reason when status='failed'. */
  failedReason: string | null;
  lastWorkerOutput: string | null;
  /** Final summary when status='done'. */
  doneSummary: string | null;
  totalCostCents: number;
}

export interface AgentSlice {
  // ── Folder / branch selection for new sessions ───────────────────────
  agentFolderPath: string | null;
  agentBranch: string | null;

  /** Per-session streaming state. Key = session_id. */
  agentStreaming: Record<string, AgentStreamingState>;
  pendingApprovals: Record<string, PendingApproval>;
  /** Maps session_id → the running loop's session id (same value today). */
  activeSessionIds: Record<string, string>;
  /** Per-session pending question from the agent's ask_user tool. */
  pendingQuestions: Record<string, { sessionId: string; question: string; options?: string[] }>;
  /** Per-session agent todo list from the todo_write tool. */
  agentTodos: Record<string, TodoItem[]>;
  /** Per-session token usage (persists after streaming clears). */
  tokenUsage: Record<string, { totalTokens: number; contextLimit: number | null; cacheReadTokens?: number; cacheCreationTokens?: number }>;
  /** Live file-watcher status per repo path, fed by `context-watcher-status` events. */
  contextWatcherStatus: Record<string, ContextWatcherStatus>;
  /** App-managed engine lifecycle status, fed by `engine:status` events (null until known). */
  engineStatus: EngineStatus | null;
  /** Latest `docker compose` progress line, fed by `engine:progress` events. */
  engineProgress: string | null;
  /** Per-Auto-turn run state. Key = run_id (one entry per Auto turn — a
   *  single session can have many). Hydrated either by live events
   *  (setAutoPlanning… setAutoDone) or by `hydrateAutoRun` from a snapshot. */
  autoRuns: Record<string, AutoRunState>;

  // ── Plan-mode flow ──────────────────────────────────────────────────
  /** Per-project completed plans from plan-mode sessions. Key = projectPath. */
  completedPlans: Record<string, { text: string; projectPath: string; planPath: string }>;
  /** Which project's plan is expanded in the side panel (null = closed). */
  activePlanProjectPath: string | null;
  /** Transient: plan data waiting for coding session confirmation after Implement click. */
  pendingPlanForCoding: { text: string; projectPath: string; planPath: string } | null;

  // ── Provider state ──────────────────────────────────────────────────
  providers: ProviderConfig[];
  selection: ModelSelection;
  providersLoaded: boolean;
  /** Resolved capability (context limit + vision) for the active model. */
  activeCapability: ModelCapability | null;

  // ── Actions ─────────────────────────────────────────────────────────
  setAgentFolderPath: (path: string | null) => void;
  setAgentBranch: (branch: string | null) => void;

  setAgentStreaming: (sessionId: string, state: AgentStreamingState) => void;
  appendTextDelta: (sessionId: string, delta: string) => void;
  setActiveTool: (sessionId: string, tool: ActiveTool | null) => void;
  addToolCall: (sessionId: string, toolCall: AgentToolCallState) => void;
  updateToolCall: (sessionId: string, toolCallId: string, success: boolean, summary: string) => void;
  setStreamingError: (sessionId: string, error: string | null) => void;
  setTokenUsage: (sessionId: string, totalTokens: number, contextLimit: number | null, cacheReadTokens?: number, cacheCreationTokens?: number) => void;
  clearTokenUsage: (sessionId: string) => void;
  setContextWatcherStatus: (repoPath: string, status: ContextWatcherStatus) => void;
  setEngineStatus: (status: EngineStatus) => void;
  setEngineProgress: (line: string) => void;
  clearAgentStreaming: (sessionId: string) => void;
  softClearAgentStreaming: (sessionId: string) => void;

  setActiveSession: (sessionId: string, loopId: string) => void;
  clearActiveSession: (sessionId: string) => void;

  addPendingApproval: (approval: PendingApproval) => void;
  removePendingApproval: (toolCallId: string) => void;

  setPendingQuestion: (sessionId: string, data: { sessionId: string; question: string; options?: string[] }) => void;
  clearPendingQuestion: (sessionId: string) => void;

  setAgentTodos: (sessionId: string, todos: TodoItem[]) => void;

  setCompletedPlan: (projectPath: string, plan: { text: string; projectPath: string; planPath: string } | null) => void;
  clearCompletedPlan: (projectPath: string) => void;
  setActivePlanProjectPath: (path: string | null) => void;
  setPendingPlanForCoding: (plan: { text: string; projectPath: string; planPath: string } | null) => void;

  loadProviders: () => Promise<void>;
  setActiveModel: (providerId: string, model: string) => Promise<void>;
  /** Resolve and cache the active model's capability (context limit + vision). */
  refreshActiveCapability: () => Promise<void>;

  // ── Auto-mode actions (P4/P6/P7) ──
  // Keyed by runId. The first per-run event is setAutoPlanning, which is the
  // only action that needs sessionId (it creates the record). All others
  // look up by runId only. hydrateAutoRun seeds state from a persisted
  // snapshot (agent_get_auto_run) for the inline-panel reload path.
  setAutoPlanning: (sessionId: string, runId: string) => void;
  setAutoPlan: (
    runId: string,
    version: number,
    reasoning: string,
    workers: WorkerSpec[],
  ) => void;
  setAutoAwaitingApproval: (runId: string) => void;
  setAutoWorkerStart: (
    runId: string,
    workerId: string,
    model: string,
    prompt: string,
  ) => void;
  appendAutoWorkerEnd: (runId: string, result: WorkerResult) => void;
  appendAutoWorkerToolStart: (
    runId: string,
    workerId: string,
    toolCallId: string,
    toolName: string,
    argsSummary: string,
  ) => void;
  patchAutoWorkerToolEnd: (
    runId: string,
    workerId: string,
    toolCallId: string,
    success: boolean,
    summary: string,
  ) => void;
  setAutoReplan: (runId: string, reason: string) => void;
  /** Transition to the paused-for-user-decision state. Called from the
   *  `agent:auto_awaiting_failure_decision` handler with the failure carried
   *  inline on the event, so no DB refetch is needed. */
  setAutoAwaitingFailureDecision: (
    runId: string,
    failure: WorkerFailureContext,
  ) => void;
  setAutoFailed: (
    runId: string,
    reason: string,
    lastWorkerOutput: string | null,
  ) => void;
  setAutoDone: (
    runId: string,
    summary: string,
    totalCostCents: number,
  ) => void;
  /** Seed state from a persisted AutoRun snapshot. Called by AutoRunPanel
   *  on mount when the runId isn't already in autoRuns (e.g. after a
   *  reload or when opening a session with historical Auto turns).
   *  No-op if the runId is already present — live events take precedence. */
  hydrateAutoRun: (snapshot: AutoRun) => void;
  clearAutoRun: (runId: string) => void;
}

// ── Helpers ────────────────────────────────────────────────────────────────

const emptyStreamingState: AgentStreamingState = {
  isStreaming: false,
  textBuffer: '',
  activeTool: null,
  toolCalls: [],
  startedAt: null,
  error: null,
  totalTokens: 0,
  contextLimit: 0,
};

/** Factory for a fresh streaming entry (isStreaming=true, startedAt=now). */
export function createInitialStreamingState(): AgentStreamingState {
  return {
    isStreaming: true,
    textBuffer: '',
    activeTool: null,
    toolCalls: [],
    startedAt: Date.now(),
    error: null,
    totalTokens: 0,
    contextLimit: 0,
  };
}

/** Patch a single field on agentStreaming[sessionId], creating the entry if missing. */
function patchStreaming(
  s: AgentSlice,
  sessionId: string,
  patch: Partial<AgentStreamingState>,
  fallback: AgentStreamingState = { ...emptyStreamingState, isStreaming: true },
): Pick<AgentSlice, 'agentStreaming'> {
  const current = s.agentStreaming[sessionId] ?? fallback;
  return {
    agentStreaming: { ...s.agentStreaming, [sessionId]: { ...current, ...patch } },
  };
}

// ── Slice Creator ──────────────────────────────────────────────────────────

export const createAgentSlice: StateCreator<AgentSlice, [], [], AgentSlice> = (set, get) => ({
  agentFolderPath: null,
  agentBranch: null,
  agentStreaming: {},
  pendingApprovals: {},
  activeSessionIds: {},
  pendingQuestions: {},
  agentTodos: {},
  tokenUsage: {},
  contextWatcherStatus: {},
  engineStatus: null,
  engineProgress: null,
  autoRuns: {},
  completedPlans: {},
  activePlanProjectPath: null,
  pendingPlanForCoding: null,
  providers: [],
  selection: { active: null, compaction: null, title: null },
  providersLoaded: false,
  activeCapability: null,

  setAgentFolderPath: (path) => set({ agentFolderPath: path }),
  setAgentBranch: (branch) => set({ agentBranch: branch }),

  setAgentStreaming: (sessionId, state) =>
    set((s) => ({ agentStreaming: { ...s.agentStreaming, [sessionId]: state } })),

  appendTextDelta: (sessionId, delta) =>
    set((s) => {
      const current = s.agentStreaming[sessionId] ?? { ...emptyStreamingState, isStreaming: true };
      return patchStreaming(s, sessionId, { textBuffer: current.textBuffer + delta });
    }),

  setActiveTool: (sessionId, tool) =>
    set((s) => patchStreaming(s, sessionId, { activeTool: tool })),

  addToolCall: (sessionId, toolCall) =>
    set((s) => {
      const current = s.agentStreaming[sessionId] ?? { ...emptyStreamingState, isStreaming: true };
      return patchStreaming(s, sessionId, { toolCalls: [...current.toolCalls, toolCall] });
    }),

  updateToolCall: (sessionId, toolCallId, success, summary) =>
    set((s) => {
      const current = s.agentStreaming[sessionId];
      if (!current) return s;
      return {
        agentStreaming: {
          ...s.agentStreaming,
          [sessionId]: {
            ...current,
            toolCalls: current.toolCalls.map((tc) =>
              tc.toolCallId === toolCallId
                ? { ...tc, status: success ? ('success' as const) : ('error' as const), summary }
                : tc,
            ),
          },
        },
      };
    }),

  setStreamingError: (sessionId, error) =>
    set((s) => patchStreaming(s, sessionId, { isStreaming: false, error }, { ...emptyStreamingState })),

  setTokenUsage: (sessionId, totalTokens, contextLimit, cacheReadTokens, cacheCreationTokens) =>
    set((s) => ({
      tokenUsage: { ...s.tokenUsage, [sessionId]: { totalTokens, contextLimit, cacheReadTokens, cacheCreationTokens } },
    })),

  clearTokenUsage: (sessionId) =>
    set((s) => {
      const { [sessionId]: _, ...rest } = s.tokenUsage;
      return { tokenUsage: rest };
    }),

  setContextWatcherStatus: (repoPath, status) =>
    set((s) => ({
      contextWatcherStatus: { ...s.contextWatcherStatus, [repoPath]: status },
    })),

  setEngineStatus: (status) => set({ engineStatus: status }),
  setEngineProgress: (line) => set({ engineProgress: line }),

  clearAgentStreaming: (sessionId) =>
    set((s) => {
      const { [sessionId]: _, ...rest } = s.agentStreaming;
      return { agentStreaming: rest };
    }),

  softClearAgentStreaming: (sessionId) =>
    set((s) => {
      const current = s.agentStreaming[sessionId];
      if (!current) return s;
      return {
        agentStreaming: {
          ...s.agentStreaming,
          [sessionId]: { ...current, isStreaming: false, activeTool: null },
        },
      };
    }),

  setActiveSession: (sessionId, loopId) =>
    set((s) => ({ activeSessionIds: { ...s.activeSessionIds, [sessionId]: loopId } })),

  clearActiveSession: (sessionId) =>
    set((s) => {
      const { [sessionId]: _, ...rest } = s.activeSessionIds;
      return { activeSessionIds: rest };
    }),

  addPendingApproval: (approval) =>
    set((s) => ({ pendingApprovals: { ...s.pendingApprovals, [approval.toolCallId]: approval } })),

  removePendingApproval: (toolCallId) =>
    set((s) => {
      const { [toolCallId]: _, ...rest } = s.pendingApprovals;
      return { pendingApprovals: rest };
    }),

  setPendingQuestion: (sessionId, data) =>
    set((s) => ({ pendingQuestions: { ...s.pendingQuestions, [sessionId]: data } })),

  clearPendingQuestion: (sessionId) =>
    set((s) => {
      const { [sessionId]: _, ...rest } = s.pendingQuestions;
      return { pendingQuestions: rest };
    }),

  setAgentTodos: (sessionId, todos) =>
    set((s) => ({ agentTodos: { ...s.agentTodos, [sessionId]: todos } })),

  setCompletedPlan: (projectPath, plan) =>
    set((s) => {
      if (plan) {
        return { completedPlans: { ...s.completedPlans, [projectPath]: plan } };
      }
      const { [projectPath]: _, ...rest } = s.completedPlans;
      return { completedPlans: rest };
    }),
  clearCompletedPlan: (projectPath) =>
    set((s) => {
      const { [projectPath]: _, ...rest } = s.completedPlans;
      return { completedPlans: rest };
    }),
  setActivePlanProjectPath: (path) => set({ activePlanProjectPath: path }),
  setPendingPlanForCoding: (plan) => set({ pendingPlanForCoding: plan }),

  loadProviders: async () => {
    try {
      const { agentTauriService } = await import('../services/agentTauriService');
      const { providers, selection } = await agentTauriService.listProviders();
      set({ providers, selection, providersLoaded: true });
      void get().refreshActiveCapability();
    } catch (e) {
      console.error('[agentSlice] Failed to load providers:', e);
    }
  },

  setActiveModel: async (providerId, model) => {
    set((s) => ({ selection: { ...s.selection, active: { providerId, model } } }));
    try {
      const { agentTauriService } = await import('../services/agentTauriService');
      await agentTauriService.setModelSelection('active', providerId, model);
      void get().refreshActiveCapability();
    } catch (e) {
      console.error('[agentSlice] Failed to set active model:', e);
    }
  },

  refreshActiveCapability: async () => {
    const active = get().selection.active;
    if (!active) {
      set({ activeCapability: null });
      return;
    }
    // Auto sentinel isn't a real provider — the backend resolver would 404.
    // Workers are full AgentLoops running whatever model the orchestrator
    // picks, some of which support images. Report vision-capable so the
    // attach affordances stay visible; the actual per-worker vision check
    // is redundant because the orchestrator routes images to a vision model.
    if (
      active.providerId === AUTO_SENTINEL_PROVIDER_ID &&
      active.model === AUTO_SENTINEL_MODEL
    ) {
      set({
        activeCapability: {
          contextLimit: null,
          supportsImages: true,
          source: 'auto',
        },
      });
      return;
    }
    try {
      const { agentTauriService } = await import('../services/agentTauriService');
      const cap = await agentTauriService.resolveModelCapability(active.providerId, active.model);
      set({ activeCapability: cap });
    } catch (e) {
      console.error('[agentSlice] Failed to resolve model capability:', e);
      set({ activeCapability: null });
    }
  },

  // ── Auto-mode actions ──
  // Keyed by runId. setAutoPlanning is the only action that needs sessionId
  // (it creates the record); the rest look up by runId only.

  setAutoPlanning: (sessionId, runId) =>
    set((s) => ({
      autoRuns: {
        ...s.autoRuns,
        [runId]: {
          runId,
          sessionId,
          status: 'planning',
          plan: null,
          workerResults: [],
          currentWorker: null,
          currentWorkerTools: [],
          replanReason: null,
          failedReason: null,
          lastWorkerOutput: null,
          doneSummary: null,
          totalCostCents: 0,
          pendingFailure: null,
        },
      },
    })),

  setAutoPlan: (runId, version, reasoning, workers) =>
    set((s) => {
      const current = s.autoRuns[runId];
      if (!current) return s;
      return {
        autoRuns: {
          ...s.autoRuns,
          [runId]: {
            ...current,
            plan: { version, reasoning, workers },
            // Plan v1 → awaiting approval; plan v2+ (replans) → keep running.
            status: version === 1 ? 'planning' : 'running',
          },
        },
      };
    }),

  setAutoAwaitingApproval: (runId) =>
    set((s) => {
      const current = s.autoRuns[runId];
      if (!current) return s;
      return {
        autoRuns: {
          ...s.autoRuns,
          [runId]: { ...current, status: 'awaiting_approval' },
        },
      };
    }),

  setAutoWorkerStart: (runId, workerId, model, prompt) =>
    set((s) => {
      const current = s.autoRuns[runId];
      if (!current) return s;
      return {
        autoRuns: {
          ...s.autoRuns,
          [runId]: {
            ...current,
            status: 'running',
            currentWorker: { workerId, model, prompt },
            // Fresh worker → fresh tool list. The previous worker's tool
            // history isn't retained (the collapsed worker card shows only
            // the total tool_count summary).
            currentWorkerTools: [],
          },
        },
      };
    }),

  appendAutoWorkerEnd: (runId, result) =>
    set((s) => {
      const current = s.autoRuns[runId];
      if (!current) return s;
      // The backend AutoWorkerEnd event doesn't include `model` (avoids
      // duplicating what AutoWorkerStart already shipped); patch it from
      // the live currentWorker so the rendered worker card can label the
      // model in its collapsed row.
      const patched = {
        ...result,
        model:
          (current.currentWorker?.workerId === result.id
            ? current.currentWorker.model
            : '') || result.model,
      };
      return {
        autoRuns: {
          ...s.autoRuns,
          [runId]: {
            ...current,
            workerResults: [...current.workerResults, patched],
            currentWorker: null,
            currentWorkerTools: [],
            totalCostCents: current.totalCostCents + result.cost_cents,
          },
        },
      };
    }),

  appendAutoWorkerToolStart: (runId, workerId, toolCallId, toolName, argsSummary) =>
    set((s) => {
      const current = s.autoRuns[runId];
      if (!current) return s;
      // Defensive: only record tools belonging to the currently-running
      // worker. If a stale event from a previous worker arrives after we've
      // already moved on, drop it.
      if (current.currentWorker?.workerId !== workerId) return s;
      return {
        autoRuns: {
          ...s.autoRuns,
          [runId]: {
            ...current,
            currentWorkerTools: [
              ...current.currentWorkerTools,
              { toolCallId, toolName, argsSummary, status: 'running' },
            ],
          },
        },
      };
    }),

  patchAutoWorkerToolEnd: (runId, workerId, toolCallId, success, summary) =>
    set((s) => {
      const current = s.autoRuns[runId];
      if (!current) return s;
      if (current.currentWorker?.workerId !== workerId) return s;
      return {
        autoRuns: {
          ...s.autoRuns,
          [runId]: {
            ...current,
            currentWorkerTools: current.currentWorkerTools.map((t) =>
              t.toolCallId === toolCallId
                ? { ...t, status: success ? 'ok' : 'error', summary }
                : t,
            ),
          },
        },
      };
    }),

  setAutoReplan: (runId, reason) =>
    set((s) => {
      const current = s.autoRuns[runId];
      if (!current) return s;
      return {
        autoRuns: {
          ...s.autoRuns,
          [runId]: {
            ...current,
            status: 'replanning',
            replanReason: reason,
          },
        },
      };
    }),

  setAutoAwaitingFailureDecision: (runId, failure) =>
    set((s) => {
      const current = s.autoRuns[runId];
      if (!current) return s;
      return {
        autoRuns: {
          ...s.autoRuns,
          [runId]: {
            ...current,
            status: 'awaiting_failure_decision',
            pendingFailure: failure,
            // The failed worker's summary was already appended to
            // workerResults by the preceding AutoWorkerEnd event; clear the
            // in-flight card so the panel doesn't show a running spinner
            // over the amber banner.
            currentWorker: null,
            currentWorkerTools: [],
          },
        },
      };
    }),

  setAutoFailed: (runId, reason, lastWorkerOutput) =>
    set((s) => {
      const current = s.autoRuns[runId];
      if (!current) return s;
      return {
        autoRuns: {
          ...s.autoRuns,
          [runId]: {
            ...current,
            status: 'failed',
            failedReason: reason,
            lastWorkerOutput,
            currentWorker: null,
          },
        },
      };
    }),

  setAutoDone: (runId, summary, totalCostCents) =>
    set((s) => {
      const current = s.autoRuns[runId];
      if (!current) return s;
      return {
        autoRuns: {
          ...s.autoRuns,
          [runId]: {
            ...current,
            status: 'done',
            doneSummary: summary,
            totalCostCents,
            currentWorker: null,
          },
        },
      };
    }),

  hydrateAutoRun: (snapshot) =>
    set((s) => {
      // Don't overwrite a live run with stale snapshot data — live events
      // are always more recent than what's on disk.
      if (s.autoRuns[snapshot.id]) return s;
      const latestPlan = snapshot.plan_versions.length
        ? snapshot.plan_versions[snapshot.plan_versions.length - 1]
        : null;
      // Map backend AutoStatus → frontend status enum. 'replanning' is a
      // live-only intermediate; a snapshot mid-replan reads as 'running'.
      const status: AutoRunState['status'] = (() => {
        switch (snapshot.status) {
          case 'planning': return 'planning';
          case 'awaiting_approval': return 'awaiting_approval';
          case 'running': return 'running';
          case 'awaiting_failure_decision': return 'awaiting_failure_decision';
          case 'done': return 'done';
          case 'failed': return 'failed';
          // 'cancelled' doesn't have a dedicated UI state; render as failed.
          case 'cancelled': return 'failed';
        }
      })();
      // The snapshot has no failure reason or done summary fields — the
      // live AutoFailed/AutoDone events carry those. For a terminal hydrated
      // run we fall back to the last worker's summary (matches what
      // executor::finalize_done uses internally for AutoDone.summary).
      const lastWorker = snapshot.worker_results.length
        ? snapshot.worker_results[snapshot.worker_results.length - 1]
        : null;
      // Map the persisted current_worker (a WorkerSpec) to the live shape
      // ({ workerId, model, prompt }). Live events use the same shape, so
      // once events resume they overlay cleanly. currentWorkerTools stays
      // empty — tool calls aren't persisted (live-only by design).
      const currentWorker = snapshot.current_worker
        ? {
            workerId: snapshot.current_worker.id,
            model: snapshot.current_worker.model,
            prompt: snapshot.current_worker.prompt,
          }
        : null;
      return {
        autoRuns: {
          ...s.autoRuns,
          [snapshot.id]: {
            runId: snapshot.id,
            sessionId: snapshot.session_id,
            status,
            plan: latestPlan
              ? {
                  version: latestPlan.version,
                  reasoning: latestPlan.reasoning,
                  workers: latestPlan.workers,
                }
              : null,
            workerResults: snapshot.worker_results,
            currentWorker,
            currentWorkerTools: [],
            replanReason: null,
            failedReason: status === 'failed' ? 'Auto run failed' : null,
            lastWorkerOutput: lastWorker?.summary ?? null,
            doneSummary: status === 'done' ? lastWorker?.summary ?? null : null,
            totalCostCents: snapshot.cost_cents,
            pendingFailure: snapshot.pending_failure,
          },
        },
      };
    }),

  clearAutoRun: (runId) =>
    set((s) => {
      if (!s.autoRuns[runId]) return s;
      const { [runId]: _, ...rest } = s.autoRuns;
      return { autoRuns: rest };
    }),
});
