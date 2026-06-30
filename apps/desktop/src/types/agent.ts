// ============================================================
// Agent TypeScript interfaces
// ============================================================

// --- Agent (same shape as WorkspaceUser from Rails get_users?filter=agents) ---

export interface Agent {
  id: number;
  agent_id?: string;
  name: string;
  email: string;
  first_name?: string;
  last_name?: string;
  avatar_url?: string;
  description?: string;
  agent_type?: string;
}

// ============================================================
// Agent Chat — Diffs, Artifacts, Messages, Threads
// ============================================================

// --- Diff primitives ---

export type FileDiffStatus = 'added' | 'modified' | 'deleted' | 'renamed';

export interface DiffLine {
  type: 'add' | 'delete' | 'context';
  content: string;
  old_line_number?: number;
  new_line_number?: number;
}

export interface DiffHunk {
  old_start: number;
  old_lines: number;
  new_start: number;
  new_lines: number;
  header: string;
  lines: DiffLine[];
}

export interface FileDiff {
  file_path: string;
  old_path?: string;
  status: FileDiffStatus;
  additions: number;
  deletions: number;
  hunks: DiffHunk[];
}

// --- Generic Artifact System ---

export type ArtifactType = 'code_changes' | 'terminal' | 'file' | 'text';

export interface BaseArtifact {
  id: string;
  type: ArtifactType;
  name: string;
  created_at: string;
}

export interface CodeChangesArtifact extends BaseArtifact {
  type: 'code_changes';
  files: FileDiff[];
  total_additions: number;
  total_deletions: number;
  files_changed?: number;
  /** Turn number this diff belongs to (undefined = cumulative/legacy). */
  turnCount?: number;
}

export interface TerminalArtifact extends BaseArtifact {
  type: 'terminal';
  command: string;
  output: string;
  exit_code: number;
}

export interface FileArtifact extends BaseArtifact {
  type: 'file';
  url: string;
  file_name: string;
  media_type: string;
  size?: number;
}

export interface TextArtifact extends BaseArtifact {
  type: 'text';
  content: string;
  format?: 'plain' | 'markdown' | 'html';
}

export type Artifact = CodeChangesArtifact | TerminalArtifact | FileArtifact | TextArtifact;

// --- Agent Messages ---

export type AgentMessageRole = 'user' | 'agent';

export interface AgentMessage {
  id: string;
  thread_id: string;
  agent_id: string;
  role: AgentMessageRole;
  text: string;
  /** Image data-URLs attached to this message (shown in the bubble). */
  images?: string[];
  artifacts: Artifact[];
  created_at: string;
  thinking?: {
    toolCalls: { toolCallId: string; toolName: string; argsSummary: string; status: string; summary?: string }[];
    durationSeconds: number;
  };
  /** Present iff this row anchors an Auto-mode turn. The thread renderer
   *  swaps the normal bubble for an inline `<AutoRunPanel runId=...>`. */
  auto_run_id?: string;
}

// --- Agent Thread ---

export interface AgentThread {
  id: string;
  agent_id: string;
  task_summary: string;
  folder_path: string;
  branch: string;
  worktree_path?: string;
  status: 'active' | 'completed' | 'error';
  is_coding_session: boolean;
  total_additions: number;
  total_deletions: number;
  /** Number of files changed in the working tree (from the git diff). */
  files_changed?: number;
  checkpoints: CheckpointSummary[];
  selectedDiffTurn: number | null;
  messages: AgentMessage[];
  /** Full plan text when this coding session was started from a plan. */
  sourcePlanText?: string;
  created_at: string;
  updated_at: string;
}

export interface CheckpointSummary {
  turn_count: number;
  checkpoint_ref: string;
  commit_sha: string;
  files: string[];
  additions: number;
  deletions: number;
  status: string;
  created_at: string;
}

// --- UI State ---

export type AgentViewMode = 'chat' | 'thread' | 'diff_review';

export interface ArtifactFileDecision {
  filePath: string;
  decision: 'pending' | 'accepted' | 'rejected';
}

// --- Agent Display Message (from SQLite via Tauri) ---

export interface AgentToolChip {
  name: string;
  summary: string;
}

export interface AgentDisplayMessage {
  id: string;
  role: 'user' | 'assistant';
  text: string;
  created_at: string;
  session_id: string;
  /** Tool calls reconstructed from the SQLite table for the "Thought for…" chips. */
  tools: AgentToolChip[];
  /** Seconds spent on tool calls before this message (for "Thought for Ns"). */
  duration_seconds: number;
  /** Image data-URLs attached to this message (rebuilt from on-disk refs). */
  images: string[];
  /** Present iff this row anchors an Auto-mode turn. The message renderer
   *  swaps the normal text bubble for an `<AutoRunPanel runId=...>` when
   *  this is set. Populated from `agent_messages.metadata.auto_run_id`. */
  auto_run_id?: string;
}

// --- Session (from the `sessions` table via Tauri) ---

export type SessionMode = 'ask' | 'plan' | 'coding';

export interface SessionRow {
  id: string;
  folder: string;
  mode: string;
  title: string | null;
  parent_session_id: string | null;
  created_at: string;
  updated_at: string;
  status: string;
  providerId?: string | null;
  model?: string | null;
}

// --- Thread Summary (from SQLite via Tauri) ---

export interface ThreadSummary {
  thread_id: string;
  is_coding_session: boolean;
  task_summary: string;
  project_path: string;
  branch: string;
  created_at: string;
  message_count: number;
}

export interface AgentToolCallState {
  toolCallId: string;
  toolName: string;
  argsSummary: string;
  status: 'running' | 'success' | 'error';
  summary?: string;
}

export type AgentSessionStatus = 'idle' | 'streaming' | 'tool_running' | 'done' | 'error';

export interface ModelProfile {
  id: string;
  display_name: string;
  provider: string;
  context_window: number;
}

/** UI provider kind → backend wire format. "openai_compatible" takes a custom base_url. */
export type ProviderKind = 'openai' | 'openai_compatible' | 'anthropic';

/** Per-model discovered/edited metadata. Mirrors Rust `ModelMeta`. */
export interface ModelMeta {
  /** Discovered context length; absent/null = unknown. */
  contextLength?: number | null;
  supportsImages?: boolean;
}

/** A saved LLM provider = an endpoint (no model bundled). Mirrors Rust `ProviderConfig`. */
export interface ProviderConfig {
  id: string;
  kind: ProviderKind;
  /** Display name — shown for openai_compatible providers; built-ins use their kind name. */
  label: string;
  baseUrl: string;
  apiKey: string;
  /** Available model ids (populated by "Fetch models" or typed). Feeds the pickers. */
  models: string[];
  /** Per-model metadata keyed by model id (discovered context length, vision). */
  modelMeta?: Record<string, ModelMeta>;
  /** Provider-level vision fallback for custom providers. */
  supportsImages?: boolean;
}

/** A model advertised by a provider's /models, with discovered context length. */
export interface FetchedModel {
  id: string;
  contextLength: number | null;
}

/** A built-in registry model for the Settings picker. Mirrors Rust `CuratedModel`. */
export interface CuratedModel {
  id: string;
  displayName: string;
  /** "openai" | "anthropic" — matches the built-in provider kind. */
  provider: string;
  contextWindow: number;
  supportsImages: boolean;
}

/** Resolved capability for the active (provider, model). Mirrors Rust `ModelCapability`. */
export interface ModelCapability {
  /** `null` = unknown → context bar shows raw count, auto-compaction off. */
  contextLimit: number | null;
  supportsImages: boolean;
  /** "known" | "discovered" | "unknown". */
  source: string;
}

/** A model on a specific provider. */
export interface ModelRef {
  providerId: string;
  model: string;
}

/** Global model selections — each picks a model from across configured providers. */
export interface ModelSelection {
  active: ModelRef | null;
  compaction: ModelRef | null;
  title: ModelRef | null;
}

export type SelectionRole = 'active' | 'compaction' | 'title';

export interface ProvidersResponse {
  providers: ProviderConfig[];
  selection: ModelSelection;
}

// ── Auto mode ──────────────────────────────────────────────────────────
// Mirrors Rust types in `agent::auto` + `agent_bridge::auto`. The sentinel
// (provider_id="auto", model="auto") is what the frontend writes when the
// user picks Auto in the picker; the backend special-cases it to dispatch
// into the orchestrator pipeline instead of a normal LLM call.

/** Sentinel constants for the "Auto" pseudo-provider/model. */
export const AUTO_SENTINEL_PROVIDER_ID = 'auto';
export const AUTO_SENTINEL_MODEL = 'auto';

/** One entry in the user-curated worker pool. Mirrors Rust `WorkerPoolEntry`. */
export interface WorkerPoolEntry {
  providerId: string;
  model: string;
  /** User-written hint the orchestrator sees when picking a model per subtask. */
  description: string;
}

/** Full Auto settings. Returned by `agent_get_auto_settings`. Mirrors Rust `AutoSettings`. */
export interface AutoSettings {
  /** Master switch; the picker shows the Auto entry iff this is true. */
  enabled: boolean;
  /** User-picked orchestrator model. `null` until configured. */
  orchestrator: ModelRef | null;
  workerPool: WorkerPoolEntry[];
}

/** Worker's `see_prior` from the orchestrator. Untagged enum on the wire:
 * `"none"` | `"all"` | `string[]` (worker ids). */
export type SeePrior = 'none' | 'all' | string[];

/** Spec for one worker in an orchestrator plan. Mirrors Rust `WorkerSpec`. */
export interface WorkerSpec {
  id: string;
  model: string;
  prompt: string;
  see_prior: SeePrior;
}

/** Terminal status of a worker. Mirrors Rust `WorkerStatus` (tagged enum). */
export type WorkerStatus =
  | { kind: 'ok' }
  | { kind: 'max_iterations' }
  | { kind: 'tool_error'; message: string }
  | { kind: 'llm_error'; message: string }
  | { kind: 'cancelled' };

/** Result of executing one worker. Mirrors Rust `WorkerResult`. */
export interface WorkerResult {
  id: string;
  model: string;
  prompt: string;
  summary: string;
  tool_count: number;
  cost_cents: number;
  status: WorkerStatus;
}

/** Terminal status of an Auto run. Mirrors Rust `AutoStatus`. */
export type AutoStatus =
  | 'planning'
  | 'awaiting_approval'
  | 'running'
  | 'awaiting_failure_decision'
  | 'done'
  | 'failed'
  | 'cancelled';

/** User-recoverable worker failure context. Mirrors Rust `WorkerFailureContext`.
 *  Present on AutoRun.pending_failure when status is awaiting_failure_decision. */
export interface WorkerFailureContext {
  worker_id: string;
  worker_model: string;
  worker_prompt: string;
  error_message: string;
}

/** One orchestrator-produced plan version. Mirrors Rust `Plan`. */
export interface Plan {
  version: number;
  reasoning: string;
  workers: WorkerSpec[];
}

/** Persisted Auto-run snapshot. Mirrors Rust `AutoRun` from the `auto_runs`
 *  table. Returned by `agent_get_auto_run(run_id)` for the AutoRunPanel to
 *  hydrate from disk on mount when in-memory state is missing. */
export interface AutoRun {
  id: string;
  session_id: string;
  status: AutoStatus;
  plan_versions: Plan[];
  worker_results: WorkerResult[];
  cost_cents: number;
  replans_used: number;
  /** Set while a worker is in flight. Cleared back to null on AutoWorkerEnd
   *  (the completed result lands in worker_results). Persists in the
   *  snapshot so a mid-flight reload shows "Running: wN (model)" instead
   *  of just "status: running". */
  current_worker: WorkerSpec | null;
  /** Set when a worker hard-fails and the run pauses awaiting user
   *  decision (Retry / Replan / Cancel from the panel banner). */
  pending_failure: WorkerFailureContext | null;
}

// Event payloads emitted on `agent:auto_*` Tauri channels. Each carries
// `thread_id` so the UI can route to the right session card stack.

export interface AutoPlanningEvent {
  thread_id: string;
  run_id: string;
}
export interface AutoPlanEvent {
  thread_id: string;
  run_id: string;
  version: number;
  reasoning: string;
  plan: WorkerSpec[];
}
export interface AutoAwaitingApprovalEvent {
  thread_id: string;
  run_id: string;
}
export interface AutoWorkerStartEvent {
  thread_id: string;
  run_id: string;
  worker_id: string;
  model: string;
  prompt: string;
}
export interface AutoWorkerEndEvent {
  thread_id: string;
  run_id: string;
  worker_id: string;
  summary: string;
  cost_cents: number;
  tool_count: number;
  status: WorkerStatus;
}
export interface AutoWorkerToolStartEvent {
  thread_id: string;
  run_id: string;
  worker_id: string;
  tool_call_id: string;
  tool_name: string;
  args_summary: string;
}
export interface AutoWorkerToolEndEvent {
  thread_id: string;
  run_id: string;
  worker_id: string;
  tool_call_id: string;
  success: boolean;
  summary: string;
}
export interface AutoReplanEvent {
  thread_id: string;
  run_id: string;
  reason: string;
}
export interface AutoFailedEvent {
  thread_id: string;
  run_id: string;
  reason: string;
  last_worker_output: string | null;
}
export interface AutoDoneEvent {
  thread_id: string;
  run_id: string;
  summary: string;
  total_cost_cents: number;
}

/** Opt-in context engine (semantic + graph code search) connection. */
export interface ContextEngineSettings {
  enabled: boolean;
  base_url: string;
}

/** Live connection probe result for the Settings panel. */
export interface ContextEngineStatus {
  connected: boolean;
  error: string | null;
}

/** A repo the app knows about + its index state on the backend. */
export interface IndexedRepo {
  path: string;
  exists: boolean;
  empty: boolean;
  repo_id: number | null;
  /** Live watcher status tag: "not_indexed" | "indexing" | "indexed" | "error:<reason>", or null if unwatched. */
  status: string | null;
  /** File count from the last full sync, when known. */
  file_count: number | null;
}

/** Context-engine lifecycle mode. `user` = connect to a self-run backend;
 *  `app` = the app runs the docker-compose stack itself. */
export type EngineMode = "user" | "app";

/** App-managed stack lifecycle status, mirrored from `engine:status` events.
 *  Tagged by `state`; only the matching variant's fields are present. */
export type EngineStatus =
  | { state: "docker_missing"; reason: string }
  | { state: "stopped" }
  | { state: "pulling"; line: string }
  | { state: "starting" }
  | { state: "running"; base_url: string }
  | { state: "error"; reason: string; logs_tail: string | null };

/** Live file-watcher status for a repo, mirrored from `context-watcher-status` events. */
export interface ContextWatcherStatus {
  /** "not_indexed" | "indexing" | "indexed" | "error" */
  status: string;
  fileCount?: number | null;
  reason?: string | null;
}

// --- Agent List API ---

export interface ListAgentsParams {
  page?: number;
  size?: number;
  status?: string;
  query?: string;
}

export interface AgentListResponse {
  success: boolean;
  agents: Agent[];
  page: number;
  size: number;
  total: number;
  hasMore: boolean;
}

// --- Agent Chat API Payloads/Responses ---

export interface SendAgentMessagePayload {
  agent_id: string;
  text: string;
  folder_path: string;
  branch: string;
  thread_id?: string;
}

export interface SendAgentMessageResponse {
  success: boolean;
  thread: AgentThread;
}

export interface AgentThreadResponse {
  success: boolean;
  thread: AgentThread;
}

export interface AgentThreadListResponse {
  success: boolean;
  threads: AgentThread[];
  page: number;
  size: number;
  total: number;
  hasMore: boolean;
}
