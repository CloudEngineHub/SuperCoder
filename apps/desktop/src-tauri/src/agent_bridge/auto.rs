//! Auto-mode Tauri bridge.
//!
//! Wires the backend `agent::auto` module (orchestrator + workers + executor)
//! into the desktop app. Phase P4 of PHASE_AUTO_MODE.md.
//!
//! Surface:
//!   - Settings persistence: orchestrator model, worker pool, replan budget
//!     stored in the SQLite settings k-v table; 4 Tauri commands surface them.
//!   - `TauriPlanApprover` — mirrors `TauriApprovalHandler` for tool approvals.
//!     Per-run_id `oneshot` channel via a shared `HashMap` on `AgentState`;
//!     `agent_approve_auto_plan` resolves it from the frontend.
//!   - `run_auto_turn` — when `run_agent_turn` detects the Auto sentinel
//!     (provider_id="auto" + model="auto"), it dispatches here instead of the
//!     regular session-manager path. Builds `ExecutorConfig`, `WorkerContext`,
//!     `LlmPlanGenerator`, `LiveWorkerRunner`, and `TauriPlanApprover`; spawns
//!     `agent::auto::run_auto` on a task whose events drain through the same
//!     `spawn_event_relay` the regular path uses.
//!
//! Cross-provider worker dispatch (P8): each `WorkerPoolEntry`'s `provider_id`
//! is resolved at run start by `build_worker_llm_configs` into a
//! `HashMap<model, LlmClientConfig>` that the executor's `WorkerContext`
//! holds. `worker::run_worker` looks up its config per spec — mixed-provider
//! pools (e.g. Anthropic orchestrator + nvidia/Nemotron worker via
//! OpenRouter) now route to the correct endpoint instead of all hitting the
//! orchestrator's API.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use tauri::{AppHandle, Emitter, Manager, State};
use tokio::sync::{mpsc, oneshot};
use tokio_util::sync::CancellationToken;
use uuid::Uuid;

use agent::agent::config::{CompactionConfig, RetryConfig};
use agent::auto::{
    self, AutoResult, AutoRun, AutoStateListener, AutoStatus, ExecutorConfig, LiveWorkerRunner,
    LlmPlanGenerator, Plan, PlanApprover, WorkerContext, WorkerPoolEntry, WorkerStatus,
};

use crate::AppState;
use super::commands::{provider_to_llm_config, AgentState, ModelRef};
use super::events::spawn_event_relay;
use super::traits::{EventEmitter, TauriEventEmitter};

// ── Constants ────────────────────────────────────────────────────────────

/// Sentinel `(provider_id, model)` pair that signals "this session is in Auto
/// mode". The model picker writes these into `sessions.provider_id` /
/// `sessions.model` (and into `ModelSelection.active`) when the user picks
/// the Auto entry. `agent_set_model_selection` special-cases these to skip
/// the usual provider-exists validation.
pub const AUTO_SENTINEL_PROVIDER_ID: &str = "auto";
pub const AUTO_SENTINEL_MODEL: &str = "auto";

/// Settings DB keys. Same k-v table that holds `llm_selection`,
/// `llm_providers`, `context_engine`. Values are JSON strings (or raw for
/// the bool/u32 keys).
const AUTO_ORCHESTRATOR_KEY: &str = "auto_orchestrator_model";
const AUTO_WORKER_POOL_KEY: &str = "auto_worker_pool";
/// Master switch. Auto entry is shown in the model picker iff this is true.
/// Prereqs (orchestrator picked + worker pool non-empty) are enforced by
/// `agent_set_auto_enabled` when flipping on, and by a live-session check
/// when flipping off (see `agent_list_sessions_using_auto`).
const AUTO_ENABLED_KEY: &str = "auto_enabled";

/// Hardcoded ceiling on automatic replans inside the executor. Set to 0
/// because P9c moves replan/retry control to the user (worker failure →
/// pause → user picks Retry / Replan / Cancel on the panel). The
/// executor's automatic replan loop is effectively disabled; this constant
/// is the safety value passed into `ExecutorConfig.replan_budget`.
pub const AUTO_REPLAN_BUDGET: u32 = 0;

/// Returns true iff `(provider_id, model)` is the Auto sentinel.
pub fn is_auto_mode(provider_id: &str, model: &str) -> bool {
    provider_id == AUTO_SENTINEL_PROVIDER_ID && model == AUTO_SENTINEL_MODEL
}

// ── Settings DTO + persistence ───────────────────────────────────────────

/// Auto settings returned by `agent_get_auto_settings` for the Settings UI
/// to consume. Replan budget removed in P9b — user is the budget now via
/// the Retry / Replan / Cancel buttons on the panel's failure banner.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct AutoSettings {
    /// Master switch. `true` ⇒ the Auto entry appears in the model picker
    /// and `run_auto_turn` accepts dispatches. `false` ⇒ both gates closed.
    pub enabled: bool,
    /// User-picked orchestrator model. `None` when Auto mode is not yet
    /// configured — `run_auto_turn` rejects with a "not configured" error
    /// in that case.
    pub orchestrator: Option<ModelRef>,
    /// User-curated pool of worker models with per-entry descriptions.
    #[serde(default)]
    pub worker_pool: Vec<WorkerPoolEntry>,
}

pub(crate) fn read_auto_orchestrator(app_state: &AppState) -> Option<ModelRef> {
    app_state
        .db
        .get_setting(AUTO_ORCHESTRATOR_KEY)
        .ok()
        .flatten()
        .and_then(|raw| serde_json::from_str(&raw).ok())
}

pub(crate) fn read_auto_worker_pool(app_state: &AppState) -> Vec<WorkerPoolEntry> {
    app_state
        .db
        .get_setting(AUTO_WORKER_POOL_KEY)
        .ok()
        .flatten()
        .and_then(|raw| serde_json::from_str(&raw).ok())
        .unwrap_or_default()
}

pub(crate) fn read_auto_enabled(app_state: &AppState) -> bool {
    app_state
        .db
        .get_setting(AUTO_ENABLED_KEY)
        .ok()
        .flatten()
        .map(|raw| raw == "true")
        .unwrap_or(false)
}

pub(crate) fn read_auto_settings(app_state: &AppState) -> AutoSettings {
    AutoSettings {
        enabled: read_auto_enabled(app_state),
        orchestrator: read_auto_orchestrator(app_state),
        worker_pool: read_auto_worker_pool(app_state),
    }
}

// ── agent_messages anchor rows (P7) ──────────────────────────────────────
//
// Every Auto turn writes one user row + one placeholder assistant row to
// `agent_messages`, both tagged with `{auto_run_id, kind:"auto"}` metadata.
// The rich snapshot lives in the `auto_runs` table; these rows are the
// chat-thread anchors that let the UI render an AutoRunPanel inline at the
// right position, and that feed the next turn's LLM context with a plain
// summary string once the run terminates.

fn build_llm_message_json(role: &str, content: &str) -> String {
    serde_json::json!({ "role": role, "content": content }).to_string()
}

fn build_auto_metadata_json(auto_run_id: &str) -> String {
    serde_json::json!({ "auto_run_id": auto_run_id, "kind": "auto" }).to_string()
}

/// Persist the user prompt + placeholder assistant row for an Auto turn.
/// Returns the assistant row's numeric id so the executor's terminal handler
/// can overwrite its `llm_message` with the final summary text.
async fn persist_auto_anchor_rows(
    db: Arc<super::db::AgentDb>,
    session_id: String,
    folder: String,
    prompt: String,
    auto_run_id: String,
) -> Result<i64, String> {
    let metadata = build_auto_metadata_json(&auto_run_id);

    let user_llm = build_llm_message_json("user", &prompt);
    let user_id = {
        let db = Arc::clone(&db);
        let sid = session_id.clone();
        let folder = folder.clone();
        let meta = metadata.clone();
        tokio::task::spawn_blocking(move || {
            db.insert_message(&sid, &folder, "user", "text", &user_llm, &meta, None)
        })
        .await
        .map_err(|e| format!("join error: {e}"))?
        .map_err(|e| format!("Failed to persist Auto user row: {e}"))?
    };

    let asst_llm = build_llm_message_json("assistant", "Auto run in progress…");
    let id_str = {
        let db = Arc::clone(&db);
        let sid = session_id.clone();
        let folder = folder.clone();
        let meta = metadata.clone();
        tokio::task::spawn_blocking(move || {
            db.insert_message(&sid, &folder, "assistant", "text", &asst_llm, &meta, None)
        })
        .await
        .map_err(|e| format!("join error: {e}"))?
        .map_err(|e| format!("Failed to persist Auto assistant row: {e}"))?
    };
    let asst_id = id_str
        .parse::<i64>()
        .map_err(|e| format!("bad insert id: {e}"))?;
    log::info!(
        "[Auto-P7] persisted anchor rows: session={} run={} user_row={} assistant_row={}",
        session_id, auto_run_id, user_id, asst_id
    );
    Ok(asst_id)
}

/// Build the per-worker LLM config map for an Auto run by snapshotting the
/// configured worker pool against the current provider settings.
///
/// Each pool entry's `provider_id` is resolved to a `ProviderConfig`, which
/// feeds `provider_to_llm_config` along with the entry's `model`. The map is
/// keyed by model name (matching `WorkerSpec.model`), so the executor's
/// per-worker dispatch (`worker::run_worker`) is a single lookup.
///
/// Fail-fast validation here surfaces config drift before any LLM call:
/// - Missing provider (user deleted it from Settings after adding to pool)
/// - Duplicate model name across different providers (ambiguous lookup —
///   `WorkerSpec` carries only `model`, so two pool entries with the same
///   model under different providers can't be told apart by the dispatcher).
fn build_worker_llm_configs(
    app_state: &AppState,
    pool: &[WorkerPoolEntry],
) -> Result<HashMap<String, agent::llm::LlmClientConfig>, String> {
    let mut map: HashMap<String, agent::llm::LlmClientConfig> = HashMap::new();
    for entry in pool {
        if map.contains_key(&entry.model) {
            return Err(format!(
                "Auto mode: worker pool has duplicate model name `{}` across providers — \
                 make pool entries' model names unique (the dispatcher looks up by model only)",
                entry.model
            ));
        }
        let provider = super::commands::provider_by_id_pub(app_state, &entry.provider_id)
            .ok_or_else(|| {
                format!(
                    "Auto mode: worker pool entry `{}` references missing provider id=`{}`. \
                     Fix Settings → Auto (remove the worker or re-add the provider).",
                    entry.model, entry.provider_id
                )
            })?;
        let mut cfg = provider_to_llm_config(&provider, &entry.model);
        // Workers are coding-mode agent loops that take many turns each; their
        // child loops opt in to cache_control on a per-message basis. We don't
        // pre-disable it here (let the per-provider LlmClient decide).
        cfg.disable_cache_control = false;
        map.insert(entry.model.clone(), cfg);
    }
    Ok(map)
}

/// Flush incremental `AutoRun` snapshots to the `auto_runs` table whenever
/// the executor mutates state (plan added, worker finished, replan, status
/// flip). Without this, the DB row stays at the initial `Planning + empty`
/// state until the run terminates — and a mid-flight reload (or the app
/// being killed) leaves the panel staring at a stale "Orchestrator is
/// planning…" forever.
///
/// **Ordering invariant**: notifies fire back-to-back (e.g. status=Planning
/// immediately followed by status=AwaitingApproval). Each `on_state` queues
/// a write onto a single-writer task via an unbounded mpsc, so writes land
/// in the exact order the executor produced them. Earlier fire-and-forget
/// spawning let later snapshots land *before* earlier ones — the older
/// state would clobber the newer one, the reaper would see stale
/// `planning` after restart, and the saved `awaiting_approval` got marked
/// failed on app start. mpsc serialization eliminates the race.
struct DbAutoStateListener {
    tx: tokio::sync::mpsc::UnboundedSender<AutoRun>,
}

impl DbAutoStateListener {
    fn new(db: Arc<super::db::AgentDb>) -> Self {
        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<AutoRun>();
        tokio::spawn(async move {
            while let Some(run) = rx.recv().await {
                let db = Arc::clone(&db);
                let run_id = run.id.clone();
                match tokio::task::spawn_blocking(move || db.update_auto_run(&run)).await {
                    Err(e) => log::warn!("[Auto-P7] DbAutoStateListener join error: {e}"),
                    Ok(Err(e)) => log::warn!(
                        "[Auto-P7] DbAutoStateListener db write failed for run={run_id}: {e}"
                    ),
                    Ok(Ok(())) => {}
                }
            }
        });
        Self { tx }
    }
}

impl AutoStateListener for DbAutoStateListener {
    fn on_state(&self, run: &AutoRun) {
        if let Err(e) = self.tx.send(run.clone()) {
            log::warn!("[Auto-P7] DbAutoStateListener queue closed: {e}");
        }
    }
}

/// Overwrite the placeholder assistant row with the run's terminal summary.
/// Best-effort: logged and swallowed on failure (the user-facing chat row
/// stays at "Auto run in progress…" but the live event stream still updates
/// the panel itself, so the loss is purely textual continuity for the next
/// LLM turn).
async fn update_auto_assistant_summary(
    db: Arc<super::db::AgentDb>,
    row_id: i64,
    summary: &str,
) {
    let llm = build_llm_message_json("assistant", summary);
    let preview: String = summary.chars().take(80).collect();
    let res = tokio::task::spawn_blocking(move || db.update_message_llm_content(row_id, &llm)).await;
    match res {
        Ok(Ok(())) => log::info!(
            "[Auto-P7] updated assistant row {} (len={}, preview={:?})",
            row_id, summary.len(), preview
        ),
        Ok(Err(e)) => log::warn!("[Auto-P7] update_message_llm_content failed row={row_id}: {e}"),
        Err(e) => log::warn!("[Auto-P7] update_message_llm_content join error row={row_id}: {e}"),
    }
}

/// Build the assistant-message text written when an Auto run is cancelled
/// mid-flight. Names the phase that was interrupted and lists the summaries
/// of any workers that completed successfully, so the chat thread keeps
/// enough context for the next LLM turn (Auto or normal) to pick up.
///
/// Format:
/// ```md
/// **Auto run stopped during {phase}.**
///
/// Completed workers (N):
/// - **w1** (`claude-haiku-4-5`): first 240 chars of summary…
/// - **w2** (`claude-opus-4-7`): ...
/// ```
///
/// When no worker completed successfully, only the header line is written.
fn build_cancelled_summary(run: &AutoRun, phase: &str) -> String {
    let mut out = format!("**Auto run stopped during {phase}.**\n");
    let completed: Vec<&agent::auto::WorkerResult> = run
        .worker_results
        .iter()
        .filter(|w| matches!(w.status, WorkerStatus::Ok))
        .collect();
    if !completed.is_empty() {
        use std::fmt::Write as _;
        let _ = write!(out, "\nCompleted workers ({}):\n", completed.len());
        for w in &completed {
            let excerpt: String = w.summary.chars().take(240).collect();
            let ellipsis = if w.summary.chars().count() > 240 { "…" } else { "" };
            let _ = write!(
                out,
                "- **{}** (`{}`): {}{}\n",
                w.id, w.model, excerpt, ellipsis
            );
        }
    }
    out
}

// ── Tauri commands: settings ─────────────────────────────────────────────

#[tauri::command]
pub async fn agent_get_auto_settings(
    app_state: State<'_, AppState>,
) -> Result<AutoSettings, String> {
    Ok(read_auto_settings(&app_state))
}

#[tauri::command]
pub async fn agent_set_auto_orchestrator_model(
    model: Option<ModelRef>,
    app_state: State<'_, AppState>,
) -> Result<(), String> {
    match model {
        Some(ref m) => {
            let raw = serde_json::to_string(m).map_err(|e| e.to_string())?;
            app_state.db.set_setting(AUTO_ORCHESTRATOR_KEY, &raw)
        }
        None => app_state.db.delete_setting(AUTO_ORCHESTRATOR_KEY),
    }
}

#[tauri::command]
pub async fn agent_set_auto_worker_pool(
    pool: Vec<WorkerPoolEntry>,
    app_state: State<'_, AppState>,
) -> Result<(), String> {
    let raw = serde_json::to_string(&pool).map_err(|e| e.to_string())?;
    app_state.db.set_setting(AUTO_WORKER_POOL_KEY, &raw)
}

/// Flip the master Auto-enable switch.
/// - On ENABLE: prerequisites must be met (orchestrator picked + non-empty
///   worker pool). The Settings UI also gates the toggle, but this is the
///   authoritative check.
/// - On DISABLE: no session may currently have Auto as its active model.
///   The Settings UI surfaces the offending sessions in a modal; the user
///   must switch them off Auto before they can disable the feature. This
///   matches the locked design's "block disable" decision.
#[tauri::command]
pub async fn agent_set_auto_enabled(
    enabled: bool,
    app_state: State<'_, AppState>,
    agent_state: State<'_, AgentState>,
) -> Result<(), String> {
    if enabled {
        // Prereqs: orchestrator picked + pool non-empty.
        if read_auto_orchestrator(&app_state).is_none() {
            return Err(
                "Cannot enable Auto: pick an orchestrator model in Settings → Auto first.".into(),
            );
        }
        if read_auto_worker_pool(&app_state).is_empty() {
            return Err(
                "Cannot enable Auto: add at least one model to the worker pool first.".into(),
            );
        }
    } else {
        // Block disable if any session still references Auto.
        let blockers = list_sessions_using_auto_impl(&agent_state).await?;
        if !blockers.is_empty() {
            return Err(format!(
                "Cannot disable Auto: {} session(s) still use it. Switch them to a regular model first.",
                blockers.len()
            ));
        }
    }
    app_state
        .db
        .set_setting(AUTO_ENABLED_KEY, if enabled { "true" } else { "false" })
}

/// List sessions whose active model is the Auto sentinel. Used by the
/// Settings UI to populate the "switch these first" modal when the user
/// tries to disable Auto. Excludes soft-deleted sessions.
#[tauri::command]
pub async fn agent_list_sessions_using_auto(
    agent_state: State<'_, AgentState>,
) -> Result<Vec<super::db::SessionRow>, String> {
    list_sessions_using_auto_impl(&agent_state).await
}

async fn list_sessions_using_auto_impl(
    agent_state: &AgentState,
) -> Result<Vec<super::db::SessionRow>, String> {
    let db = Arc::clone(&agent_state.db);
    let all = tokio::task::spawn_blocking(move || db.list_sessions())
        .await
        .map_err(|e| format!("join error: {e}"))?
        .map_err(|e| format!("Failed to list sessions: {e}"))?;
    Ok(all
        .into_iter()
        .filter(|s| match (s.provider_id.as_deref(), s.model.as_deref()) {
            (Some(p), Some(m)) => is_auto_mode(p, m),
            _ => false,
        })
        .collect())
}

// ── Plan approval ────────────────────────────────────────────────────────

/// Pending plan approvals keyed by `run_id`. Lives on `AgentState`; populated
/// by `TauriPlanApprover::approve` (waits) and drained by
/// `agent_approve_auto_plan` (resolves).
pub type AutoApprovalPending = Arc<Mutex<HashMap<String, oneshot::Sender<bool>>>>;

pub fn new_pending_map() -> AutoApprovalPending {
    Arc::new(Mutex::new(HashMap::new()))
}

/// Cancellation tokens for in-flight Auto runs, keyed by `session_id`. Only
/// one Auto run per session at a time (enforced by the folder lock), so the
/// session id is a sufficient key. `agent_cancel_session` looks here when
/// the stop button fires.
pub type AutoCancelRegistry = Arc<Mutex<HashMap<String, CancellationToken>>>;

pub fn new_cancel_registry() -> AutoCancelRegistry {
    Arc::new(Mutex::new(HashMap::new()))
}

/// `PlanApprover` impl that blocks on a `oneshot` channel keyed by `run_id`.
/// The frontend resolves the wait via `agent_approve_auto_plan`. Mirrors the
/// shape of `TauriApprovalHandler` in `events.rs`.
///
/// Note: the executor already emits `AutoAwaitingApproval` immediately before
/// calling `approve`, so this approver does NOT emit any event of its own —
/// it just waits.
pub struct TauriPlanApprover {
    run_id: String,
    pending: AutoApprovalPending,
}

impl TauriPlanApprover {
    pub fn new(run_id: String, pending: AutoApprovalPending) -> Self {
        Self { run_id, pending }
    }
}

#[async_trait]
impl PlanApprover for TauriPlanApprover {
    async fn approve(&self, _plan: &Plan) -> bool {
        let (tx, rx) = oneshot::channel();
        self.pending.lock().unwrap().insert(self.run_id.clone(), tx);
        // Wait. If the sender is dropped (e.g. the run was cancelled before
        // the user clicked anything), treat as a denial.
        rx.await.unwrap_or(false)
    }
}

/// Fetch a persisted `auto_runs` snapshot by `run_id`. The AutoRunPanel
/// calls this on mount to hydrate from disk when its in-memory state for
/// that run is empty (e.g. after a page reload, or when opening a session
/// that has prior Auto turns in its history).
#[tauri::command]
pub async fn agent_get_auto_run(
    run_id: String,
    agent_state: State<'_, AgentState>,
) -> Result<Option<AutoRun>, String> {
    let db = Arc::clone(&agent_state.db);
    let run_id_for_log = run_id.clone();
    let snapshot = tokio::task::spawn_blocking(move || db.get_auto_run(&run_id))
        .await
        .map_err(|e| format!("join error: {e}"))?
        .map_err(|e| format!("Failed to load auto_run: {e}"))?;
    log::info!(
        "[Auto-P7] agent_get_auto_run({}) -> {}",
        run_id_for_log,
        snapshot.as_ref().map(|s| format!("status={:?} workers={} plans={}",
            s.status, s.worker_results.len(), s.plan_versions.len())).unwrap_or_else(|| "None".into())
    );
    Ok(snapshot)
}

/// Resolve a pending plan approval. Called by the frontend after the user
/// clicks Run (`approved=true`) or Cancel (`approved=false`).
///
/// Two paths:
/// 1. **Live executor** — a `TauriPlanApprover` is parked on a oneshot
///    keyed by `run_id`. Resolving it unblocks the executor.
/// 2. **Resume after restart** — no live oneshot exists because the prior
///    executor task died (app restart, crash). The saved plan is on disk
///    in `auto_runs.plan_versions`. On `approved=true` we spawn a fresh
///    executor seeded with the saved snapshot (no orchestrator re-call).
///    On `approved=false` we just mark the snapshot as failed.
#[tauri::command]
pub async fn agent_approve_auto_plan(
    app_handle: AppHandle,
    run_id: String,
    approved: bool,
    app_state: State<'_, AppState>,
    agent_state: State<'_, AgentState>,
) -> Result<(), String> {
    let sender = agent_state.auto_plan_pending.lock().unwrap().remove(&run_id);
    if let Some(tx) = sender {
        let _ = tx.send(approved);
        return Ok(());
    }

    // No live oneshot → resume path.
    log::info!("[Auto-P7] approve fallback: no live oneshot for run={run_id}, treating as resume (approved={approved})");
    let snapshot = {
        let db = Arc::clone(&agent_state.db);
        let run_id_clone = run_id.clone();
        tokio::task::spawn_blocking(move || db.get_auto_run(&run_id_clone))
            .await
            .map_err(|e| format!("join error: {e}"))?
            .map_err(|e| format!("Failed to load auto_run: {e}"))?
            .ok_or_else(|| format!("No saved Auto run with id={run_id}"))?
    };
    if !matches!(snapshot.status, AutoStatus::AwaitingApproval) {
        return Err(format!(
            "Auto run id={run_id} is in status {:?}; resume is only allowed from awaiting_approval",
            snapshot.status
        ));
    }
    if !approved {
        // User clicked Cancel on a stale awaiting — just mark failed.
        let db = Arc::clone(&agent_state.db);
        let mut snap = snapshot;
        snap.status = AutoStatus::Failed;
        let _ = tokio::task::spawn_blocking(move || db.update_auto_run(&snap)).await;
        return Ok(());
    }

    // Spawn a fresh executor seeded with the saved snapshot.
    resume_auto_turn(app_handle, &app_state, &agent_state, snapshot, ResumeAction::ApproveSavedPlan).await
}

/// P9c: resolve a paused-for-failure Auto run with the user's chosen action.
/// The snapshot status must be `AwaitingFailureDecision`. Actions:
///   - `"retry"` → spawn a fresh executor that skips the orchestrator and
///     restarts the failed worker (via `ResumeAction::RetryWorker`).
///   - `"replan"` → spawn a fresh executor that calls the orchestrator with
///     the failure context as the replan reason
///     (via `ResumeAction::ReplanAfterFailure`).
///   - `"cancel"` → mark the run terminally Failed in the DB; no executor.
#[tauri::command]
pub async fn agent_resolve_worker_failure(
    app_handle: AppHandle,
    run_id: String,
    action: String,
    app_state: State<'_, AppState>,
    agent_state: State<'_, AgentState>,
) -> Result<(), String> {
    log::info!("[Auto-P7] agent_resolve_worker_failure run={run_id} action={action}");
    let snapshot = {
        let db = Arc::clone(&agent_state.db);
        let run_id_clone = run_id.clone();
        tokio::task::spawn_blocking(move || db.get_auto_run(&run_id_clone))
            .await
            .map_err(|e| format!("join error: {e}"))?
            .map_err(|e| format!("Failed to load auto_run: {e}"))?
            .ok_or_else(|| format!("No saved Auto run with id={run_id}"))?
    };
    if !matches!(snapshot.status, AutoStatus::AwaitingFailureDecision) {
        return Err(format!(
            "Auto run id={run_id} is in status {:?}; failure-resolve is only allowed from awaiting_failure_decision",
            snapshot.status
        ));
    }
    let failure = snapshot
        .pending_failure
        .as_ref()
        .ok_or_else(|| format!("Auto run id={run_id} is awaiting_failure_decision but has no pending_failure context"))?
        .clone();

    match action.as_str() {
        "cancel" => {
            // 1. Flip the DB row to terminally Failed and clear the pause context.
            let db = Arc::clone(&agent_state.db);
            let session_id = snapshot.session_id.clone();
            let reason = format!(
                "Cancelled by user after worker {} ({}) failed: {}",
                failure.worker_id, failure.worker_model, failure.error_message
            );
            let mut snap = snapshot;
            snap.status = AutoStatus::Failed;
            snap.pending_failure = None;
            let _ = tokio::task::spawn_blocking(move || db.update_auto_run(&snap)).await;

            // 2. Emit AutoFailed so the frontend's existing handler runs:
            // setAutoFailed flips the panel to the terminal red banner, and
            // the post-event snapshot refetch sees pending_failure=null and
            // doesn't re-hydrate back to amber. Without this emit, the panel
            // sits on the stale amber banner until the session is reloaded.
            if let Err(e) = app_handle.emit(
                "agent:auto_failed",
                serde_json::json!({
                    "thread_id": session_id,
                    "run_id": run_id,
                    "reason": reason,
                    "last_worker_output": serde_json::Value::Null,
                }),
            ) {
                log::warn!("[Auto-P9c] failed to emit agent:auto_failed on cancel: {e}");
            }
            Ok(())
        }
        "retry" => {
            resume_auto_turn(
                app_handle,
                &app_state,
                &agent_state,
                snapshot,
                ResumeAction::RetryWorker { worker_id: failure.worker_id },
            )
            .await
        }
        "replan" => {
            resume_auto_turn(
                app_handle,
                &app_state,
                &agent_state,
                snapshot,
                ResumeAction::ReplanAfterFailure { reason: failure.error_message },
            )
            .await
        }
        other => Err(format!("Unknown action `{other}`; expected one of: retry, replan, cancel")),
    }
}

// ── Spawn entry point ────────────────────────────────────────────────────

/// Dispatched from `run_agent_turn` when the session is in Auto mode.
/// Validates Auto settings, builds the executor pipeline, persists the
/// initial `AutoRun` row, and spawns the executor on a background task.
/// The task drains its events through the standard `spawn_event_relay` so
/// the existing frontend wiring (`agent:text_delta`, `agent:tool_start`,
/// and the new `agent:auto_*` events forwarded in P4) all flow uniformly.
///
/// Releases the folder lock on completion via the supplied `on_complete`
/// callback (mirrors the regular path's monitor task).
#[allow(clippy::too_many_arguments)]
pub async fn run_auto_turn(
    app_handle: AppHandle,
    app_state: &AppState,
    agent_state: &AgentState,
    session_id: String,
    folder: String,
    message: String,
    on_complete: impl FnOnce() + Send + 'static,
) -> Result<(), String> {
    log::info!(
        "[Auto-P7] run_auto_turn START session={} folder={} prompt_len={}",
        session_id, folder, message.len()
    );
    // ── 1. Validate settings ──
    let settings = read_auto_settings(app_state);
    if !settings.enabled {
        return Err(
            "Auto mode is disabled in Settings. Re-enable it, or pick a regular model.".into(),
        );
    }
    let Some(orchestrator_ref) = settings.orchestrator else {
        return Err("Auto mode: orchestrator model not configured in Settings → Auto".into());
    };
    if settings.worker_pool.is_empty() {
        return Err("Auto mode: worker pool is empty in Settings → Auto".into());
    }
    let orchestrator_provider = super::commands::provider_by_id_pub(app_state, &orchestrator_ref.provider_id)
        .ok_or_else(|| {
            format!(
                "Auto mode: orchestrator provider `{}` is not configured",
                orchestrator_ref.provider_id
            )
        })?;

    // ── 2. Build orchestrator + per-worker LLM configs ──
    let mut orchestrator_llm = provider_to_llm_config(&orchestrator_provider, &orchestrator_ref.model);
    // The orchestrator is a one-shot tool-use call — explicit cache_control
    // is meaningless for it and the auto::orchestrator IO layer doesn't
    // emit any cache markers anyway.
    orchestrator_llm.disable_cache_control = true;
    // Snapshot the worker pool to a model→LlmClientConfig map. Failures here
    // (missing provider, duplicate model) reject the whole run before any
    // LLM call. See `build_worker_llm_configs` for the rules.
    let worker_llm_configs = build_worker_llm_configs(app_state, &settings.worker_pool)?;
    log::info!(
        "[Auto-P7] built worker_llm_configs ({} models): {:?}",
        worker_llm_configs.len(),
        worker_llm_configs.keys().collect::<Vec<_>>()
    );

    // ── 2b. Context engine pass-through ──
    // When semantic search is ON in Settings, give every worker the
    // codebase_search + codebase_graph tools (same wiring the regular
    // run_agent_turn path does). Workers each spawn their own AgentLoop in
    // Coding mode, which auto-registers those tools when
    // `AgentConfig.context_engine` is set.
    let ce_settings = super::commands::read_context_engine(app_state);
    let ce_for_workers: Option<Arc<dyn agent::context_engine::ContextEngineApi>> =
        if ce_settings.enabled {
            let cfg = agent::context_engine::ContextEngineConfig {
                base_url: super::commands::normalize_base_url(&ce_settings.base_url),
                user_id: "local".to_string(),
                workspace_id: 0,
                machine_id: super::commands::machine_id(app_state),
                repo_path: folder.clone(),
                auth_token: String::new(),
            };
            Some(Arc::new(agent::context_engine::ContextEngineClient::new(cfg)))
        } else {
            None
        };
    let ce_repo_path: Option<PathBuf> =
        if ce_settings.enabled { Some(PathBuf::from(&folder)) } else { None };

    // Keep the index live for this repo while the Auto task runs. Idempotent
    // — bumps last_used_at + triggers a fresh full_sync + starts the fs
    // watcher if it wasn't already. Same pattern as run_agent_turn.
    if ce_settings.enabled {
        let wm = app_handle
            .state::<Arc<crate::context_watcher::WatcherManager>>()
            .inner()
            .clone();
        let folder_owned = folder.clone();
        tokio::spawn(async move {
            if let Err(e) = wm.start_watching(&folder_owned).await {
                log::warn!("[ContextWatcher] start_watching failed for {folder_owned}: {e}");
            }
        });
    }

    // ── 2c. Session title (parity with run_agent_turn) ──
    // First-turn Auto sessions used to keep the 60-char substring fallback
    // forever because dispatch skipped run_agent_turn's title block. Mirror
    // the same shape here so the sidebar upgrades to a clean 3–6 word title.
    {
        let session_row = {
            let db = Arc::clone(&agent_state.db);
            let sid = session_id.clone();
            tokio::task::spawn_blocking(move || db.get_session(&sid))
                .await
                .map_err(|e| format!("join error loading session: {e}"))?
                .map_err(|e| format!("failed to load session for title check: {e}"))?
        };
        let needs_title = session_row
            .as_ref()
            .and_then(|s| s.title.as_deref())
            .unwrap_or("")
            .is_empty();
        let fallback: String = message.chars().take(60).collect();
        {
            let db = Arc::clone(&agent_state.db);
            let sid = session_id.clone();
            let fb = fallback.clone();
            let _ = tokio::task::spawn_blocking(move || {
                let _ = db.set_session_status(&sid, "active");
                if needs_title && !fb.trim().is_empty() {
                    let _ = db.set_session_title(&sid, fb.trim());
                }
            })
            .await;
        }
        if needs_title && !message.trim().is_empty() {
            // Title model: user's configured title model if any, else fall
            // back to the orchestrator's real provider+model. Never the
            // Auto sentinel — spawn_title_generation needs a real endpoint.
            let (title_provider, title_model) = super::commands::read_selection(app_state)
                .title
                .and_then(|r| super::commands::resolve_ref(app_state, &r))
                .unwrap_or_else(|| {
                    (orchestrator_provider.clone(), orchestrator_ref.model.clone())
                });
            super::commands::spawn_title_generation(
                app_handle.clone(),
                Arc::clone(&agent_state.db),
                session_id.clone(),
                message.clone(),
                title_provider,
                title_model,
            );
        }
    }

    // ── 3. Build event channel + relay ──
    // The relay runs in parallel with the executor; both end-of-Auto event
    // (AutoDone / AutoFailed) and ToolEnd-from-workers are forwarded to the
    // frontend via the existing pipeline.
    let (event_tx, event_rx) = mpsc::channel::<agent::types::AgentEvent>(256);
    let emitter: Arc<dyn EventEmitter> = Arc::new(TauriEventEmitter::new(app_handle.clone()));
    let relay_handle = spawn_event_relay(
        emitter,
        event_rx,
        session_id.clone(),
        Some(Arc::clone(&agent_state.db)),
        folder.clone(),
        None, // Auto mode: no per-turn checkpoint capture for v1
    );

    // ── 4. Build WorkerContext ──
    let work_dir = PathBuf::from(&folder);
    let cancel = CancellationToken::new();
    let auto_run_id = Uuid::new_v4().to_string();

    let worker_ctx = WorkerContext {
        session_id: session_id.clone(),
        run_id: auto_run_id.clone(),
        working_dir: work_dir.clone(),
        event_tx: event_tx.clone(),
        cancel_token: cancel.clone(),
        worker_llm_configs,
        retry_config: RetryConfig::auto_worker(),
        compaction_config: CompactionConfig::default(),
        compaction_llm: None,
        // Worker max_iter matches AgentConfig default per PHASE_AUTO_MODE.md.
        // 100 is the same ceiling a normal agent run gets.
        max_iterations: 100,
        context_engine: ce_for_workers,
        context_engine_repo_path: ce_repo_path,
        persister: None,
        approval_handler: None,
        checkpoint_dir: None,
    };

    // ── 5. Build executor pipeline ──
    let plan_gen = Arc::new(LlmPlanGenerator { llm: orchestrator_llm });
    let runner = Arc::new(LiveWorkerRunner { ctx: worker_ctx });
    let approver = Arc::new(TauriPlanApprover::new(
        auto_run_id.clone(),
        Arc::clone(&agent_state.auto_plan_pending),
    ));

    let exec_config = ExecutorConfig {
        task: message.clone(),
        auto_run_id: auto_run_id.clone(),
        session_id: session_id.clone(),
        worker_pool: settings.worker_pool,
        replan_budget: AUTO_REPLAN_BUDGET,
        resume_from: None,
        start_at_worker_id: None,
        initial_replan_reason: None,
    };

    // ── 6. Persist initial AutoRun row + agent_messages anchor pair ──
    // The auto_runs row is the rich snapshot the AutoRunPanel hydrates from.
    // The two agent_messages rows are the chat-thread anchors: the user row
    // shows the prompt, and the placeholder assistant row gets overwritten
    // with the terminal summary (so the next turn's LLM context sees a plain
    // continuation, not a hole in history).
    let initial_run = AutoRun {
        id: auto_run_id.clone(),
        session_id: session_id.clone(),
        status: AutoStatus::Planning,
        plan_versions: Vec::new(),
        worker_results: Vec::new(),
        cost_cents: 0,
        replans_used: 0,
        current_worker: None,
        pending_failure: None,
    };
    let db = Arc::clone(&agent_state.db);
    let initial_run_for_db = initial_run.clone();
    tokio::task::spawn_blocking(move || db.insert_auto_run(&initial_run_for_db))
        .await
        .map_err(|e| format!("join error: {e}"))?
        .map_err(|e| format!("Failed to insert auto_run: {e}"))?;

    let assistant_row_id = persist_auto_anchor_rows(
        Arc::clone(&agent_state.db),
        session_id.clone(),
        folder.clone(),
        message.clone(),
        auto_run_id.clone(),
    )
    .await?;

    // Register the cancellation token so `agent_cancel_session` can trigger
    // it when the user hits the stop button. Removed in the terminal handler
    // below regardless of how the run ends.
    agent_state
        .auto_cancellation
        .lock()
        .unwrap()
        .insert(session_id.clone(), cancel.clone());
    log::info!("[Auto-P7] registered cancel token for session={}", session_id);

    // ── 7. Spawn the executor task ──
    // Keep a clone of event_tx for the terminal handler — `run_auto` consumes
    // its own copy, so we need this clone to emit the synthetic AutoFailed
    // event on user cancellation (executor::finalize_cancelled is silent by
    // design — terminal cancel signalling is a Tauri-side concern).
    let event_tx_for_cancel = event_tx.clone();
    let db_for_task = Arc::clone(&agent_state.db);
    let pending_for_cancel = Arc::clone(&agent_state.auto_plan_pending);
    let cancel_registry = Arc::clone(&agent_state.auto_cancellation);
    let auto_run_id_for_task = auto_run_id.clone();
    let session_id_for_task = session_id.clone();
    // Persist intermediate snapshots so reload during a long-running Auto
    // turn shows the actual progress (plan, completed workers) instead of
    // the initial Planning state.
    let state_listener: Option<Arc<dyn AutoStateListener>> = Some(Arc::new(DbAutoStateListener::new(
        Arc::clone(&agent_state.db),
    )));

    tokio::spawn(async move {
        let result = auto::run_auto(
            exec_config,
            event_tx,
            plan_gen,
            runner,
            approver,
            cancel,
            state_listener,
        )
        .await;

        log::info!(
            "[Auto-P7] executor returned: run={} variant={}",
            auto_run_id_for_task,
            match &result {
                AutoResult::Done { .. } => "Done",
                AutoResult::Failed { .. } => "Failed",
                AutoResult::Cancelled { .. } => "Cancelled",
            }
        );

        // Persist terminal state.
        let final_run = match &result {
            AutoResult::Done { run, .. }
            | AutoResult::Failed { run, .. }
            | AutoResult::Cancelled { run, .. } => run.clone(),
        };
        let _ = tokio::task::spawn_blocking({
            let db_for_run = Arc::clone(&db_for_task);
            let final_run = final_run.clone();
            move || db_for_run.update_auto_run(&final_run)
        })
        .await;

        // Overwrite the placeholder assistant row with the terminal summary
        // so the next LLM turn (Auto or normal) sees the run as a normal
        // chat exchange instead of "Auto run in progress…".
        let summary_text = match &result {
            AutoResult::Done { summary, .. } => summary.clone(),
            AutoResult::Failed { reason, .. } => format!("Auto run failed: {reason}"),
            AutoResult::Cancelled { run, phase } => build_cancelled_summary(run, phase),
        };
        update_auto_assistant_summary(
            Arc::clone(&db_for_task),
            assistant_row_id,
            &summary_text,
        )
        .await;

        // User-cancelled runs don't emit a terminal event from the executor.
        // Synthesize an AutoFailed so the frontend's existing handler fires
        // (stops the thinking placeholder, marks the panel terminal, clears
        // approvals) — same UX as a regular session interrupt.
        if matches!(&result, AutoResult::Cancelled { .. }) {
            let _ = event_tx_for_cancel
                .send(agent::types::AgentEvent::AutoFailed {
                    session_id: session_id_for_task.clone(),
                    run_id: auto_run_id_for_task.clone(),
                    reason: "Interrupted by user".to_string(),
                    last_worker_output: None,
                })
                .await;
        }
        // Drop our clone so the relay's event_rx closes once executor's tx
        // is also dropped — relay terminates cleanly.
        drop(event_tx_for_cancel);

        // Clean up any leftover pending approval (defensive — the executor
        // shouldn't terminate while still holding one, but a cancelled run
        // mid-approval would).
        pending_for_cancel
            .lock()
            .unwrap()
            .remove(&auto_run_id_for_task);
        // Drop the cancellation token registration.
        cancel_registry
            .lock()
            .unwrap()
            .remove(&session_id_for_task);

        // Await the relay so the final AutoDone/AutoFailed event has been
        // emitted to the frontend before we touch session state and release
        // the folder lock — matches the ordering the regular agent path uses.
        let _ = relay_handle.await;

        // Flip session status idle so the sidebar's "running" badge clears.
        // Regular run_agent_turn does this in its monitor task right after
        // relay_handle.await; we mirror that here.
        let _ = tokio::task::spawn_blocking({
            let db_for_status = Arc::clone(&db_for_task);
            let sid = session_id_for_task.clone();
            move || db_for_status.set_session_status(&sid, "idle")
        })
        .await;

        on_complete();
    });

    Ok(())
}

/// Spawn a fresh executor that picks up where a saved `auto_runs` snapshot
/// left off. Used by `agent_approve_auto_plan`'s fallback path when a user
/// clicks Run on an `awaiting_approval` panel after the original executor
/// task is gone (app restart, crash). The saved plan is reused; the
/// orchestrator is NOT called on the first iteration. Replans inside the
/// resumed run go through the orchestrator normally.
///
/// Duplicates most of `run_auto_turn`'s body for now — both should be
/// folded into a shared `spawn_auto_executor` helper in a follow-up.
/// What kind of resume this is — drives how the executor's first iteration
/// behaves. `ApproveSavedPlan` is the original P7 awaiting_approval resume;
/// the two `Retry…` and `Replan…` variants are P9c's user-driven worker
/// failure recovery.
#[derive(Debug, Clone)]
pub(crate) enum ResumeAction {
    /// User clicked Run on the saved awaiting_approval plan. Executor uses
    /// `plan_versions.last()` as-is and runs from worker[0] forward.
    ApproveSavedPlan,
    /// User clicked Retry on the awaiting_failure_decision banner. Executor
    /// uses the saved plan; the worker loop skips ahead to `worker_id`
    /// instead of starting at index 0.
    RetryWorker { worker_id: String },
    /// User clicked Replan on the awaiting_failure_decision banner. Executor
    /// invokes the orchestrator on iteration 1 with `reason` baked into the
    /// replan prompt (saved plan ignored, prior worker results preserved).
    ReplanAfterFailure { reason: String },
}

async fn resume_auto_turn(
    app_handle: AppHandle,
    app_state: &AppState,
    agent_state: &AgentState,
    snapshot: AutoRun,
    action: ResumeAction,
) -> Result<(), String> {
    let session_id = snapshot.session_id.clone();
    let auto_run_id = snapshot.id.clone();
    log::info!(
        "[Auto-P7] resume_auto_turn START session={} run={} plans={} workers={} action={:?}",
        session_id, auto_run_id, snapshot.plan_versions.len(), snapshot.worker_results.len(), action
    );
    // Translate the high-level ResumeAction into the executor's lower-level
    // start_at_worker_id / initial_replan_reason fields.
    let (resume_start_at_worker_id, resume_initial_replan_reason) = match &action {
        ResumeAction::ApproveSavedPlan => (None, None),
        ResumeAction::RetryWorker { worker_id } => (Some(worker_id.clone()), None),
        ResumeAction::ReplanAfterFailure { reason } => (None, Some(reason.clone())),
    };

    // ── 1. Look up session (folder) + chat anchors ──
    let db = Arc::clone(&agent_state.db);
    let session_row = {
        let sid = session_id.clone();
        tokio::task::spawn_blocking(move || db.get_session(&sid))
            .await
            .map_err(|e| format!("join error: {e}"))?
            .map_err(|e| format!("Failed to load session: {e}"))?
            .ok_or_else(|| format!("Session {session_id} not found"))?
    };
    let folder = session_row.folder.clone();
    let db = Arc::clone(&agent_state.db);
    let assistant_row_id = {
        let run_id_clone = auto_run_id.clone();
        tokio::task::spawn_blocking(move || db.find_auto_assistant_row_id(&run_id_clone))
            .await
            .map_err(|e| format!("join error: {e}"))?
            .map_err(|e| format!("Failed to look up assistant row: {e}"))?
            .ok_or_else(|| format!("No assistant anchor row for run_id={auto_run_id}"))?
    };
    let db = Arc::clone(&agent_state.db);
    let task = {
        let run_id_clone = auto_run_id.clone();
        tokio::task::spawn_blocking(move || db.find_auto_user_prompt(&run_id_clone))
            .await
            .map_err(|e| format!("join error: {e}"))?
            .map_err(|e| format!("Failed to look up user prompt: {e}"))?
            .ok_or_else(|| format!("No user prompt found for run_id={auto_run_id}"))?
    };

    // ── 2. Validate Auto settings still configured ──
    let settings = read_auto_settings(app_state);
    let Some(orchestrator_ref) = settings.orchestrator else {
        return Err("Resume: orchestrator model not configured in Settings → Auto".into());
    };
    if settings.worker_pool.is_empty() {
        return Err("Resume: worker pool is empty in Settings → Auto".into());
    }
    let orchestrator_provider = super::commands::provider_by_id_pub(app_state, &orchestrator_ref.provider_id)
        .ok_or_else(|| format!("Resume: orchestrator provider `{}` is not configured", orchestrator_ref.provider_id))?;

    // ── 3. Build LLM + context engine + worker context (same as run_auto_turn) ──
    let mut orchestrator_llm = provider_to_llm_config(&orchestrator_provider, &orchestrator_ref.model);
    orchestrator_llm.disable_cache_control = true;
    let worker_llm_configs = build_worker_llm_configs(app_state, &settings.worker_pool)?;
    log::info!(
        "[Auto-P7] resume built worker_llm_configs ({} models): {:?}",
        worker_llm_configs.len(),
        worker_llm_configs.keys().collect::<Vec<_>>()
    );

    let ce_settings = super::commands::read_context_engine(app_state);
    let ce_for_workers: Option<Arc<dyn agent::context_engine::ContextEngineApi>> =
        if ce_settings.enabled {
            let cfg = agent::context_engine::ContextEngineConfig {
                base_url: super::commands::normalize_base_url(&ce_settings.base_url),
                user_id: "local".to_string(),
                workspace_id: 0,
                machine_id: super::commands::machine_id(app_state),
                repo_path: folder.clone(),
                auth_token: String::new(),
            };
            Some(Arc::new(agent::context_engine::ContextEngineClient::new(cfg)))
        } else { None };
    let ce_repo_path: Option<PathBuf> =
        if ce_settings.enabled { Some(PathBuf::from(&folder)) } else { None };
    if ce_settings.enabled {
        let wm = app_handle
            .state::<Arc<crate::context_watcher::WatcherManager>>()
            .inner()
            .clone();
        let folder_owned = folder.clone();
        tokio::spawn(async move {
            if let Err(e) = wm.start_watching(&folder_owned).await {
                log::warn!("[ContextWatcher] start_watching failed for {folder_owned}: {e}");
            }
        });
    }

    // ── 4. Folder lock (one active session per folder) ──
    {
        let mut running = agent_state.running_folders.write().await;
        if running.contains(&folder) {
            return Err(format!("A session is already running for {folder}"));
        }
        running.insert(folder.clone());
    }
    let running_for_release = Arc::clone(&agent_state.running_folders);
    let folder_for_release = folder.clone();
    let release_folder = move || {
        let running = running_for_release;
        let folder = folder_for_release;
        tokio::spawn(async move {
            running.write().await.remove(&folder);
        });
    };

    // ── 5. Event channel + relay ──
    let (event_tx, event_rx) = mpsc::channel::<agent::types::AgentEvent>(256);
    let emitter: Arc<dyn EventEmitter> = Arc::new(TauriEventEmitter::new(app_handle.clone()));
    let relay_handle = spawn_event_relay(
        emitter,
        event_rx,
        session_id.clone(),
        Some(Arc::clone(&agent_state.db)),
        folder.clone(),
        None,
    );

    // ── 6. Worker context + pipeline ──
    let work_dir = PathBuf::from(&folder);
    let cancel = CancellationToken::new();

    let worker_ctx = WorkerContext {
        session_id: session_id.clone(),
        run_id: auto_run_id.clone(),
        working_dir: work_dir.clone(),
        event_tx: event_tx.clone(),
        cancel_token: cancel.clone(),
        worker_llm_configs,
        retry_config: RetryConfig::auto_worker(),
        compaction_config: CompactionConfig::default(),
        compaction_llm: None,
        max_iterations: 100,
        context_engine: ce_for_workers,
        context_engine_repo_path: ce_repo_path,
        persister: None,
        approval_handler: None,
        checkpoint_dir: None,
    };

    let plan_gen = Arc::new(LlmPlanGenerator { llm: orchestrator_llm });
    let runner = Arc::new(LiveWorkerRunner { ctx: worker_ctx });
    // Resume = user already approved by clicking Run on the panel.
    let approver: Arc<dyn PlanApprover> = Arc::new(agent::auto::AutoApprove);

    let exec_config = ExecutorConfig {
        task,
        auto_run_id: auto_run_id.clone(),
        session_id: session_id.clone(),
        worker_pool: settings.worker_pool,
        replan_budget: AUTO_REPLAN_BUDGET,
        resume_from: Some(snapshot),
        // Default resume = AwaitingApproval flow (user clicked Run on the
        // saved plan). P9c overrides these when resuming after a worker
        // failure: RetryWorker sets start_at_worker_id, ReplanAfterFailure
        // sets initial_replan_reason. See `resume_auto_turn`'s callers.
        start_at_worker_id: resume_start_at_worker_id,
        initial_replan_reason: resume_initial_replan_reason,
    };

    // Register cancel token + state listener.
    agent_state
        .auto_cancellation
        .lock()
        .unwrap()
        .insert(session_id.clone(), cancel.clone());
    let state_listener: Option<Arc<dyn AutoStateListener>> = Some(Arc::new(DbAutoStateListener::new(
        Arc::clone(&agent_state.db),
    )));

    // ── 7. Spawn executor (terminal handler mirrors run_auto_turn) ──
    let event_tx_for_cancel = event_tx.clone();
    let db_for_task = Arc::clone(&agent_state.db);
    let pending_for_cancel = Arc::clone(&agent_state.auto_plan_pending);
    let cancel_registry = Arc::clone(&agent_state.auto_cancellation);
    let auto_run_id_for_task = auto_run_id.clone();
    let session_id_for_task = session_id.clone();

    tokio::spawn(async move {
        let result = auto::run_auto(
            exec_config, event_tx, plan_gen, runner, approver, cancel, state_listener,
        )
        .await;
        log::info!(
            "[Auto-P7] resume executor returned: run={} variant={}",
            auto_run_id_for_task,
            match &result {
                AutoResult::Done { .. } => "Done",
                AutoResult::Failed { .. } => "Failed",
                AutoResult::Cancelled { .. } => "Cancelled",
            }
        );

        let final_run = match &result {
            AutoResult::Done { run, .. }
            | AutoResult::Failed { run, .. }
            | AutoResult::Cancelled { run, .. } => run.clone(),
        };
        let _ = tokio::task::spawn_blocking({
            let db = Arc::clone(&db_for_task);
            let r = final_run.clone();
            move || db.update_auto_run(&r)
        })
        .await;

        let summary_text = match &result {
            AutoResult::Done { summary, .. } => summary.clone(),
            AutoResult::Failed { reason, .. } => format!("Auto run failed: {reason}"),
            AutoResult::Cancelled { run, phase } => build_cancelled_summary(run, phase),
        };
        update_auto_assistant_summary(
            Arc::clone(&db_for_task),
            assistant_row_id,
            &summary_text,
        )
        .await;

        if matches!(&result, AutoResult::Cancelled { .. }) {
            let _ = event_tx_for_cancel
                .send(agent::types::AgentEvent::AutoFailed {
                    session_id: session_id_for_task.clone(),
                    run_id: auto_run_id_for_task.clone(),
                    reason: "Interrupted by user".to_string(),
                    last_worker_output: None,
                })
                .await;
        }
        drop(event_tx_for_cancel);

        pending_for_cancel.lock().unwrap().remove(&auto_run_id_for_task);
        cancel_registry.lock().unwrap().remove(&session_id_for_task);

        let _ = relay_handle.await;

        let _ = tokio::task::spawn_blocking({
            let db = Arc::clone(&db_for_task);
            let sid = session_id_for_task.clone();
            move || db.set_session_status(&sid, "idle")
        })
        .await;

        release_folder();
    });

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use parking_lot::Mutex as PlMutex;
    use rusqlite::Connection;

    /// In-memory `Database` with the settings table pre-created. Lets us
    /// exercise the settings k-v helpers without touching the real app data
    /// dir or going through Tauri's `State` wrapper.
    fn mk_app_state() -> AppState {
        let conn = Connection::open_in_memory().unwrap();
        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS settings (
                key        TEXT PRIMARY KEY,
                value      TEXT NOT NULL,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP
            );",
        )
        .unwrap();
        AppState {
            db: Arc::new(crate::Database { conn: PlMutex::new(conn) }),
        }
    }

    // ── is_auto_mode ──

    #[test]
    fn is_auto_mode_recognizes_only_the_sentinel_pair() {
        assert!(is_auto_mode(AUTO_SENTINEL_PROVIDER_ID, AUTO_SENTINEL_MODEL));
        assert!(!is_auto_mode("auto", "something-else"));
        assert!(!is_auto_mode("anthropic", "auto"));
        assert!(!is_auto_mode("anthropic", "claude-sonnet-4-6"));
        assert!(!is_auto_mode("", ""));
    }

    // ── defaults ──

    #[test]
    fn read_auto_settings_returns_documented_defaults_when_unset() {
        let state = mk_app_state();
        let s = read_auto_settings(&state);
        assert!(s.orchestrator.is_none());
        assert!(s.worker_pool.is_empty());
    }

    // ── orchestrator round-trip ──

    #[test]
    fn orchestrator_model_writes_then_reads_unchanged() {
        let state = mk_app_state();
        let model = ModelRef {
            provider_id: "anthropic".into(),
            model: "claude-sonnet-4-6".into(),
        };
        let raw = serde_json::to_string(&model).unwrap();
        state.db.set_setting(AUTO_ORCHESTRATOR_KEY, &raw).unwrap();

        let loaded = read_auto_orchestrator(&state).expect("should round-trip");
        assert_eq!(loaded.provider_id, "anthropic");
        assert_eq!(loaded.model, "claude-sonnet-4-6");
    }

    #[test]
    fn orchestrator_returns_none_when_value_is_corrupt() {
        let state = mk_app_state();
        state
            .db
            .set_setting(AUTO_ORCHESTRATOR_KEY, "not valid json")
            .unwrap();
        assert!(read_auto_orchestrator(&state).is_none());
    }

    // ── worker pool round-trip ──

    #[test]
    fn worker_pool_writes_then_reads_unchanged() {
        let state = mk_app_state();
        let pool = vec![
            WorkerPoolEntry {
                provider_id: "anthropic".into(),
                model: "claude-haiku-4-5".into(),
                description: "cheap exploration".into(),
            },
            WorkerPoolEntry {
                provider_id: "anthropic".into(),
                model: "claude-sonnet-4-6".into(),
                description: "design + synthesis".into(),
            },
        ];
        let raw = serde_json::to_string(&pool).unwrap();
        state.db.set_setting(AUTO_WORKER_POOL_KEY, &raw).unwrap();

        let loaded = read_auto_worker_pool(&state);
        assert_eq!(loaded.len(), 2);
        assert_eq!(loaded[0].model, "claude-haiku-4-5");
        assert_eq!(loaded[0].description, "cheap exploration");
        assert_eq!(loaded[1].model, "claude-sonnet-4-6");
    }

    #[test]
    fn worker_pool_defaults_to_empty_when_corrupt() {
        let state = mk_app_state();
        state.db.set_setting(AUTO_WORKER_POOL_KEY, "{broken").unwrap();
        assert!(read_auto_worker_pool(&state).is_empty());
    }

    // ── composite read_auto_settings ──

    #[test]
    fn read_auto_settings_assembles_all_keys() {
        let state = mk_app_state();
        let model = ModelRef {
            provider_id: "anthropic".into(),
            model: "claude-sonnet-4-6".into(),
        };
        state
            .db
            .set_setting(AUTO_ORCHESTRATOR_KEY, &serde_json::to_string(&model).unwrap())
            .unwrap();
        let pool = vec![WorkerPoolEntry {
            provider_id: "anthropic".into(),
            model: "claude-haiku-4-5".into(),
            description: "cheap".into(),
        }];
        state
            .db
            .set_setting(AUTO_WORKER_POOL_KEY, &serde_json::to_string(&pool).unwrap())
            .unwrap();
        state.db.set_setting(AUTO_ENABLED_KEY, "true").unwrap();

        let s = read_auto_settings(&state);
        assert!(s.enabled);
        assert!(s.orchestrator.is_some());
        assert_eq!(s.orchestrator.unwrap().model, "claude-sonnet-4-6");
        assert_eq!(s.worker_pool.len(), 1);
    }

    // ── enabled flag ──

    #[test]
    fn enabled_defaults_false_when_unset() {
        let state = mk_app_state();
        assert!(!read_auto_enabled(&state));
        assert!(!read_auto_settings(&state).enabled);
    }

    #[test]
    fn enabled_roundtrips_both_directions() {
        let state = mk_app_state();
        state.db.set_setting(AUTO_ENABLED_KEY, "true").unwrap();
        assert!(read_auto_enabled(&state));
        state.db.set_setting(AUTO_ENABLED_KEY, "false").unwrap();
        assert!(!read_auto_enabled(&state));
    }

    #[test]
    fn enabled_defaults_false_when_value_is_garbage() {
        // Anything other than "true" → false. No partial-truth like "yes".
        let state = mk_app_state();
        state.db.set_setting(AUTO_ENABLED_KEY, "yes").unwrap();
        assert!(!read_auto_enabled(&state));
        state.db.set_setting(AUTO_ENABLED_KEY, "1").unwrap();
        assert!(!read_auto_enabled(&state));
    }
}
