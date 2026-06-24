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
//! v1 limitation: all workers in the pool are dispatched through the
//! orchestrator's provider config (model name is overridden per worker, but
//! base_url + api_key + provider come from the orchestrator). Cross-provider
//! worker pools will work only if the orchestrator's endpoint accepts every
//! model name; otherwise the failing worker triggers a reactive replan. True
//! multi-provider dispatch is deferred to a future phase.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use tauri::{AppHandle, State};
use tokio::sync::{mpsc, oneshot};
use tokio_util::sync::CancellationToken;
use uuid::Uuid;

use agent::agent::config::{CompactionConfig, RetryConfig};
use agent::auto::{
    self, AutoResult, AutoRun, AutoStatus, ExecutorConfig, LiveWorkerRunner, LlmPlanGenerator,
    Plan, PlanApprover, WorkerContext, WorkerPoolEntry,
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
/// `llm_providers`, `context_engine`. Values are JSON strings.
const AUTO_ORCHESTRATOR_KEY: &str = "auto_orchestrator_model";
const AUTO_WORKER_POOL_KEY: &str = "auto_worker_pool";
const AUTO_REPLAN_BUDGET_KEY: &str = "auto_replan_budget";

/// Default replan budget when the user has not configured it.
/// Matches PHASE_AUTO_MODE.md "Replan budget" decision.
pub const DEFAULT_REPLAN_BUDGET: u32 = 2;

/// Returns true iff `(provider_id, model)` is the Auto sentinel.
pub fn is_auto_mode(provider_id: &str, model: &str) -> bool {
    provider_id == AUTO_SENTINEL_PROVIDER_ID && model == AUTO_SENTINEL_MODEL
}

// ── Settings DTO + persistence ───────────────────────────────────────────

/// All three Auto settings in a single struct, returned by
/// `agent_get_auto_settings` for the Settings UI to consume.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct AutoSettings {
    /// User-picked orchestrator model. `None` when Auto mode is not yet
    /// configured — `run_auto_turn` rejects with a "not configured" error
    /// in that case.
    pub orchestrator: Option<ModelRef>,
    /// User-curated pool of worker models with per-entry descriptions.
    #[serde(default)]
    pub worker_pool: Vec<WorkerPoolEntry>,
    /// Max reactive replans per task; total orchestrator calls = 1 + budget.
    pub replan_budget: u32,
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

pub(crate) fn read_auto_replan_budget(app_state: &AppState) -> u32 {
    app_state
        .db
        .get_setting(AUTO_REPLAN_BUDGET_KEY)
        .ok()
        .flatten()
        .and_then(|raw| raw.parse::<u32>().ok())
        .unwrap_or(DEFAULT_REPLAN_BUDGET)
}

pub(crate) fn read_auto_settings(app_state: &AppState) -> AutoSettings {
    AutoSettings {
        orchestrator: read_auto_orchestrator(app_state),
        worker_pool: read_auto_worker_pool(app_state),
        replan_budget: read_auto_replan_budget(app_state),
    }
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

#[tauri::command]
pub async fn agent_set_auto_replan_budget(
    budget: u32,
    app_state: State<'_, AppState>,
) -> Result<(), String> {
    // Per PHASE_AUTO_MODE.md the Settings slider exposes range 0–5.
    if budget > 5 {
        return Err(format!(
            "replan budget {budget} exceeds max (5); see PHASE_AUTO_MODE.md"
        ));
    }
    app_state.db.set_setting(AUTO_REPLAN_BUDGET_KEY, &budget.to_string())
}

// ── Plan approval ────────────────────────────────────────────────────────

/// Pending plan approvals keyed by `run_id`. Lives on `AgentState`; populated
/// by `TauriPlanApprover::approve` (waits) and drained by
/// `agent_approve_auto_plan` (resolves).
pub type AutoApprovalPending = Arc<Mutex<HashMap<String, oneshot::Sender<bool>>>>;

pub fn new_pending_map() -> AutoApprovalPending {
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

/// Resolve a pending plan approval. Called by the frontend after the user
/// clicks Run (`approved=true`) or Cancel (`approved=false`).
#[tauri::command]
pub async fn agent_approve_auto_plan(
    run_id: String,
    approved: bool,
    agent_state: State<'_, AgentState>,
) -> Result<(), String> {
    let sender = agent_state.auto_plan_pending.lock().unwrap().remove(&run_id);
    match sender {
        Some(tx) => {
            let _ = tx.send(approved);
            Ok(())
        }
        None => Err(format!(
            "No pending Auto-mode approval for run_id={run_id} (already resolved or expired)"
        )),
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
    // ── 1. Validate settings ──
    let settings = read_auto_settings(app_state);
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

    // ── 2. Build orchestrator + worker LLM configs ──
    let mut orchestrator_llm = provider_to_llm_config(&orchestrator_provider, &orchestrator_ref.model);
    // The orchestrator is a one-shot tool-use call — explicit cache_control
    // is meaningless for it and the auto::orchestrator IO layer doesn't
    // emit any cache markers anyway.
    orchestrator_llm.disable_cache_control = true;
    // v1 limitation noted at the top of this file: workers all use the
    // orchestrator's provider config; only the model field is overridden.
    let worker_llm_base = orchestrator_llm.clone();

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
        llm_base_config: worker_llm_base,
        retry_config: RetryConfig::default(),
        compaction_config: CompactionConfig::default(),
        compaction_llm: None,
        // Worker max_iter matches AgentConfig default per PHASE_AUTO_MODE.md.
        // 100 is the same ceiling a normal agent run gets.
        max_iterations: 100,
        context_engine: None,
        context_engine_repo_path: None,
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
        task: message,
        auto_run_id: auto_run_id.clone(),
        session_id: session_id.clone(),
        worker_pool: settings.worker_pool,
        replan_budget: settings.replan_budget,
    };

    // ── 6. Persist initial AutoRun row ──
    let initial_run = AutoRun {
        id: auto_run_id.clone(),
        session_id: session_id.clone(),
        status: AutoStatus::Planning,
        plan_versions: Vec::new(),
        worker_results: Vec::new(),
        cost_cents: 0,
        replans_used: 0,
    };
    let db = Arc::clone(&agent_state.db);
    let initial_run_for_db = initial_run.clone();
    tokio::task::spawn_blocking(move || db.insert_auto_run(&initial_run_for_db))
        .await
        .map_err(|e| format!("join error: {e}"))?
        .map_err(|e| format!("Failed to insert auto_run: {e}"))?;

    // ── 7. Spawn the executor task ──
    let db_for_task = Arc::clone(&agent_state.db);
    let pending_for_cancel = Arc::clone(&agent_state.auto_plan_pending);
    let auto_run_id_for_task = auto_run_id.clone();
    tokio::spawn(async move {
        let result = auto::run_auto(exec_config, event_tx, plan_gen, runner, approver, cancel).await;

        // Persist terminal state.
        let final_run = match &result {
            AutoResult::Done { run, .. }
            | AutoResult::Failed { run, .. }
            | AutoResult::Cancelled { run } => run.clone(),
        };
        let _ = tokio::task::spawn_blocking({
            let final_run = final_run.clone();
            move || db_for_task.update_auto_run(&final_run)
        })
        .await;

        // Clean up any leftover pending approval (defensive — the executor
        // shouldn't terminate while still holding one, but a cancelled run
        // mid-approval would).
        pending_for_cancel
            .lock()
            .unwrap()
            .remove(&auto_run_id_for_task);

        // Await the relay so the final AutoDone/AutoFailed event has been
        // emitted to the frontend before we release the folder lock.
        let _ = relay_handle.await;
        on_complete();
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
        assert_eq!(s.replan_budget, DEFAULT_REPLAN_BUDGET);
        assert_eq!(DEFAULT_REPLAN_BUDGET, 2, "default budget pinned by PHASE_AUTO_MODE.md");
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

    // ── replan budget round-trip ──

    #[test]
    fn replan_budget_writes_then_reads_unchanged() {
        let state = mk_app_state();
        state.db.set_setting(AUTO_REPLAN_BUDGET_KEY, "5").unwrap();
        assert_eq!(read_auto_replan_budget(&state), 5);
        state.db.set_setting(AUTO_REPLAN_BUDGET_KEY, "0").unwrap();
        assert_eq!(read_auto_replan_budget(&state), 0);
    }

    #[test]
    fn replan_budget_defaults_when_unparseable() {
        let state = mk_app_state();
        state.db.set_setting(AUTO_REPLAN_BUDGET_KEY, "garbage").unwrap();
        assert_eq!(read_auto_replan_budget(&state), DEFAULT_REPLAN_BUDGET);
    }

    // ── composite read_auto_settings ──

    #[test]
    fn read_auto_settings_assembles_all_three_keys() {
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
        state.db.set_setting(AUTO_REPLAN_BUDGET_KEY, "3").unwrap();

        let s = read_auto_settings(&state);
        assert!(s.orchestrator.is_some());
        assert_eq!(s.orchestrator.unwrap().model, "claude-sonnet-4-6");
        assert_eq!(s.worker_pool.len(), 1);
        assert_eq!(s.replan_budget, 3);
    }
}
