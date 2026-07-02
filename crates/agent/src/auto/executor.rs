//! Executor: tie the orchestrator + workers + replan loop together.
//!
//! This module is the Auto-mode entry point. It owns the `AutoRun` state,
//! drives the orchestrator → workers → (maybe replan) cycle, and emits the
//! `Auto*` events the UI consumes.
//!
//! ## Why traits?
//!
//! `executor::run` depends on three traits — `PlanGenerator`, `WorkerRunner`,
//! `PlanApprover` — rather than concrete implementations. This is the only
//! way to write meaningful tests for the sequencer: the production
//! `LlmPlanGenerator` makes real HTTP calls to Anthropic/OpenAI, and the
//! production `LiveWorkerRunner` spawns an `AgentLoop`. Neither is mockable
//! from a unit test. With these traits, headless tests inject canned plans
//! and canned worker results to exercise every branch of the sequencer.
//!
//! Production wiring (P4) constructs:
//!   - `LlmPlanGenerator { llm: orchestrator_model_config }`
//!   - `LiveWorkerRunner { ctx: worker_context }`
//!   - A `PlanApprover` that emits `AutoAwaitingApproval` and awaits the
//!     user's Run-button click via a Tauri channel.
//!
//! ## Replan semantics
//!
//! On a hard worker failure within the replan budget, the executor re-invokes
//! the orchestrator with the prior worker results + failure reason. The
//! orchestrator returns a NEW plan; the executor runs it top-to-bottom from
//! index 0. The orchestrator's system prompt instructs it to return only
//! REMAINING work (not redo successful workers), but the executor trusts the
//! orchestrator's decision — it doesn't defensively dedup by worker id.
//!
//! `prior_results` accumulates monotonically across replans so the
//! orchestrator (and `see_prior`-using workers) can reference results from
//! any prior plan version.
//!
//! Phase: P3 (PHASE_AUTO_MODE.md). P4 wires this to the Tauri backend.

use std::sync::Arc;

use async_trait::async_trait;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

use crate::llm::LlmClientConfig;
use crate::types::AgentEvent;

use super::orchestrator::{generate_plan as orchestrator_generate_plan, OrchestratorError, OrchestratorRequest};
use super::types::{
    AutoResult, AutoRun, AutoStatus, Plan, WorkerFailureContext, WorkerPoolEntry, WorkerResult,
    WorkerSpec, WorkerStatus,
};
use super::worker::{run_worker as live_run_worker, WorkerContext};

// ── Traits ───────────────────────────────────────────────────────────────

/// Produces a `Plan` from a task + worker pool. The production impl
/// (`LlmPlanGenerator`) calls the orchestrator LLM; tests pass mocks that
/// return canned plans / errors.
#[async_trait]
pub trait PlanGenerator: Send + Sync {
    async fn generate(
        &self,
        req: &OrchestratorRequest<'_>,
        cancel: Option<&CancellationToken>,
    ) -> Result<Plan, OrchestratorError>;
}

/// Runs a single worker to completion. The production impl
/// (`LiveWorkerRunner`) spawns an `AgentLoop`; tests pass mocks that return
/// canned `WorkerResult`s (success / hard failure / cancel).
#[async_trait]
pub trait WorkerRunner: Send + Sync {
    async fn run(&self, spec: &WorkerSpec, prior_results: &[WorkerResult]) -> WorkerResult;
}

/// Gate the plan on user approval. Production impl (added in P4) shows the
/// plan UI and awaits the Run button. Tests use `AutoApprove` (always true)
/// or `AlwaysReject` (always false).
#[async_trait]
pub trait PlanApprover: Send + Sync {
    /// Returns `true` to proceed, `false` to abort the task.
    async fn approve(&self, plan: &Plan) -> bool;
}

/// Fire-and-forget hook called by the executor whenever `auto_run`'s state
/// changes meaningfully (plan added, worker finished, replan, status flip).
/// The production impl persists the snapshot to the `auto_runs` table so a
/// page reload or app restart sees the latest known state — without it, the
/// row stays frozen at the initial `Planning` until the run terminates.
///
/// Sync by design — the executor never awaits on persistence. Impls that
/// need IO should spawn their own task.
pub trait AutoStateListener: Send + Sync {
    fn on_state(&self, run: &AutoRun);
}

// ── Production-ready impls ───────────────────────────────────────────────

/// Production `PlanGenerator` — calls the orchestrator LLM via `auto::orchestrator`.
pub struct LlmPlanGenerator {
    pub llm: LlmClientConfig,
}

#[async_trait]
impl PlanGenerator for LlmPlanGenerator {
    async fn generate(
        &self,
        req: &OrchestratorRequest<'_>,
        cancel: Option<&CancellationToken>,
    ) -> Result<Plan, OrchestratorError> {
        orchestrator_generate_plan(&self.llm, req, cancel).await
    }
}

/// Production `WorkerRunner` — spawns an `AgentLoop` per worker via `auto::worker`.
pub struct LiveWorkerRunner {
    pub ctx: WorkerContext,
}

#[async_trait]
impl WorkerRunner for LiveWorkerRunner {
    async fn run(&self, spec: &WorkerSpec, prior_results: &[WorkerResult]) -> WorkerResult {
        live_run_worker(spec, prior_results, &self.ctx).await
    }
}

/// `PlanApprover` that always returns `true`. Useful for headless tests, the
/// bench-runner, and any CLI mode that should bypass interactive approval.
pub struct AutoApprove;

#[async_trait]
impl PlanApprover for AutoApprove {
    async fn approve(&self, _: &Plan) -> bool {
        true
    }
}

// ── Configuration ────────────────────────────────────────────────────────

/// What the executor needs to know about this Auto task.
pub struct ExecutorConfig {
    pub task: String,
    pub auto_run_id: String,
    pub session_id: String,
    pub worker_pool: Vec<WorkerPoolEntry>,
    /// Max reactive replans after the initial plan. Total orchestrator calls
    /// per task = 1 + replan_budget. 0 = strict single-shot.
    pub replan_budget: u32,
    /// When `Some`, the executor resumes from this saved snapshot instead
    /// of starting fresh: `plan_versions.last()` becomes the first plan
    /// (orchestrator NOT called for that iteration). Replans inside the
    /// resumed run still go through the orchestrator normally. Used by the
    /// "Resume" button on the failed-run banner when the prior executor
    /// task was killed (app restart, crash) before user approval.
    pub resume_from: Option<AutoRun>,
    /// P9c: when set, the worker loop starts execution at the named worker
    /// instead of `plan.workers[0]`. Used by the user-driven Retry path
    /// after a hard worker failure: the prior workers already ran (their
    /// results are restored from `resume_from`), only the failed one and
    /// any subsequent workers need to run.
    pub start_at_worker_id: Option<String>,
    /// P9c: when set, the executor treats its first iteration as a replan —
    /// invokes the orchestrator with this reason in the prompt (instead of
    /// the saved plan). Used by the user-driven Replan path after a hard
    /// worker failure. Mutually exclusive with `start_at_worker_id`.
    pub initial_replan_reason: Option<String>,
}

// ── Main entry ───────────────────────────────────────────────────────────

/// Drive an Auto task from start to terminal `AutoResult`.
///
/// Emits the `Auto*` events on `event_tx`. The executor never silently falls
/// back to the regular agent loop; a fully exhausted replan budget produces
/// `AutoResult::Failed` (and an `AutoFailed` event).
pub async fn run(
    config: ExecutorConfig,
    event_tx: mpsc::Sender<AgentEvent>,
    plan_gen: Arc<dyn PlanGenerator>,
    worker_runner: Arc<dyn WorkerRunner>,
    approver: Arc<dyn PlanApprover>,
    cancel: CancellationToken,
    state_listener: Option<Arc<dyn AutoStateListener>>,
) -> AutoResult {
    // Resume path: seed the in-memory state from the saved snapshot so the
    // executor's view of plan_versions / worker_results / replans_used
    // matches what's on disk. The orchestrator is short-circuited on the
    // first loop iteration (see `resume_plan` below) — replans behave
    // normally from there.
    // P9c: ReplanAfterFailure forces a fresh orchestrator call (the saved
    // plan is NOT reused; the failure reason is fed in as replan context).
    // Mutually exclusive with RetryWorker.
    let force_replan_on_first_iter = config.initial_replan_reason.is_some();
    let (mut auto_run, mut resume_plan, mut current_version) = match config.resume_from.clone() {
        Some(snap) => {
            // current_version is "the version of the plan we're about to
            // run". For a resume from awaiting_approval the snapshot has
            // exactly that plan as its latest entry, so we re-use that
            // version number (don't increment) — unless we're replanning
            // after failure, in which case the next plan is a new version.
            let v_existing = snap.plan_versions.last().map(|p| p.version).unwrap_or(1);
            if force_replan_on_first_iter {
                (snap, None, v_existing + 1)
            } else {
                let initial = snap.plan_versions.last().cloned();
                (snap, initial, v_existing)
            }
        }
        None => (
            AutoRun {
                id: config.auto_run_id.clone(),
                session_id: config.session_id.clone(),
                status: AutoStatus::Planning,
                plan_versions: Vec::new(),
                worker_results: Vec::new(),
                cost_cents: 0,
                replans_used: 0,
                current_worker: None,
                pending_failure: None,
            },
            None,
            1u32,
        ),
    };

    // Notify the listener (if any) of every state mutation. Used by the
    // Tauri layer to flush incremental snapshots to `auto_runs` so a reload
    // mid-flight reads the latest known state instead of the initial row.
    let notify = |run: &AutoRun| {
        if let Some(l) = &state_listener {
            l.on_state(run);
        }
    };

    // Restore prior worker outputs so already-completed workers are still
    // visible to subsequent workers' `see_prior` and to the orchestrator on
    // any future replan. Empty on a fresh run; non-empty on resume.
    let mut prior_results: Vec<WorkerResult> = auto_run.worker_results.clone();
    // P9c: clear any prior pending_failure on resume — the user has chosen
    // an action (Retry/Replan) by spawning us, so the failure is being
    // addressed and shouldn't appear in the snapshot anymore.
    auto_run.pending_failure = None;
    // P9c: ReplanAfterFailure seeds the replan_reason so the orchestrator
    // call on iteration 1 includes the failure context.
    let mut replan_reason: Option<String> = config.initial_replan_reason.clone();
    // P9c: RetryWorker: skip earlier workers in the saved plan that
    // already completed. Only the named worker and subsequent ones run.
    // Cleared (via Option::take) after the first iteration uses it so any
    // subsequent replan iteration starts at index 0 of the new plan.
    let mut start_at_worker_id: Option<String> = config.start_at_worker_id.clone();

    loop {
        // ── 1. Planning ──
        // Resume short-circuit: take the saved plan instead of calling the
        // orchestrator. It's already in plan_versions on the snapshot, so
        // don't re-push. Still emit AutoPlan so the frontend's live
        // reducer matches the hydrated state. After this iteration
        // resume_plan is None and the loop falls back to normal planning.
        let plan = if let Some(p) = resume_plan.take() {
            log::info!(
                "[Auto-P7] executor resuming from saved plan v{} ({} workers); skipping orchestrator",
                p.version,
                p.workers.len(),
            );
            emit(
                &event_tx,
                AgentEvent::AutoPlan {
                    session_id: config.session_id.clone(),
                    run_id: config.auto_run_id.clone(),
                    version: p.version,
                    reasoning: p.reasoning.clone(),
                    plan: p.workers.clone(),
                },
            )
            .await;
            p
        } else {
            emit(
                &event_tx,
                AgentEvent::AutoPlanning {
                    session_id: config.session_id.clone(),
                    run_id: config.auto_run_id.clone(),
                },
            )
            .await;
            auto_run.status = AutoStatus::Planning;
            notify(&auto_run);

            if cancel.is_cancelled() {
                let phase = planning_phase_label(&auto_run);
                return finalize_cancelled(auto_run, phase, &notify);
            }

            let req = OrchestratorRequest {
                task: &config.task,
                worker_pool: &config.worker_pool,
                prior_results: &prior_results,
                replan_reason: replan_reason.as_deref(),
                plan_version: current_version,
            };

            let phase_for_cancel = planning_phase_label(&auto_run);
            let p = match plan_gen.generate(&req, Some(&cancel)).await {
                Ok(p) => p,
                Err(OrchestratorError::Cancelled) => return finalize_cancelled(auto_run, phase_for_cancel, &notify),
                Err(e) => {
                    let reason = format!("orchestrator error: {e}");
                    return finalize_failed(auto_run, &event_tx, &config, reason, None, &notify).await;
                }
            };

            auto_run.plan_versions.push(p.clone());
            notify(&auto_run);
            emit(
                &event_tx,
                AgentEvent::AutoPlan {
                    session_id: config.session_id.clone(),
                    run_id: config.auto_run_id.clone(),
                    version: p.version,
                    reasoning: p.reasoning.clone(),
                    plan: p.workers.clone(),
                },
            )
            .await;
            p
        };

        // ── 2. Approval (initial plan only; replans auto-proceed) ──
        if current_version == 1 {
            auto_run.status = AutoStatus::AwaitingApproval;
            notify(&auto_run);
            emit(
                &event_tx,
                AgentEvent::AutoAwaitingApproval {
                    session_id: config.session_id.clone(),
                    run_id: config.auto_run_id.clone(),
                },
            )
            .await;

            let approved = tokio::select! {
                a = approver.approve(&plan) => a,
                _ = cancel.cancelled() => {
                    let phase = format!("Approval for plan v{}", plan.version);
                    return finalize_cancelled(auto_run, phase, &notify);
                }
            };
            if !approved {
                let reason = "plan rejected by user".to_string();
                return finalize_failed(auto_run, &event_tx, &config, reason, None, &notify).await;
            }
        }

        // ── 3. Sequential worker execution ──
        auto_run.status = AutoStatus::Running;
        notify(&auto_run);
        let mut hard_failure_pending: Option<WorkerFailureContext> = None;
        let mut last_summary: Option<String> = None;

        // P9c: RetryWorker resume — skip past workers that already
        // completed; start at the one the user asked to retry. Only applies
        // on the first plan iteration; subsequent replans always start at
        // index 0 of the new plan (Option::take clears the binding).
        let start_idx = if let Some(target_id) = start_at_worker_id.take() {
            plan.workers.iter().position(|w| w.id == target_id).unwrap_or(0)
        } else {
            0
        };

        for spec in plan.workers.iter().skip(start_idx) {
            if cancel.is_cancelled() {
                let phase = format!("Worker {} ({})", spec.id, spec.model);
                return finalize_cancelled(auto_run, phase, &notify);
            }

            // Mark this worker as in flight BEFORE we block on its execution
            // so a reload during the (potentially long) worker run shows
            // "Running: wN (model)" instead of just "status: running".
            auto_run.current_worker = Some(spec.clone());
            notify(&auto_run);

            let result = worker_runner.run(spec, &prior_results).await;
            last_summary = Some(result.summary.clone());
            auto_run.cost_cents = auto_run.cost_cents.saturating_add(result.cost_cents);
            // Worker is no longer in flight; clear before pushing the result
            // so the next persisted snapshot is internally consistent.
            auto_run.current_worker = None;

            if result.status == WorkerStatus::Cancelled {
                let phase = format!("Worker {} ({})", spec.id, spec.model);
                auto_run.worker_results.push(result);
                return finalize_cancelled(auto_run, phase, &notify);
            }

            if result.status.is_hard_failure() {
                // P9c: capture the failure context for the panel's user-driven
                // recovery banner instead of recording the partial result in
                // the ledger. The Tauri side spawns a fresh executor with the
                // user's choice (Retry / Replan / Cancel).
                hard_failure_pending = Some(WorkerFailureContext {
                    worker_id: spec.id.clone(),
                    worker_model: spec.model.clone(),
                    worker_prompt: spec.prompt.clone(),
                    error_message: format!(
                        "worker {} ({}) failed [{}]: {}",
                        spec.id,
                        spec.model,
                        status_label(&result.status),
                        truncate(&result.summary, 240),
                    ),
                });
                break;
            }

            // success — feed into both prior_results (visible to subsequent
            // workers + next orchestrator call) and auto_run.worker_results
            // (the persisted ledger).
            prior_results.push(result.clone());
            auto_run.worker_results.push(result);
            notify(&auto_run);
        }

        // ── 4. Terminal or replan? ──
        match hard_failure_pending {
            None => {
                // All workers in this plan ran successfully.
                return finalize_done(auto_run, &event_tx, &config, last_summary, &notify).await;
            }
            Some(ctx) => {
                // P9c: replan_budget defaults to 0 in production (auto.rs
                // sets AUTO_REPLAN_BUDGET=0). The user becomes the budget —
                // pause the run by stamping pending_failure + status
                // AwaitingFailureDecision into the snapshot. The Tauri side
                // surfaces a Retry/Replan/Cancel banner and spawns a fresh
                // executor on the user's choice. Tests can still pass a
                // nonzero replan_budget to exercise the auto-replan path.
                if auto_run.replans_used >= config.replan_budget {
                    return finalize_awaiting_failure_decision(
                        auto_run, &event_tx, &config, ctx, &notify,
                    )
                    .await;
                }
                let reason = ctx.error_message.clone();
                auto_run.replans_used += 1;
                current_version += 1;
                notify(&auto_run);
                emit(
                    &event_tx,
                    AgentEvent::AutoReplan {
                        session_id: config.session_id.clone(),
                        run_id: config.auto_run_id.clone(),
                        reason: reason.clone(),
                    },
                )
                .await;
                replan_reason = Some(reason);
                continue;
            }
        }
    }
}

/// P9c: terminate the run as "Failed" but tag it with `pending_failure` and
/// `status = AwaitingFailureDecision`. The frontend renders this as a
/// recoverable failure (Retry / Replan / Cancel banner); the user's choice
/// spawns a fresh executor via `agent_resolve_worker_failure`.
async fn finalize_awaiting_failure_decision(
    mut run: AutoRun,
    event_tx: &mpsc::Sender<AgentEvent>,
    config: &ExecutorConfig,
    failure: WorkerFailureContext,
    notify: &(dyn Fn(&AutoRun) + Send + Sync),
) -> AutoResult {
    let reason = failure.error_message.clone();
    run.status = AutoStatus::AwaitingFailureDecision;
    run.pending_failure = Some(failure);
    notify(&run);
    // Reuse the existing AutoFailed event channel — the frontend will read
    // the snapshot's `pending_failure` field via hydration to know it's
    // user-recoverable (vs. a terminally-Failed run). Avoids adding yet
    // another event variant + 4-tier UI dispatch.
    emit(
        event_tx,
        AgentEvent::AutoFailed {
            session_id: config.session_id.clone(),
            run_id: config.auto_run_id.clone(),
            reason: reason.clone(),
            last_worker_output: None,
        },
    )
    .await;
    AutoResult::Failed { reason, run }
}

// ── Finalizers ───────────────────────────────────────────────────────────

async fn finalize_done(
    mut run: AutoRun,
    event_tx: &mpsc::Sender<AgentEvent>,
    config: &ExecutorConfig,
    summary_hint: Option<String>,
    notify: &(dyn Fn(&AutoRun) + Send + Sync),
) -> AutoResult {
    run.status = AutoStatus::Done;
    notify(&run);
    let summary = summary_hint.unwrap_or_else(|| "Auto task complete.".to_string());
    emit(
        event_tx,
        AgentEvent::AutoDone {
            session_id: config.session_id.clone(),
            run_id: config.auto_run_id.clone(),
            summary: summary.clone(),
            total_cost_cents: run.cost_cents,
        },
    )
    .await;
    AutoResult::Done { summary, run }
}

async fn finalize_failed(
    mut run: AutoRun,
    event_tx: &mpsc::Sender<AgentEvent>,
    config: &ExecutorConfig,
    reason: String,
    last_worker_output: Option<String>,
    notify: &(dyn Fn(&AutoRun) + Send + Sync),
) -> AutoResult {
    run.status = AutoStatus::Failed;
    notify(&run);
    emit(
        event_tx,
        AgentEvent::AutoFailed {
            session_id: config.session_id.clone(),
            run_id: config.auto_run_id.clone(),
            reason: reason.clone(),
            last_worker_output: last_worker_output.clone(),
        },
    )
    .await;
    AutoResult::Failed { reason, run }
}

fn finalize_cancelled(
    mut run: AutoRun,
    phase: String,
    notify: &dyn Fn(&AutoRun),
) -> AutoResult {
    run.status = AutoStatus::Cancelled;
    notify(&run);
    // No AgentEvent::AutoCancelled in the enum; the UI infers cancel from the
    // user's Cancel click. We could re-purpose AutoFailed with reason
    // "cancelled" if downstream surfaces need an explicit signal — for now,
    // returning the result is enough; AutoRun.status carries the state.
    // `phase` captures what was happening when cancel fired so the persisted
    // assistant summary can name it (Planning vN, Worker wN, etc.).
    AutoResult::Cancelled { run, phase }
}

// ── Helpers ──────────────────────────────────────────────────────────────

async fn emit(tx: &mpsc::Sender<AgentEvent>, ev: AgentEvent) {
    let _ = tx.send(ev).await;
}

fn status_label(s: &WorkerStatus) -> &'static str {
    match s {
        WorkerStatus::Ok => "ok",
        WorkerStatus::MaxIterations => "max_iterations",
        WorkerStatus::ToolError { .. } => "tool_error",
        WorkerStatus::LlmError { .. } => "llm_error",
        WorkerStatus::Cancelled => "cancelled",
    }
}

fn truncate(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        s.to_string()
    } else {
        let truncated: String = s.chars().take(max).collect();
        format!("{truncated}…")
    }
}

/// Label for the current planning attempt. First plan → "Planning vN";
/// any subsequent iteration (user- or auto-replan) → "Replanning vN".
/// Called at cancel sites in the planning branch so the persisted summary
/// tells the user exactly what was interrupted.
fn planning_phase_label(run: &AutoRun) -> String {
    let version = run.plan_versions.len() + 1;
    if run.plan_versions.is_empty() {
        format!("Planning v{version}")
    } else {
        format!("Replanning v{version}")
    }
}

// ── Tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::auto::types::{SeePrior, SeePriorKeyword};
    use std::sync::{Arc, Mutex};
    use std::time::Duration;

    // ── Mock impls ──

    /// Returns pre-queued plans in order. Errors if asked for more plans than
    /// were queued (catches off-by-one bugs in the replan loop).
    struct MockPlanGen {
        queue: Mutex<Vec<Result<Plan, OrchestratorError>>>,
        calls: Mutex<u32>,
    }
    impl MockPlanGen {
        fn new(plans: Vec<Result<Plan, OrchestratorError>>) -> Self {
            Self {
                queue: Mutex::new(plans),
                calls: Mutex::new(0),
            }
        }
        fn call_count(&self) -> u32 {
            *self.calls.lock().unwrap()
        }
    }
    #[async_trait]
    impl PlanGenerator for MockPlanGen {
        async fn generate(
            &self,
            _req: &OrchestratorRequest<'_>,
            _cancel: Option<&CancellationToken>,
        ) -> Result<Plan, OrchestratorError> {
            *self.calls.lock().unwrap() += 1;
            let mut q = self.queue.lock().unwrap();
            if q.is_empty() {
                panic!("MockPlanGen: no more plans queued");
            }
            q.remove(0)
        }
    }

    /// Returns pre-queued `WorkerResult`s in order, keyed by worker id with
    /// fallthrough. If the executor asks for a worker id not in the canned
    /// map, returns a generic Ok.
    struct MockRunner {
        results_by_id: Mutex<std::collections::HashMap<String, Vec<WorkerResult>>>,
        invocations: Mutex<Vec<String>>,
    }
    impl MockRunner {
        fn new() -> Self {
            Self {
                results_by_id: Mutex::new(Default::default()),
                invocations: Mutex::new(Vec::new()),
            }
        }
        fn enqueue(&self, id: &str, result: WorkerResult) {
            self.results_by_id.lock().unwrap().entry(id.into()).or_default().push(result);
        }
        fn invocations(&self) -> Vec<String> {
            self.invocations.lock().unwrap().clone()
        }
    }
    #[async_trait]
    impl WorkerRunner for MockRunner {
        async fn run(&self, spec: &WorkerSpec, _prior: &[WorkerResult]) -> WorkerResult {
            self.invocations.lock().unwrap().push(spec.id.clone());
            let mut map = self.results_by_id.lock().unwrap();
            if let Some(queue) = map.get_mut(&spec.id) {
                if !queue.is_empty() {
                    return queue.remove(0);
                }
            }
            // Fallthrough: generic Ok.
            WorkerResult {
                id: spec.id.clone(),
                model: spec.model.clone(),
                prompt: spec.prompt.clone(),
                summary: format!("ok: {}", spec.prompt),
                tool_count: 0,
                cost_cents: 0,
                status: WorkerStatus::Ok,
            }
        }
    }

    /// Approver that always returns false.
    struct AlwaysReject;
    #[async_trait]
    impl PlanApprover for AlwaysReject {
        async fn approve(&self, _: &Plan) -> bool {
            false
        }
    }

    // ── Fixtures ──

    fn plan(version: u32, ids: &[&str]) -> Plan {
        Plan {
            version,
            reasoning: format!("plan v{version}"),
            workers: ids
                .iter()
                .map(|id| WorkerSpec {
                    id: (*id).into(),
                    model: "test-model".into(),
                    prompt: format!("task for {id}"),
                    see_prior: SeePrior::Keyword(SeePriorKeyword::None),
                })
                .collect(),
        }
    }

    fn worker_ok(id: &str, summary: &str) -> WorkerResult {
        WorkerResult {
            id: id.into(),
            model: "test-model".into(),
            prompt: "p".into(),
            summary: summary.into(),
            tool_count: 1,
            cost_cents: 5,
            status: WorkerStatus::Ok,
        }
    }
    fn worker_fail(id: &str, kind: WorkerStatus) -> WorkerResult {
        WorkerResult {
            id: id.into(),
            model: "test-model".into(),
            prompt: "p".into(),
            summary: format!("worker {id} failed"),
            tool_count: 0,
            cost_cents: 1,
            status: kind,
        }
    }

    fn cfg(replan_budget: u32) -> (ExecutorConfig, mpsc::Sender<AgentEvent>, mpsc::Receiver<AgentEvent>) {
        let (tx, rx) = mpsc::channel(128);
        (
            ExecutorConfig {
                task: "test task".into(),
                auto_run_id: "auto-run-test".into(),
                session_id: "sess-test".into(),
                worker_pool: vec![WorkerPoolEntry {
                    provider_id: "anthropic".into(),
                    model: "test-model".into(),
                    description: "the only worker".into(),
                }],
                replan_budget,
                resume_from: None,
                start_at_worker_id: None,
                initial_replan_reason: None,
            },
            tx,
            rx,
        )
    }

    /// Collect events until terminal (AutoDone or AutoFailed) or a timeout.
    async fn drain(mut rx: mpsc::Receiver<AgentEvent>) -> Vec<AgentEvent> {
        let mut out = Vec::new();
        while let Ok(Some(ev)) =
            tokio::time::timeout(Duration::from_millis(500), rx.recv()).await
        {
            let terminal =
                matches!(ev, AgentEvent::AutoDone { .. } | AgentEvent::AutoFailed { .. });
            out.push(ev);
            if terminal {
                break;
            }
        }
        out
    }

    // ── Tests ──

    #[tokio::test]
    async fn happy_path_all_workers_ok_yields_autodone() {
        let plan_gen = Arc::new(MockPlanGen::new(vec![Ok(plan(1, &["w1", "w2"]))]));
        let runner = Arc::new(MockRunner::new());
        runner.enqueue("w1", worker_ok("w1", "explored 12 files"));
        runner.enqueue("w2", worker_ok("w2", "synthesized result"));

        let (config, tx, rx) = cfg(2);
        let result = run(
            config,
            tx,
            plan_gen.clone(),
            runner.clone(),
            Arc::new(AutoApprove),
            CancellationToken::new(),
            None,
        )
        .await;

        let events = drain(rx).await;

        match result {
            AutoResult::Done { ref summary, ref run } => {
                assert!(summary.contains("synthesized"), "summary={summary}");
                assert_eq!(run.status, AutoStatus::Done);
                assert_eq!(run.worker_results.len(), 2);
                assert_eq!(run.replans_used, 0);
                assert_eq!(run.plan_versions.len(), 1);
                assert_eq!(run.cost_cents, 10); // 5 + 5
            }
            other => panic!("expected Done, got {other:?}"),
        }
        assert_eq!(plan_gen.call_count(), 1, "no replans expected on happy path");
        assert_eq!(runner.invocations(), vec!["w1", "w2"]);

        // Event ordering sanity.
        let has = |needle: &str| {
            events
                .iter()
                .any(|e| format!("{e:?}").contains(needle))
        };
        assert!(has("AutoPlanning"));
        assert!(has("AutoPlan {"));
        assert!(has("AutoAwaitingApproval"));
        assert!(has("AutoDone"));
        assert!(!has("AutoReplan"), "no replan expected");
        assert!(!has("AutoFailed"), "no failure expected");
    }

    #[tokio::test]
    async fn worker_hard_failure_triggers_replan_then_succeeds() {
        // Plan v1 has [w1, w2]. w1 succeeds; w2 fails (max_iter).
        // Plan v2 has [w2_retry]. w2_retry succeeds.
        let plan_gen = Arc::new(MockPlanGen::new(vec![
            Ok(plan(1, &["w1", "w2"])),
            Ok(plan(2, &["w2_retry"])),
        ]));
        let runner = Arc::new(MockRunner::new());
        runner.enqueue("w1", worker_ok("w1", "step 1 done"));
        runner.enqueue("w2", worker_fail("w2", WorkerStatus::MaxIterations));
        runner.enqueue("w2_retry", worker_ok("w2_retry", "step 2 recovered"));

        let (config, tx, rx) = cfg(2);
        let result = run(
            config,
            tx,
            plan_gen.clone(),
            runner.clone(),
            Arc::new(AutoApprove),
            CancellationToken::new(),
            None,
        )
        .await;
        let events = drain(rx).await;

        assert!(
            matches!(result, AutoResult::Done { .. }),
            "expected Done after successful replan, got {result:?}"
        );
        assert_eq!(plan_gen.call_count(), 2);
        assert_eq!(runner.invocations(), vec!["w1", "w2", "w2_retry"]);

        let auto_run = match &result {
            AutoResult::Done { run, .. } => run,
            _ => unreachable!(),
        };
        assert_eq!(auto_run.replans_used, 1);
        assert_eq!(auto_run.plan_versions.len(), 2);
        // P9c: failed worker results are no longer pushed into worker_results
        // (they live in pending_failure context instead). After a successful
        // replan the ledger contains only the OK workers: w1 + w2_retry.
        assert_eq!(auto_run.worker_results.len(), 2);

        assert!(events.iter().any(|e| matches!(e, AgentEvent::AutoReplan { .. })));
    }

    #[tokio::test]
    async fn replan_budget_exhaustion_yields_autofailed() {
        // Budget = 1. Plan v1 fails, plan v2 also fails → AutoFailed.
        let plan_gen = Arc::new(MockPlanGen::new(vec![
            Ok(plan(1, &["w1"])),
            Ok(plan(2, &["w1_retry"])),
        ]));
        let runner = Arc::new(MockRunner::new());
        runner.enqueue("w1", worker_fail("w1", WorkerStatus::ToolError { message: "boom".into() }));
        runner.enqueue(
            "w1_retry",
            worker_fail("w1_retry", WorkerStatus::ToolError { message: "still boom".into() }),
        );

        let (config, tx, rx) = cfg(1);
        let result = run(
            config,
            tx,
            plan_gen,
            runner,
            Arc::new(AutoApprove),
            CancellationToken::new(),
            None,
        )
        .await;
        let events = drain(rx).await;

        match &result {
            AutoResult::Failed { reason, run } => {
                assert!(reason.contains("w1_retry"), "reason should mention the last failed worker: {reason}");
                assert_eq!(run.replans_used, 1, "budget=1 means exactly 1 replan attempted");
                assert_eq!(run.plan_versions.len(), 2);
                // P9c: budget exhaustion now parks the run in
                // AwaitingFailureDecision with pending_failure set so the
                // user can pick Retry / Replan / Cancel from the panel.
                assert_eq!(run.status, AutoStatus::AwaitingFailureDecision);
                assert!(run.pending_failure.is_some(), "pending_failure must be populated for user recovery");
            }
            other => panic!("expected Failed, got {other:?}"),
        }
        assert!(events.iter().any(|e| matches!(e, AgentEvent::AutoFailed { .. })));
    }

    #[tokio::test]
    async fn zero_replan_budget_fails_immediately_on_worker_failure() {
        let plan_gen = Arc::new(MockPlanGen::new(vec![Ok(plan(1, &["w1"]))]));
        let runner = Arc::new(MockRunner::new());
        runner.enqueue("w1", worker_fail("w1", WorkerStatus::LlmError { message: "5xx".into() }));

        let (config, tx, rx) = cfg(0);
        let result = run(config, tx, plan_gen, runner, Arc::new(AutoApprove), CancellationToken::new(), None).await;
        let _ = drain(rx).await;

        match result {
            AutoResult::Failed { run, .. } => {
                assert_eq!(run.replans_used, 0);
                assert_eq!(run.plan_versions.len(), 1);
                // P9c: budget=0 path also parks in AwaitingFailureDecision now.
                assert_eq!(run.status, AutoStatus::AwaitingFailureDecision);
                assert!(run.pending_failure.is_some());
            }
            other => panic!("expected immediate Failed with budget=0, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn user_rejects_plan_yields_failed_with_rejection_reason() {
        let plan_gen = Arc::new(MockPlanGen::new(vec![Ok(plan(1, &["w1"]))]));
        let runner = Arc::new(MockRunner::new());

        let (config, tx, rx) = cfg(2);
        let result = run(config, tx, plan_gen, runner.clone(), Arc::new(AlwaysReject), CancellationToken::new(), None).await;
        let events = drain(rx).await;

        match result {
            AutoResult::Failed { reason, run } => {
                assert!(reason.contains("rejected"), "reason: {reason}");
                assert_eq!(run.plan_versions.len(), 1, "plan was still generated and emitted");
                assert!(run.worker_results.is_empty(), "no workers should run if rejected");
            }
            other => panic!("expected Failed on plan rejection, got {other:?}"),
        }
        assert!(runner.invocations().is_empty(), "runner must not be invoked on rejection");
        assert!(events.iter().any(|e| matches!(e, AgentEvent::AutoFailed { .. })));
    }

    #[tokio::test]
    async fn orchestrator_error_yields_failed_no_replan() {
        let plan_gen = Arc::new(MockPlanGen::new(vec![Err(OrchestratorError::Parse {
            attempts: 3,
            message: "all parse attempts failed".into(),
        })]));
        let runner = Arc::new(MockRunner::new());

        let (config, tx, rx) = cfg(2);
        let result = run(config, tx, plan_gen, runner.clone(), Arc::new(AutoApprove), CancellationToken::new(), None).await;
        let _ = drain(rx).await;

        match result {
            AutoResult::Failed { reason, run } => {
                assert!(reason.contains("orchestrator error"), "reason: {reason}");
                assert!(run.plan_versions.is_empty(), "no plan was produced");
                assert_eq!(run.replans_used, 0);
            }
            other => panic!("expected Failed on orchestrator error, got {other:?}"),
        }
        assert!(runner.invocations().is_empty());
    }

    #[tokio::test]
    async fn cancel_during_worker_run_yields_cancelled() {
        // First worker reports Cancelled status (simulates user cancelling
        // mid-worker — the child agent loop's cancel token propagates).
        let plan_gen = Arc::new(MockPlanGen::new(vec![Ok(plan(1, &["w1", "w2"]))]));
        let runner = Arc::new(MockRunner::new());
        runner.enqueue("w1", worker_fail("w1", WorkerStatus::Cancelled));

        let (config, tx, _rx) = cfg(2);
        let result = run(config, tx, plan_gen, runner.clone(), Arc::new(AutoApprove), CancellationToken::new(), None).await;

        match result {
            AutoResult::Cancelled { run, phase } => {
                assert_eq!(run.status, AutoStatus::Cancelled);
                assert_eq!(run.worker_results.len(), 1, "only w1 ran before cancel");
                assert_eq!(runner.invocations(), vec!["w1"]);
                assert!(
                    phase.starts_with("Worker w1"),
                    "phase should name the cancelled worker, got {phase:?}"
                );
            }
            other => panic!("expected Cancelled, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn cancel_during_second_worker_preserves_prior_success() {
        // Two-worker plan: w1 completes cleanly, w2 gets cancelled. We assert
        // both that w1's Ok result stays in worker_results (so a downstream
        // Retry/Replan sees it as prior context) AND that the phase string
        // names w2 specifically.
        let plan_gen = Arc::new(MockPlanGen::new(vec![Ok(plan(1, &["w1", "w2"]))]));
        let runner = Arc::new(MockRunner::new());
        runner.enqueue("w1", worker_ok("w1", "explored the repo"));
        runner.enqueue("w2", worker_fail("w2", WorkerStatus::Cancelled));

        let (config, tx, _rx) = cfg(2);
        let result = run(config, tx, plan_gen, runner.clone(), Arc::new(AutoApprove), CancellationToken::new(), None).await;

        match result {
            AutoResult::Cancelled { run, phase } => {
                assert_eq!(run.status, AutoStatus::Cancelled);
                assert_eq!(run.worker_results.len(), 2, "w1 Ok + w2 Cancelled both recorded");
                assert_eq!(run.worker_results[0].id, "w1");
                assert_eq!(run.worker_results[0].status, WorkerStatus::Ok);
                assert_eq!(run.worker_results[1].id, "w2");
                assert_eq!(run.worker_results[1].status, WorkerStatus::Cancelled);
                assert!(
                    phase.starts_with("Worker w2"),
                    "phase should name w2 (the cancelled one), got {phase:?}"
                );
                assert_eq!(runner.invocations(), vec!["w1", "w2"]);
            }
            other => panic!("expected Cancelled, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn cancel_token_fired_before_orchestrator_yields_cancelled() {
        let plan_gen = Arc::new(MockPlanGen::new(vec![Ok(plan(1, &["w1"]))]));
        let runner = Arc::new(MockRunner::new());

        let (config, tx, _rx) = cfg(2);
        let cancel = CancellationToken::new();
        cancel.cancel();
        let result = run(config, tx, plan_gen.clone(), runner.clone(), Arc::new(AutoApprove), cancel, None).await;

        match result {
            AutoResult::Cancelled { run, phase } => {
                assert!(run.plan_versions.is_empty(), "no plan should be generated");
                assert_eq!(phase, "Planning v1", "first-plan cancel should tag Planning v1");
            }
            other => panic!("expected Cancelled with pre-fired token, got {other:?}"),
        }
        assert_eq!(plan_gen.call_count(), 0, "orchestrator must not be called when cancelled");
    }
}
