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
    AutoResult, AutoRun, AutoStatus, Plan, WorkerPoolEntry, WorkerResult, WorkerSpec, WorkerStatus,
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
) -> AutoResult {
    let mut auto_run = AutoRun {
        id: config.auto_run_id.clone(),
        session_id: config.session_id.clone(),
        status: AutoStatus::Planning,
        plan_versions: Vec::new(),
        worker_results: Vec::new(),
        cost_cents: 0,
        replans_used: 0,
    };

    let mut prior_results: Vec<WorkerResult> = Vec::new();
    let mut current_version: u32 = 1;
    let mut replan_reason: Option<String> = None;

    loop {
        // ── 1. Planning ──
        emit(
            &event_tx,
            AgentEvent::AutoPlanning {
                session_id: config.session_id.clone(),
                run_id: config.auto_run_id.clone(),
            },
        )
        .await;
        auto_run.status = AutoStatus::Planning;

        if cancel.is_cancelled() {
            return finalize_cancelled(auto_run);
        }

        let req = OrchestratorRequest {
            task: &config.task,
            worker_pool: &config.worker_pool,
            prior_results: &prior_results,
            replan_reason: replan_reason.as_deref(),
            plan_version: current_version,
        };

        let plan = match plan_gen.generate(&req, Some(&cancel)).await {
            Ok(p) => p,
            Err(OrchestratorError::Cancelled) => return finalize_cancelled(auto_run),
            Err(e) => {
                let reason = format!("orchestrator error: {e}");
                return finalize_failed(auto_run, &event_tx, &config, reason, None).await;
            }
        };

        auto_run.plan_versions.push(plan.clone());
        emit(
            &event_tx,
            AgentEvent::AutoPlan {
                session_id: config.session_id.clone(),
                run_id: config.auto_run_id.clone(),
                version: plan.version,
                reasoning: plan.reasoning.clone(),
                plan: plan.workers.clone(),
            },
        )
        .await;

        // ── 2. Approval (initial plan only; replans auto-proceed) ──
        if current_version == 1 {
            auto_run.status = AutoStatus::AwaitingApproval;
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
                _ = cancel.cancelled() => return finalize_cancelled(auto_run),
            };
            if !approved {
                let reason = "plan rejected by user".to_string();
                return finalize_failed(auto_run, &event_tx, &config, reason, None).await;
            }
        }

        // ── 3. Sequential worker execution ──
        auto_run.status = AutoStatus::Running;
        let mut hard_failure: Option<String> = None;
        let mut last_summary: Option<String> = None;

        for spec in plan.workers.iter() {
            if cancel.is_cancelled() {
                return finalize_cancelled(auto_run);
            }

            let result = worker_runner.run(spec, &prior_results).await;
            last_summary = Some(result.summary.clone());
            auto_run.cost_cents = auto_run.cost_cents.saturating_add(result.cost_cents);

            if result.status == WorkerStatus::Cancelled {
                auto_run.worker_results.push(result);
                return finalize_cancelled(auto_run);
            }

            if result.status.is_hard_failure() {
                hard_failure = Some(format!(
                    "worker {} ({}) failed [{}]: {}",
                    spec.id,
                    spec.model,
                    status_label(&result.status),
                    truncate(&result.summary, 240),
                ));
                auto_run.worker_results.push(result);
                break;
            }

            // success — feed into both prior_results (visible to subsequent
            // workers + next orchestrator call) and auto_run.worker_results
            // (the persisted ledger).
            prior_results.push(result.clone());
            auto_run.worker_results.push(result);
        }

        // ── 4. Terminal or replan? ──
        match hard_failure {
            None => {
                // All workers in this plan ran successfully.
                return finalize_done(auto_run, &event_tx, &config, last_summary).await;
            }
            Some(reason) => {
                if auto_run.replans_used >= config.replan_budget {
                    return finalize_failed(auto_run, &event_tx, &config, reason, last_summary).await;
                }
                auto_run.replans_used += 1;
                current_version += 1;
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

// ── Finalizers ───────────────────────────────────────────────────────────

async fn finalize_done(
    mut run: AutoRun,
    event_tx: &mpsc::Sender<AgentEvent>,
    config: &ExecutorConfig,
    summary_hint: Option<String>,
) -> AutoResult {
    run.status = AutoStatus::Done;
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
) -> AutoResult {
    run.status = AutoStatus::Failed;
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

fn finalize_cancelled(mut run: AutoRun) -> AutoResult {
    run.status = AutoStatus::Cancelled;
    // No AgentEvent::AutoCancelled in the enum; the UI infers cancel from the
    // user's Cancel click. We could re-purpose AutoFailed with reason
    // "cancelled" if downstream surfaces need an explicit signal — for now,
    // returning the result is enough; AutoRun.status carries the state.
    AutoResult::Cancelled { run }
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
        assert_eq!(auto_run.worker_results.len(), 3); // w1 ok, w2 failed, w2_retry ok

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
        )
        .await;
        let events = drain(rx).await;

        match &result {
            AutoResult::Failed { reason, run } => {
                assert!(reason.contains("w1_retry"), "reason should mention the last failed worker: {reason}");
                assert_eq!(run.replans_used, 1, "budget=1 means exactly 1 replan attempted");
                assert_eq!(run.plan_versions.len(), 2);
                assert_eq!(run.status, AutoStatus::Failed);
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
        let result = run(config, tx, plan_gen, runner, Arc::new(AutoApprove), CancellationToken::new()).await;
        let _ = drain(rx).await;

        match result {
            AutoResult::Failed { run, .. } => {
                assert_eq!(run.replans_used, 0);
                assert_eq!(run.plan_versions.len(), 1);
            }
            other => panic!("expected immediate Failed with budget=0, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn user_rejects_plan_yields_failed_with_rejection_reason() {
        let plan_gen = Arc::new(MockPlanGen::new(vec![Ok(plan(1, &["w1"]))]));
        let runner = Arc::new(MockRunner::new());

        let (config, tx, rx) = cfg(2);
        let result = run(config, tx, plan_gen, runner.clone(), Arc::new(AlwaysReject), CancellationToken::new()).await;
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
        let result = run(config, tx, plan_gen, runner.clone(), Arc::new(AutoApprove), CancellationToken::new()).await;
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
        let result = run(config, tx, plan_gen, runner.clone(), Arc::new(AutoApprove), CancellationToken::new()).await;

        match result {
            AutoResult::Cancelled { run } => {
                assert_eq!(run.status, AutoStatus::Cancelled);
                assert_eq!(run.worker_results.len(), 1, "only w1 ran before cancel");
                assert_eq!(runner.invocations(), vec!["w1"]);
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
        let result = run(config, tx, plan_gen.clone(), runner.clone(), Arc::new(AutoApprove), cancel).await;

        match result {
            AutoResult::Cancelled { run } => {
                assert!(run.plan_versions.is_empty(), "no plan should be generated");
            }
            other => panic!("expected Cancelled with pre-fired token, got {other:?}"),
        }
        assert_eq!(plan_gen.call_count(), 0, "orchestrator must not be called when cancelled");
    }
}
