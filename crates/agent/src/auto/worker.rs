//! Ephemeral worker runtime.
//!
//! A worker is the unit the orchestrator dispatches to. It runs as a brand-new
//! `AgentLoop` (no parent message history, no shared state besides the working
//! directory and the inherited host context). The orchestrator supplies the
//! model and the natural-language prompt; the worker's system prompt is the
//! standard Coding-mode prompt — the orchestrator instructs via the user
//! message, not via system overrides.
//!
//! This module reuses the spawning skeleton pattern from `crate::subagents::tool`
//! (child `AgentLoop::with_provider`, drained event channel, cancel-child token)
//! but is **not coupled** to `SubagentRegistry` — workers are constructed from a
//! `WorkerSpec` the orchestrator emits at runtime, not from a markdown-defined
//! persona registry.
//!
//! Phase: P1 (PHASE_AUTO_MODE.md). Sequencing across workers + the orchestrator
//! itself land in P2/P3.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;
use uuid::Uuid;

use crate::agent::config::{CompactionConfig, RetryConfig};
use crate::agent::{AgentConfig, AgentLoop};
use crate::approval::ApprovalHandler;
use crate::context_engine::ContextEngineApi;
use crate::error::AgentError;
use crate::llm::client::LlmProvider;
use crate::llm::types::ChatMessage;
use crate::llm::{LlmClient, LlmClientConfig};
use crate::persistence::MessagePersister;
use crate::tool::{ToolMode, ToolPolicy, ToolRegistry};
use crate::types::{AgentEvent, AgentResult};

use super::types::{SeePrior, SeePriorKeyword, WorkerResult, WorkerSpec, WorkerStatus};

/// Marker prefix the `AgentLoop` writes into the final summary when it hits
/// `max_iterations` (see `crates/agent/src/agent/loop_.rs` exit path). The
/// worker classifier uses this to map that result to `WorkerStatus::MaxIterations`
/// so the executor can trigger a reactive replan.
///
/// Fragile string-match. If the AgentLoop ever rephrases that message, the
/// classifier silently misroutes hard failures as `Ok`. The unit test
/// `max_iterations_summary_maps_to_status` pins this contract.
const MAX_ITER_SUMMARY_PREFIX: &str = "Reached maximum of";

/// Host-supplied context every worker needs: inherited LLM client shape,
/// working directory, channels, cancellation, and the optional integrations
/// (context engine / persister / approval handler).
///
/// Cloned/borrowed per worker call. The executor (P3) builds one of these
/// from the parent session and reuses it across all workers in a single
/// `AutoRun`.
pub struct WorkerContext {
    pub session_id: String,
    pub run_id: String,
    pub working_dir: PathBuf,
    pub event_tx: mpsc::Sender<AgentEvent>,
    pub cancel_token: CancellationToken,

    /// Per-worker LLM configs keyed by model name. Built once at run start
    /// (`auto.rs::build_worker_llm_configs`) by resolving each `WorkerPoolEntry`
    /// to its provider's `LlmClientConfig`. The config's `model` field already
    /// matches its key, so dispatch is a clone of the entry — no override.
    /// Missing key = orchestrator picked a model not in the pool → worker
    /// returns `WorkerStatus::LlmError` and the executor's replan path kicks in.
    pub worker_llm_configs: HashMap<String, LlmClientConfig>,
    pub retry_config: RetryConfig,
    pub compaction_config: CompactionConfig,
    pub compaction_llm: Option<LlmClientConfig>,
    pub max_iterations: u32,

    pub context_engine: Option<Arc<dyn ContextEngineApi>>,
    pub context_engine_repo_path: Option<PathBuf>,
    pub persister: Option<Arc<dyn MessagePersister>>,
    pub approval_handler: Option<Arc<dyn ApprovalHandler>>,
    pub checkpoint_dir: Option<PathBuf>,
}

/// Run a single worker to completion (or hard failure).
///
/// Emits `AutoWorkerStart` on the parent channel before kicking off, drains the
/// child's event stream (counting tool calls), runs the child `AgentLoop`, then
/// emits `AutoWorkerEnd` and returns the populated `WorkerResult`.
///
/// `prior_results` is the full history of completed workers in this `AutoRun`;
/// `spec.see_prior` decides which subset gets injected into the user message
/// (paper's `access_list`).
pub async fn run_worker(
    spec: &WorkerSpec,
    prior_results: &[WorkerResult],
    ctx: &WorkerContext,
) -> WorkerResult {
    // 1. Announce the start to the parent UI.
    let _ = ctx
        .event_tx
        .send(AgentEvent::AutoWorkerStart {
            session_id: ctx.session_id.clone(),
            run_id: ctx.run_id.clone(),
            worker_id: spec.id.clone(),
            model: spec.model.clone(),
            prompt: spec.prompt.clone(),
        })
        .await;

    // 2. Build the user message: prior outputs (filtered by see_prior) +
    //    the orchestrator's task prompt.
    let user_text = build_worker_user_message(spec, prior_results);

    // 3. Look up this worker's LlmClientConfig from the per-provider map.
    //    Miss means the orchestrator hallucinated a model that isn't in the
    //    pool — surface as a hard failure so the executor's replan path
    //    kicks in with the failure context.
    let child_llm = match ctx.worker_llm_configs.get(&spec.model) {
        Some(cfg) => cfg.clone(),
        None => {
            let msg = format!(
                "model `{}` is not in the configured worker pool — \
                 orchestrator picked an unknown model",
                spec.model
            );
            let status = WorkerStatus::LlmError { message: msg.clone() };
            let _ = ctx
                .event_tx
                .send(AgentEvent::AutoWorkerEnd {
                    session_id: ctx.session_id.clone(),
                    run_id: ctx.run_id.clone(),
                    worker_id: spec.id.clone(),
                    summary: msg.clone(),
                    cost_cents: 0,
                    tool_count: 0,
                    status: status.clone(),
                })
                .await;
            return WorkerResult {
                id: spec.id.clone(),
                model: spec.model.clone(),
                prompt: spec.prompt.clone(),
                summary: msg,
                tool_count: 0,
                cost_cents: 0,
                status,
            };
        }
    };

    // 4. Build the child AgentConfig. system_prompt=None → AgentLoop builds
    //    the standard Coding-mode prompt internally. No skills/subagents
    //    nesting in v1 (depth-1 enforced same way subagents do it).
    let child_config = AgentConfig {
        llm: child_llm.clone(),
        working_dir: ctx.working_dir.clone(),
        mode: ToolMode::Coding,
        max_iterations: ctx.max_iterations,
        system_prompt: None,
        retry_config: ctx.retry_config.clone(),
        compaction_config: ctx.compaction_config.clone(),
        compaction_llm: ctx.compaction_llm.clone(),
        context_engine: ctx.context_engine.clone(),
        context_engine_repo_path: ctx.context_engine_repo_path.clone(),
        skills: None,
        subagents: None,
        subagent_inheritance: None,
        checkpoint_dir: ctx.checkpoint_dir.clone(),
        tool_policy: ToolPolicy::default(),
    };

    // 5. Coding-mode tool registry, wired to the parent's context engine if any.
    let context_engine_arg = ctx.context_engine.as_ref().and_then(|e| {
        ctx.context_engine_repo_path
            .as_ref()
            .map(|p| (e.clone(), p.clone()))
    });
    let child_registry = ToolRegistry::for_mode(ToolMode::Coding, context_engine_arg, None);

    // 6. Child event channel + drain task. Most child events are discarded
    //    at the relay layer (workers don't stream text into the parent chat),
    //    but ToolStart/ToolEnd are re-emitted as wrapped Auto* events on the
    //    parent's channel so the UI can render live tool activity inside the
    //    running worker card. The drain also counts ToolEnd for the worker's
    //    `tool_count` summary.
    let (child_tx, mut child_rx) = mpsc::channel::<AgentEvent>(256);
    let parent_tx_for_drain = ctx.event_tx.clone();
    let parent_session = ctx.session_id.clone();
    let parent_run = ctx.run_id.clone();
    let drain_worker_id = spec.id.clone();
    let drain_handle = tokio::spawn(async move {
        let mut tool_count = 0u32;
        while let Some(ev) = child_rx.recv().await {
            match ev {
                AgentEvent::ToolStart { tool_call_id, tool_name, args_summary, .. } => {
                    let _ = parent_tx_for_drain
                        .send(AgentEvent::AutoWorkerToolStart {
                            session_id: parent_session.clone(),
                            run_id: parent_run.clone(),
                            worker_id: drain_worker_id.clone(),
                            tool_call_id,
                            tool_name,
                            args_summary,
                        })
                        .await;
                }
                AgentEvent::ToolEnd { tool_call_id, success, summary, .. } => {
                    tool_count += 1;
                    let _ = parent_tx_for_drain
                        .send(AgentEvent::AutoWorkerToolEnd {
                            session_id: parent_session.clone(),
                            run_id: parent_run.clone(),
                            worker_id: drain_worker_id.clone(),
                            tool_call_id,
                            success,
                            summary,
                        })
                        .await;
                }
                _ => {
                    // Other child events (TextDelta, ThinkingDelta, TokenUsage,
                    // TurnCompleted, Done, etc.) intentionally NOT forwarded —
                    // the parent UI shows worker output via the final
                    // AutoWorkerEnd summary only.
                }
            }
        }
        tool_count
    });

    // 7. Run the child loop. Cancel propagates from parent → child via
    //    `child_token()` so user-cancel kills the in-flight worker mid-tool.
    let child_session_id = Uuid::new_v4().to_string();
    let child_cancel = ctx.cancel_token.child_token();
    let provider: Box<dyn LlmProvider> = Box::new(LlmClient::new(child_llm));
    let mut child_loop = AgentLoop::with_provider(
        child_config,
        provider,
        child_registry,
        child_cancel,
        child_tx,
        child_session_id.clone(),
    );
    if let Some(p) = ctx.persister.clone() {
        child_loop = child_loop.with_persister(p, child_session_id);
    }
    if let Some(h) = ctx.approval_handler.clone() {
        child_loop = child_loop.with_approval_handler(h);
    }
    let run_result = child_loop.run(ChatMessage::user(user_text)).await;
    drop(child_loop);
    let tool_count = drain_handle.await.unwrap_or(0);

    // 8. Map AgentLoop outcome to (summary, WorkerStatus).
    let (summary, status) = classify_outcome(run_result);

    // 9. P1: cost is not yet computed (no pricing table wired). Surface 0 and
    //    let the executor / UI display "—" until cost telemetry lands in P7.
    let cost_cents = 0;

    // 10. Announce the end.
    let _ = ctx
        .event_tx
        .send(AgentEvent::AutoWorkerEnd {
            session_id: ctx.session_id.clone(),
            run_id: ctx.run_id.clone(),
            worker_id: spec.id.clone(),
            summary: summary.clone(),
            cost_cents,
            tool_count,
            status: status.clone(),
        })
        .await;

    WorkerResult {
        id: spec.id.clone(),
        model: spec.model.clone(),
        prompt: spec.prompt.clone(),
        summary,
        tool_count,
        cost_cents,
        status,
    }
}

/// Build the user message handed to the worker: prior outputs (filtered by
/// `see_prior`) followed by the orchestrator's task prompt. When no priors
/// are visible, returns the prompt verbatim.
fn build_worker_user_message(spec: &WorkerSpec, prior: &[WorkerResult]) -> String {
    let included: Vec<&WorkerResult> = match &spec.see_prior {
        SeePrior::Keyword(SeePriorKeyword::None) => Vec::new(),
        SeePrior::Keyword(SeePriorKeyword::All) => prior.iter().collect(),
        SeePrior::Specific(ids) => prior.iter().filter(|r| ids.iter().any(|i| i == &r.id)).collect(),
    };

    if included.is_empty() {
        return spec.prompt.clone();
    }

    let mut s = String::new();
    for r in included {
        s.push_str(&format!("<prior_worker id=\"{}\">\n", r.id));
        s.push_str(&r.summary);
        s.push_str("\n</prior_worker>\n\n");
    }
    s.push_str("<task>\n");
    s.push_str(&spec.prompt);
    s.push_str("\n</task>");
    s
}

/// Translate the `AgentLoop`'s terminal outcome into a worker-facing
/// `(summary, WorkerStatus)` pair. `WorkerStatus::is_hard_failure()` (in
/// `types.rs`) decides whether the executor will trigger a reactive replan.
fn classify_outcome(
    outcome: Result<AgentResult, AgentError>,
) -> (String, WorkerStatus) {
    match outcome {
        Ok(AgentResult::Done { summary }) => {
            // The AgentLoop encodes max_iterations exhaustion as Ok(Done) with
            // a known prefix; rescue it as a hard failure here so the
            // executor can replan.
            if summary.starts_with(MAX_ITER_SUMMARY_PREFIX) {
                (summary, WorkerStatus::MaxIterations)
            } else {
                (summary, WorkerStatus::Ok)
            }
        }
        Ok(AgentResult::AskUser { question, .. }) => (
            format!("worker yielded AskUser unexpectedly: {question}"),
            WorkerStatus::ToolError {
                message: "unexpected AskUser yield from a worker — Auto mode does not pass questions through".into(),
            },
        ),
        Ok(AgentResult::PlanReady { .. }) => (
            "worker yielded PlanReady unexpectedly".into(),
            WorkerStatus::ToolError {
                message: "unexpected PlanReady yield from a worker — Auto mode does not surface plans".into(),
            },
        ),
        Err(AgentError::Cancelled) => ("cancelled".into(), WorkerStatus::Cancelled),
        Err(e) => {
            let msg = e.to_string();
            (format!("worker failed: {msg}"), WorkerStatus::LlmError { message: msg })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::auto::types::{SeePrior, SeePriorKeyword};
    use crate::llm::client::LlmPolicy;
    use crate::llm::Provider;
    use crate::test_util::{text_response, MockLlm};

    fn wr(id: &str, summary: &str) -> WorkerResult {
        WorkerResult {
            id: id.into(),
            model: "test".into(),
            prompt: "p".into(),
            summary: summary.into(),
            tool_count: 0,
            cost_cents: 0,
            status: WorkerStatus::Ok,
        }
    }

    fn spec(id: &str, see: SeePrior) -> WorkerSpec {
        WorkerSpec {
            id: id.into(),
            model: "test-model".into(),
            prompt: "do the task".into(),
            see_prior: see,
        }
    }

    // ── build_worker_user_message ──

    #[test]
    fn user_message_with_no_priors_is_just_the_prompt() {
        let s = spec("w2", SeePrior::Keyword(SeePriorKeyword::None));
        let priors = vec![wr("w1", "prior summary here")];
        let msg = build_worker_user_message(&s, &priors);
        assert_eq!(msg, "do the task");
    }

    #[test]
    fn user_message_with_all_injects_every_prior_in_order() {
        let s = spec("w3", SeePrior::Keyword(SeePriorKeyword::All));
        let priors = vec![wr("w1", "first"), wr("w2", "second")];
        let msg = build_worker_user_message(&s, &priors);
        assert!(msg.contains("<prior_worker id=\"w1\">\nfirst\n</prior_worker>"));
        assert!(msg.contains("<prior_worker id=\"w2\">\nsecond\n</prior_worker>"));
        assert!(msg.contains("<task>\ndo the task\n</task>"));
        // Ordering: w1 must appear before w2.
        assert!(msg.find("w1").unwrap() < msg.find("w2").unwrap());
    }

    #[test]
    fn user_message_with_specific_filters_to_listed_priors_only() {
        let s = spec("w4", SeePrior::Specific(vec!["w2".into()]));
        let priors = vec![
            wr("w1", "first"),
            wr("w2", "second"),
            wr("w3", "third"),
        ];
        let msg = build_worker_user_message(&s, &priors);
        assert!(!msg.contains("first"), "w1 must be excluded");
        assert!(msg.contains("second"), "w2 must be present");
        assert!(!msg.contains("third"), "w3 must be excluded");
    }

    #[test]
    fn user_message_with_specific_but_no_matches_emits_just_prompt() {
        // Orchestrator references a worker id that doesn't exist yet (shouldn't
        // happen if the executor validates plans, but be defensive).
        let s = spec("w5", SeePrior::Specific(vec!["w99".into()]));
        let priors = vec![wr("w1", "first")];
        let msg = build_worker_user_message(&s, &priors);
        assert_eq!(msg, "do the task");
    }

    // ── classify_outcome ──

    #[test]
    fn done_classified_as_ok() {
        let (summary, status) = classify_outcome(Ok(AgentResult::Done {
            summary: "completed normally".into(),
        }));
        assert_eq!(summary, "completed normally");
        assert_eq!(status, WorkerStatus::Ok);
    }

    #[test]
    fn max_iterations_summary_maps_to_status() {
        // Pins the contract between AgentLoop's max-iter exit summary and the
        // worker classifier. If this test breaks, the AgentLoop rephrased its
        // exit message — update MAX_ITER_SUMMARY_PREFIX accordingly.
        let summary = format!("{} 100 steps. Progress preserved.", MAX_ITER_SUMMARY_PREFIX);
        let (out_summary, status) = classify_outcome(Ok(AgentResult::Done {
            summary: summary.clone(),
        }));
        assert_eq!(out_summary, summary);
        assert_eq!(status, WorkerStatus::MaxIterations);
        assert!(status.is_hard_failure());
    }

    #[test]
    fn cancelled_classified_correctly() {
        let (_, status) = classify_outcome(Err(AgentError::Cancelled));
        assert_eq!(status, WorkerStatus::Cancelled);
        assert!(!status.is_hard_failure(), "Cancelled is NOT a hard failure (no replan)");
    }

    #[test]
    fn llm_error_classified_as_hard_failure() {
        let (_, status) = classify_outcome(Err(AgentError::LlmApiError {
            status: 500,
            body: "boom".into(),
        }));
        assert!(matches!(status, WorkerStatus::LlmError { .. }));
        assert!(status.is_hard_failure());
    }

    #[test]
    fn unexpected_ask_user_yield_treated_as_tool_error() {
        let (_, status) = classify_outcome(Ok(AgentResult::AskUser {
            question: "?".into(),
            options: None,
        }));
        assert!(matches!(status, WorkerStatus::ToolError { .. }));
        assert!(status.is_hard_failure());
    }

    // ── run_worker end-to-end with MockLlm (no API) ──

    fn make_ctx(
        provider: Provider,
        event_tx: mpsc::Sender<AgentEvent>,
        working_dir: PathBuf,
    ) -> WorkerContext {
        // The only test caller (start_event_carries_...) uses spec.model="gpt-test",
        // so pre-populate the per-worker map with that key. The LLM call still
        // fails at localhost (which is what the test pins).
        let mut worker_llm_configs = HashMap::new();
        worker_llm_configs.insert(
            "gpt-test".to_string(),
            LlmClientConfig {
                provider,
                base_url: "http://localhost".into(),
                model: "gpt-test".into(),
                api_key: String::new(),
                temperature: None,
                max_completion_tokens: None,
                extra_headers: vec![],
                thinking: None,
                disable_cache_control: true,
                policy: LlmPolicy::default(),
            },
        );
        WorkerContext {
            session_id: "parent-session".into(),
            run_id: "run-1".into(),
            working_dir,
            event_tx,
            cancel_token: CancellationToken::new(),
            worker_llm_configs,
            retry_config: RetryConfig::default(),
            compaction_config: CompactionConfig::default(),
            compaction_llm: None,
            max_iterations: 5,
            context_engine: None,
            context_engine_repo_path: None,
            persister: None,
            approval_handler: None,
            checkpoint_dir: None,
        }
    }

    /// Verify run_worker drives a child AgentLoop end-to-end and emits the
    /// AutoWorker{Start,End} bracket around it. We bypass the production
    /// LlmClient by constructing the loop via with_provider in a parallel
    /// path — but since run_worker doesn't expose that knob, this test
    /// instead checks the outer plumbing by issuing a worker that the
    /// classifier maps to Ok via a one-shot text response.
    ///
    /// We can't easily intercept the inner AgentLoop's provider from outside
    /// run_worker without changing the public API. So this test focuses on
    /// the parts run_worker fully owns: event emission + result shape +
    /// see_prior injection. End-to-end with a real LLM lives in the
    /// integration test (#[ignore]'d).
    #[tokio::test]
    async fn start_event_carries_worker_id_model_and_prompt() {
        let (tx, mut rx) = mpsc::channel::<AgentEvent>(32);
        let tmp = std::env::temp_dir();
        let ctx = make_ctx(Provider::OpenAI, tx, tmp);
        let spec = WorkerSpec {
            id: "w1".into(),
            model: "gpt-test".into(),
            prompt: "explore the auth files".into(),
            see_prior: SeePrior::Keyword(SeePriorKeyword::None),
        };

        // Spawn run_worker; we expect the AutoWorkerStart event to land on
        // the channel before the child loop completes (or fails) — and even
        // if the LLM call fails (no real API at http://localhost), the start
        // event still goes out first, which is the contract we're pinning.
        let handle = tokio::spawn(async move { run_worker(&spec, &[], &ctx).await });

        // First event must be AutoWorkerStart with the spec's identity.
        let first = tokio::time::timeout(
            std::time::Duration::from_secs(5),
            rx.recv(),
        )
        .await
        .expect("AutoWorkerStart must arrive within 5s")
        .expect("channel closed before start event");
        match first {
            AgentEvent::AutoWorkerStart { worker_id, model, prompt, run_id, session_id } => {
                assert_eq!(worker_id, "w1");
                assert_eq!(model, "gpt-test");
                assert_eq!(prompt, "explore the auth files");
                assert_eq!(run_id, "run-1");
                assert_eq!(session_id, "parent-session");
            }
            other => panic!("expected AutoWorkerStart, got {other:?}"),
        }

        // Wait for the worker to finish (it will fail because the LlmClient
        // tries to hit http://localhost). We don't care about the failure
        // mode — only that the result shape is well-formed and the end
        // event fires.
        let result = handle.await.expect("worker task panicked");
        assert_eq!(result.id, "w1");
        assert_eq!(result.model, "gpt-test");
        assert_eq!(result.prompt, "explore the auth files");
        assert!(result.status.is_hard_failure(), "localhost LLM should fail");

        // Drain the rest of the events; AutoWorkerEnd must be in there.
        let mut saw_end = false;
        while let Ok(Some(ev)) = tokio::time::timeout(
            std::time::Duration::from_millis(200),
            rx.recv(),
        )
        .await
        {
            if matches!(ev, AgentEvent::AutoWorkerEnd { .. }) {
                saw_end = true;
                break;
            }
        }
        assert!(saw_end, "AutoWorkerEnd must be emitted even on failure");
    }

    /// Headless end-to-end exercising the full classify→emit path against
    /// MockLlm. Builds an AgentLoop directly (bypassing run_worker's internal
    /// `LlmClient::new`) so we can prove the loop-side contract: a one-shot
    /// text response → AgentResult::Done → would map to WorkerStatus::Ok.
    /// This is the contract run_worker depends on; if it breaks, run_worker
    /// breaks.
    #[tokio::test]
    async fn mocklm_one_shot_response_yields_done() {
        use crate::tool::{ToolMode, ToolRegistry};

        let tmp = tempfile::tempdir().unwrap();
        let (tx, _rx) = mpsc::channel::<AgentEvent>(32);
        let cancel = CancellationToken::new();
        let mock = MockLlm::new(vec![Ok(text_response("done — auth flow uses bcrypt + sessions"))]);
        let config = AgentConfig {
            llm: LlmClientConfig {
                provider: Provider::OpenAI,
                base_url: "http://localhost".into(),
                model: "mock".into(),
                api_key: String::new(),
                temperature: None,
                max_completion_tokens: None,
                extra_headers: vec![],
                thinking: None,
                disable_cache_control: true,
                policy: LlmPolicy::default(),
            },
            working_dir: tmp.path().to_path_buf(),
            mode: ToolMode::Coding,
            max_iterations: 5,
            system_prompt: None,
            retry_config: RetryConfig::default(),
            compaction_config: CompactionConfig::default(),
            compaction_llm: None,
            context_engine: None,
            context_engine_repo_path: None,
            skills: None,
            subagents: None,
            subagent_inheritance: None,
            checkpoint_dir: None,
            tool_policy: ToolPolicy::default(),
        };
        let registry = ToolRegistry::for_mode(ToolMode::Coding, None, None);
        let mut loop_ = AgentLoop::with_provider(
            config,
            Box::new(mock),
            registry,
            cancel,
            tx,
            "child-session".into(),
        );
        let result = loop_
            .run(ChatMessage::user("describe the auth flow"))
            .await
            .expect("loop should succeed");
        let (summary, status) = classify_outcome(Ok(result));
        assert!(summary.contains("bcrypt"));
        assert_eq!(status, WorkerStatus::Ok);
    }
}
