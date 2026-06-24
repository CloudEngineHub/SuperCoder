//! Orchestrator: build the prompt, call the orchestrator LLM with forced
//! `submit_plan` tool use, parse the result into a `Plan`.
//!
//! Two layers:
//!   - **Pure**: `build_orchestrator_messages`, `parse_plan_from_tool_input`,
//!     and the formatters. No IO; unit-tested exhaustively.
//!   - **IO**: `generate_plan` — direct reqwest calls per provider. Not
//!     unit-tested (lives behind the live integration test).
//!
//! IO design choice: we do NOT route through `LlmClient::chat_completion`.
//! The orchestrator is a one-shot, non-streaming call with a forced tool
//! choice — different enough from the streaming agent-loop path that
//! threading a `force_tool` parameter through the trait + every mock + every
//! call site felt heavier than just building the body directly here.
//! Provider differences (URL, auth header, tool_choice shape, response shape)
//! are isolated in `call_anthropic` and `call_openai`.
//!
//! Phase: P2 (PHASE_AUTO_MODE.md). Executor (P3) is the only intended caller.

use std::time::Duration;

use serde_json::{json, Value};
use tokio_util::sync::CancellationToken;

use crate::auto::schema::{submit_plan_tool, SUBMIT_PLAN_TOOL_NAME};
use crate::auto::types::{Plan, WorkerPoolEntry, WorkerResult, WorkerSpec};
use crate::llm::anthropic::ANTHROPIC_VERSION;
use crate::llm::{LlmClientConfig, Provider};

/// System prompt template with `{{worker_pool_listing}}` placeholder.
/// Loaded at compile time.
const SYSTEM_PROMPT_TEMPLATE: &str = include_str!("prompts/orchestrator.txt");

/// Default cap on output tokens for the orchestrator call. The plan itself is
/// usually <2KB; this leaves room for the reasoning field and any unforeseen
/// growth. Lower than typical agent calls because the orchestrator is
/// one-shot.
const DEFAULT_MAX_TOKENS: u32 = 4096;

/// Number of parse retries on top of the initial attempt. Separate from the
/// REPLAN budget (which counts orchestrator re-invocations after worker
/// failures). With forced tool_choice + the input_schema, parse failures
/// should be rare; this is just a safety net.
const PARSE_RETRY_BUDGET: u32 = 2;

/// Network timeout for one orchestrator HTTP request. Orchestrator calls are
/// short — most plans should arrive within seconds. 60s is generous; past that
/// we bail rather than wedge the executor.
const REQUEST_TIMEOUT: Duration = Duration::from_secs(60);

/// Everything the orchestrator needs to know to produce a plan.
///
/// Construction is the executor's job (P3). On the initial call, `prior_results`
/// is empty and `replan_reason` is None. On a reactive replan, both are populated.
#[derive(Debug, Clone)]
pub struct OrchestratorRequest<'a> {
    /// The user's original task verbatim.
    pub task: &'a str,
    /// User-curated worker pool (Settings → Auto → Worker pool).
    pub worker_pool: &'a [WorkerPoolEntry],
    /// Outputs of workers that already executed (empty on initial call).
    pub prior_results: &'a [WorkerResult],
    /// Why we're being re-invoked. Present iff this is a replan.
    pub replan_reason: Option<&'a str>,
    /// Version to stamp on the produced Plan. 1 for initial; increments per replan.
    pub plan_version: u32,
}

/// Errors surfaced by `generate_plan`. The executor catches these and may
/// emit `AutoFailed` (no silent fallback per locked design).
#[derive(Debug, thiserror::Error)]
pub enum OrchestratorError {
    #[error("worker pool is empty — orchestrator cannot pick from zero models")]
    EmptyWorkerPool,
    #[error("orchestrator returned an unparseable plan after {attempts} attempt(s); last error: {message}")]
    Parse { attempts: u32, message: String },
    #[error("orchestrator HTTP error: {0}")]
    Http(String),
    #[error("orchestrator call cancelled")]
    Cancelled,
    #[error("provider returned no submit_plan tool_use: {0}")]
    NoToolUse(String),
}

// ── Pure: prompt building ─────────────────────────────────────────────────

/// Build the two-message conversation handed to the orchestrator LLM:
///   1. system: the orchestrator instructions with worker pool injected
///   2. user: the task plus (on replan) prior outputs + failure context
///
/// Returns plain `(system, user)` strings so the IO layer can package them
/// per-provider (Anthropic puts system at the top level; OpenAI inlines it
/// as the first message).
pub fn build_orchestrator_messages(req: &OrchestratorRequest) -> (String, String) {
    let system = format_system_prompt(req.worker_pool);
    let user = format_user_message(req);
    (system, user)
}

fn format_system_prompt(pool: &[WorkerPoolEntry]) -> String {
    let listing = if pool.is_empty() {
        "(empty — caller must reject before this point)".to_string()
    } else {
        pool.iter()
            .map(|e| format!("- `{}` ({}) — {}", e.model, e.provider_id, e.description))
            .collect::<Vec<_>>()
            .join("\n")
    };
    SYSTEM_PROMPT_TEMPLATE.replace("{{worker_pool_listing}}", &listing)
}

fn format_user_message(req: &OrchestratorRequest) -> String {
    // Initial call: just the task.
    if req.prior_results.is_empty() && req.replan_reason.is_none() {
        return format!("# Task\n\n{}\n", req.task);
    }

    // Replan: surface task, prior outputs, and the failure context.
    let mut s = String::new();
    s.push_str("# Task\n\n");
    s.push_str(req.task);
    s.push_str("\n\n");

    if !req.prior_results.is_empty() {
        s.push_str("# Prior worker outputs\n\n");
        for r in req.prior_results {
            s.push_str(&format!(
                "## `{}` (model: `{}`) — status: {}\n\n{}\n\n",
                r.id,
                r.model,
                worker_status_short(&r.status),
                r.summary,
            ));
        }
    }

    if let Some(reason) = req.replan_reason {
        s.push_str("# Replan reason\n\n");
        s.push_str(reason);
        s.push_str(
            "\n\nRevise the REMAINING plan. Workers already completed (status ok) \
             will not re-execute. Add new workers (`w<next>+`) or reissue the failed \
             worker with a corrected prompt. Reuse a failed worker's id if you want \
             it to replace the failed attempt.\n",
        );
    }

    s
}

fn worker_status_short(status: &crate::auto::types::WorkerStatus) -> &'static str {
    use crate::auto::types::WorkerStatus::*;
    match status {
        Ok => "ok",
        MaxIterations => "max_iterations",
        ToolError { .. } => "tool_error",
        LlmError { .. } => "llm_error",
        Cancelled => "cancelled",
    }
}

// ── Pure: parse the tool input as a Plan ──────────────────────────────────

/// Deserialize the orchestrator's `submit_plan` tool input into a `Plan`.
/// `version` is supplied by the caller — the orchestrator doesn't know which
/// plan version it's producing; only the executor maintains that counter.
///
/// Returns Err with a human-readable message on shape mismatch. The
/// orchestrator IO layer turns this into a parse retry; if all retries
/// exhaust, the executor surfaces `OrchestratorError::Parse`.
pub fn parse_plan_from_tool_input(input: &Value, version: u32) -> Result<Plan, String> {
    let reasoning = input
        .get("reasoning")
        .and_then(|v| v.as_str())
        .ok_or_else(|| "missing `reasoning` field (must be string)".to_string())?;

    let plan_arr = input
        .get("plan")
        .and_then(|v| v.as_array())
        .ok_or_else(|| "missing `plan` field (must be array)".to_string())?;

    if plan_arr.is_empty() {
        return Err("`plan` array is empty; orchestrator must return at least one worker".into());
    }

    let workers: Vec<WorkerSpec> = plan_arr
        .iter()
        .enumerate()
        .map(|(i, w)| {
            serde_json::from_value::<WorkerSpec>(w.clone())
                .map_err(|e| format!("worker[{i}] shape invalid: {e}"))
        })
        .collect::<Result<Vec<_>, String>>()?;

    Ok(Plan {
        version,
        reasoning: reasoning.to_string(),
        workers,
    })
}

// ── IO: generate_plan ─────────────────────────────────────────────────────

/// Run the orchestrator end-to-end: build prompt, call LLM with forced
/// `submit_plan`, parse result. Retries on malformed output up to
/// `PARSE_RETRY_BUDGET` times (separate from the executor's replan budget).
///
/// `llm` is the orchestrator's LLM config — the user picks this in Settings →
/// Auto → Orchestrator model. It is NOT the same client as the workers use;
/// the orchestrator typically wants a strong reasoning model regardless of
/// what workers use.
pub async fn generate_plan(
    llm: &LlmClientConfig,
    req: &OrchestratorRequest<'_>,
    cancel: Option<&CancellationToken>,
) -> Result<Plan, OrchestratorError> {
    if req.worker_pool.is_empty() {
        return Err(OrchestratorError::EmptyWorkerPool);
    }

    let (system, user) = build_orchestrator_messages(req);
    let http = build_http_client()?;

    let mut last_parse_err: Option<String> = None;
    for attempt in 0..=PARSE_RETRY_BUDGET {
        let raw_input = match llm.provider {
            Provider::Anthropic => call_anthropic(&http, llm, &system, &user, cancel).await?,
            Provider::OpenAI => call_openai(&http, llm, &system, &user, cancel).await?,
        };

        match parse_plan_from_tool_input(&raw_input, req.plan_version) {
            Ok(plan) => return Ok(plan),
            Err(e) => {
                log::warn!(
                    "[auto::orchestrator] parse attempt {} failed: {e}",
                    attempt + 1
                );
                last_parse_err = Some(e);
                // Falls through to next iteration; final iteration's Err
                // exits the loop via the `attempts` counter below.
            }
        }
    }

    Err(OrchestratorError::Parse {
        attempts: PARSE_RETRY_BUDGET + 1,
        message: last_parse_err.unwrap_or_else(|| "unknown parse error".to_string()),
    })
}

fn build_http_client() -> Result<reqwest::Client, OrchestratorError> {
    reqwest::Client::builder()
        .timeout(REQUEST_TIMEOUT)
        .build()
        .map_err(|e| OrchestratorError::Http(format!("build http client: {e}")))
}

// ── Anthropic IO ──

async fn call_anthropic(
    http: &reqwest::Client,
    llm: &LlmClientConfig,
    system: &str,
    user: &str,
    cancel: Option<&CancellationToken>,
) -> Result<Value, OrchestratorError> {
    let tool = submit_plan_tool();
    let body = json!({
        "model": llm.model,
        "max_tokens": llm.max_completion_tokens.unwrap_or(DEFAULT_MAX_TOKENS),
        "system": system,
        "messages": [
            {"role": "user", "content": user}
        ],
        "tools": [{
            "name": tool.function.name,
            "description": tool.function.description,
            "input_schema": tool.function.parameters,
        }],
        "tool_choice": {"type": "tool", "name": SUBMIT_PLAN_TOOL_NAME},
        "stream": false,
    });

    let url = format!("{}/v1/messages", llm.base_url.trim_end_matches('/'));
    let mut req = http
        .post(&url)
        .header("content-type", "application/json")
        .header("anthropic-version", ANTHROPIC_VERSION)
        .json(&body);
    if !llm.api_key.is_empty() {
        req = req.header("x-api-key", &llm.api_key);
    }
    for (k, v) in &llm.extra_headers {
        req = req.header(k.as_str(), v.as_str());
    }

    let resp = send_with_cancel(req, cancel).await?;
    let value = parse_http_response(resp).await?;
    extract_anthropic_tool_input(&value)
}

fn extract_anthropic_tool_input(resp: &Value) -> Result<Value, OrchestratorError> {
    let content = resp
        .get("content")
        .and_then(|v| v.as_array())
        .ok_or_else(|| OrchestratorError::NoToolUse("response missing `content` array".into()))?;
    for block in content {
        let is_tool_use = block.get("type").and_then(|v| v.as_str()) == Some("tool_use");
        let is_submit = block.get("name").and_then(|v| v.as_str()) == Some(SUBMIT_PLAN_TOOL_NAME);
        if is_tool_use && is_submit {
            return block
                .get("input")
                .cloned()
                .ok_or_else(|| OrchestratorError::NoToolUse("tool_use block has no `input`".into()));
        }
    }
    Err(OrchestratorError::NoToolUse(
        "no submit_plan tool_use in response (model may have refused / replied as text)".into(),
    ))
}

// ── OpenAI IO ──

async fn call_openai(
    http: &reqwest::Client,
    llm: &LlmClientConfig,
    system: &str,
    user: &str,
    cancel: Option<&CancellationToken>,
) -> Result<Value, OrchestratorError> {
    let body = json!({
        "model": llm.model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user",   "content": user}
        ],
        "tools": [submit_plan_tool()],
        "tool_choice": {"type": "function", "function": {"name": SUBMIT_PLAN_TOOL_NAME}},
        "stream": false,
    });

    let url = format!("{}/chat/completions", llm.base_url.trim_end_matches('/'));
    let mut req = http
        .post(&url)
        .header("content-type", "application/json")
        .json(&body);
    if !llm.api_key.is_empty() {
        req = req.bearer_auth(&llm.api_key);
    }
    for (k, v) in &llm.extra_headers {
        req = req.header(k.as_str(), v.as_str());
    }

    let resp = send_with_cancel(req, cancel).await?;
    let value = parse_http_response(resp).await?;
    extract_openai_tool_input(&value)
}

fn extract_openai_tool_input(resp: &Value) -> Result<Value, OrchestratorError> {
    let calls = resp
        .pointer("/choices/0/message/tool_calls")
        .and_then(|v| v.as_array())
        .ok_or_else(|| {
            OrchestratorError::NoToolUse("response missing /choices/0/message/tool_calls".into())
        })?;
    for call in calls {
        let name = call.pointer("/function/name").and_then(|v| v.as_str());
        if name == Some(SUBMIT_PLAN_TOOL_NAME) {
            let args_str = call
                .pointer("/function/arguments")
                .and_then(|v| v.as_str())
                .ok_or_else(|| {
                    OrchestratorError::NoToolUse(
                        "submit_plan tool_call has no string `function.arguments`".into(),
                    )
                })?;
            return serde_json::from_str(args_str).map_err(|e| {
                OrchestratorError::NoToolUse(format!(
                    "submit_plan arguments are not valid JSON: {e}"
                ))
            });
        }
    }
    Err(OrchestratorError::NoToolUse(
        "no submit_plan in tool_calls (model may have refused)".into(),
    ))
}

// ── HTTP helpers ──

async fn send_with_cancel(
    req: reqwest::RequestBuilder,
    cancel: Option<&CancellationToken>,
) -> Result<reqwest::Response, OrchestratorError> {
    let fut = req.send();
    match cancel {
        Some(token) => tokio::select! {
            r = fut => r.map_err(|e| OrchestratorError::Http(e.to_string())),
            _ = token.cancelled() => Err(OrchestratorError::Cancelled),
        },
        None => fut.await.map_err(|e| OrchestratorError::Http(e.to_string())),
    }
}

async fn parse_http_response(resp: reqwest::Response) -> Result<Value, OrchestratorError> {
    let status = resp.status();
    let text = resp
        .text()
        .await
        .map_err(|e| OrchestratorError::Http(format!("read response body: {e}")))?;
    if !status.is_success() {
        return Err(OrchestratorError::Http(format!(
            "HTTP {status}: {}",
            &text[..text.len().min(500)]
        )));
    }
    serde_json::from_str(&text)
        .map_err(|e| OrchestratorError::Http(format!("parse response JSON: {e}")))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::auto::types::{SeePrior, SeePriorKeyword, WorkerStatus};

    fn pool_one() -> Vec<WorkerPoolEntry> {
        vec![
            WorkerPoolEntry {
                provider_id: "anthropic".into(),
                model: "claude-haiku-4-5".into(),
                description: "fast + cheap, best for reads and verification".into(),
            },
            WorkerPoolEntry {
                provider_id: "anthropic".into(),
                model: "claude-sonnet-4-6".into(),
                description: "strong reasoning, best for synthesis and edits".into(),
            },
        ]
    }

    fn wr(id: &str, summary: &str, status: WorkerStatus) -> WorkerResult {
        WorkerResult {
            id: id.into(),
            model: "claude-sonnet-4-6".into(),
            prompt: "p".into(),
            summary: summary.into(),
            tool_count: 1,
            cost_cents: 0,
            status,
        }
    }

    // ── format_system_prompt ──

    #[test]
    fn system_prompt_lists_pool_with_descriptions() {
        let s = format_system_prompt(&pool_one());
        assert!(s.contains("claude-haiku-4-5"));
        assert!(s.contains("fast + cheap"));
        assert!(s.contains("claude-sonnet-4-6"));
        assert!(s.contains("strong reasoning"));
        assert!(!s.contains("{{worker_pool_listing}}"), "template placeholder must be substituted");
    }

    #[test]
    fn system_prompt_handles_empty_pool_with_fallback_string() {
        // generate_plan should reject before we get here, but the formatter
        // must not panic on an empty pool.
        let s = format_system_prompt(&[]);
        assert!(s.contains("(empty"));
        assert!(!s.contains("{{worker_pool_listing}}"));
    }

    // ── format_user_message ──

    #[test]
    fn user_message_initial_call_is_just_the_task() {
        let req = OrchestratorRequest {
            task: "refactor auth",
            worker_pool: &pool_one(),
            prior_results: &[],
            replan_reason: None,
            plan_version: 1,
        };
        let msg = format_user_message(&req);
        assert!(msg.starts_with("# Task"));
        assert!(msg.contains("refactor auth"));
        assert!(!msg.contains("Prior worker outputs"));
        assert!(!msg.contains("Replan reason"));
    }

    #[test]
    fn user_message_replan_includes_prior_outputs_and_reason() {
        let priors = vec![
            wr("w1", "found 12 files", WorkerStatus::Ok),
            wr("w2", "tool failed: nosuch", WorkerStatus::ToolError { message: "x".into() }),
        ];
        let req = OrchestratorRequest {
            task: "refactor auth",
            worker_pool: &pool_one(),
            prior_results: &priors,
            replan_reason: Some("worker w2 failed: tool_error"),
            plan_version: 2,
        };
        let msg = format_user_message(&req);
        assert!(msg.contains("# Prior worker outputs"));
        assert!(msg.contains("`w1`"));
        assert!(msg.contains("status: ok"));
        assert!(msg.contains("`w2`"));
        assert!(msg.contains("status: tool_error"));
        assert!(msg.contains("# Replan reason"));
        assert!(msg.contains("w2 failed"));
    }

    // ── parse_plan_from_tool_input ──

    #[test]
    fn parse_accepts_well_formed_plan() {
        let raw = json!({
            "reasoning": "two-stage decomposition",
            "plan": [
                {"id": "w1", "model": "claude-haiku-4-5", "prompt": "explore",  "see_prior": "none"},
                {"id": "w2", "model": "claude-sonnet-4-6", "prompt": "design",  "see_prior": ["w1"]}
            ]
        });
        let plan = parse_plan_from_tool_input(&raw, 1).expect("plan should parse");
        assert_eq!(plan.version, 1);
        assert_eq!(plan.reasoning, "two-stage decomposition");
        assert_eq!(plan.workers.len(), 2);
        assert_eq!(plan.workers[0].id, "w1");
        assert!(matches!(
            plan.workers[0].see_prior,
            SeePrior::Keyword(SeePriorKeyword::None)
        ));
        assert!(matches!(
            &plan.workers[1].see_prior,
            SeePrior::Specific(ids) if ids == &["w1"]
        ));
    }

    #[test]
    fn parse_rejects_missing_reasoning() {
        let raw = json!({"plan": [{"id":"w1","model":"m","prompt":"p","see_prior":"none"}]});
        let err = parse_plan_from_tool_input(&raw, 1).unwrap_err();
        assert!(err.contains("reasoning"), "got: {err}");
    }

    #[test]
    fn parse_rejects_empty_plan_array() {
        let raw = json!({"reasoning": "r", "plan": []});
        let err = parse_plan_from_tool_input(&raw, 1).unwrap_err();
        assert!(err.contains("empty"), "got: {err}");
    }

    #[test]
    fn parse_rejects_worker_missing_required_field() {
        let raw = json!({
            "reasoning": "r",
            "plan": [
                {"id": "w1", "model": "m", "prompt": "p"}  // missing see_prior
            ]
        });
        let err = parse_plan_from_tool_input(&raw, 1).unwrap_err();
        assert!(err.contains("worker[0]"), "got: {err}");
    }

    #[test]
    fn parse_stamps_caller_provided_version() {
        let raw = json!({
            "reasoning": "r",
            "plan": [{"id":"w1","model":"m","prompt":"p","see_prior":"all"}]
        });
        let plan = parse_plan_from_tool_input(&raw, 7).unwrap();
        assert_eq!(plan.version, 7, "version must come from caller, not from input");
    }

    // ── extract_anthropic_tool_input ──

    #[test]
    fn anthropic_extracts_input_from_tool_use_block() {
        let resp = json!({
            "content": [
                {"type": "text", "text": "ignored"},
                {"type": "tool_use", "id": "tu1", "name": "submit_plan", "input": {"reasoning": "r", "plan": []}}
            ]
        });
        let v = extract_anthropic_tool_input(&resp).unwrap();
        assert_eq!(v["reasoning"], "r");
    }

    #[test]
    fn anthropic_errors_when_no_tool_use() {
        let resp = json!({
            "content": [
                {"type": "text", "text": "I refuse"}
            ]
        });
        let err = extract_anthropic_tool_input(&resp).unwrap_err();
        assert!(matches!(err, OrchestratorError::NoToolUse(_)));
    }

    #[test]
    fn anthropic_errors_when_wrong_tool_name() {
        let resp = json!({
            "content": [
                {"type": "tool_use", "name": "other_tool", "input": {}}
            ]
        });
        let err = extract_anthropic_tool_input(&resp).unwrap_err();
        assert!(matches!(err, OrchestratorError::NoToolUse(_)));
    }

    // ── extract_openai_tool_input ──

    #[test]
    fn openai_extracts_arguments_from_tool_call() {
        let args_json = r#"{"reasoning":"r","plan":[]}"#;
        let resp = json!({
            "choices": [{
                "message": {
                    "tool_calls": [
                        {"id": "c1", "function": {"name": "submit_plan", "arguments": args_json}}
                    ]
                }
            }]
        });
        let v = extract_openai_tool_input(&resp).unwrap();
        assert_eq!(v["reasoning"], "r");
    }

    #[test]
    fn openai_errors_when_arguments_is_invalid_json() {
        let resp = json!({
            "choices": [{
                "message": {
                    "tool_calls": [
                        {"id": "c1", "function": {"name": "submit_plan", "arguments": "not json"}}
                    ]
                }
            }]
        });
        let err = extract_openai_tool_input(&resp).unwrap_err();
        assert!(matches!(err, OrchestratorError::NoToolUse(_)));
    }

    #[test]
    fn openai_errors_when_no_tool_calls() {
        let resp = json!({"choices": [{"message": {"content": "plain text"}}]});
        let err = extract_openai_tool_input(&resp).unwrap_err();
        assert!(matches!(err, OrchestratorError::NoToolUse(_)));
    }

    // ── generate_plan: rejects empty pool without hitting the network ──

    #[tokio::test]
    async fn generate_plan_rejects_empty_worker_pool() {
        let llm = LlmClientConfig {
            provider: Provider::Anthropic,
            base_url: "http://localhost".into(),
            model: "x".into(),
            api_key: String::new(),
            temperature: None,
            max_completion_tokens: None,
            extra_headers: vec![],
            thinking: None,
            disable_cache_control: true,
            policy: Default::default(),
        };
        let req = OrchestratorRequest {
            task: "anything",
            worker_pool: &[],
            prior_results: &[],
            replan_reason: None,
            plan_version: 1,
        };
        let err = generate_plan(&llm, &req, None).await.unwrap_err();
        assert!(matches!(err, OrchestratorError::EmptyWorkerPool));
    }
}
