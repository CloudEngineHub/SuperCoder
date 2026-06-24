//! Public types for Auto mode (orchestrator + workers).
//!
//! Pure data definitions. No behavior. Logic lives in `orchestrator.rs` (P2),
//! `executor.rs` (P3), `worker.rs` (P1). See PHASE_AUTO_MODE.md for the full
//! design.

use serde::{Deserialize, Serialize};

/// One worker's specification, emitted by the orchestrator as part of a Plan.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct WorkerSpec {
    /// Stable id like "w1", "w2". Used in `see_prior` references.
    pub id: String,
    /// Model identifier the orchestrator picked from the eligible pool.
    pub model: String,
    /// Natural-language task for this worker. Becomes its user prompt.
    pub prompt: String,
    /// Which prior worker outputs (if any) get injected into this worker's prompt.
    pub see_prior: SeePrior,
}

/// Paper's `access_list`: which prior worker summaries this worker can see.
/// Serializes as either a keyword string (`"none"` / `"all"`) or an explicit
/// array of worker ids (`["w1", "w2"]`). Untagged so the orchestrator's JSON
/// output can use either form per the locked schema.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(untagged)]
pub enum SeePrior {
    Keyword(SeePriorKeyword),
    Specific(Vec<String>),
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum SeePriorKeyword {
    None,
    All,
}

/// A complete orchestrator-produced plan. One Plan per orchestrator call;
/// successive replans produce new Plans with incremented `version`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Plan {
    /// 1-indexed. Initial plan is version 1; each replan increments.
    pub version: u32,
    /// Brief why-this-plan rationale (1-3 sentences).
    pub reasoning: String,
    /// Workers in execution order.
    pub workers: Vec<WorkerSpec>,
}

/// Result of executing a single worker. Populated by `worker::run_worker` (P1).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct WorkerResult {
    pub id: String,
    pub model: String,
    pub prompt: String,
    /// Final summary text returned by the worker's agent loop.
    pub summary: String,
    pub tool_count: u32,
    pub cost_cents: u32,
    pub status: WorkerStatus,
}

/// Terminal status of a single worker. Hard-failure variants trigger reactive
/// replans (executor.rs, P3); non-hard variants pass through to the next worker.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum WorkerStatus {
    /// Worker completed normally.
    Ok,
    /// Worker hit max_iterations (100 per AgentConfig default).
    MaxIterations,
    /// A tool call returned an error the worker couldn't recover from.
    ToolError { message: String },
    /// Worker's underlying LLM call failed (network, 5xx, parse, etc).
    LlmError { message: String },
    /// User cancelled (or the parent task was cancelled).
    Cancelled,
}

impl WorkerStatus {
    /// True for statuses that trigger a reactive replan in the executor.
    pub fn is_hard_failure(&self) -> bool {
        matches!(
            self,
            WorkerStatus::MaxIterations
                | WorkerStatus::ToolError { .. }
                | WorkerStatus::LlmError { .. }
        )
    }
}

/// Persisted per-Auto-task state. One row per Auto invocation in the
/// `auto_runs` table (schema added in P4).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct AutoRun {
    pub id: String,
    pub session_id: String,
    pub status: AutoStatus,
    /// Ordered; index 0 is the initial plan, subsequent entries are replans.
    pub plan_versions: Vec<Plan>,
    pub worker_results: Vec<WorkerResult>,
    pub cost_cents: u32,
    pub replans_used: u32,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum AutoStatus {
    Planning,
    AwaitingApproval,
    Running,
    Done,
    Failed,
    Cancelled,
}

/// Terminal result of an Auto task. Returned by `executor::run` (P3).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum AutoResult {
    /// All workers ran successfully (possibly after replans).
    Done { summary: String, run: AutoRun },
    /// Replan budget exhausted; no silent fallback.
    Failed { reason: String, run: AutoRun },
    /// User cancelled mid-execution.
    Cancelled { run: AutoRun },
}

/// One entry in the user's curated worker pool (Settings → Auto → Worker pool).
/// `description` is fed verbatim to the orchestrator so it can pick wisely.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct WorkerPoolEntry {
    pub provider_id: String,
    pub model: String,
    /// User-written hint, e.g. "best for cheap exploration, weak on synthesis".
    pub description: String,
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn see_prior_keyword_serializes_as_string() {
        let none = serde_json::to_value(SeePrior::Keyword(SeePriorKeyword::None)).unwrap();
        assert_eq!(none, json!("none"));
        let all = serde_json::to_value(SeePrior::Keyword(SeePriorKeyword::All)).unwrap();
        assert_eq!(all, json!("all"));
    }

    #[test]
    fn see_prior_specific_serializes_as_array() {
        let s = SeePrior::Specific(vec!["w1".into(), "w3".into()]);
        let v = serde_json::to_value(&s).unwrap();
        assert_eq!(v, json!(["w1", "w3"]));
    }

    #[test]
    fn see_prior_roundtrips_all_forms() {
        for form in [
            json!("none"),
            json!("all"),
            json!([]),
            json!(["w1"]),
            json!(["w1", "w2"]),
        ] {
            let parsed: SeePrior = serde_json::from_value(form.clone()).unwrap();
            let back = serde_json::to_value(&parsed).unwrap();
            assert_eq!(back, form, "round-trip mismatch for {form}");
        }
    }

    #[test]
    fn worker_status_serializes_with_kind_tag() {
        assert_eq!(
            serde_json::to_value(WorkerStatus::Ok).unwrap(),
            json!({"kind": "ok"})
        );
        assert_eq!(
            serde_json::to_value(WorkerStatus::MaxIterations).unwrap(),
            json!({"kind": "max_iterations"})
        );
        assert_eq!(
            serde_json::to_value(WorkerStatus::ToolError { message: "boom".into() }).unwrap(),
            json!({"kind": "tool_error", "message": "boom"})
        );
    }

    #[test]
    fn is_hard_failure_classifies_correctly() {
        assert!(!WorkerStatus::Ok.is_hard_failure());
        assert!(!WorkerStatus::Cancelled.is_hard_failure());
        assert!(WorkerStatus::MaxIterations.is_hard_failure());
        assert!(WorkerStatus::ToolError { message: "x".into() }.is_hard_failure());
        assert!(WorkerStatus::LlmError { message: "x".into() }.is_hard_failure());
    }

    #[test]
    fn plan_full_roundtrip_matches_orchestrator_schema() {
        // Mirrors the example payload in PHASE_AUTO_MODE.md "Orchestrator
        // output schema". If this breaks, either the type or the doc moved
        // and the other needs to catch up.
        let raw = json!({
            "version": 1,
            "reasoning": "Two-stage: cheap exploration first, then expensive synthesis.",
            "workers": [
                {
                    "id": "w1",
                    "model": "claude-haiku-4-5",
                    "prompt": "Read all files matching src/auth/**.ts and summarize the auth flow.",
                    "see_prior": "none"
                },
                {
                    "id": "w2",
                    "model": "claude-sonnet-4-6",
                    "prompt": "Given the prior summary, design a JWT-based replacement.",
                    "see_prior": ["w1"]
                }
            ]
        });
        let plan: Plan = serde_json::from_value(raw.clone()).unwrap();
        assert_eq!(plan.workers.len(), 2);
        assert_eq!(plan.workers[0].id, "w1");
        assert!(matches!(plan.workers[0].see_prior, SeePrior::Keyword(SeePriorKeyword::None)));
        assert!(matches!(&plan.workers[1].see_prior, SeePrior::Specific(ids) if ids == &["w1"]));
        let back = serde_json::to_value(&plan).unwrap();
        assert_eq!(back, raw);
    }

    #[test]
    fn auto_status_serializes_snake_case() {
        assert_eq!(serde_json::to_value(AutoStatus::AwaitingApproval).unwrap(), json!("awaiting_approval"));
        assert_eq!(serde_json::to_value(AutoStatus::Failed).unwrap(), json!("failed"));
    }
}
