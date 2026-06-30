use serde::Serialize;

use crate::auto::{WorkerSpec, WorkerStatus};

/// Events emitted by the agent loop for UI consumption.
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type")]
pub enum AgentEvent {
    TextDelta {
        session_id: String,
        delta: String,
    },
    ThinkingDelta {
        session_id: String,
        delta: String,
    },
    ToolStart {
        session_id: String,
        tool_call_id: String,
        tool_name: String,
        args_summary: String,
    },
    ToolStatus {
        session_id: String,
        tool_call_id: String,
        status: String,
    },
    ToolEnd {
        session_id: String,
        tool_call_id: String,
        success: bool,
        summary: String,
        /// Files modified by this tool (for write/edit tools).
        #[serde(skip_serializing_if = "Option::is_none")]
        modified_files: Option<Vec<String>>,
    },
    Error {
        session_id: String,
        message: String,
        retrying: bool,
    },
    Done {
        session_id: String,
        summary: Option<String>,
    },
    Compaction {
        session_id: String,
    },
    /// Token usage update after each LLM call.
    TokenUsage {
        session_id: String,
        total_tokens: u32,
        /// Known/discovered context window. `None` for models with an unknown
        /// limit — the UI then shows the raw token count with no max/percentage.
        context_limit: Option<u32>,
        /// Tokens served from prompt cache this call (read tier).
        /// None when caching isn't active or the provider didn't report it.
        cache_read_tokens: Option<u32>,
        /// Tokens written to prompt cache this call (write tier, Anthropic).
        cache_creation_tokens: Option<u32>,
    },
    /// Fired at the end of each agent loop iteration (after all tool calls complete).
    /// Carries the turn number and list of files modified during this turn.
    TurnCompleted {
        session_id: String,
        turn_count: u32,
        modified_files: Vec<String>,
    },
    /// The agent is asking the user a clarifying question (yield).
    UserQuestionAsked {
        session_id: String,
        question: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        options: Option<Vec<String>>,
    },
    /// The agent updated its todo list.
    TodoUpdated {
        session_id: String,
        todos: Vec<TodoItem>,
    },
    /// The plan-mode agent saved an implementation plan (yield).
    PlanReady {
        session_id: String,
        plan: String,
        plan_path: String,
        project_path: String,
    },
    /// The `skill` tool loaded a skill body into the conversation.
    SkillLoaded {
        session_id: String,
        skill_name: String,
    },
    /// A `spawn_subagent` tool call has started a child AgentLoop. Emitted on
    /// the PARENT's event_tx. Per DEC-1, the child's stream is NOT forwarded
    /// to the parent UI — this event (plus SubagentEnd) is the sole parent-
    /// visible representation. `prompt_preview` is the first ~120 chars of
    /// the user-facing prompt, truncated for chip display.
    SubagentStart {
        session_id: String,
        parent_tool_call_id: String,
        child_session_id: String,
        subagent_name: String,
        prompt_preview: String,
    },
    /// The child AgentLoop finished. `success` is false on error or cancel;
    /// `summary` is the child's final assistant text (or an error message).
    SubagentEnd {
        session_id: String,
        parent_tool_call_id: String,
        child_session_id: String,
        success: bool,
        summary: String,
    },

    // ── Auto mode (orchestrator + workers) ──
    // See PHASE_AUTO_MODE.md "AgentEvent additions" for the canonical list.

    /// Auto: the orchestrator LLM call is in flight.
    AutoPlanning {
        session_id: String,
        run_id: String,
    },
    /// Auto: orchestrator emitted a (new or revised) plan. `version` matches
    /// `Plan::version` — initial plan is 1; replans increment.
    AutoPlan {
        session_id: String,
        run_id: String,
        version: u32,
        reasoning: String,
        plan: Vec<WorkerSpec>,
    },
    /// Auto: plan rendered to the user; executor is blocked on the Run button.
    AutoAwaitingApproval {
        session_id: String,
        run_id: String,
    },
    /// Auto: a worker has started executing.
    AutoWorkerStart {
        session_id: String,
        run_id: String,
        worker_id: String,
        model: String,
        prompt: String,
    },
    /// Auto: a worker finished (success or failure). Hard-failure statuses
    /// trigger a reactive replan in the executor.
    AutoWorkerEnd {
        session_id: String,
        run_id: String,
        worker_id: String,
        summary: String,
        cost_cents: u32,
        tool_count: u32,
        status: WorkerStatus,
    },
    /// Auto: a worker's child AgentLoop is about to execute a tool. Lets the
    /// UI render live tool activity inside the running worker card. Re-emitted
    /// by `auto::worker`'s drain task — the worker's own ToolStart event
    /// fires on its private channel, this is the parent-visible echo.
    AutoWorkerToolStart {
        session_id: String,
        run_id: String,
        worker_id: String,
        tool_call_id: String,
        tool_name: String,
        args_summary: String,
    },
    /// Auto: a worker's tool call finished. Paired with `AutoWorkerToolStart`.
    AutoWorkerToolEnd {
        session_id: String,
        run_id: String,
        worker_id: String,
        tool_call_id: String,
        success: bool,
        summary: String,
    },
    /// Auto: a worker failed and the orchestrator is being re-invoked.
    AutoReplan {
        session_id: String,
        run_id: String,
        reason: String,
    },
    /// Auto: replan budget exhausted. No silent fallback — user decides next move.
    AutoFailed {
        session_id: String,
        run_id: String,
        reason: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        last_worker_output: Option<String>,
    },
    /// Auto: all workers finished successfully (possibly after replans).
    AutoDone {
        session_id: String,
        run_id: String,
        summary: String,
        total_cost_cents: u32,
    },
}

/// A single todo item tracked by the agent.
#[derive(Debug, Clone, Serialize, serde::Deserialize)]
pub struct TodoItem {
    pub id: String,
    pub content: String,
    pub status: String,
}

/// Result returned by the agent loop when it finishes.
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type")]
pub enum AgentResult {
    /// Agent completed normally with a final summary.
    Done { summary: String },
    /// Agent is asking the user a clarifying question (yield from ask/plan mode).
    AskUser {
        question: String,
        options: Option<Vec<String>>,
    },
    /// Plan-mode agent saved a plan and yielded for user approval.
    PlanReady {
        plan: String,
        plan_path: String,
    },
}

impl AgentResult {
    /// Extract the summary from a Done result, panicking if it's not Done.
    /// Useful in tests.
    #[cfg(test)]
    pub fn unwrap_done(self) -> String {
        match self {
            AgentResult::Done { summary } => summary,
            other => panic!("Expected AgentResult::Done, got {:?}", other),
        }
    }
}
