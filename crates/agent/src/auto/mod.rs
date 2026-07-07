//! Auto mode — orchestrator-driven multi-model coding.
//!
//! See PHASE_AUTO_MODE.md for the locked design + phase plan. This crate
//! sub-module is architecturally distinct from `crate::subagents` (no
//! `SubagentRegistry` coupling) and reuses only the spawning skeleton.
//!
//! Phase status:
//!   P0 (current): types + schema only. No behavior.
//!   P1: `worker.rs` — ephemeral worker runtime.
//!   P2: `orchestrator.rs` — orchestrator LLM call + plan parsing (forced tool use).
//!   P3: `executor.rs` — sequencer + reactive replan loop.

pub mod executor;
pub mod orchestrator;
pub mod schema;
pub mod types;
pub mod worker;

pub use executor::{
    run as run_auto, AutoApprove, AutoStateListener, ExecutorConfig, LiveWorkerRunner,
    LlmPlanGenerator, PlanApprover, PlanGenerator, WorkerRunner,
};
pub use orchestrator::{
    build_orchestrator_messages, generate_plan, parse_plan_from_tool_input, OrchestratorError,
    OrchestratorRequest,
};
pub use schema::{submit_plan_input_schema, submit_plan_tool, SUBMIT_PLAN_TOOL_NAME};
pub use types::{
    AutoResult, AutoRun, AutoStatus, Plan, SeePrior, SeePriorKeyword, WorkerFailureContext,
    WorkerPoolEntry, WorkerResult, WorkerSpec, WorkerStatus,
};
pub use worker::{run_worker, WorkerContext};
