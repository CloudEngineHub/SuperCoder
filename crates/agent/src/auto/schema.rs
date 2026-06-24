//! JSON schema for the `submit_plan` tool. The orchestrator is forced to call
//! this tool as its only allowed action (forced `tool_choice` in P2), making
//! the plan structure provider-validated on both Anthropic and OpenAI.
//!
//! Canonical schema definition: PHASE_AUTO_MODE.md "Orchestrator output schema".

use serde_json::{json, Value};

use crate::llm::types::{FunctionDefinition, ToolDefinition};

pub const SUBMIT_PLAN_TOOL_NAME: &str = "submit_plan";

/// Build the `submit_plan` ToolDefinition. Used by `orchestrator::generate_plan`
/// (P2) when invoking the orchestrator LLM with forced tool_choice.
pub fn submit_plan_tool() -> ToolDefinition {
    ToolDefinition {
        type_: "function".into(),
        function: FunctionDefinition {
            name: SUBMIT_PLAN_TOOL_NAME.into(),
            description: Some(
                "Submit the worker plan for this task. The plan is an ordered \
                 list of workers; each worker has a model id, a natural-language \
                 prompt, and a see_prior field declaring which prior worker \
                 outputs it may consume."
                    .into(),
            ),
            parameters: Some(submit_plan_input_schema()),
        },
        cache_control: None,
    }
}

/// Just the JSON Schema for the tool's input. Split out so tests / docs can
/// inspect it without going through the ToolDefinition wrapper.
pub fn submit_plan_input_schema() -> Value {
    json!({
        "type": "object",
        "additionalProperties": false,
        "required": ["reasoning", "plan"],
        "properties": {
            "reasoning": {
                "type": "string",
                "description": "Brief why-this-plan rationale (1-3 sentences)."
            },
            "plan": {
                "type": "array",
                "minItems": 1,
                "items": {
                    "type": "object",
                    "additionalProperties": false,
                    "required": ["id", "model", "prompt", "see_prior"],
                    "properties": {
                        "id":     { "type": "string", "pattern": "^w[0-9]+$" },
                        "model":  { "type": "string" },
                        "prompt": { "type": "string", "minLength": 1 },
                        "see_prior": {
                            "oneOf": [
                                { "type": "string", "enum": ["none", "all"] },
                                { "type": "array",  "items": { "type": "string" } }
                            ]
                        }
                    }
                }
            }
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn submit_plan_tool_has_correct_name_and_schema() {
        let tool = submit_plan_tool();
        assert_eq!(tool.type_, "function");
        assert_eq!(tool.function.name, SUBMIT_PLAN_TOOL_NAME);
        assert!(tool.function.description.is_some());
        let params = tool.function.parameters.expect("schema present");
        // Sanity: schema has the top-level required fields the orchestrator
        // must return.
        let required = params["required"].as_array().expect("required array");
        let names: Vec<&str> = required.iter().filter_map(|v| v.as_str()).collect();
        assert!(names.contains(&"reasoning"));
        assert!(names.contains(&"plan"));
    }

    #[test]
    fn submit_plan_tool_serializes_as_provider_function_shape() {
        // The wire format expected by both Anthropic (after translate_tool) and
        // OpenAI (chat-completions tools array). If this breaks the LLM client
        // would silently reject the tool.
        let tool = submit_plan_tool();
        let v = serde_json::to_value(&tool).unwrap();
        assert_eq!(v["type"], "function");
        assert_eq!(v["function"]["name"], SUBMIT_PLAN_TOOL_NAME);
        assert!(v["function"]["parameters"]["properties"]["plan"].is_object());
        assert!(v.get("cache_control").is_none(), "cache_control must be omitted when None");
    }

    #[test]
    fn submit_plan_schema_accepts_valid_example_payload() {
        // The example payload from PHASE_AUTO_MODE.md must structurally match
        // the schema. We don't run a JSON Schema validator here (no dep);
        // instead we deserialize as Plan, which validates the Rust-side
        // contract that mirrors the schema.
        let payload = serde_json::json!({
            "reasoning": "Two-stage decomposition.",
            "plan": [
                {"id": "w1", "model": "haiku", "prompt": "explore",  "see_prior": "none"},
                {"id": "w2", "model": "sonnet", "prompt": "design",  "see_prior": ["w1"]},
                {"id": "w3", "model": "sonnet", "prompt": "synthesize", "see_prior": "all"}
            ]
        });
        // The Plan type doesn't have a `version` in the tool payload — version
        // is assigned by the executor at insertion time. So we deserialize
        // just the inner shape.
        let workers = payload["plan"].as_array().unwrap();
        for w in workers {
            let _: crate::auto::WorkerSpec = serde_json::from_value(w.clone()).unwrap();
        }
    }
}
