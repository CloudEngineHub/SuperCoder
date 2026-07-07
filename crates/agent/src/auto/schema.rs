//! JSON schema for the `submit_plan` tool. The orchestrator is forced to call
//! this tool as its only allowed action (forced `tool_choice` in P2), making
//! the plan structure provider-validated on both Anthropic and OpenAI.
//!
//! Canonical schema definition: PHASE_AUTO_MODE.md "Orchestrator output schema".

use serde_json::{json, Value};

use crate::llm::types::{FunctionDefinition, ToolDefinition};

pub const SUBMIT_PLAN_TOOL_NAME: &str = "submit_plan";

/// Build the `submit_plan` ToolDefinition for the **Anthropic** orchestrator
/// path. Schema is rich (uses `pattern`, `minLength`, `minItems`, `oneOf`)
/// because Anthropic treats `input_schema` as a hint, not server-enforced —
/// extra constraints help the model produce well-formed output without
/// causing rejections.
pub fn submit_plan_tool() -> ToolDefinition {
    ToolDefinition {
        type_: "function".into(),
        function: FunctionDefinition {
            name: SUBMIT_PLAN_TOOL_NAME.into(),
            description: Some(SUBMIT_PLAN_DESCRIPTION.into()),
            parameters: Some(submit_plan_input_schema()),
            strict: None,
        },
        cache_control: None,
    }
}

/// Build the `submit_plan` ToolDefinition for the **OpenAI** orchestrator
/// path with `strict: true`. Uses the stripped schema that conforms to
/// OpenAI's structured-outputs subset (no `pattern` / `minLength` /
/// `minItems`; `additionalProperties: false` at every level; all
/// `properties` keys appear in `required`; `oneOf` → `anyOf`). With strict
/// mode the provider GUARANTEES the tool_call arguments match the schema
/// via constrained decoding — no `see_prior`-style drift.
pub fn submit_plan_tool_strict() -> ToolDefinition {
    ToolDefinition {
        type_: "function".into(),
        function: FunctionDefinition {
            name: SUBMIT_PLAN_TOOL_NAME.into(),
            description: Some(SUBMIT_PLAN_DESCRIPTION.into()),
            parameters: Some(submit_plan_input_schema_strict()),
            strict: Some(true),
        },
        cache_control: None,
    }
}

const SUBMIT_PLAN_DESCRIPTION: &str =
    "Submit the worker plan for this task. The plan is an ordered list of \
     workers; each worker has a model id, a natural-language prompt, and a \
     see_prior field declaring which prior worker outputs it may consume.";

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

/// Stripped schema compatible with OpenAI's structured-outputs strict mode:
/// drops `pattern`, `minLength`, `minItems`; swaps `oneOf` → `anyOf`. The
/// stripped constraints are re-enforced by `parse_plan_from_tool_input` and
/// by the executor (e.g. an empty plan array is rejected at parse time).
pub fn submit_plan_input_schema_strict() -> Value {
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
                "items": {
                    "type": "object",
                    "additionalProperties": false,
                    "required": ["id", "model", "prompt", "see_prior"],
                    "properties": {
                        "id":     { "type": "string", "description": "Stable id like 'w1', 'w2', sequential starting at w1." },
                        "model":  { "type": "string" },
                        "prompt": { "type": "string" },
                        "see_prior": {
                            "anyOf": [
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
    fn strict_tool_emits_strict_true_and_drops_unsupported_constraints() {
        let tool = submit_plan_tool_strict();
        let v = serde_json::to_value(&tool).unwrap();
        assert_eq!(v["type"], "function");
        assert_eq!(v["function"]["name"], SUBMIT_PLAN_TOOL_NAME);
        assert_eq!(
            v["function"]["strict"], true,
            "OpenAI structured-outputs requires strict: true"
        );
        // Confirm the schema-subset rules:
        let plan = &v["function"]["parameters"]["properties"]["plan"];
        assert!(
            plan.get("minItems").is_none(),
            "minItems unsupported by OpenAI strict mode"
        );
        let item_props = &plan["items"]["properties"];
        assert!(
            item_props["id"].get("pattern").is_none(),
            "pattern unsupported by OpenAI strict mode"
        );
        assert!(
            item_props["prompt"].get("minLength").is_none(),
            "minLength unsupported by OpenAI strict mode"
        );
        assert!(
            item_props["see_prior"].get("oneOf").is_none(),
            "oneOf at property level unsupported; swap to anyOf"
        );
        assert!(
            item_props["see_prior"]["anyOf"].is_array(),
            "see_prior must use anyOf in strict mode"
        );
    }

    #[test]
    fn non_strict_tool_omits_strict_field_on_wire() {
        // Anthropic path: the field must not appear at all (skip_serializing_if).
        let tool = submit_plan_tool();
        let v = serde_json::to_value(&tool).unwrap();
        assert!(
            v["function"].get("strict").is_none(),
            "non-strict tool must omit `strict` on the wire so Anthropic doesn't choke"
        );
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
