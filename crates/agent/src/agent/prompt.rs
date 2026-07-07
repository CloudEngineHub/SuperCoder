use std::path::Path;

use crate::llm::types::CacheControl;
use crate::skills::SkillRegistry;
use crate::subagents::SubagentRegistry;
use crate::tool::ToolMode;

/// Raw prompt templates embedded at compile time from .txt files.
/// Templates contain the STATIC body only (no env section) plus a few inline
/// placeholders like `{{working_dir}}` that are session-stable.
const ASK_PROMPT_TEMPLATE: &str = include_str!("ask_prompt.txt");
const CODING_PROMPT_TEMPLATE: &str = include_str!("coding_prompt.txt");
const PLAN_PROMPT_TEMPLATE: &str = include_str!("plan_prompt.txt");

/// Injected after the static body when a context-engine index is available.
/// The base templates advertise only grep/glob and instruct the model to "use
/// glob and grep to find what you need", so without this block models never reach
/// for the `codebase_search`/`codebase_graph` tools (registered by
/// `ToolRegistry::for_mode`) even though they're more capable. This block
/// advertises the index tools and steers exploration to `codebase_search` first.
const INDEX_PROMPT_BLOCK: &str = "# Codebase Index (semantic search — PREFER THIS for finding code)\n\
\n\
This project is indexed. Two extra tools are available and are faster and more accurate \
than grep/glob for locating code:\n\
\n\
- `codebase_search` — semantic, AI-ranked search across the whole codebase. Handles \
conceptual queries (e.g. \"color parsing\", \"rate limiting logic\") and finds relevant \
code even when you don't know the exact symbol or string.\n\
- `codebase_graph` — call/dependency graph: find a function's callers, callees, \
definitions, and references.\n\
\n\
Exploration workflow (this OVERRIDES the grep/glob guidance in the sections below): when \
you need to find where something is implemented or understand how a feature works, call \
`codebase_search` FIRST (and `codebase_graph` for relationships), then `read` the files it \
points to. Use `grep`/`glob` only for exact-string matches, or if the index tool reports \
it is unavailable.\n";

/// One block of the system prompt. Anthropic sends these in order; each block
/// can independently carry a `cache_control` marker.
#[derive(Debug, Clone)]
pub struct SystemBlock {
    pub text: String,
    pub cache_control: Option<CacheControl>,
}

/// Build a system prompt for the given mode as an ordered list of blocks.
///
/// Returns 2-4 blocks depending on which optional sections apply:
///   [0] static body — everything except the Environment section, with
///       `{{working_dir}}` interpolated inline.
///   [1] (optional) Context-engine index guidance — when `context_engine_enabled`.
///   [2] (optional) Skills + Subagents combined — when either registry has entries.
///   [last] environment section — Working directory, branch, project note,
///       date, OS/arch. NOT cached: `{{date}}` rotates daily and we don't
///       want a fresh cache write every midnight.
///
/// Caching: exactly ONE `cache_control: ephemeral` marker is placed on the LAST
/// cacheable block (i.e. the block immediately before env). Anthropic's prompt
/// cache extends as a prefix from the start of the request through each
/// breakpoint, so a single marker at the end of the prefix caches the whole
/// static+index+skills span as one entry — and keeps the system prompt's
/// contribution to Anthropic's 4-cache_control limit at exactly 1, leaving
/// headroom for the conversation-level and tools-level breakpoints.
pub fn build_system_prompt(
    mode: ToolMode,
    working_dir: &Path,
    branch: Option<&str>,
    project_note: Option<&str>,
    skills: Option<&SkillRegistry>,
    subagents: Option<&SubagentRegistry>,
    context_engine_enabled: bool,
) -> Vec<SystemBlock> {
    let template = match mode {
        ToolMode::Ask => ASK_PROMPT_TEMPLATE,
        ToolMode::Coding => CODING_PROMPT_TEMPLATE,
        ToolMode::Plan => PLAN_PROMPT_TEMPLATE,
    };

    let static_body = template.replace("{{working_dir}}", &working_dir.display().to_string());

    let date = chrono::Local::now().format("%Y-%m-%d").to_string();
    let os = std::env::consts::OS;
    let arch = std::env::consts::ARCH;
    let branch_line = branch.map(|b| format!("- Git branch: {b}\n")).unwrap_or_default();
    let note_line = project_note.map(|n| format!("- {n}\n")).unwrap_or_default();
    let env_section = format!(
        "# Environment\n- Working directory: {}\n{}{}- Date: {}\n- OS: {}/{}\n",
        working_dir.display(),
        branch_line,
        note_line,
        date,
        os,
        arch,
    );

    let mut blocks = vec![SystemBlock {
        text: static_body,
        cache_control: None,
    }];

    if context_engine_enabled {
        blocks.push(SystemBlock {
            text: INDEX_PROMPT_BLOCK.to_string(),
            cache_control: None,
        });
    }

    let skills_entries = skills.map(|r| r.list_for_prompt()).unwrap_or_default();
    let subagents_entries = subagents.map(|r| r.list_for_prompt()).unwrap_or_default();

    if !skills_entries.is_empty() || !subagents_entries.is_empty() {
        let mut combined = String::new();
        if !skills_entries.is_empty() {
            combined.push_str(
                "# Available Skills\n\
                 Skills are instruction packs loaded on demand via the `skill` tool. \
                 The descriptions below are a MENU — they do not contain the rules \
                 themselves. Whenever a task matches a skill by name, topic, or \
                 intent — or the user explicitly mentions one — CALL THE `skill` \
                 TOOL FIRST and follow the body verbatim. Do not guess at the rules \
                 from the description. Skip the tool only when no skill below is \
                 clearly relevant.\n\n\
                 Available:\n",
            );
            for (name, description) in &skills_entries {
                combined.push_str(&format!("- {name}: {description}\n"));
            }
        }
        if !subagents_entries.is_empty() {
            if !combined.is_empty() {
                combined.push('\n');
            }
            combined.push_str(
                "# Available Subagents\n\
                 Subagents currently enabled for this turn (see \"Default Subagents\" \
                 in the main prompt for routing guidance):\n",
            );
            for (name, description) in &subagents_entries {
                combined.push_str(&format!("- {name}: {description}\n"));
            }
        }
        log::info!(
            "[prompt] skills+subagents block: {} skill(s), {} subagent(s), {} chars",
            skills_entries.len(),
            subagents_entries.len(),
            combined.len()
        );
        blocks.push(SystemBlock {
            text: combined,
            cache_control: None,
        });
    }

    // Single cache breakpoint at the end of the cacheable prefix. Caches the
    // whole static+index+skills span as one entry regardless of which optional
    // blocks are present, and leaves room for the conversation + tools markers
    // within Anthropic's 4-marker cap.
    if let Some(last) = blocks.last_mut() {
        last.cache_control = Some(CacheControl::ephemeral());
    }

    blocks.push(SystemBlock {
        text: env_section,
        cache_control: None,
    });

    blocks
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn joined(blocks: &[SystemBlock]) -> String {
        blocks.iter().map(|b| b.text.as_str()).collect::<Vec<_>>().join("\n")
    }

    use crate::skills::{registry::SkillInput, SkillRegistry};
    use std::collections::HashSet;

    fn mk_skill_registry() -> SkillRegistry {
        let input = SkillInput {
            raw: "---\nname: hello\ndescription: A greeting skill.\n---\nbody\n".to_string(),
            path: PathBuf::from("/hello"),
        };
        SkillRegistry::new(vec![], vec![input], vec![], &HashSet::new())
    }

    // ── Structural: two-block layout with correct cache markers ──

    #[test]
    fn test_build_system_prompt_splits_into_two_blocks() {
        for mode in [ToolMode::Ask, ToolMode::Coding, ToolMode::Plan] {
            let blocks = build_system_prompt(mode, &PathBuf::from("/p"), Some("main"), None, None, None, false);
            assert_eq!(blocks.len(), 2, "{:?}: expected exactly 2 blocks", mode);
            assert!(blocks[0].cache_control.is_some(), "{:?}: block 0 must be cached", mode);
            assert!(blocks[1].cache_control.is_none(), "{:?}: block 1 must NOT be cached", mode);
        }
    }

    #[test]
    fn test_skills_block_inserted_when_registry_non_empty() {
        let registry = mk_skill_registry();
        for mode in [ToolMode::Ask, ToolMode::Coding, ToolMode::Plan] {
            let blocks = build_system_prompt(
                mode,
                &PathBuf::from("/p"),
                Some("main"),
                None,
                Some(&registry),
                None,
                false,
            );
            assert_eq!(blocks.len(), 3, "{:?}: expected 3 blocks with skills", mode);
            // Single cache breakpoint at the end of the cacheable prefix —
            // static body is part of the same cached span but carries no
            // marker of its own.
            assert!(blocks[0].cache_control.is_none(), "{:?}: static body no marker", mode);
            assert!(blocks[1].cache_control.is_some(), "{:?}: last cacheable block carries marker", mode);
            assert!(blocks[2].cache_control.is_none(), "{:?}: env uncached", mode);
            assert!(blocks[1].text.contains("# Available Skills"));
            assert!(blocks[1].text.contains("hello: A greeting skill."));
        }
    }

    // ── Anthropic 4-cache_control invariant: system contributes exactly 1 ──

    #[test]
    fn test_exactly_one_cache_marker_across_all_configs() {
        let registry = mk_skill_registry();
        let empty = SkillRegistry::new(vec![], vec![], vec![], &HashSet::new());
        // (semantic_on, skills_registry) — env block never carries a marker, so
        // every config must total exactly one cache_control across all blocks.
        let cases: &[(bool, Option<&SkillRegistry>)] = &[
            (false, None),
            (true, None),
            (false, Some(&empty)),
            (true, Some(&empty)),
            (false, Some(&registry)),
            (true, Some(&registry)),
        ];
        for (semantic_on, skills) in cases {
            for mode in [ToolMode::Ask, ToolMode::Coding, ToolMode::Plan] {
                let blocks = build_system_prompt(
                    mode,
                    &PathBuf::from("/p"),
                    Some("main"),
                    None,
                    *skills,
                    None,
                    *semantic_on,
                );
                let markers = blocks.iter().filter(|b| b.cache_control.is_some()).count();
                assert_eq!(
                    markers, 1,
                    "{:?} semantic_on={} skills={}: expected exactly 1 cache_control marker, got {} (blocks={})",
                    mode,
                    semantic_on,
                    skills.is_some(),
                    markers,
                    blocks.len(),
                );
                // Marker must be on the last *cacheable* block — i.e. the one
                // immediately before env. Env itself must never be marked.
                let last_idx = blocks.len() - 1;
                assert!(blocks[last_idx].cache_control.is_none(), "env block must stay uncached");
                assert!(
                    blocks[last_idx - 1].cache_control.is_some(),
                    "marker must sit on the last cacheable block"
                );
            }
        }
    }

    #[test]
    fn test_skills_block_omitted_when_registry_empty() {
        let empty = SkillRegistry::new(vec![], vec![], vec![], &HashSet::new());
        let blocks = build_system_prompt(
            ToolMode::Ask,
            &PathBuf::from("/p"),
            Some("main"),
            None,
            Some(&empty),
            None,
            false,
        );
        assert_eq!(blocks.len(), 2, "empty registry must not add a block");
    }

    #[test]
    fn test_index_block_inserted_only_when_context_engine_enabled() {
        for mode in [ToolMode::Ask, ToolMode::Coding, ToolMode::Plan] {
            let on = build_system_prompt(mode, &PathBuf::from("/p"), Some("main"), None, None, None, true);
            assert!(
                on.iter().any(|b| b.text.contains("codebase_search")),
                "{:?}: index block must advertise codebase_search when engine enabled", mode
            );
            let off = build_system_prompt(mode, &PathBuf::from("/p"), Some("main"), None, None, None, false);
            assert!(
                !off.iter().any(|b| b.text.contains("codebase_search")),
                "{:?}: no index block when engine disabled", mode
            );
        }
    }

    #[test]
    fn test_static_body_excludes_date() {
        let today = chrono::Local::now().format("%Y-%m-%d").to_string();
        for mode in [ToolMode::Ask, ToolMode::Coding, ToolMode::Plan] {
            let blocks = build_system_prompt(mode, &PathBuf::from("/p"), Some("main"), None, None, None, false);
            assert!(
                !blocks[0].text.contains(&today),
                "{:?}: static body must not contain today's date (would invalidate cache daily)",
                mode
            );
        }
    }

    #[test]
    fn test_env_block_contains_date_and_working_dir() {
        let today = chrono::Local::now().format("%Y-%m-%d").to_string();
        for mode in [ToolMode::Ask, ToolMode::Coding, ToolMode::Plan] {
            let blocks = build_system_prompt(mode, &PathBuf::from("/home/user/project"), Some("main"), None, None, None, false);
            assert!(blocks[1].text.contains(&today), "{:?}: env must contain date", mode);
            assert!(blocks[1].text.contains("/home/user/project"), "{:?}: env must contain working_dir", mode);
            assert!(blocks[1].text.contains("main"), "{:?}: env must contain branch when provided", mode);
        }
    }

    // ── Legacy coverage: working_dir injection, placeholder resolution ──

    #[test]
    fn test_working_dir_injected_all_modes() {
        for mode in [ToolMode::Ask, ToolMode::Coding, ToolMode::Plan] {
            let prompt = joined(&build_system_prompt(mode, &PathBuf::from("/home/user/project"), None, None, None, None, false));
            assert!(prompt.contains("/home/user/project"), "Mode {:?} missing working_dir", mode);
            assert!(!prompt.contains("{{working_dir}}"), "Mode {:?} has unresolved placeholder", mode);
        }
    }

    #[test]
    fn test_branch_injected() {
        let prompt = joined(&build_system_prompt(ToolMode::Coding, &PathBuf::from("/tmp"), Some("feature/login"), None, None, None, false));
        assert!(prompt.contains("feature/login"));
    }

    #[test]
    fn test_branch_omitted_when_none() {
        let prompt = joined(&build_system_prompt(ToolMode::Ask, &PathBuf::from("/tmp"), None, None, None, None, false));
        assert!(!prompt.contains("Git branch"));
    }

    #[test]
    fn test_date_injected() {
        let prompt = joined(&build_system_prompt(ToolMode::Ask, &PathBuf::from("/tmp"), None, None, None, None, false));
        let today = chrono::Local::now().format("%Y-%m-%d").to_string();
        assert!(prompt.contains(&today));
    }

    #[test]
    fn test_os_arch_injected() {
        for mode in [ToolMode::Ask, ToolMode::Coding, ToolMode::Plan] {
            let prompt = joined(&build_system_prompt(mode, &PathBuf::from("/tmp"), None, None, None, None, false));
            assert!(prompt.contains(std::env::consts::OS), "Mode {:?} missing OS", mode);
            assert!(prompt.contains(std::env::consts::ARCH), "Mode {:?} missing ARCH", mode);
        }
    }

    #[test]
    fn test_no_unresolved_placeholders() {
        for mode in [ToolMode::Ask, ToolMode::Coding, ToolMode::Plan] {
            let prompt = joined(&build_system_prompt(mode, &PathBuf::from("/tmp"), Some("main"), None, None, None, false));
            assert!(!prompt.contains("{{"), "Mode {:?} has unresolved placeholder: {}", mode,
                prompt.find("{{").map(|i| &prompt[i..(i+30).min(prompt.len())]).unwrap_or(""));
        }
    }
}
