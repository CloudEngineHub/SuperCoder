/**
 * Suggested descriptions for well-known models, pre-filled in the worker-pool
 * "Add worker" dialog. Users can edit before saving. Unknown models start
 * empty.
 *
 * These descriptions are what the orchestrator LLM sees when it picks a model
 * per subtask — keep them concise and capability-focused, not marketing-y.
 *
 * Key = model id (the same string the LLM will receive in the plan), not the
 * display name.
 */
export const KNOWN_MODEL_DESCRIPTIONS: Record<string, string> = {
  // Anthropic
  'claude-opus-4-8':
    'Strongest reasoning + coding. Best for complex synthesis, multi-file refactors, and tricky debugging. Expensive — reserve for genuinely hard work.',
  'claude-sonnet-4-6':
    'Balanced reasoning + coding workhorse. Strong code quality at mid-tier cost. Good default for design + most coding subtasks.',
  'claude-haiku-4-5':
    'Fast and cheap. Best for reads, greps, file exploration, simple writes, and verification (e.g., running tests). Weak on complex synthesis.',

  // OpenAI
  'gpt-5':
    'Strong synthesis + coding. Comparable to Opus for hard reasoning tasks. Higher cost.',
  'gpt-4o-mini':
    'Cheap and fast. Good for simple text generation, tool calls, and light edits. Limited on complex multi-step reasoning.',
};

/**
 * Look up a suggested description for a model id, or an empty string if we
 * don't recognize it. Case-sensitive — matches the exact model id strings
 * providers emit.
 */
export function suggestedDescription(model: string): string {
  return KNOWN_MODEL_DESCRIPTIONS[model] ?? '';
}
