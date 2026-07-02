import { useEffect, useMemo, useState } from 'react';
import { Button, Select, Switch, Tooltip, message as toast } from 'antd';
import { Plus, Sparkles, Trash2 } from 'lucide-react';
import { agentTauriService } from '@/services/agentTauriService';
import { useAppStore } from '@/store';
import type {
  AutoSettings,
  ModelRef,
  ProviderConfig,
  SessionRow,
  WorkerPoolEntry,
} from '@/types/agent';
import AddWorkerDialog from './AddWorkerDialog';
import DisableConfirmModal from './DisableConfirmModal';

/** Same shape used by ModelPicker + Settings page. Keep in sync when
 * refactoring — a shared helper would be nicer but isn't wired yet. */
function providerName(p: ProviderConfig): string {
  if (p.kind === 'openai') return 'OpenAI';
  if (p.kind === 'anthropic') return 'Anthropic';
  if (p.label?.trim()) return p.label.trim();
  try {
    return new URL(p.baseUrl).host || 'OpenAI-compatible';
  } catch {
    return 'OpenAI-compatible';
  }
}

/**
 * "Auto" Settings section. Sits at the end of the Settings page (after
 * Semantic search) per the locked design — opt-in advanced feature.
 *
 * Gating:
 *   - Enable toggle is disabled (with a tooltip) unless: ≥1 provider with
 *     models exists AND orchestrator picked AND worker pool non-empty.
 *   - Disable is blocked if any session still has Auto as its active model
 *     (modal lists them and tells the user to switch first).
 *
 * Auto-fill: when a user adds a known model (e.g. claude-sonnet-4-6) to the
 * worker pool, the description pre-fills with a sensible suggestion they can
 * edit. See `knownDescriptions.ts`.
 */
export default function AutoSettingsSection() {
  const providers = useAppStore((s) => s.providers);
  const providersLoaded = useAppStore((s) => s.providersLoaded);
  const loadProviders = useAppStore((s) => s.loadProviders);

  const [settings, setSettings] = useState<AutoSettings | null>(null);
  const [loading, setLoading] = useState(true);
  const [addOpen, setAddOpen] = useState(false);
  const [blockingSessions, setBlockingSessions] = useState<SessionRow[]>([]);

  // Initial load: settings + providers (the latter only if not already cached).
  useEffect(() => {
    if (!providersLoaded) loadProviders();
    agentTauriService
      .getAutoSettings()
      .then((s) => setSettings(s))
      .catch((e) => toast.error(`Failed to load Auto settings: ${e}`))
      .finally(() => setLoading(false));
  }, [providersLoaded, loadProviders]);

  // ── Derived: prereq state for the enable toggle ──
  const hasProviderWithModels = providers.some((p) => p.models.length > 0);
  const hasOrchestrator = !!settings?.orchestrator;
  const hasWorkers = (settings?.workerPool.length ?? 0) > 0;
  const prereqsMet = hasProviderWithModels && hasOrchestrator && hasWorkers;
  const prereqHint = !hasProviderWithModels
    ? 'Add a provider with at least one model first.'
    : !hasOrchestrator
    ? 'Pick an orchestrator model below first.'
    : !hasWorkers
    ? 'Add at least one model to the worker pool first.'
    : '';

  // ── Options for the orchestrator dropdown: flat list grouped by provider. ──
  const orchestratorOptions = useMemo(() => {
    return providers
      .filter((p) => p.models.length > 0)
      .map((p) => ({
        label: p.label?.trim() || p.kind,
        options: p.models.map((m) => ({
          value: `${p.id}::${m}`,
          label: m,
        })),
      }));
  }, [providers]);

  const orchestratorValue = settings?.orchestrator
    ? `${settings.orchestrator.providerId}::${settings.orchestrator.model}`
    : undefined;

  // ── Handlers ──

  const handleToggleEnabled = async (checked: boolean) => {
    if (!settings) return;
    try {
      await agentTauriService.setAutoEnabled(checked);
      setSettings({ ...settings, enabled: checked });
      toast.success(checked ? 'Auto mode enabled' : 'Auto mode disabled');
    } catch (e) {
      // Backend rejects disable when sessions still use Auto. Surface them
      // in the modal so the user knows what to fix.
      if (!checked) {
        try {
          const blockers = await agentTauriService.listSessionsUsingAuto();
          if (blockers.length > 0) {
            setBlockingSessions(blockers);
            return;
          }
        } catch {
          /* fall through to the generic error */
        }
      }
      toast.error(`${e}`);
    }
  };

  /**
   * Try to flip Auto off because a prerequisite was just invalidated (worker
   * pool emptied or orchestrator cleared). If the backend gate refuses
   * because sessions still use Auto, leave it enabled and surface the
   * blockers — same path as a manual disable attempt.
   */
  const tryAutoDisable = async (reason: string) => {
    try {
      await agentTauriService.setAutoEnabled(false);
      setSettings((prev) => (prev ? { ...prev, enabled: false } : prev));
      toast.warning(reason);
    } catch {
      try {
        const blockers = await agentTauriService.listSessionsUsingAuto();
        if (blockers.length > 0) {
          setBlockingSessions(blockers);
          toast.warning(`${reason} Switch the listed sessions off Auto first to disable.`);
          return;
        }
      } catch {
        /* fall through */
      }
      toast.warning(`${reason} Auto stays enabled — tasks will fail until you fix it.`);
    }
  };

  const handleOrchestratorChange = async (value: string | undefined) => {
    if (!settings) return;
    const ref: ModelRef | null = value
      ? (() => {
          const [providerId, model] = value.split('::');
          return { providerId, model };
        })()
      : null;
    try {
      await agentTauriService.setAutoOrchestratorModel(ref);
      setSettings({ ...settings, orchestrator: ref });
      toast.success(ref ? `Orchestrator set to ${ref.model}` : 'Orchestrator cleared');
      // Clearing the orchestrator invalidates Auto — flip it off so the
      // model picker drops the ✨ Auto entry and no future turn errors
      // on dispatch.
      if (!ref && settings.enabled) {
        await tryAutoDisable('Orchestrator cleared — Auto disabled.');
      }
    } catch (e) {
      toast.error(`Failed to set orchestrator: ${e}`);
    }
  };

  const handleAddWorker = async (entry: WorkerPoolEntry) => {
    if (!settings) return;
    const next = [...settings.workerPool, entry];
    try {
      await agentTauriService.setAutoWorkerPool(next);
      setSettings({ ...settings, workerPool: next });
      setAddOpen(false);
      toast.success(`Added ${entry.model} to worker pool`);
    } catch (e) {
      toast.error(`Failed to save worker pool: ${e}`);
    }
  };

  const handleRemoveWorker = async (idx: number) => {
    if (!settings) return;
    const removed = settings.workerPool[idx];
    const next = settings.workerPool.filter((_, i) => i !== idx);
    try {
      await agentTauriService.setAutoWorkerPool(next);
      setSettings({ ...settings, workerPool: next });
      toast.success(`Removed ${removed.model} from worker pool`);
      // Empty pool invalidates Auto — flip it off via the same gate the
      // manual disable uses (will surface blocking sessions if any).
      if (next.length === 0 && settings.enabled) {
        await tryAutoDisable('Worker pool is empty — Auto disabled.');
      }
    } catch (e) {
      toast.error(`Failed to update worker pool: ${e}`);
    }
  };

  const handleEditDescription = async (idx: number, description: string) => {
    if (!settings) return;
    // Skip save+toast when blur fires without an actual change (common when
    // the user clicks the textarea and then clicks away).
    if (settings.workerPool[idx]?.description === description) return;
    const next = settings.workerPool.map((e, i) =>
      i === idx ? { ...e, description } : e,
    );
    try {
      await agentTauriService.setAutoWorkerPool(next);
      setSettings({ ...settings, workerPool: next });
      toast.success(`Updated description for ${next[idx].model}`);
    } catch (e) {
      toast.error(`Failed to update description: ${e}`);
    }
  };

  // ── Render ──

  if (loading || !settings) {
    return (
      <div className="mt-8 text-sm text-[var(--text-secondary)]">
        Loading Auto settings…
      </div>
    );
  }

  const enableSwitch = (
    <Switch
      checked={settings.enabled}
      disabled={!settings.enabled && !prereqsMet}
      onChange={handleToggleEnabled}
    />
  );

  return (
    <>
      <div className="flex items-center justify-between mt-8 mb-1">
        <h2 className="text-base font-semibold text-[var(--text-primary)] flex items-center gap-2">
          <Sparkles className="w-4 h-4 text-[var(--text-secondary)]" />
          Auto
        </h2>
        {prereqHint && !settings.enabled ? (
          <Tooltip title={prereqHint} placement="left">
            <span>{enableSwitch}</span>
          </Tooltip>
        ) : (
          enableSwitch
        )}
      </div>
      <p className="text-sm text-[var(--text-secondary)] mb-4 leading-relaxed">
        Multi-model orchestration. An orchestrator LLM decomposes your request
        into subtasks and dispatches each to the best worker model from your
        configured pool. Power-user feature — keep disabled if you prefer
        picking one model per session.
      </p>

      <div className="flex flex-col gap-5 border border-[var(--border)] rounded-lg p-5">
        {/* Step 1: Orchestrator model */}
        <div>
          <div className="flex items-baseline justify-between mb-1">
            <label className="text-sm font-medium text-[var(--text-primary)]">
              Orchestrator model
            </label>
            <span className="text-xs text-[var(--text-secondary)]">
              The model that plans the work.
            </span>
          </div>
          <Select
            className="w-full"
            value={orchestratorValue}
            onChange={handleOrchestratorChange}
            options={orchestratorOptions}
            placeholder={
              hasProviderWithModels
                ? 'Select a model'
                : 'Configure a provider with models first'
            }
            allowClear
            showSearch
            disabled={!hasProviderWithModels}
            notFoundContent="Fetch models on a provider first"
          />
        </div>

        {/* Step 2: Worker pool */}
        <div>
          <div className="flex items-baseline justify-between mb-1">
            <label className="text-sm font-medium text-[var(--text-primary)]">
              Worker pool
            </label>
            <span className="text-xs text-[var(--text-secondary)]">
              The orchestrator picks from this list per subtask.
            </span>
          </div>
          {settings.workerPool.length === 0 ? (
            <div className="text-sm text-[var(--text-secondary)] border border-dashed border-[var(--border)] rounded-md p-4 text-center">
              No workers yet. Add one to get started.
            </div>
          ) : (
            <div className="border border-[var(--border)] rounded-md divide-y divide-[var(--border)]">
              {settings.workerPool.map((entry, idx) => {
                // Resolve the pool entry's providerId to its friendly name.
                // Falls back to a marker + the raw id if the underlying
                // provider was deleted after the entry was added (rare —
                // AutoRunTurn's build_worker_llm_configs will fail-fast on
                // this too so the user sees a clear error at run time).
                const provider = providers.find((p) => p.id === entry.providerId);
                const providerLabel = provider
                  ? providerName(provider)
                  : `(unknown provider ${entry.providerId.slice(0, 8)}…)`;
                return (
                <div key={`${entry.providerId}-${entry.model}`} className="p-3">
                  <div className="flex items-start justify-between gap-3 mb-2">
                    <div className="flex-1 min-w-0">
                      <div className="text-sm font-medium text-[var(--text-primary)] truncate">
                        {entry.model}
                      </div>
                      <div className="text-xs text-[var(--text-secondary)]">
                        {providerLabel}
                      </div>
                    </div>
                    <Tooltip title="Remove from pool">
                      <Button
                        type="text"
                        size="small"
                        icon={<Trash2 className="w-4 h-4" />}
                        onClick={() => handleRemoveWorker(idx)}
                      />
                    </Tooltip>
                  </div>
                  <textarea
                    className="w-full text-xs p-2 rounded border border-[var(--border)] bg-[var(--bg-secondary)] text-[var(--text-primary)] resize-y"
                    rows={2}
                    value={entry.description}
                    onChange={(e) => {
                      // Optimistic local update on every keystroke so the
                      // textarea stays responsive; persist on blur.
                      const next = settings.workerPool.map((x, i) =>
                        i === idx ? { ...x, description: e.target.value } : x,
                      );
                      setSettings({ ...settings, workerPool: next });
                    }}
                    onBlur={(e) => handleEditDescription(idx, e.target.value)}
                    placeholder="When should the orchestrator pick this model?"
                  />
                </div>
                );
              })}
            </div>
          )}
          <Button
            className="mt-3"
            icon={<Plus className="w-4 h-4" />}
            onClick={() => setAddOpen(true)}
            disabled={!hasProviderWithModels}
          >
            Add worker
          </Button>
        </div>

        {/* Replan budget removed in P9b — Auto now pauses on worker failure
            and shows Retry / Replan / Cancel buttons inline. The user IS
            the budget; no slider needed. */}
      </div>

      <AddWorkerDialog
        open={addOpen}
        providers={providers}
        existingPool={settings.workerPool}
        onCancel={() => setAddOpen(false)}
        onAdd={handleAddWorker}
      />
      <DisableConfirmModal
        open={blockingSessions.length > 0}
        sessions={blockingSessions}
        onClose={() => setBlockingSessions([])}
      />
    </>
  );
}
