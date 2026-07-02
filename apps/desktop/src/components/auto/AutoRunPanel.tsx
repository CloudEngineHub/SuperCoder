import { useEffect, useState } from 'react';
import { Button, message as toast } from 'antd';
import {
  AlertCircle,
  CheckCircle2,
  ChevronDown,
  ChevronRight,
  Loader2,
  Play,
  RotateCcw,
  Sparkles,
  X,
  XCircle,
} from 'lucide-react';
import { useAppStore } from '@/store';
import { agentTauriService } from '@/services/agentTauriService';
import Markdown from '@/components/common/Markdown/Markdown';
import type { AutoToolActivity } from '@/store/agentSlice';
import type { WorkerResult, WorkerSpec, WorkerStatus } from '@/types/agent';

interface Props {
  runId: string;
}

/**
 * Renders the live state of one Auto-mode run, keyed by `runId`. Reads from
 * `useAppStore().autoRuns[runId]`; if missing (page reload, opening a session
 * with prior Auto turns), self-hydrates via `agent_get_auto_run` so the
 * persisted snapshot fills the panel until live events take over.
 *
 * Lifecycle states the panel handles:
 *   - planning            → "Planning…" spinner
 *   - awaiting_approval   → plan card + Run / Cancel
 *   - running             → plan card (collapsed) + worker cards streaming
 *   - replanning          → replan banner + worker cards so far
 *   - done                → done banner + collapsed worker cards
 *   - failed              → failure banner + Retry
 */
export default function AutoRunPanel({ runId }: Props) {
  const run = useAppStore((s) => s.autoRuns[runId]);
  const hydrateAutoRun = useAppStore((s) => s.hydrateAutoRun);
  const clearAutoRun = useAppStore((s) => s.clearAutoRun);

  // Lazy-hydrate from disk when state for this run isn't in memory yet.
  // No-op on the slice side if it's already present (live events win).
  useEffect(() => {
    console.info('[Auto-P7] panel mount runId=%s hasState=%s', runId, !!run);
    if (run) return;
    let cancelled = false;
    agentTauriService
      .getAutoRun(runId)
      .then((snapshot) => {
        if (cancelled || !snapshot) {
          console.info('[Auto-P7] panel hydrate runId=%s -> %s', runId, snapshot ? 'cancelled' : 'null snapshot');
          return;
        }
        console.info('[Auto-P7] panel hydrate runId=%s status=%s workers=%d plans=%d',
          runId, snapshot.status, snapshot.worker_results.length, snapshot.plan_versions.length);
        hydrateAutoRun(snapshot);
      })
      .catch((e) => {
        console.warn('[Auto-P7] panel hydrate failed runId=%s err=%o', runId, e);
      });
    return () => {
      cancelled = true;
    };
  }, [runId, run, hydrateAutoRun]);

  if (!run) return null;

  return (
    <div className="my-4 flex flex-col gap-3">
      {/* Plan card — visible at all stages once a plan exists, collapsed
          after approval to avoid taking too much vertical room. */}
      {run.plan && (
        <PlanCard
          plan={run.plan}
          showActions={run.status === 'awaiting_approval'}
          runId={run.runId}
          collapsed={run.status === 'running' || run.status === 'replanning' || run.status === 'done'}
        />
      )}

      {/* Status banners by lifecycle state. */}
      {run.status === 'planning' && (
        <StatusBanner
          icon={<Loader2 className="w-4 h-4 animate-spin" />}
          tone="info"
          text="Orchestrator is planning…"
        />
      )}
      {run.status === 'replanning' && (
        <StatusBanner
          icon={<RotateCcw className="w-4 h-4" />}
          tone="warning"
          text={`Replanning — ${run.replanReason ?? 'worker failed'}`}
        />
      )}

      {/* Completed worker cards (collapsed by default). */}
      {run.workerResults.map((result, idx) => (
        <WorkerCard key={`${run.runId}-${idx}`} result={result} />
      ))}

      {/* Currently-running worker (no result yet). */}
      {run.currentWorker && (
        <RunningWorkerCard
          key={run.currentWorker.workerId}
          workerId={run.currentWorker.workerId}
          model={run.currentWorker.model}
          prompt={run.currentWorker.prompt}
          tools={run.currentWorkerTools}
        />
      )}

      {/* Terminal banners. */}
      {run.status === 'awaiting_failure_decision' && run.pendingFailure && (
        <PendingFailureBanner
          runId={runId}
          failure={run.pendingFailure}
        />
      )}
      {run.status === 'failed' && (
        <FailedBanner
          reason={run.failedReason ?? 'unknown'}
          lastWorkerOutput={run.lastWorkerOutput}
          onDismiss={() => clearAutoRun(runId)}
          sessionId={run.sessionId}
          runId={runId}
        />
      )}
      {run.status === 'done' && (
        <DoneBanner summary={run.doneSummary ?? 'Done.'} />
      )}
    </div>
  );
}

// ── PendingFailureBanner ──────────────────────────────────────────────────
// P9c: when a worker hard-fails the executor pauses the run and surfaces
// this banner with three actions. Retry re-runs the same failed worker
// (cheapest; useful for transient infra errors). Replan kicks the
// orchestrator with the failure context (when the picked model just can't
// do it). Cancel marks the run terminally Failed.

function PendingFailureBanner({
  runId,
  failure,
}: {
  runId: string;
  failure: { worker_id: string; worker_model: string; error_message: string };
}) {
  const [submitting, setSubmitting] = useState<'retry' | 'replan' | 'cancel' | null>(null);
  const dispatch = async (action: 'retry' | 'replan' | 'cancel') => {
    if (submitting) return;
    setSubmitting(action);
    try {
      await agentTauriService.resolveWorkerFailure(runId, action);
    } catch (e) {
      toast.error(`${action} failed: ${e}`);
    } finally {
      // Always clear so the banner isn't stuck if the spawned executor
      // crashes before emitting any event. On success the panel re-renders
      // to the next status (running / failed) and this banner unmounts;
      // on error the user can retry the action.
      setSubmitting(null);
    }
  };
  return (
    <div className="border border-amber-300 bg-amber-50 dark:border-amber-700 dark:bg-amber-900/20 rounded-lg p-3">
      <div className="flex items-start gap-2 mb-2">
        <AlertCircle className="w-4 h-4 text-amber-600 mt-0.5" />
        <div className="flex-1">
          <div className="text-sm font-medium text-[var(--text-primary)]">
            Worker {failure.worker_id} ({failure.worker_model}) failed
          </div>
          <div className="text-xs text-[var(--text-secondary)] mt-0.5 break-words">
            {failure.error_message}
          </div>
        </div>
      </div>
      <div className="flex gap-2 mt-3">
        <Button
          size="small"
          type="primary"
          onClick={() => dispatch('retry')}
          loading={submitting === 'retry'}
          disabled={!!submitting}
        >
          Retry worker
        </Button>
        <Button
          size="small"
          onClick={() => dispatch('replan')}
          loading={submitting === 'replan'}
          disabled={!!submitting}
        >
          Ask orchestrator to replan
        </Button>
        <Button
          size="small"
          type="text"
          danger
          onClick={() => dispatch('cancel')}
          loading={submitting === 'cancel'}
          disabled={!!submitting}
        >
          Cancel run
        </Button>
      </div>
    </div>
  );
}

// ── PlanCard ──────────────────────────────────────────────────────────────

interface PlanCardProps {
  plan: { version: number; reasoning: string; workers: WorkerSpec[] };
  showActions: boolean;
  collapsed: boolean;
  runId: string;
}

function PlanCard({ plan, showActions, collapsed, runId }: PlanCardProps) {
  const [open, setOpen] = useState(!collapsed);
  const [submitting, setSubmitting] = useState<'run' | 'cancel' | null>(null);

  const handle = async (approved: boolean) => {
    setSubmitting(approved ? 'run' : 'cancel');
    try {
      await agentTauriService.approveAutoPlan(runId, approved);
    } catch (e) {
      toast.error(`${e}`);
    } finally {
      setSubmitting(null);
    }
  };

  return (
    <div className="border border-[var(--border)] rounded-lg bg-[var(--bg-secondary)]">
      <button
        className="w-full flex items-center gap-2 px-3 py-2 text-left"
        onClick={() => setOpen((v) => !v)}
      >
        {open ? (
          <ChevronDown className="w-4 h-4 text-[var(--text-secondary)]" />
        ) : (
          <ChevronRight className="w-4 h-4 text-[var(--text-secondary)]" />
        )}
        <Sparkles className="w-4 h-4 text-amber-500" />
        <span className="font-medium text-sm text-[var(--text-primary)]">
          Auto plan v{plan.version}
        </span>
        <span className="text-xs text-[var(--text-secondary)] ml-1">
          ({plan.workers.length} worker{plan.workers.length === 1 ? '' : 's'})
        </span>
      </button>
      {open && (
        <div className="px-3 pb-3 pt-0">
          <p className="text-xs text-[var(--text-secondary)] mb-3 italic">
            {plan.reasoning}
          </p>
          <ol className="flex flex-col gap-1.5">
            {plan.workers.map((w, i) => (
              <li
                key={w.id}
                className="text-xs flex gap-2 items-start text-[var(--text-primary)]"
              >
                <span className="text-[var(--text-secondary)] font-mono mt-0.5">
                  {i + 1}.
                </span>
                <span className="font-mono text-[var(--text-secondary)]">
                  {w.id}
                </span>
                <span className="text-[var(--text-secondary)]">·</span>
                <span className="font-mono text-amber-600 dark:text-amber-400">
                  {w.model}
                </span>
                <span className="flex-1 text-[var(--text-primary)]">
                  {w.prompt}
                </span>
              </li>
            ))}
          </ol>
          {showActions && (
            <div className="flex gap-2 mt-4">
              <Button
                type="primary"
                icon={<Play className="w-3.5 h-3.5" />}
                loading={submitting === 'run'}
                disabled={submitting !== null}
                onClick={() => handle(true)}
              >
                Run
              </Button>
              <Button
                icon={<X className="w-3.5 h-3.5" />}
                loading={submitting === 'cancel'}
                disabled={submitting !== null}
                onClick={() => handle(false)}
              >
                Cancel
              </Button>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

// ── WorkerCard (completed) ────────────────────────────────────────────────

function WorkerCard({ result }: { result: WorkerResult }) {
  const [open, setOpen] = useState(false);
  const ok = result.status.kind === 'ok';
  const cancelled = result.status.kind === 'cancelled';
  const statusIcon = ok ? (
    <CheckCircle2 className="w-4 h-4 text-green-600" />
  ) : cancelled ? (
    <X className="w-4 h-4 text-gray-400" />
  ) : (
    <XCircle className="w-4 h-4 text-red-600" />
  );

  return (
    <div className="border border-[var(--border)] rounded-lg bg-[var(--bg-secondary)]">
      <button
        className="w-full flex items-center gap-2 px-3 py-2 text-left"
        onClick={() => setOpen((v) => !v)}
      >
        {open ? (
          <ChevronDown className="w-4 h-4 text-[var(--text-secondary)]" />
        ) : (
          <ChevronRight className="w-4 h-4 text-[var(--text-secondary)]" />
        )}
        {statusIcon}
        <span className="font-mono text-xs text-[var(--text-secondary)]">
          {result.id}
        </span>
        {result.model && (
          <>
            <span className="text-xs text-[var(--text-secondary)]">·</span>
            <span className="font-mono text-xs text-amber-600 dark:text-amber-400">
              {result.model}
            </span>
          </>
        )}
        <span className="text-xs text-[var(--text-secondary)]">·</span>
        <span className="text-xs text-[var(--text-secondary)]">
          {result.tool_count} tool{result.tool_count === 1 ? '' : 's'}
        </span>
        <span className="flex-1 truncate text-sm text-[var(--text-primary)] ml-2">
          {firstLine(result.summary)}
        </span>
      </button>
      {open && (
        <div className="px-3 pb-3">
          <div className="text-xs text-[var(--text-secondary)] mb-2">
            Status: {statusLabel(result.status)}
          </div>
          <Markdown>{result.summary}</Markdown>
        </div>
      )}
    </div>
  );
}

// ── RunningWorkerCard ─────────────────────────────────────────────────────

const TOOL_TAIL_LIMIT = 5;

function RunningWorkerCard({
  workerId,
  model,
  prompt,
  tools,
}: {
  workerId: string;
  model: string;
  prompt: string;
  tools: AutoToolActivity[];
}) {
  // Default expanded — live progress is the whole point of this card while a
  // worker is running. User can collapse to reclaim space.
  const [open, setOpen] = useState(true);
  // Only the tail is visible by default. A long-running worker can rack up
  // dozens of tool calls; showing them all pushes the chat composer off-screen.
  const [showAllTools, setShowAllTools] = useState(false);

  const hidden = Math.max(0, tools.length - TOOL_TAIL_LIMIT);
  const visibleTools =
    showAllTools || tools.length <= TOOL_TAIL_LIMIT
      ? tools
      : tools.slice(-TOOL_TAIL_LIMIT);

  return (
    <div className="border border-[var(--border)] rounded-lg bg-[var(--bg-secondary)]">
      <button
        className="w-full flex items-start gap-2 px-3 py-2 text-left"
        onClick={() => setOpen((v) => !v)}
      >
        {open ? (
          <ChevronDown className="w-4 h-4 text-[var(--text-secondary)] mt-0.5 shrink-0" />
        ) : (
          <ChevronRight className="w-4 h-4 text-[var(--text-secondary)] mt-0.5 shrink-0" />
        )}
        <Loader2 className="w-4 h-4 text-amber-500 animate-spin mt-0.5 shrink-0" />
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 text-xs text-[var(--text-secondary)]">
            <span className="font-mono">{workerId}</span>
            <span>·</span>
            <span className="font-mono text-amber-600 dark:text-amber-400">
              {model}
            </span>
            <span>· running…</span>
            {tools.length > 0 && (
              <>
                <span>·</span>
                <span>
                  {tools.length} tool{tools.length === 1 ? '' : 's'}
                </span>
              </>
            )}
          </div>
          <div className="text-sm text-[var(--text-primary)] mt-1 truncate">
            {firstLine(prompt)}
          </div>
        </div>
      </button>

      {open && tools.length > 0 && (
        <div className="px-3 pb-3 ml-6 flex flex-col gap-1.5">
          {hidden > 0 && !showAllTools && (
            <button
              className="text-xs text-[var(--text-secondary)] hover:text-[var(--text-primary)] text-left underline-offset-2 hover:underline"
              onClick={() => setShowAllTools(true)}
            >
              + {hidden} earlier tool call{hidden === 1 ? '' : 's'}
            </button>
          )}
          {visibleTools.map((t) => (
            <ToolActivityRow key={t.toolCallId} tool={t} />
          ))}
        </div>
      )}
    </div>
  );
}

/**
 * One row in the live tool list under the running worker. Status icon mirrors
 * what completed worker cards use, with an in-flight spinner for running tools.
 */
function ToolActivityRow({ tool }: { tool: AutoToolActivity }) {
  const icon =
    tool.status === 'running' ? (
      <Loader2 className="w-3 h-3 text-[var(--text-secondary)] animate-spin shrink-0" />
    ) : tool.status === 'ok' ? (
      <CheckCircle2 className="w-3 h-3 text-green-600 shrink-0" />
    ) : (
      <XCircle className="w-3 h-3 text-red-600 shrink-0" />
    );
  return (
    <div className="flex items-center gap-1.5 text-xs text-[var(--text-secondary)] min-w-0">
      {icon}
      <span className="font-mono">{tool.toolName}</span>
      {(tool.summary || tool.argsSummary) && (
        <span className="truncate text-[var(--text-secondary)]">
          {tool.summary || tool.argsSummary}
        </span>
      )}
    </div>
  );
}

// ── Status / failure / done banners ───────────────────────────────────────

function StatusBanner({
  icon,
  tone,
  text,
}: {
  icon: React.ReactNode;
  tone: 'info' | 'warning';
  text: string;
}) {
  const cls =
    tone === 'warning'
      ? 'border-amber-300 bg-amber-50 dark:border-amber-700 dark:bg-amber-900/20'
      : 'border-[var(--border)] bg-[var(--bg-secondary)]';
  return (
    <div className={`border rounded-lg px-3 py-2 flex items-center gap-2 text-sm ${cls}`}>
      {icon}
      <span className="text-[var(--text-primary)]">{text}</span>
    </div>
  );
}

function FailedBanner({
  reason,
  lastWorkerOutput,
  onDismiss,
  sessionId,
  runId,
}: {
  reason: string;
  lastWorkerOutput: string | null;
  onDismiss: () => void;
  sessionId: string;
  runId: string;
}) {
  // Look up the original user prompt from the chat thread — the user row
  // tagged with this auto_run_id (written by run_auto_turn at run start)
  // carries the prompt verbatim as its text.
  const originalPrompt = useAppStore((s) => {
    const thread = s.agentThreads[sessionId];
    if (!thread) return null;
    const userMsg = thread.messages.find(
      (m) => m.role === 'user' && m.auto_run_id === runId,
    );
    return userMsg?.text ?? null;
  });
  const [retrying, setRetrying] = useState(false);

  const handleRetry = async () => {
    if (!originalPrompt || retrying) return;
    setRetrying(true);
    try {
      // Re-send the same prompt → run_auto_turn dispatches a fresh
      // executor with a new run_id. The original failed panel stays
      // (Dismiss still works); the new run renders its own panel below.
      // Note: the orchestrator runs again, so the new plan may differ.
      await agentTauriService.sendMessage(sessionId, originalPrompt);
    } catch (e) {
      toast.error(`Retry failed: ${e}`);
    } finally {
      setRetrying(false);
    }
  };

  return (
    <div className="border border-red-300 bg-red-50 dark:border-red-700 dark:bg-red-900/20 rounded-lg p-3">
      <div className="flex items-start gap-2 mb-2">
        <AlertCircle className="w-4 h-4 text-red-600 mt-0.5" />
        <div className="flex-1">
          <div className="text-sm font-medium text-[var(--text-primary)]">
            Auto mode failed
          </div>
          <div className="text-xs text-[var(--text-secondary)] mt-0.5">
            {reason}
          </div>
        </div>
        {originalPrompt && (
          <Button
            size="small"
            type="text"
            onClick={handleRetry}
            loading={retrying}
            disabled={retrying}
          >
            Retry
          </Button>
        )}
        <Button size="small" type="text" onClick={onDismiss}>
          Dismiss
        </Button>
      </div>
      {lastWorkerOutput && (
        <details className="text-xs text-[var(--text-secondary)] mt-2">
          <summary className="cursor-pointer">Last worker output</summary>
          <pre className="whitespace-pre-wrap mt-2 text-[var(--text-primary)]">
            {lastWorkerOutput}
          </pre>
        </details>
      )}
    </div>
  );
}

function DoneBanner({ summary }: { summary: string }) {
  return (
    <div className="border border-green-300 bg-green-50 dark:border-green-700 dark:bg-green-900/20 rounded-lg p-3 flex items-start gap-2">
      <CheckCircle2 className="w-4 h-4 text-green-600 mt-0.5 shrink-0" />
      <div className="flex-1 min-w-0">
        <div className="text-sm font-medium text-[var(--text-primary)] mb-1">
          Auto mode complete
        </div>
        <Markdown>{summary}</Markdown>
      </div>
    </div>
  );
}

// ── Helpers ──────────────────────────────────────────────────────────────

function statusLabel(status: WorkerStatus): string {
  switch (status.kind) {
    case 'ok':
      return 'ok';
    case 'max_iterations':
      return 'max iterations reached';
    case 'tool_error':
      return `tool error: ${status.message}`;
    case 'llm_error':
      return `llm error: ${status.message}`;
    case 'cancelled':
      return 'cancelled';
  }
}

function firstLine(s: string): string {
  return s.split('\n', 1)[0];
}
