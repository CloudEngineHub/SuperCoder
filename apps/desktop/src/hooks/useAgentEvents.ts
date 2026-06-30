import { useEffect } from "react";
import { listen, type UnlistenFn } from "@tauri-apps/api/event";
import { useAppStore } from "@/store";
import { createInitialStreamingState } from "@/store/agentSlice";
import { agentTauriService } from "@/services/agentTauriService";
import { parseThinkingMarkers, buildAgentMessage, buildThinkingMeta, displayToAgentMessage } from "@/utils/agentMessageAdapter";
import type {
  AutoAwaitingApprovalEvent,
  AutoDoneEvent,
  AutoFailedEvent,
  AutoPlanEvent,
  AutoPlanningEvent,
  AutoReplanEvent,
  AutoWorkerEndEvent,
  AutoWorkerStartEvent,
  AutoWorkerToolEndEvent,
  AutoWorkerToolStartEvent,
  EngineStatus,
} from "@/types/agent";
import type {
  ApprovalNeededPayload,
  TextDeltaPayload,
  ToolStartPayload,
  ToolEndPayload,
  MessageCompletePayload,
  DonePayload,
  ErrorPayload,
  SessionCompletePayload,
  TurnDiffCompletedPayload,
  CheckpointRestoredPayload,
  QuestionAskedPayload,
  TodoUpdatedPayload,
  PlanReadyPayload,
  TokenUsagePayload,
  SubagentStartPayload,
  SubagentEndPayload,
} from "@/types/agentContract";

/** Clear pending approvals belonging to a session. */
function clearApprovalsForSession(sessionId: string): void {
  const store = useAppStore.getState();
  for (const [id, approval] of Object.entries(store.pendingApprovals)) {
    if (approval.threadId === sessionId) store.removePendingApproval(id);
  }
}

/**
 * Listens to namespaced Tauri events from the local Rust agent and dispatches
 * them to the Zustand store. Mount once at the app root.
 *
 * Every event carries `thread_id` which now holds the `session_id`.
 */
export function useAgentEvents() {
  useEffect(() => {
    const listeners: Promise<UnlistenFn>[] = [];

    // ── agent:text_delta (buffered ~30fps) ─────────────────────────────
    const deltaBuffers = new Map<string, string>();
    let deltaFlushTimer: ReturnType<typeof setTimeout> | null = null;

    function flushDeltas() {
      deltaFlushTimer = null;
      const store = useAppStore.getState();
      for (const [sid, buffered] of deltaBuffers) {
        if (!store.activeSessionIds[sid]) continue;
        store.appendTextDelta(sid, buffered);
      }
      deltaBuffers.clear();
    }

    listeners.push(
      listen<TextDeltaPayload>("agent:text_delta", (event) => {
        const { thread_id, delta } = event.payload;
        deltaBuffers.set(thread_id, (deltaBuffers.get(thread_id) ?? "") + delta);
        if (!deltaFlushTimer) deltaFlushTimer = setTimeout(flushDeltas, 32);
      }),
    );

    // ── agent:approval_needed ──────────────────────────────────────────
    listeners.push(
      listen<ApprovalNeededPayload>("agent:approval_needed", (event) => {
        const { thread_id, tool_call_id, tool_name, description, args } = event.payload;
        let rawCommand: string | undefined;
        if (args) {
          if (tool_name === "bash") rawCommand = (args as any).command;
          else if (tool_name === "grep")
            rawCommand = `grep "${(args as any).pattern}"${(args as any).path ? ` ${(args as any).path}` : ""}`;
          else if (tool_name === "git") rawCommand = `git ${(args as any).command}`;
          else if (tool_name === "glob") rawCommand = (args as any).pattern;
        }
        useAppStore.getState().addPendingApproval({
          threadId: thread_id,
          toolCallId: tool_call_id,
          toolName: tool_name,
          description,
          rawCommand,
          args: args ?? undefined,
        });
      }),
    );

    // ── agent:tool_start ───────────────────────────────────────────────
    listeners.push(
      listen<ToolStartPayload>("agent:tool_start", (event) => {
        const { thread_id, tool_call_id, tool_name, args_summary } = event.payload;
        const store = useAppStore.getState();
        store.setActiveTool(thread_id, {
          toolCallId: tool_call_id,
          toolName: tool_name,
          argsSummary: args_summary,
        });
        store.addToolCall(thread_id, {
          toolCallId: tool_call_id,
          toolName: tool_name,
          argsSummary: args_summary,
          status: "running",
        });
      }),
    );

    // ── agent:tool_end ─────────────────────────────────────────────────
    listeners.push(
      listen<ToolEndPayload>("agent:tool_end", (event) => {
        const { thread_id, tool_call_id, success, summary } = event.payload;
        const store = useAppStore.getState();
        store.setActiveTool(thread_id, null);
        store.updateToolCall(thread_id, tool_call_id, success, summary);
      }),
    );

    // ── agent:message_complete ─────────────────────────────────────────
    listeners.push(
      listen<MessageCompletePayload>("agent:message_complete", (event) => {
        const { thread_id, message, content: legacyContent } = event.payload;
        const content = legacyContent ?? message?.content ?? "";
        const store = useAppStore.getState();

        const { cleanText, thinkingSteps, durationSeconds } = parseThinkingMarkers(content);
        const displayText = cleanText || content;

        // Internal markers handled by dedicated UI — don't render as a message.
        if (displayText.startsWith("Starting coding session:") || displayText.startsWith("Plan saved:")) {
          store.clearAgentStreaming(thread_id);
          return;
        }

        const streaming = store.agentStreaming[thread_id];
        const thinking = buildThinkingMeta(
          thinkingSteps,
          durationSeconds,
          streaming?.toolCalls,
          streaming?.startedAt,
        ) ?? undefined;

        store.addMessageToThread(
          thread_id,
          buildAgentMessage(`${thread_id}-agent-${Date.now()}`, displayText, "agent", thread_id, "", undefined, thinking),
        );
        store.clearAgentStreaming(thread_id);
      }),
    );

    // ── agent:done ─────────────────────────────────────────────────────
    listeners.push(
      listen<DonePayload>("agent:done", (event) => {
        const { thread_id } = event.payload;
        const store = useAppStore.getState();
        // Keep textBuffer visible until message_complete replaces it.
        store.softClearAgentStreaming(thread_id);
        store.updateThreadStatus(thread_id, "completed");
        store.clearActiveSession(thread_id);

        // Refresh client-generated message IDs with real SQLite row IDs.
        agentTauriService
          .getMessages(thread_id)
          .then((msgs) => useAppStore.getState().refreshMessageIds(thread_id, msgs.map((m) => ({ id: m.id, role: m.role, text: m.text }))))
          .catch(() => {});

        clearApprovalsForSession(thread_id);
      }),
    );

    // ── agent:error ────────────────────────────────────────────────────
    listeners.push(
      listen<ErrorPayload>("agent:error", (event) => {
        const { thread_id, message, retrying } = event.payload;
        if (retrying) {
          console.warn("[Agent] Error (retrying):", message);
          return;
        }
        const store = useAppStore.getState();
        let displayMessage = message;
        if (message.includes("prompt is too long") || message.includes("too many tokens")) {
          displayMessage = "Prompt is too long — clear or compact the context and try again.";
        }
        store.setStreamingError(thread_id, displayMessage);
        store.clearActiveSession(thread_id);
        store.updateThreadStatus(thread_id, "error");
        clearApprovalsForSession(thread_id);
        console.error("[Agent] Error:", message);
      }),
    );

    // ── agent:turn_diff_completed ──────────────────────────────────────
    listeners.push(
      listen<TurnDiffCompletedPayload>("agent:turn_diff_completed", (event) => {
        const { thread_id, turn_count, files, additions, deletions, status } = event.payload;
        useAppStore.getState().addCheckpoint(thread_id, {
          turn_count,
          checkpoint_ref: `snapshot/${thread_id}/turn-${turn_count}`,
          commit_sha: "",
          files,
          additions,
          deletions,
          status,
          created_at: new Date().toISOString(),
        });
      }),
    );

    // ── agent:checkpoint_restored ──────────────────────────────────────
    listeners.push(
      listen<CheckpointRestoredPayload>("agent:checkpoint_restored", async (event) => {
        const { thread_id, turn_count } = event.payload;
        const store = useAppStore.getState();
        store.truncateCheckpoints(thread_id, turn_count);
        try {
          const messages = await agentTauriService.getMessages(thread_id);
          store.replaceThreadMessages(thread_id, messages.map(displayToAgentMessage));
        } catch (err) {
          console.error("[agent:checkpoint_restored] reload failed:", err);
        }
      }),
    );

    // ── agent-session-complete (plan → coding handoff) ─────────────────
    listeners.push(
      listen<SessionCompletePayload>("agent-session-complete", (event) => {
        const store = useAppStore.getState();
        const { session_id, project_path, task_summary } = event.payload;
        const pendingPlan = store.pendingPlanForCoding;

        store.upsertAgentThread({
          id: session_id,
          agent_id: "",
          task_summary: task_summary || "Coding session",
          folder_path: project_path,
          branch: store.agentBranch ?? "main",
          status: "active",
          is_coding_session: true,
          total_additions: 0,
          total_deletions: 0,
          checkpoints: [],
          selectedDiffTurn: null,
          messages: [],
          sourcePlanText: pendingPlan?.text,
          created_at: new Date().toISOString(),
          updated_at: new Date().toISOString(),
        });
        store.openAgentThread(session_id);
        store.setActiveSession(session_id, session_id);
        store.setAgentStreaming(session_id, createInitialStreamingState());
        store.loadSessions();

        if (pendingPlan) {
          store.clearCompletedPlan(pendingPlan.projectPath);
          store.setActivePlanProjectPath(null);
          store.setPendingPlanForCoding(null);
        }
      }),
    );

    // ── agent:session_title (auto-generated title is ready) ────────────
    listeners.push(
      listen<{ session_id: string; title: string }>("agent:session_title", () => {
        useAppStore.getState().loadSessions();
      }),
    );

    // ── agent:question_asked ───────────────────────────────────────────
    listeners.push(
      listen<QuestionAskedPayload>("agent:question_asked", (event) => {
        const { thread_id, session_id, question, options } = event.payload;
        useAppStore.getState().setPendingQuestion(thread_id, {
          sessionId: session_id,
          question,
          options: options ?? undefined,
        });
      }),
    );

    // ── agent:plan_ready ───────────────────────────────────────────────
    listeners.push(
      listen<PlanReadyPayload>("agent:plan_ready", (event) => {
        const { thread_id, plan, plan_path, project_path } = event.payload;
        const store = useAppStore.getState();
        if (plan && project_path) {
          store.setCompletedPlan(project_path, { text: plan, projectPath: project_path, planPath: plan_path });
          store.setActivePlanProjectPath(project_path);
        }
        store.softClearAgentStreaming(thread_id);
        store.clearActiveSession(thread_id);
      }),
    );

    // ── agent:todo_updated ─────────────────────────────────────────────
    listeners.push(
      listen<TodoUpdatedPayload>("agent:todo_updated", (event) => {
        const { thread_id, todos } = event.payload;
        useAppStore.getState().setAgentTodos(thread_id, todos);
      }),
    );

    // ── agent:token_usage ──────────────────────────────────────────────
    listeners.push(
      listen<TokenUsagePayload>("agent:token_usage", (event) => {
        const { thread_id, total_tokens, context_limit, cache_read_tokens, cache_creation_tokens } = event.payload;
        useAppStore.getState().setTokenUsage(thread_id, total_tokens, context_limit, cache_read_tokens, cache_creation_tokens);
      }),
    );

    // ── agent:subagent_start / agent:subagent_end ──────────────────────
    listeners.push(
      listen<SubagentStartPayload>("agent:subagent_start", (event) => {
        const { thread_id, parent_tool_call_id, subagent_name, prompt_preview } = event.payload;
        useAppStore.getState().addToolCall(thread_id, {
          toolCallId: parent_tool_call_id,
          toolName: `Subagent: ${subagent_name}`,
          argsSummary: prompt_preview,
          status: "running",
        });
      }),
    );
    listeners.push(
      listen<SubagentEndPayload>("agent:subagent_end", (event) => {
        const { thread_id, parent_tool_call_id, success, summary } = event.payload;
        const existing = useAppStore.getState().agentStreaming[thread_id]?.toolCalls ?? [];
        if (existing.some((tc) => tc.toolCallId === parent_tool_call_id)) {
          useAppStore.getState().updateToolCall(thread_id, parent_tool_call_id, success, summary);
        } else {
          useAppStore.getState().addToolCall(thread_id, {
            toolCallId: parent_tool_call_id,
            toolName: "Subagent",
            argsSummary: "",
            status: success ? "success" : "error",
            summary,
          });
        }
      }),
    );

    // ── context-watcher-status (file-watcher index lifecycle) ──────────
    // Payload is the flattened IndexWatcherStatus: { repo_path, status, file_count?, reason? }.
    listeners.push(
      listen<{ repo_path: string; status: string; file_count?: number | null; reason?: string | null }>(
        "context-watcher-status",
        (event) => {
          const { repo_path, status, file_count, reason } = event.payload;
          useAppStore.getState().setContextWatcherStatus(repo_path, {
            status,
            fileCount: file_count ?? null,
            reason: reason ?? null,
          });
        },
      ),
    );

    // ── engine:status / engine:progress (app-managed stack lifecycle) ──
    listeners.push(
      listen<EngineStatus>("engine:status", (event) => {
        useAppStore.getState().setEngineStatus(event.payload);
      }),
    );
    listeners.push(
      listen<string>("engine:progress", (event) => {
        useAppStore.getState().setEngineProgress(event.payload);
      }),
    );

    // ── agent:auto_* (Auto-mode orchestrator + worker lifecycle, P4/P6/P7) ─
    // Eight events form one task's lifecycle: planning → plan → awaiting →
    // worker_start/end (×N) → optional replan → done/failed. Keyed by run_id
    // in the slice (one entry per Auto turn). thread_id is only used for
    // session-scoped side-effects (Thinking placeholder, approvals).
    listeners.push(
      listen<AutoPlanningEvent>("agent:auto_planning", (event) => {
        const { thread_id, run_id } = event.payload;
        console.info("[Auto-P7] evt auto_planning session=%s run=%s", thread_id, run_id);
        const store = useAppStore.getState();
        store.setAutoPlanning(thread_id, run_id);
        // Auto turns persist their user + placeholder assistant rows in
        // run_auto_turn BEFORE emitting auto_planning, so refreshing the
        // thread now picks up real DB ids — replaces the optimistic user
        // bubble and adds the AutoRunPanel anchor inline at the right
        // position. Same pattern as agent:checkpoint_restored.
        agentTauriService
          .getMessages(thread_id)
          .then((msgs) => {
            const autoRows = msgs.filter((m) => m.auto_run_id);
            console.info("[Auto-P7] refreshed messages count=%d auto_anchors=%d",
              msgs.length, autoRows.length);
            useAppStore
              .getState()
              .replaceThreadMessages(thread_id, msgs.map(displayToAgentMessage));
          })
          .catch((e) => console.warn("[Auto-P7] refresh failed:", e));
      }),
    );
    listeners.push(
      listen<AutoPlanEvent>("agent:auto_plan", (event) => {
        const { run_id, version, reasoning, plan } = event.payload;
        useAppStore.getState().setAutoPlan(run_id, version, reasoning, plan);
      }),
    );
    listeners.push(
      listen<AutoAwaitingApprovalEvent>(
        "agent:auto_awaiting_approval",
        (event) => {
          const { run_id } = event.payload;
          useAppStore.getState().setAutoAwaitingApproval(run_id);
        },
      ),
    );
    listeners.push(
      listen<AutoWorkerStartEvent>("agent:auto_worker_start", (event) => {
        const { run_id, worker_id, model, prompt } = event.payload;
        useAppStore
          .getState()
          .setAutoWorkerStart(run_id, worker_id, model, prompt);
      }),
    );
    listeners.push(
      listen<AutoWorkerEndEvent>("agent:auto_worker_end", (event) => {
        const {
          run_id,
          worker_id,
          summary,
          cost_cents,
          tool_count,
          status,
        } = event.payload;
        useAppStore.getState().appendAutoWorkerEnd(run_id, {
          id: worker_id,
          model: "", // model is in setAutoWorkerStart; not duplicated on end
          prompt: "",
          summary,
          cost_cents,
          tool_count,
          status,
        });
      }),
    );
    listeners.push(
      listen<AutoWorkerToolStartEvent>(
        "agent:auto_worker_tool_start",
        (event) => {
          const { run_id, worker_id, tool_call_id, tool_name, args_summary } =
            event.payload;
          useAppStore
            .getState()
            .appendAutoWorkerToolStart(
              run_id,
              worker_id,
              tool_call_id,
              tool_name,
              args_summary,
            );
        },
      ),
    );
    listeners.push(
      listen<AutoWorkerToolEndEvent>(
        "agent:auto_worker_tool_end",
        (event) => {
          const { run_id, worker_id, tool_call_id, success, summary } =
            event.payload;
          useAppStore
            .getState()
            .patchAutoWorkerToolEnd(
              run_id,
              worker_id,
              tool_call_id,
              success,
              summary,
            );
        },
      ),
    );
    listeners.push(
      listen<AutoReplanEvent>("agent:auto_replan", (event) => {
        const { run_id, reason } = event.payload;
        useAppStore.getState().setAutoReplan(run_id, reason);
      }),
    );
    listeners.push(
      listen<AutoFailedEvent>("agent:auto_failed", (event) => {
        const { thread_id, run_id, reason, last_worker_output } = event.payload;
        console.info("[Auto-P7] evt auto_failed session=%s run=%s reason=%s", thread_id, run_id, reason);
        const store = useAppStore.getState();
        store.setAutoFailed(run_id, reason, last_worker_output);
        // Terminal: stop the "Thinking…" placeholder + tear down any tool
        // activity state that was set when the turn started. Mirrors what
        // the regular `agent:done` handler does for normal sessions.
        store.softClearAgentStreaming(thread_id);
        clearApprovalsForSession(thread_id);
        // P9c: AutoFailed is reused for the new "pause for user decision"
        // path too — refetch the snapshot to see if pending_failure is set,
        // and if so, re-hydrate to awaiting_failure_decision state so the
        // panel renders the Retry / Replan / Cancel banner.
        agentTauriService
          .getAutoRun(run_id)
          .then((snapshot) => {
            if (snapshot?.pending_failure) {
              console.info("[Auto-P7] auto_failed has pending_failure → switching to awaiting_failure_decision");
              // Clear the now-incorrect 'failed' state and re-hydrate from
              // the snapshot (hydrateAutoRun maps awaiting_failure_decision
              // and populates pendingFailure).
              useAppStore.getState().clearAutoRun(run_id);
              useAppStore.getState().hydrateAutoRun(snapshot);
            }
          })
          .catch((e) => console.warn("[Auto-P7] post-auto_failed snapshot fetch failed:", e));
      }),
    );
    listeners.push(
      listen<AutoDoneEvent>("agent:auto_done", (event) => {
        const { thread_id, run_id, summary, total_cost_cents } = event.payload;
        console.info("[Auto-P7] evt auto_done session=%s run=%s cost=%d summary_len=%d",
          thread_id, run_id, total_cost_cents, summary?.length ?? 0);
        const store = useAppStore.getState();
        store.setAutoDone(run_id, summary, total_cost_cents);
        store.softClearAgentStreaming(thread_id);
        clearApprovalsForSession(thread_id);
      }),
    );

    return () => {
      if (deltaFlushTimer) {
        clearTimeout(deltaFlushTimer);
        deltaFlushTimer = null;
        flushDeltas();
      }
      listeners.forEach((p) => p.then((fn) => fn()).catch(() => {}));
    };
  }, []);
}
