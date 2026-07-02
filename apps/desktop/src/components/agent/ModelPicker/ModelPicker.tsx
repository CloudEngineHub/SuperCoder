import { useEffect, useMemo, useCallback, useState } from "react";
import { Bot, ChevronDown, Sparkles } from "lucide-react";
import { useAppStore } from "@/store";
import { agentTauriService } from "@/services/agentTauriService";
import ActionChip from "@/components/common/ActionChip/ActionChip";
import CustomDropdown from "@/components/common/CustomDropdown/CustomDropdown";
import type { CustomDropdownItem } from "@/components/common/CustomDropdown/types";
import {
  AUTO_SENTINEL_MODEL,
  AUTO_SENTINEL_PROVIDER_ID,
  type ProviderConfig,
} from "@/types/agent";

/** Display name for a provider group: kind label, or the host for OpenAI-compatible. */
function providerName(p: ProviderConfig): string {
  if (p.kind === "openai") return "OpenAI";
  if (p.kind === "anthropic") return "Anthropic";
  if (p.label?.trim()) return p.label.trim();
  try {
    return new URL(p.baseUrl).host || "OpenAI-compatible";
  } catch {
    return "OpenAI-compatible";
  }
}

/** Picks the coding model across all configured providers (grouped). When a
 * session is open the picker controls THAT session's model (re-pins it); with no
 * session open it sets the global default that new sessions snapshot. */
export default function ModelPicker() {
  const providers = useAppStore((s) => s.providers);
  const selection = useAppStore((s) => s.selection);
  const providersLoaded = useAppStore((s) => s.providersLoaded);
  const loadProviders = useAppStore((s) => s.loadProviders);
  const setActiveModel = useAppStore((s) => s.setActiveModel);
  const setSessionModel = useAppStore((s) => s.setSessionModel);
  const syncPickerToSession = useAppStore((s) => s.syncPickerToSession);
  const activeThreadId = useAppStore((s) => s.activeAgentThreadId);
  // The open session's pinned model (if any) — the picker reflects/controls it.
  const openSession = useAppStore((s) =>
    activeThreadId ? s.sessions.find((x) => x.id === activeThreadId) : undefined,
  );

  // Auto-mode availability is a Settings flag, not derived from providers.
  // Re-fetched whenever the dropdown opens so a toggle change in Settings
  // shows up without a full reload.
  const [autoEnabled, setAutoEnabled] = useState(false);

  useEffect(() => {
    if (!providersLoaded) loadProviders();
  }, [providersLoaded, loadProviders]);

  // Initial fetch of auto_enabled. Errors are silent (the picker just won't
  // show Auto) — Settings is the place to surface them.
  useEffect(() => {
    agentTauriService
      .getAutoSettings()
      .then((s) => setAutoEnabled(s.enabled))
      .catch(() => setAutoEnabled(false));
  }, []);

  // When a session opens, align the in-memory active model + vision gating to it.
  useEffect(() => {
    if (activeThreadId) syncPickerToSession(activeThreadId);
  }, [activeThreadId, openSession?.model, openSession?.providerId, syncPickerToSession]);

  const handleSelect = useCallback(
    (providerId: string, model: string) => {
      if (activeThreadId) setSessionModel(activeThreadId, providerId, model);
      else setActiveModel(providerId, model);
    },
    [activeThreadId, setSessionModel, setActiveModel],
  );

  // Flatten providers → models as grouped dropdown items (header + model rows).
  // When Auto is enabled in Settings AND the open session is in Code mode,
  // a magic ✨ entry sits at the top with a divider before the first regular
  // provider. Auto is scoped to Code because the orchestrator-of-workers
  // pattern is code-execution-oriented; Plan/Ask surfaces stay picker-only.
  const showAuto = autoEnabled && openSession?.mode === "coding";
  const dropdownItems: CustomDropdownItem[] = useMemo(() => {
    const items: CustomDropdownItem[] = [];
    if (showAuto) {
      items.push({
        key: `${AUTO_SENTINEL_PROVIDER_ID}::${AUTO_SENTINEL_MODEL}`,
        label: "Auto",
        // Sparkles sized + colored to match the Bot icon used by regular
        // models (text-gray-500, w-4 h-4) — Auto is a peer of the other
        // models in the picker, not a banner. The chip prefix when Auto is
        // active uses the same muted color for consistency.
        icon: <Sparkles className="w-4 h-4 text-gray-500" />,
        onClick: () =>
          handleSelect(AUTO_SENTINEL_PROVIDER_ID, AUTO_SENTINEL_MODEL),
      });
    }
    let firstProviderHeader = true;
    for (const p of providers) {
      if (p.models.length === 0) continue;
      items.push({
        key: `hdr-${p.id}`,
        label: providerName(p),
        disabled: true,
        // Visual break between the Auto entry and the regular providers.
        dividerBefore: showAuto && firstProviderHeader,
      });
      firstProviderHeader = false;
      for (const m of p.models) {
        items.push({
          key: `${p.id}::${m}`,
          label: m,
          icon: <Bot className="w-4 h-4 text-gray-500" />,
          onClick: () => handleSelect(p.id, m),
        });
      }
    }
    return items;
  }, [showAuto, providers, handleSelect]);

  // Active model resolution. Auto sentinel is shown as "✨ Auto" (we never
  // want the literal sentinel string to leak into the chip).
  const activeProviderId = openSession?.providerId ?? selection.active?.providerId;
  const activeModel = openSession?.model ?? selection.active?.model;
  const isAuto =
    activeProviderId === AUTO_SENTINEL_PROVIDER_ID &&
    activeModel === AUTO_SENTINEL_MODEL;
  const displayLabel = isAuto ? "Auto" : activeModel ?? "Select model";

  return (
    <CustomDropdown
      items={dropdownItems}
      placement="topLeft"
      searchable
      trigger={
        <ActionChip
          label={displayLabel}
          prefix={
            isAuto ? (
              <Sparkles className="w-4 h-4 text-gray-500" />
            ) : (
              <Bot className="w-4 h-4 text-gray-500" />
            )
          }
          suffix={<ChevronDown className="w-4 h-4 text-gray-500" />}
        />
      }
    />
  );
}
