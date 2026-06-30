import { useEffect, useMemo, useState } from 'react';
import { Modal, Select, Input, Button } from 'antd';
import type { ProviderConfig, WorkerPoolEntry } from '@/types/agent';
import { suggestedDescription } from './knownDescriptions';

interface Props {
  open: boolean;
  providers: ProviderConfig[];
  /** Existing pool — used to prevent adding a duplicate {provider, model}. */
  existingPool: WorkerPoolEntry[];
  onCancel: () => void;
  onAdd: (entry: WorkerPoolEntry) => void;
}

/**
 * Modal for adding one worker to the pool. Constrained dropdowns by design:
 * the user picks from already-configured providers, then from that provider's
 * curated model list. This catches typos that would otherwise surface as
 * runtime "model not found" failures during a real Auto task.
 *
 * On known models the description auto-fills from `KNOWN_MODEL_DESCRIPTIONS`;
 * the user can edit before saving.
 */
export default function AddWorkerDialog({
  open,
  providers,
  existingPool,
  onCancel,
  onAdd,
}: Props) {
  const [providerId, setProviderId] = useState<string | undefined>();
  const [model, setModel] = useState<string | undefined>();
  const [description, setDescription] = useState<string>('');

  // Reset state every time the modal opens so a previous edit doesn't bleed in.
  useEffect(() => {
    if (open) {
      setProviderId(undefined);
      setModel(undefined);
      setDescription('');
    }
  }, [open]);

  const providerOptions = useMemo(
    () =>
      providers
        // Only providers with at least one configured model are useful.
        .filter((p) => p.models.length > 0)
        .map((p) => ({
          value: p.id,
          label: p.label?.trim() || p.kind || p.id,
        })),
    [providers],
  );

  const modelOptions = useMemo(() => {
    const provider = providers.find((p) => p.id === providerId);
    if (!provider) return [];
    // Hide duplicates already in the pool (per-provider).
    const alreadyInPool = new Set(
      existingPool
        .filter((e) => e.providerId === providerId)
        .map((e) => e.model),
    );
    return provider.models
      .filter((m) => !alreadyInPool.has(m))
      .map((m) => ({ value: m, label: m }));
  }, [providers, providerId, existingPool]);

  const handleModelChange = (m: string) => {
    setModel(m);
    // Pre-fill suggested description for known models; never clobber an
    // edit the user has already typed.
    if (description.trim() === '') {
      setDescription(suggestedDescription(m));
    }
  };

  const canAdd =
    !!providerId && !!model && description.trim().length > 0;

  const handleAdd = () => {
    if (!providerId || !model) return;
    onAdd({ providerId, model, description: description.trim() });
  };

  return (
    <Modal
      open={open}
      title="Add worker to pool"
      onCancel={onCancel}
      footer={
        <>
          <Button onClick={onCancel}>Cancel</Button>
          <Button type="primary" disabled={!canAdd} onClick={handleAdd}>
            Add
          </Button>
        </>
      }
      destroyOnClose
    >
      <div className="flex flex-col gap-4 mt-2">
        <div>
          <label className="block text-xs text-[var(--text-secondary)] mb-1">
            Provider
          </label>
          <Select
            className="w-full"
            value={providerId}
            options={providerOptions}
            placeholder="Pick a provider"
            onChange={(v) => {
              setProviderId(v);
              setModel(undefined);
            }}
            notFoundContent="Configure a provider first in Settings → LLM Providers"
          />
        </div>

        <div>
          <label className="block text-xs text-[var(--text-secondary)] mb-1">
            Model
          </label>
          <Select
            className="w-full"
            value={model}
            options={modelOptions}
            placeholder={
              providerId ? 'Pick a model' : 'Pick a provider first'
            }
            disabled={!providerId}
            onChange={handleModelChange}
            showSearch
            notFoundContent={
              providerId
                ? "This provider has no models left to add — they're all already in the pool."
                : 'Pick a provider first'
            }
          />
        </div>

        <div>
          <label className="block text-xs text-[var(--text-secondary)] mb-1">
            Description
          </label>
          <Input.TextArea
            value={description}
            onChange={(e) => setDescription(e.target.value)}
            placeholder="e.g. Fast and cheap; best for reads and simple writes."
            autoSize={{ minRows: 2, maxRows: 5 }}
          />
          <div className="text-[11px] text-[var(--text-secondary)] mt-1">
            The orchestrator sees this when picking models. Keep it short and
            capability-focused — when to use, what it's bad at.
          </div>
        </div>
      </div>
    </Modal>
  );
}
