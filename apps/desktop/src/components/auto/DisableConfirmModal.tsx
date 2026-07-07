import { Modal, Button } from 'antd';
import type { SessionRow } from '@/types/agent';

interface Props {
  open: boolean;
  sessions: SessionRow[];
  onClose: () => void;
}

/**
 * Shown when the user tries to flip Auto OFF while one or more sessions still
 * have it as their active model. Per the locked design, disable is BLOCKED in
 * this case — the user must manually switch (or archive) those sessions first.
 *
 * Read-only here: we list the offending sessions but don't offer to switch
 * them from inside the modal (one task at a time; switching is done from the
 * session list / chat view).
 */
export default function DisableConfirmModal({
  open,
  sessions,
  onClose,
}: Props) {
  return (
    <Modal
      open={open}
      title="Can't disable Auto yet"
      footer={<Button onClick={onClose}>OK</Button>}
      onCancel={onClose}
    >
      <p className="text-sm text-[var(--text-secondary)] mb-3">
        {sessions.length === 1 ? '1 session is' : `${sessions.length} sessions are`}{' '}
        still using Auto. Switch{' '}
        {sessions.length === 1 ? 'it' : 'them'} to a regular model (or archive)
        before disabling.
      </p>
      <ul className="text-sm text-[var(--text-primary)] max-h-64 overflow-auto border border-[var(--border)] rounded-md divide-y divide-[var(--border)]">
        {sessions.map((s) => (
          <li key={s.id} className="px-3 py-2">
            <div className="font-medium truncate">{s.title ?? s.id}</div>
            <div className="text-xs text-[var(--text-secondary)] truncate">
              {s.folder}
            </div>
          </li>
        ))}
      </ul>
    </Modal>
  );
}
