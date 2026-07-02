use std::path::Path;
use std::sync::Arc;

use async_trait::async_trait;
use parking_lot::Mutex;
use rusqlite::Connection;

use agent::persistence::{
    self, AgentMessage, MessagePersister, MessageRole, MessageType, PersistError, PersistResult,
    Sender,
};

// ── AgentDb ────────────────────────────────────────────────────────────────
//
// Greenfield single-user local store. Schema v1 — no migration ladder carried
// over from the chat product. Sessions are the unit of work (folder + mode);
// messages are keyed purely by `session_id`. Checkpoints live outside SQLite,
// in the file-snapshot dir managed by `git-ops`.

/// Local SQLite database for agent sessions + LLM message history.
pub struct AgentDb {
    conn: Mutex<Connection>,
}

/// A row from the `sessions` table — the unit shown in the session-list sidebar.
#[derive(Debug, Clone, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SessionRow {
    pub id: String,
    pub folder: String,
    /// "ask" | "plan" | "coding"
    pub mode: String,
    pub title: Option<String>,
    pub parent_session_id: Option<String>,
    pub created_at: String,
    pub updated_at: String,
    /// "active" | "idle" | "error"
    pub status: String,
    /// Provider + model this session was created with (resume uses the same one).
    /// `None` for legacy sessions → falls back to the active selection.
    pub provider_id: Option<String>,
    pub model: Option<String>,
}

/// Raw row from the `agent_messages` table.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct StoredMessage {
    pub id: i64,
    pub session_id: String,
    pub project_path: String,
    pub role: String,
    pub type_: String,
    pub llm_message: String,
    pub metadata: String,
    pub created_at: String,
    pub rewound_at: Option<String>,
    pub turn_count: Option<u32>,
}

/// Error type for AgentDb operations.
#[derive(Debug, thiserror::Error)]
pub enum AgentDbError {
    #[error("SQLite error: {0}")]
    Sqlite(#[from] rusqlite::Error),
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),
}

impl AgentDb {
    /// Open or create `agent_data.db` in the given directory with the v1 schema.
    pub fn new(data_dir: &Path) -> Result<Self, AgentDbError> {
        std::fs::create_dir_all(data_dir)?;
        let db_path = data_dir.join("agent_data.db");
        let conn = Connection::open(&db_path)?;

        conn.execute_batch(
            "PRAGMA journal_mode = WAL;
             PRAGMA busy_timeout = 5000;
             PRAGMA synchronous = NORMAL;
             PRAGMA wal_autocheckpoint = 400;
             PRAGMA cache_size = -32768;",
        )?;

        // Fresh-DB tuning: only set page_size before the first table is written.
        let initialized: bool = conn
            .query_row(
                "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='sessions'",
                [],
                |row| row.get::<_, i64>(0),
            )
            .map(|c| c > 0)?;
        if !initialized {
            conn.execute_batch("PRAGMA page_size = 8192;")?;
        }

        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS sessions (
                 id                TEXT PRIMARY KEY,
                 folder            TEXT NOT NULL,
                 mode              TEXT NOT NULL,
                 title             TEXT,
                 parent_session_id TEXT,
                 created_at        TEXT NOT NULL,
                 updated_at        TEXT NOT NULL,
                 status            TEXT NOT NULL DEFAULT 'active',
                 provider_id       TEXT,
                 model             TEXT,
                 deleted_at        TEXT
             );
             CREATE INDEX IF NOT EXISTS idx_sessions_updated ON sessions(updated_at DESC);
             CREATE INDEX IF NOT EXISTS idx_sessions_folder  ON sessions(folder);

             CREATE TABLE IF NOT EXISTS agent_messages (
                 id            INTEGER PRIMARY KEY AUTOINCREMENT,
                 session_id    TEXT NOT NULL DEFAULT '',
                 project_path  TEXT NOT NULL DEFAULT '',
                 role          TEXT NOT NULL,
                 type          TEXT NOT NULL,
                 llm_message   TEXT NOT NULL,
                 metadata      TEXT NOT NULL DEFAULT '{}',
                 created_at    TEXT NOT NULL,
                 rewound_at    TEXT DEFAULT NULL,
                 turn_count    INTEGER DEFAULT NULL
             );
             CREATE INDEX IF NOT EXISTS idx_msgs_session
                 ON agent_messages(session_id, created_at);
             CREATE INDEX IF NOT EXISTS idx_msgs_session_active
                 ON agent_messages(session_id, rewound_at);
             CREATE INDEX IF NOT EXISTS idx_msgs_type
                 ON agent_messages(session_id, type);

             CREATE TABLE IF NOT EXISTS skill_prefs (
                 skill_name TEXT PRIMARY KEY,
                 enabled    INTEGER NOT NULL DEFAULT 1
             );

             CREATE TABLE IF NOT EXISTS subagent_prefs (
                 subagent_name TEXT PRIMARY KEY,
                 enabled       INTEGER NOT NULL DEFAULT 1
             );

             CREATE TABLE IF NOT EXISTS auto_runs (
                 id              TEXT PRIMARY KEY,
                 session_id      TEXT NOT NULL,
                 status          TEXT NOT NULL,
                 plan_versions   TEXT NOT NULL DEFAULT '[]',
                 worker_results  TEXT NOT NULL DEFAULT '[]',
                 cost_cents      INTEGER NOT NULL DEFAULT 0,
                 replans_used    INTEGER NOT NULL DEFAULT 0,
                 current_worker  TEXT NOT NULL DEFAULT 'null',
                 pending_failure TEXT NOT NULL DEFAULT 'null',
                 created_at      TEXT NOT NULL,
                 updated_at      TEXT NOT NULL
             );
             CREATE INDEX IF NOT EXISTS idx_auto_runs_session
                 ON auto_runs(session_id, updated_at DESC);

             PRAGMA user_version = 1;",
        )?;

        // Idempotent migrations: add `provider_id` + `model` to sessions tables
        // created before multi-provider support. Ignore "duplicate column".
        for col in ["provider_id", "model", "deleted_at"] {
            if let Err(e) = conn.execute(&format!("ALTER TABLE sessions ADD COLUMN {col} TEXT"), []) {
                if !e.to_string().contains("duplicate column") {
                    return Err(e.into());
                }
            }
        }
        // Idempotent migration: `current_worker` lets a mid-flight reload
        // show "Running: wN (model)" instead of just "status: running".
        // Stored as JSON; literal 'null' means no worker in flight.
        if let Err(e) = conn.execute(
            "ALTER TABLE auto_runs ADD COLUMN current_worker TEXT NOT NULL DEFAULT 'null'",
            [],
        ) {
            if !e.to_string().contains("duplicate column") {
                return Err(e.into());
            }
        }
        // Idempotent migration: `pending_failure` carries the
        // WorkerFailureContext for the user-driven Retry/Replan/Cancel
        // banner (P9c). JSON; literal 'null' means no pending failure.
        if let Err(e) = conn.execute(
            "ALTER TABLE auto_runs ADD COLUMN pending_failure TEXT NOT NULL DEFAULT 'null'",
            [],
        ) {
            if !e.to_string().contains("duplicate column") {
                return Err(e.into());
            }
        }

        Ok(Self {
            conn: Mutex::new(conn),
        })
    }

    #[allow(dead_code)]
    pub fn schema_version(&self) -> Result<i32, AgentDbError> {
        let conn = self.conn.lock();
        Ok(conn.query_row("PRAGMA user_version", [], |row| row.get(0))?)
    }

    // ── Sessions ─────────────────────────────────────────────────────────

    /// Insert a new session row. `created_at`/`updated_at` are set to now.
    #[allow(clippy::too_many_arguments)]
    pub fn create_session(
        &self,
        id: &str,
        folder: &str,
        mode: &str,
        title: Option<&str>,
        parent_session_id: Option<&str>,
        provider_id: Option<&str>,
        model: Option<&str>,
    ) -> Result<(), AgentDbError> {
        let conn = self.conn.lock();
        let now = now_iso();
        conn.execute(
            "INSERT INTO sessions (id, folder, mode, title, parent_session_id, created_at, updated_at, status, provider_id, model)
             VALUES (?, ?, ?, ?, ?, ?, ?, 'idle', ?, ?)",
            rusqlite::params![id, folder, mode, title, parent_session_id, now, now, provider_id, model],
        )?;
        Ok(())
    }

    /// Get a single session by id.
    pub fn get_session(&self, id: &str) -> Result<Option<SessionRow>, AgentDbError> {
        let conn = self.conn.lock();
        let result = conn.query_row(
            "SELECT id, folder, mode, title, parent_session_id, created_at, updated_at, status, provider_id, model
             FROM sessions WHERE id = ?",
            [id],
            map_session_row,
        );
        match result {
            Ok(row) => Ok(Some(row)),
            Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
            Err(e) => Err(e.into()),
        }
    }

    /// List all sessions, most-recently-updated first.
    pub fn list_sessions(&self) -> Result<Vec<SessionRow>, AgentDbError> {
        let conn = self.conn.lock();
        let mut stmt = conn.prepare(
            "SELECT id, folder, mode, title, parent_session_id, created_at, updated_at, status, provider_id, model
             FROM sessions WHERE deleted_at IS NULL ORDER BY updated_at DESC",
        )?;
        let rows = stmt.query_map([], map_session_row)?;
        let mut out = Vec::new();
        for r in rows {
            out.push(r?);
        }
        Ok(out)
    }

    /// Set a session's status and bump `updated_at`.
    pub fn set_session_status(&self, id: &str, status: &str) -> Result<(), AgentDbError> {
        let conn = self.conn.lock();
        conn.execute(
            "UPDATE sessions SET status = ?, updated_at = ? WHERE id = ?",
            rusqlite::params![status, now_iso(), id],
        )?;
        Ok(())
    }

    /// Soft-delete a session: stamp `deleted_at` so it drops out of the sidebar
    /// list while its messages + checkpoints stay on disk (recoverable).
    pub fn delete_session(&self, id: &str) -> Result<(), AgentDbError> {
        let conn = self.conn.lock();
        conn.execute(
            "UPDATE sessions SET deleted_at = ?, updated_at = ? WHERE id = ?",
            rusqlite::params![now_iso(), now_iso(), id],
        )?;
        Ok(())
    }

    /// Set a session's title (first message preview, etc.) and bump `updated_at`.
    pub fn set_session_title(&self, id: &str, title: &str) -> Result<(), AgentDbError> {
        let conn = self.conn.lock();
        conn.execute(
            "UPDATE sessions SET title = ?, updated_at = ? WHERE id = ?",
            rusqlite::params![title, now_iso(), id],
        )?;
        Ok(())
    }

    /// Re-pin a session's provider + model (when the user switches the picker for
    /// an open session). Subsequent turns + context sizing use the new model.
    pub fn set_session_model(&self, id: &str, provider_id: &str, model: &str) -> Result<(), AgentDbError> {
        let conn = self.conn.lock();
        conn.execute(
            "UPDATE sessions SET provider_id = ?, model = ?, updated_at = ? WHERE id = ?",
            rusqlite::params![provider_id, model, now_iso(), id],
        )?;
        Ok(())
    }

    /// Set a session's current mode ("ask" | "plan" | "coding") and bump `updated_at`.
    pub fn set_session_mode(&self, id: &str, mode: &str) -> Result<(), AgentDbError> {
        let conn = self.conn.lock();
        conn.execute(
            "UPDATE sessions SET mode = ?, updated_at = ? WHERE id = ?",
            rusqlite::params![mode, now_iso(), id],
        )?;
        Ok(())
    }

    /// Bump a session's `updated_at` to now (recency for the sidebar).
    pub fn touch_session(&self, id: &str) -> Result<(), AgentDbError> {
        let conn = self.conn.lock();
        conn.execute(
            "UPDATE sessions SET updated_at = ? WHERE id = ?",
            rusqlite::params![now_iso(), id],
        )?;
        Ok(())
    }

    /// Return the id of an active session for the given folder, if any.
    /// Used to enforce one active session per folder.
    pub fn active_session_for_folder(&self, folder: &str) -> Result<Option<String>, AgentDbError> {
        let conn = self.conn.lock();
        let result = conn.query_row(
            "SELECT id FROM sessions WHERE folder = ? AND status = 'active' AND deleted_at IS NULL ORDER BY updated_at DESC LIMIT 1",
            [folder],
            |row| row.get::<_, String>(0),
        );
        match result {
            Ok(id) => Ok(Some(id)),
            Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
            Err(e) => Err(e.into()),
        }
    }

    // ── Skill / subagent prefs ───────────────────────────────────────────

    pub fn load_disabled_skills(&self) -> Result<std::collections::HashSet<String>, AgentDbError> {
        let conn = self.conn.lock();
        let mut stmt = conn.prepare("SELECT skill_name FROM skill_prefs WHERE enabled = 0")?;
        let rows = stmt.query_map([], |r| r.get::<_, String>(0))?;
        Ok(rows.filter_map(Result::ok).collect())
    }

    pub fn set_skill_enabled(&self, name: &str, enabled: bool) -> Result<(), AgentDbError> {
        let conn = self.conn.lock();
        conn.execute(
            "INSERT INTO skill_prefs (skill_name, enabled) VALUES (?1, ?2)
             ON CONFLICT(skill_name) DO UPDATE SET enabled = excluded.enabled",
            rusqlite::params![name, enabled as i32],
        )?;
        Ok(())
    }

    pub fn load_disabled_subagents(
        &self,
    ) -> Result<std::collections::HashSet<String>, AgentDbError> {
        let conn = self.conn.lock();
        let mut stmt = conn.prepare("SELECT subagent_name FROM subagent_prefs WHERE enabled = 0")?;
        let rows = stmt.query_map([], |r| r.get::<_, String>(0))?;
        Ok(rows.filter_map(Result::ok).collect())
    }

    pub fn set_subagent_enabled(&self, name: &str, enabled: bool) -> Result<(), AgentDbError> {
        let conn = self.conn.lock();
        conn.execute(
            "INSERT INTO subagent_prefs (subagent_name, enabled) VALUES (?1, ?2)
             ON CONFLICT(subagent_name) DO UPDATE SET enabled = excluded.enabled",
            rusqlite::params![name, enabled as i32],
        )?;
        Ok(())
    }

    // ── Messages ─────────────────────────────────────────────────────────

    /// Insert a message for a session. Returns the row id as a string.
    #[allow(clippy::too_many_arguments)]
    pub fn insert_message(
        &self,
        session_id: &str,
        project_path: &str,
        role: &str,
        type_: &str,
        llm_message: &str,
        metadata: &str,
        turn_count: Option<u32>,
    ) -> Result<String, AgentDbError> {
        let conn = self.conn.lock();
        conn.execute(
            "INSERT INTO agent_messages (session_id, project_path, role, type, llm_message, metadata, created_at, turn_count)
             VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            rusqlite::params![session_id, project_path, role, type_, llm_message, metadata, now_iso(), turn_count],
        )?;
        Ok(conn.last_insert_rowid().to_string())
    }

    /// Load all active (non-rewound) rows for a session, ordered by id ASC.
    pub fn load_session_messages(
        &self,
        session_id: &str,
    ) -> Result<Vec<StoredMessage>, AgentDbError> {
        let conn = self.conn.lock();
        let mut stmt = conn.prepare(
            "SELECT id, session_id, project_path, role, type, llm_message, metadata, created_at, rewound_at, turn_count
             FROM agent_messages
             WHERE session_id = ? AND rewound_at IS NULL
             ORDER BY id ASC",
        )?;
        let rows = stmt.query_map([session_id], map_stored_message)?;
        let mut out = Vec::new();
        for r in rows {
            out.push(r?);
        }
        Ok(out)
    }

    /// Load a session's messages optimized for LLM context (compaction-aware).
    /// Mirrors the historical behaviour: load from the last compaction marker
    /// (honoring `kept_before_count`), else fall back to the most recent
    /// `fallback_limit` rows; then trim leading orphaned tool rows.
    pub fn load_session_for_context(
        &self,
        session_id: &str,
        fallback_limit: u32,
    ) -> Result<Vec<StoredMessage>, AgentDbError> {
        let conn = self.conn.lock();

        let reset_floor: i64 = conn
            .query_row(
                "SELECT COALESCE(MAX(id), 0) FROM agent_messages
                 WHERE session_id = ? AND type = 'context_reset' AND rewound_at IS NULL",
                [session_id],
                |row| row.get(0),
            )
            .unwrap_or(0);

        let last_compaction_id: Option<i64> = conn
            .query_row(
                "SELECT MAX(id) FROM agent_messages
                 WHERE session_id = ? AND type = 'compaction' AND rewound_at IS NULL AND id > ?",
                rusqlite::params![session_id, reset_floor],
                |row| row.get(0),
            )
            .unwrap_or(None);

        let mut messages = if let Some(compaction_id) = last_compaction_id {
            let kept_before: i64 = conn
                .query_row(
                    "SELECT COALESCE(json_extract(metadata, '$.kept_before_count'), 0) FROM agent_messages WHERE id = ?",
                    [compaction_id],
                    |row| row.get(0),
                )
                .unwrap_or(0);

            let start_id: i64 = if kept_before > 0 {
                conn.query_row(
                    "SELECT COALESCE(MIN(id), ?) FROM (
                        SELECT id FROM agent_messages
                        WHERE session_id = ? AND id < ? AND id > ?
                          AND type NOT IN ('compaction', 'context_usage', 'context_reset')
                          AND rewound_at IS NULL
                        ORDER BY id DESC LIMIT ?
                    )",
                    rusqlite::params![compaction_id, session_id, compaction_id, reset_floor, kept_before],
                    |row| row.get(0),
                )
                .unwrap_or(compaction_id)
            } else {
                compaction_id
            };

            let mut stmt = conn.prepare(
                "SELECT id, session_id, project_path, role, type, llm_message, metadata, created_at, rewound_at, turn_count
                 FROM agent_messages
                 WHERE session_id = ? AND id >= ?
                   AND type NOT IN ('context_usage', 'context_reset')
                   AND rewound_at IS NULL
                 ORDER BY id ASC",
            )?;
            let rows = stmt.query_map(rusqlite::params![session_id, start_id], map_stored_message)?;
            let mut msgs = Vec::new();
            for r in rows {
                msgs.push(r?);
            }
            msgs
        } else {
            let mut stmt = conn.prepare(
                "SELECT id, session_id, project_path, role, type, llm_message, metadata, created_at, rewound_at, turn_count
                 FROM agent_messages
                 WHERE session_id = ? AND id > ?
                   AND type NOT IN ('context_usage', 'context_reset')
                   AND rewound_at IS NULL
                 ORDER BY id DESC LIMIT ?",
            )?;
            let rows = stmt.query_map(
                rusqlite::params![session_id, reset_floor, fallback_limit],
                map_stored_message,
            )?;
            let mut msgs = Vec::new();
            for r in rows {
                msgs.push(r?);
            }
            msgs.reverse();
            msgs
        };

        // Trim leading orphaned tool rows so the context starts clean.
        while let Some(first) = messages.first() {
            if first.role == "tool" || (first.role == "assistant" && first.type_ == "tool_call") {
                messages.remove(0);
            } else {
                break;
            }
        }

        Ok(messages)
    }

    /// Get the latest `session_init` record for a session (if any).
    pub fn get_session_init(
        &self,
        session_id: &str,
    ) -> Result<Option<StoredMessage>, AgentDbError> {
        let conn = self.conn.lock();
        let result = conn.query_row(
            "SELECT id, session_id, project_path, role, type, llm_message, metadata, created_at, rewound_at, turn_count
             FROM agent_messages
             WHERE session_id = ? AND type = 'session_init'
             ORDER BY id DESC LIMIT 1",
            [session_id],
            map_stored_message,
        );
        match result {
            Ok(msg) => Ok(Some(msg)),
            Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
            Err(e) => Err(e.into()),
        }
    }

    /// Get a single message by SQLite row id.
    pub fn get_message_by_id(&self, id: i64) -> Result<Option<StoredMessage>, AgentDbError> {
        let conn = self.conn.lock();
        let result = conn.query_row(
            "SELECT id, session_id, project_path, role, type, llm_message, metadata, created_at, rewound_at, turn_count
             FROM agent_messages WHERE id = ?",
            [id],
            map_stored_message,
        );
        match result {
            Ok(msg) => Ok(Some(msg)),
            Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
            Err(e) => Err(e.into()),
        }
    }

    /// Soft-delete all messages with id >= from_id for a session (rewind).
    pub fn rewind_messages(
        &self,
        session_id: &str,
        from_id: i64,
    ) -> Result<usize, AgentDbError> {
        let conn = self.conn.lock();
        conn.execute(
            "UPDATE agent_messages SET rewound_at = ?
             WHERE session_id = ? AND id >= ? AND rewound_at IS NULL",
            rusqlite::params![now_iso(), session_id, from_id],
        )?;
        Ok(conn.changes() as usize)
    }

    /// Overwrite the `llm_message` JSON of an existing row. Used by the Auto
    /// path to update a placeholder assistant row's content once the run
    /// terminates (last worker summary, failure reason, or "cancelled").
    pub fn update_message_llm_content(
        &self,
        row_id: i64,
        llm_message: &str,
    ) -> Result<(), AgentDbError> {
        let conn = self.conn.lock();
        conn.execute(
            "UPDATE agent_messages SET llm_message = ? WHERE id = ?",
            rusqlite::params![llm_message, row_id],
        )?;
        Ok(())
    }

    /// Soft-delete all messages with turn_count >= from_turn for a session.
    /// Used after a checkpoint restore so the conversation matches the files.
    pub fn rewind_from_turn(
        &self,
        session_id: &str,
        from_turn: u32,
    ) -> Result<usize, AgentDbError> {
        let conn = self.conn.lock();
        conn.execute(
            "UPDATE agent_messages SET rewound_at = ?
             WHERE session_id = ? AND turn_count >= ? AND rewound_at IS NULL",
            rusqlite::params![now_iso(), session_id, from_turn],
        )?;
        Ok(conn.changes() as usize)
    }

    // ── Context usage / reset ────────────────────────────────────────────

    /// Upsert token-usage stats for a session (one row per session).
    pub fn upsert_context_usage(
        &self,
        session_id: &str,
        project_path: &str,
        total_tokens: u32,
        context_limit: u32,
        message_count: u32,
    ) -> Result<(), AgentDbError> {
        let conn = self.conn.lock();
        let metadata = serde_json::json!({
            "total_tokens": total_tokens,
            "context_limit": context_limit,
            "message_count": message_count,
        })
        .to_string();

        let tx = conn.unchecked_transaction()?;
        tx.execute(
            "DELETE FROM agent_messages WHERE session_id = ? AND type = 'context_usage'",
            [session_id],
        )?;
        tx.execute(
            "INSERT INTO agent_messages (session_id, project_path, role, type, llm_message, metadata, created_at)
             VALUES (?, ?, 'system', 'context_usage', '{}', ?, ?)",
            rusqlite::params![session_id, project_path, metadata, now_iso()],
        )?;
        tx.commit()?;
        Ok(())
    }

    /// Get persisted (total_tokens, context_limit, message_count) for a session.
    pub fn get_context_usage(
        &self,
        session_id: &str,
    ) -> Result<Option<(u32, u32, u32)>, AgentDbError> {
        let conn = self.conn.lock();
        let result = conn.query_row(
            "SELECT metadata FROM agent_messages
             WHERE session_id = ? AND type = 'context_usage' LIMIT 1",
            [session_id],
            |row| row.get::<_, String>(0),
        );
        match result {
            Ok(meta_str) => {
                let meta: serde_json::Value = serde_json::from_str(&meta_str).unwrap_or_default();
                Ok(Some((
                    meta["total_tokens"].as_u64().unwrap_or(0) as u32,
                    meta["context_limit"].as_u64().unwrap_or(0) as u32,
                    meta["message_count"].as_u64().unwrap_or(0) as u32,
                )))
            }
            Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
            Err(e) => Err(e.into()),
        }
    }

    /// Delete the context_usage row for a session (after manual compaction).
    pub fn delete_context_usage(&self, session_id: &str) -> Result<(), AgentDbError> {
        let conn = self.conn.lock();
        conn.execute(
            "DELETE FROM agent_messages WHERE session_id = ? AND type = 'context_usage'",
            [session_id],
        )?;
        Ok(())
    }

    /// Insert a context_reset marker (hard floor for context loading) and clear usage.
    pub fn insert_context_reset(
        &self,
        session_id: &str,
        project_path: &str,
    ) -> Result<(), AgentDbError> {
        let conn = self.conn.lock();
        conn.execute(
            "INSERT INTO agent_messages (session_id, project_path, role, type, llm_message, metadata, created_at)
             VALUES (?, ?, 'system', 'context_reset', '{}', '{}', ?)",
            rusqlite::params![session_id, project_path, now_iso()],
        )?;
        conn.execute(
            "DELETE FROM agent_messages WHERE session_id = ? AND type = 'context_usage'",
            [session_id],
        )?;
        Ok(())
    }

    /// Insert a compaction record. Mirrors the agent loop's own persisted format.
    pub fn insert_compaction(
        &self,
        session_id: &str,
        project_path: &str,
        summary: &str,
        kept_before_count: u32,
    ) -> Result<String, AgentDbError> {
        let metadata = serde_json::json!({
            "version": 1,
            "kept_before_count": kept_before_count,
        })
        .to_string();
        let llm = serde_json::json!({"role": "system", "content": summary}).to_string();
        self.insert_message(session_id, project_path, "system", "compaction", &llm, &metadata, None)
    }

    // ── Auto-mode runs ────────────────────────────────────────────────────

    /// Insert a fresh `auto_runs` row when a new Auto task starts.
    /// `plan_versions` and `worker_results` start as empty JSON arrays; the
    /// executor calls `update_auto_run` after each phase to flush state.
    pub fn insert_auto_run(&self, run: &agent::auto::AutoRun) -> Result<(), AgentDbError> {
        let conn = self.conn.lock();
        let plan_versions = serde_json::to_string(&run.plan_versions)?;
        let worker_results = serde_json::to_string(&run.worker_results)?;
        let current_worker = serde_json::to_string(&run.current_worker)?;
        let pending_failure = serde_json::to_string(&run.pending_failure)?;
        let status = serde_json::to_string(&run.status)?
            .trim_matches('"')
            .to_string();
        let now = chrono::Utc::now().to_rfc3339();
        conn.execute(
            "INSERT INTO auto_runs
                 (id, session_id, status, plan_versions, worker_results,
                  cost_cents, replans_used, current_worker, pending_failure,
                  created_at, updated_at)
             VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            rusqlite::params![
                run.id,
                run.session_id,
                status,
                plan_versions,
                worker_results,
                run.cost_cents as i64,
                run.replans_used as i64,
                current_worker,
                pending_failure,
                now,
                now,
            ],
        )?;
        Ok(())
    }

    /// Overwrite an existing `auto_runs` row with the executor's latest snapshot.
    /// Called after each plan / worker / replan / terminal transition so the
    /// row mirrors the in-memory `AutoRun`.
    pub fn update_auto_run(&self, run: &agent::auto::AutoRun) -> Result<(), AgentDbError> {
        let conn = self.conn.lock();
        let plan_versions = serde_json::to_string(&run.plan_versions)?;
        let worker_results = serde_json::to_string(&run.worker_results)?;
        let current_worker = serde_json::to_string(&run.current_worker)?;
        let pending_failure = serde_json::to_string(&run.pending_failure)?;
        let status = serde_json::to_string(&run.status)?
            .trim_matches('"')
            .to_string();
        let now = chrono::Utc::now().to_rfc3339();
        conn.execute(
            "UPDATE auto_runs SET
                 status = ?, plan_versions = ?, worker_results = ?,
                 cost_cents = ?, replans_used = ?, current_worker = ?,
                 pending_failure = ?, updated_at = ?
             WHERE id = ?",
            rusqlite::params![
                status,
                plan_versions,
                worker_results,
                run.cost_cents as i64,
                run.replans_used as i64,
                current_worker,
                pending_failure,
                now,
                run.id,
            ],
        )?;
        Ok(())
    }

    /// Load a single `auto_runs` row by id. Returns `None` if missing.
    pub fn get_auto_run(&self, id: &str) -> Result<Option<agent::auto::AutoRun>, AgentDbError> {
        let conn = self.conn.lock();
        let mut stmt = conn.prepare(
            "SELECT id, session_id, status, plan_versions, worker_results,
                    cost_cents, replans_used, current_worker, pending_failure
             FROM auto_runs WHERE id = ?",
        )?;
        let mut rows = stmt.query(rusqlite::params![id])?;
        if let Some(row) = rows.next()? {
            let status_raw: String = row.get(2)?;
            let plan_versions_raw: String = row.get(3)?;
            let worker_results_raw: String = row.get(4)?;
            let current_worker_raw: String = row.get(7)?;
            let pending_failure_raw: String = row.get(8)?;
            // Status round-trip: serde wraps strings in quotes, so we add them
            // back before deserializing the snake_case-renamed enum.
            let status: agent::auto::AutoStatus =
                serde_json::from_str(&format!("\"{}\"", status_raw))?;
            let plan_versions = serde_json::from_str(&plan_versions_raw)?;
            let worker_results = serde_json::from_str(&worker_results_raw)?;
            let current_worker = serde_json::from_str(&current_worker_raw)?;
            let pending_failure = serde_json::from_str(&pending_failure_raw)?;
            Ok(Some(agent::auto::AutoRun {
                id: row.get(0)?,
                session_id: row.get(1)?,
                status,
                plan_versions,
                worker_results,
                cost_cents: row.get::<_, i64>(5)? as u32,
                replans_used: row.get::<_, i64>(6)? as u32,
                current_worker,
                pending_failure,
            }))
        } else {
            Ok(None)
        }
    }

    /// Mark zombie `auto_runs` rows as `failed` at app startup. Reaps rows
    /// where there was an actively-executing task that died (planning =
    /// orchestrator in flight, running = workers in flight). Deliberately
    /// SKIPS `awaiting_approval` and `awaiting_failure_decision` — those
    /// rows have no in-flight work and the resume path spawns a fresh
    /// executor on the user's button click. Returns rows updated.
    pub fn mark_zombie_auto_runs_failed(&self) -> Result<usize, AgentDbError> {
        let conn = self.conn.lock();
        let n = conn.execute(
            "UPDATE auto_runs
               SET status = 'failed', updated_at = ?
             WHERE status IN ('planning', 'running')",
            rusqlite::params![now_iso()],
        )?;
        Ok(n)
    }

    /// Find the `agent_messages.id` of the placeholder assistant row that
    /// anchors a given Auto run, used by the Resume path to reuse the
    /// existing chat-thread anchor instead of creating duplicates.
    pub fn find_auto_assistant_row_id(&self, auto_run_id: &str) -> Result<Option<i64>, AgentDbError> {
        let conn = self.conn.lock();
        let result = conn.query_row(
            "SELECT id FROM agent_messages
             WHERE role = 'assistant'
               AND json_extract(metadata, '$.auto_run_id') = ?
               AND rewound_at IS NULL
             ORDER BY id ASC LIMIT 1",
            [auto_run_id],
            |row| row.get::<_, i64>(0),
        );
        match result {
            Ok(id) => Ok(Some(id)),
            Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
            Err(e) => Err(e.into()),
        }
    }

    /// Recover the original user prompt for an Auto run (the user row
    /// `run_auto_turn` persisted at start). Needed by the Resume path so the
    /// executor can call the orchestrator on replan with the same task.
    pub fn find_auto_user_prompt(&self, auto_run_id: &str) -> Result<Option<String>, AgentDbError> {
        let conn = self.conn.lock();
        let result = conn.query_row(
            "SELECT llm_message FROM agent_messages
             WHERE role = 'user'
               AND json_extract(metadata, '$.auto_run_id') = ?
               AND rewound_at IS NULL
             ORDER BY id ASC LIMIT 1",
            [auto_run_id],
            |row| row.get::<_, String>(0),
        );
        let llm_json = match result {
            Ok(j) => j,
            Err(rusqlite::Error::QueryReturnedNoRows) => return Ok(None),
            Err(e) => return Err(e.into()),
        };
        // llm_message is `{"role":"user","content":"<prompt>"}`.
        let v: serde_json::Value = serde_json::from_str(&llm_json)?;
        Ok(v.get("content").and_then(|c| c.as_str()).map(String::from))
    }

    /// List `auto_runs` rows for a session, most-recently-updated first.
    /// Used by the UI to surface prior Auto invocations under a session.
    pub fn list_auto_runs_for_session(
        &self,
        session_id: &str,
    ) -> Result<Vec<agent::auto::AutoRun>, AgentDbError> {
        let conn = self.conn.lock();
        let mut stmt = conn.prepare(
            "SELECT id, session_id, status, plan_versions, worker_results,
                    cost_cents, replans_used, current_worker, pending_failure
             FROM auto_runs WHERE session_id = ?
             ORDER BY updated_at DESC",
        )?;
        let rows = stmt.query_map(rusqlite::params![session_id], |row| {
            // Defer serde to outside the closure; capture raw strings here so
            // the closure stays rusqlite::Error-typed.
            let status_raw: String = row.get(2)?;
            let plan_versions_raw: String = row.get(3)?;
            let worker_results_raw: String = row.get(4)?;
            let current_worker_raw: String = row.get(7)?;
            let pending_failure_raw: String = row.get(8)?;
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, String>(1)?,
                status_raw,
                plan_versions_raw,
                worker_results_raw,
                row.get::<_, i64>(5)?,
                row.get::<_, i64>(6)?,
                current_worker_raw,
                pending_failure_raw,
            ))
        })?;
        let mut out = Vec::new();
        for row in rows {
            let (id, session_id, status_raw, plans_raw, results_raw, cost, replans, current_raw, pending_raw) = row?;
            let status: agent::auto::AutoStatus =
                serde_json::from_str(&format!("\"{}\"", status_raw))?;
            out.push(agent::auto::AutoRun {
                id,
                session_id,
                status,
                plan_versions: serde_json::from_str(&plans_raw)?,
                worker_results: serde_json::from_str(&results_raw)?,
                cost_cents: cost as u32,
                replans_used: replans as u32,
                current_worker: serde_json::from_str(&current_raw)?,
                pending_failure: serde_json::from_str(&pending_raw)?,
            });
        }
        Ok(out)
    }
}

// ── Row mappers ──────────────────────────────────────────────────────────────

fn now_iso() -> String {
    chrono::Utc::now()
        .format("%Y-%m-%dT%H:%M:%S%.3fZ")
        .to_string()
}

fn map_session_row(row: &rusqlite::Row) -> rusqlite::Result<SessionRow> {
    Ok(SessionRow {
        id: row.get(0)?,
        folder: row.get(1)?,
        mode: row.get(2)?,
        title: row.get(3)?,
        parent_session_id: row.get(4)?,
        created_at: row.get(5)?,
        updated_at: row.get(6)?,
        status: row.get(7)?,
        provider_id: row.get(8)?,
        model: row.get(9)?,
    })
}

/// Columns: id, session_id, project_path, role, type, llm_message, metadata,
/// created_at, rewound_at, turn_count (10 columns).
fn map_stored_message(row: &rusqlite::Row) -> rusqlite::Result<StoredMessage> {
    Ok(StoredMessage {
        id: row.get(0)?,
        session_id: row.get(1)?,
        project_path: row.get(2)?,
        role: row.get(3)?,
        type_: row.get(4)?,
        llm_message: row.get(5)?,
        metadata: row.get(6)?,
        created_at: row.get(7)?,
        rewound_at: row.get(8)?,
        turn_count: row.get(9)?,
    })
}

// ── Role/Type string conversions ───────────────────────────────────────────

fn role_to_str(role: MessageRole) -> &'static str {
    match role {
        MessageRole::User => "user",
        MessageRole::Assistant => "assistant",
        MessageRole::Tool => "tool",
        MessageRole::System => "system",
    }
}

fn type_to_str(mt: MessageType) -> &'static str {
    match mt {
        MessageType::Text => "text",
        MessageType::SessionInit => "session_init",
        MessageType::Compaction => "compaction",
        MessageType::ToolCall => "tool_call",
        MessageType::ToolResult => "tool_result",
        MessageType::CompletionSummary => "completion_summary",
    }
}

fn str_to_role(s: &str) -> MessageRole {
    persistence::str_to_role(s)
}

fn str_to_type(s: &str) -> MessageType {
    persistence::str_to_type(s)
}

// ── Context reconstruction ─────────────────────────────────────────────────

/// Reconstruct LLM context from stored messages, applying compaction.
/// Finds the last compaction record, reads `kept_before_count`, and keeps that
/// many non-compaction messages before it plus everything after.
// ── Image persistence (disk-backed attachments) ────────────────────────────
//
// Image bytes are stored on disk under <appdata>/images/<session>/<uuid>.<ext>;
// the persisted message keeps only a `supercoder-image:<file>` reference so
// SQLite rows don't bloat with base64. The reference is rebuilt into a data-URL
// before the message reaches the LLM client (the wire format is unchanged).

const IMAGE_REF_PREFIX: &str = "supercoder-image:";

/// On-disk image directory for a session.
pub fn images_dir(session_id: &str) -> std::path::PathBuf {
    crate::app_data_dir().join("images").join(session_id)
}

fn ext_for_media(media: &str) -> &'static str {
    match media {
        "image/png" => "png",
        "image/jpeg" | "image/jpg" => "jpg",
        "image/gif" => "gif",
        "image/webp" => "webp",
        "image/svg+xml" => "svg",
        "image/bmp" => "bmp",
        _ => "img",
    }
}

fn media_for_ext(ext: &str) -> String {
    match ext {
        "png" => "image/png",
        "jpg" | "jpeg" => "image/jpeg",
        "gif" => "image/gif",
        "webp" => "image/webp",
        "svg" => "image/svg+xml",
        "bmp" => "image/bmp",
        _ => "application/octet-stream",
    }
    .to_string()
}

fn parse_data_uri(url: &str) -> Option<(String, String)> {
    let rest = url.strip_prefix("data:")?;
    let (meta, data) = rest.split_once(',')?;
    let media = meta.strip_suffix(";base64").unwrap_or(meta);
    if media.is_empty() {
        return None;
    }
    Some((media.to_string(), data.to_string()))
}

/// Replace inline `data:` image URLs with on-disk references before persisting.
/// Best-effort: on any IO/parse failure the original data-URL is left inline.
fn externalize_images(llm_message: &mut serde_json::Value, session_id: &str) {
    let Some(blocks) = llm_message.get_mut("content").and_then(|c| c.as_array_mut()) else {
        return;
    };
    for block in blocks {
        let url = match block.get("image_url").and_then(|i| i.get("url")).and_then(|u| u.as_str()) {
            Some(u) if u.starts_with("data:") => u.to_string(),
            _ => continue,
        };
        let Some((media, data)) = parse_data_uri(&url) else { continue };
        use base64::Engine;
        let Ok(bytes) = base64::engine::general_purpose::STANDARD.decode(data.as_bytes()) else {
            continue;
        };
        let dir = images_dir(session_id);
        if std::fs::create_dir_all(&dir).is_err() {
            continue;
        }
        let file = format!("{}.{}", uuid::Uuid::new_v4(), ext_for_media(&media));
        if std::fs::write(dir.join(&file), &bytes).is_err() {
            continue;
        }
        block["image_url"]["url"] = serde_json::Value::String(format!("{IMAGE_REF_PREFIX}{file}"));
    }
}

/// Rebuild `data:` image URLs from on-disk references before the message is used.
/// Legacy inline data-URLs pass through untouched; missing files are left as-is.
fn inline_images(llm_message: &mut serde_json::Value, session_id: &str) {
    let Some(blocks) = llm_message.get_mut("content").and_then(|c| c.as_array_mut()) else {
        return;
    };
    for block in blocks {
        let file = match block.get("image_url").and_then(|i| i.get("url")).and_then(|u| u.as_str()) {
            Some(u) => match u.strip_prefix(IMAGE_REF_PREFIX) {
                Some(f) => f.to_string(),
                None => continue,
            },
            None => continue,
        };
        let path = images_dir(session_id).join(&file);
        let Ok(bytes) = std::fs::read(&path) else { continue };
        use base64::Engine;
        let encoded = base64::engine::general_purpose::STANDARD.encode(&bytes);
        let ext = std::path::Path::new(&file).extension().and_then(|e| e.to_str()).unwrap_or("img");
        let data_url = format!("data:{};base64,{}", media_for_ext(ext), encoded);
        block["image_url"]["url"] = serde_json::Value::String(data_url);
    }
}

/// Extract the display text and image data-URLs from a stored `llm_message` JSON.
/// Handles both plain-string content and the multimodal block array (text +
/// image_url). On-disk image refs are rebuilt into data-URLs so the UI can show
/// them. Used by the message-list command to render images in the chat bubble.
pub fn extract_display_content(llm_message_json: &str, session_id: &str) -> (String, Vec<String>) {
    let mut val: serde_json::Value =
        serde_json::from_str(llm_message_json).unwrap_or(serde_json::json!({}));
    inline_images(&mut val, session_id);

    let content = &val["content"];
    if let Some(s) = content.as_str() {
        return (s.to_string(), Vec::new());
    }
    let Some(blocks) = content.as_array() else {
        return (String::new(), Vec::new());
    };

    let mut text = String::new();
    let mut images = Vec::new();
    for block in blocks {
        match block.get("type").and_then(|t| t.as_str()) {
            Some("text") => {
                if let Some(t) = block.get("text").and_then(|t| t.as_str()) {
                    text.push_str(t);
                }
            }
            Some("image_url") => {
                if let Some(u) = block
                    .get("image_url")
                    .and_then(|i| i.get("url"))
                    .and_then(|u| u.as_str())
                {
                    images.push(u.to_string());
                }
            }
            _ => {}
        }
    }
    (text, images)
}

pub fn reconstruct_context(stored: Vec<StoredMessage>) -> Vec<AgentMessage> {
    if stored.is_empty() {
        return Vec::new();
    }

    let last_compaction_pos = stored.iter().rposition(|m| m.type_ == "compaction");

    match last_compaction_pos {
        Some(pos) => {
            let kept_before_count = serde_json::from_str::<serde_json::Value>(&stored[pos].metadata)
                .ok()
                .and_then(|meta| meta["kept_before_count"].as_u64())
                .unwrap_or(0) as usize;

            let mut count = 0;
            let mut start = pos;
            while start > 0 && count < kept_before_count {
                start -= 1;
                if stored[start].type_ != "compaction" {
                    count += 1;
                }
            }

            stored
                .into_iter()
                .skip(start)
                .map(stored_to_agent_message)
                .collect()
        }
        None => stored.into_iter().map(stored_to_agent_message).collect(),
    }
}

fn stored_to_agent_message(stored: StoredMessage) -> AgentMessage {
    let mut llm_message: serde_json::Value =
        serde_json::from_str(&stored.llm_message).unwrap_or(serde_json::json!({}));
    // Rebuild any disk-backed image references into data-URLs before the LLM sees them.
    inline_images(&mut llm_message, &stored.session_id);
    let metadata: serde_json::Value =
        serde_json::from_str(&stored.metadata).unwrap_or(serde_json::json!({}));
    let content = llm_message["content"].as_str().unwrap_or("").to_string();
    let sender = if stored.role == "user" {
        Sender::HumanUser
    } else {
        Sender::Agent
    };

    AgentMessage {
        content,
        llm_message,
        metadata,
        role: str_to_role(&stored.role),
        message_type: str_to_type(&stored.type_),
        sender,
        turn_count: stored.turn_count,
    }
}

// ── SqliteMessagePersister ─────────────────────────────────────────────────

/// Implements [`MessagePersister`] backed by [`AgentDb`]. Messages are keyed by
/// the `session_id` passed to each call; `project_path` is stamped on every row
/// for reference. One instance is shared across a session and its subagents —
/// children persist under their own `session_id` (the crate stamps the parent
/// link into message metadata).
pub struct SqliteMessagePersister {
    db: Arc<AgentDb>,
    project_path: String,
}

impl SqliteMessagePersister {
    pub fn new(db: Arc<AgentDb>, project_path: String) -> Self {
        Self { db, project_path }
    }

    pub fn project_path(&self) -> &str {
        &self.project_path
    }
}

#[async_trait]
impl MessagePersister for SqliteMessagePersister {
    async fn persist_message(
        &self,
        message: &AgentMessage,
        session_id: &str,
    ) -> Result<PersistResult, PersistError> {
        let db = Arc::clone(&self.db);
        let sid = session_id.to_string();
        let project_path = self.project_path.clone();
        let role = role_to_str(message.role).to_string();
        let type_ = type_to_str(message.message_type).to_string();
        let turn_count = message.turn_count;
        // Externalize inline image bytes to disk, keeping only a reference in SQLite.
        let mut llm_value = message.llm_message.clone();
        externalize_images(&mut llm_value, session_id);
        let llm_message = serde_json::to_string(&llm_value)
            .map_err(|e| PersistError::Storage(e.to_string()))?;
        let metadata = serde_json::to_string(&message.metadata)
            .map_err(|e| PersistError::Storage(e.to_string()))?;

        let id = tokio::task::spawn_blocking(move || {
            db.insert_message(&sid, &project_path, &role, &type_, &llm_message, &metadata, turn_count)
        })
        .await
        .map_err(|e| PersistError::Storage(e.to_string()))?
        .map_err(|e| PersistError::Storage(e.to_string()))?;

        Ok(PersistResult { id })
    }

    async fn load_context(&self, session_id: &str) -> Result<Vec<AgentMessage>, PersistError> {
        let db = Arc::clone(&self.db);
        let sid = session_id.to_string();
        let stored = tokio::task::spawn_blocking(move || db.load_session_for_context(&sid, 500))
            .await
            .map_err(|e| PersistError::Storage(e.to_string()))?
            .map_err(|e| PersistError::Storage(e.to_string()))?;
        Ok(reconstruct_context(stored))
    }
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use tempfile::tempdir;

    fn make_db() -> (tempfile::TempDir, AgentDb) {
        let dir = tempdir().unwrap();
        let db = AgentDb::new(dir.path()).unwrap();
        (dir, db)
    }

    fn make_persister() -> (tempfile::TempDir, Arc<AgentDb>, SqliteMessagePersister) {
        let dir = tempdir().unwrap();
        let db = Arc::new(AgentDb::new(dir.path()).unwrap());
        let persister = SqliteMessagePersister::new(Arc::clone(&db), "/proj".into());
        (dir, db, persister)
    }

    fn llm_json(role: &str, content: &str) -> String {
        json!({"role": role, "content": content}).to_string()
    }

    fn insert_text(db: &AgentDb, session_id: &str, content: &str) -> String {
        db.insert_message(session_id, "/proj", "user", "text", &llm_json("user", content), "{}", None)
            .unwrap()
    }

    fn agent_msg(content: &str, role: MessageRole, mt: MessageType) -> AgentMessage {
        AgentMessage {
            content: content.into(),
            llm_message: json!({"role": role_to_str(role), "content": content}),
            metadata: json!({}),
            role,
            message_type: mt,
            sender: if matches!(role, MessageRole::User) { Sender::HumanUser } else { Sender::Agent },
            turn_count: None,
        }
    }

    #[test]
    fn test_schema_v1() {
        let (_d, db) = make_db();
        assert_eq!(db.schema_version().unwrap(), 1);
    }

    #[test]
    fn test_session_crud() {
        let (_d, db) = make_db();
        db.create_session("s1", "/proj", "coding", Some("Fix bug"), None, None, None).unwrap();
        let s = db.get_session("s1").unwrap().unwrap();
        assert_eq!(s.folder, "/proj");
        assert_eq!(s.mode, "coding");
        // A new session starts "idle" — "active" means a turn is currently running
        // (set on run start, cleared on run end). So nothing is active yet.
        assert_eq!(s.status, "idle");
        assert!(db.active_session_for_folder("/proj").unwrap().is_none());

        // one-active-per-folder lookup tracks the running session
        db.set_session_status("s1", "active").unwrap();
        assert_eq!(db.active_session_for_folder("/proj").unwrap().as_deref(), Some("s1"));
        db.set_session_status("s1", "idle").unwrap();
        assert!(db.active_session_for_folder("/proj").unwrap().is_none());

        let all = db.list_sessions().unwrap();
        assert_eq!(all.len(), 1);
    }

    #[test]
    fn test_session_isolation() {
        let (_d, db) = make_db();
        insert_text(&db, "a", "msg-a");
        insert_text(&db, "b", "msg-b");
        let a = db.load_session_messages("a").unwrap();
        assert_eq!(a.len(), 1);
        assert!(a[0].llm_message.contains("msg-a"));
    }

    #[tokio::test]
    async fn test_persister_roundtrip() {
        let (_d, _db, p) = make_persister();
        let msg = agent_msg("hello", MessageRole::Assistant, MessageType::Text);
        p.persist_message(&msg, "s1").await.unwrap();
        let loaded = p.load_context("s1").await.unwrap();
        assert_eq!(loaded.len(), 1);
        assert_eq!(loaded[0].content, "hello");
        assert_eq!(loaded[0].role, MessageRole::Assistant);
    }

    #[test]
    fn test_compaction_reconstruction() {
        let (_d, db) = make_db();
        for i in 1..=5 {
            insert_text(&db, "t1", &format!("msg-{i}"));
        }
        db.insert_compaction("t1", "/proj", "Summary of 1-3", 2).unwrap();
        for i in 6..=7 {
            insert_text(&db, "t1", &format!("msg-{i}"));
        }
        let stored = db.load_session_messages("t1").unwrap();
        let rec = reconstruct_context(stored);
        let contents: Vec<String> = rec.iter().map(|m| m.content.clone()).collect();
        for i in 1..=3 {
            assert!(!contents.contains(&format!("msg-{i}")), "msg-{i} should be compacted");
        }
        for i in 4..=7 {
            assert!(contents.contains(&format!("msg-{i}")), "msg-{i} should be present");
        }
        assert!(contents.iter().any(|c| c.contains("Summary of 1-3")));
    }

    #[test]
    fn test_context_usage_roundtrip() {
        let (_d, db) = make_db();
        db.upsert_context_usage("s1", "/proj", 1200, 128000, 8).unwrap();
        let usage = db.get_context_usage("s1").unwrap().unwrap();
        assert_eq!(usage, (1200, 128000, 8));
        db.delete_context_usage("s1").unwrap();
        assert!(db.get_context_usage("s1").unwrap().is_none());
    }

    #[test]
    fn test_rewind() {
        let (_d, db) = make_db();
        let _id1 = insert_text(&db, "t1", "keep");
        let id2: i64 = insert_text(&db, "t1", "drop").parse().unwrap();
        insert_text(&db, "t1", "also drop");
        let n = db.rewind_messages("t1", id2).unwrap();
        assert_eq!(n, 2);
        let remaining = db.load_session_messages("t1").unwrap();
        assert_eq!(remaining.len(), 1);
        assert!(remaining[0].llm_message.contains("keep"));
    }

    // ── auto_runs (P4) ────────────────────────────────────────────────────

    /// End-to-end CRUD on the `auto_runs` table. Pins:
    ///   - status enum round-trip (snake_case in DB, restored to variant)
    ///   - plan_versions + worker_results JSON columns survive non-trivial
    ///     nested payloads (Plan with workers, WorkerResult with status)
    ///   - update overwrites status / cost / replans correctly
    ///   - list_for_session ordering is by updated_at DESC
    ///   - get_auto_run returns None for unknown id
    #[test]
    fn test_auto_runs_crud_roundtrip() {
        use agent::auto::{
            AutoRun, AutoStatus, Plan, SeePrior, SeePriorKeyword, WorkerResult, WorkerSpec,
            WorkerStatus,
        };

        let (_d, db) = make_db();

        let initial = AutoRun {
            id: "auto-1".into(),
            session_id: "sess-1".into(),
            status: AutoStatus::Planning,
            plan_versions: Vec::new(),
            worker_results: Vec::new(),
            cost_cents: 0,
            replans_used: 0,
            current_worker: None,
            pending_failure: None,
        };
        db.insert_auto_run(&initial).unwrap();

        // Fresh insert is readable.
        let loaded = db.get_auto_run("auto-1").unwrap().expect("row should exist");
        assert_eq!(loaded.id, "auto-1");
        assert_eq!(loaded.session_id, "sess-1");
        assert_eq!(loaded.status, AutoStatus::Planning);
        assert!(loaded.plan_versions.is_empty());
        assert!(loaded.worker_results.is_empty());

        // Update with non-trivial nested payloads.
        let updated = AutoRun {
            id: "auto-1".into(),
            session_id: "sess-1".into(),
            status: AutoStatus::Done,
            plan_versions: vec![Plan {
                version: 1,
                reasoning: "decompose into one cheap worker".into(),
                workers: vec![WorkerSpec {
                    id: "w1".into(),
                    model: "claude-haiku-4-5".into(),
                    prompt: "read README".into(),
                    see_prior: SeePrior::Keyword(SeePriorKeyword::None),
                }],
            }],
            worker_results: vec![WorkerResult {
                id: "w1".into(),
                model: "claude-haiku-4-5".into(),
                prompt: "read README".into(),
                summary: "README describes a Rust project".into(),
                tool_count: 1,
                cost_cents: 3,
                status: WorkerStatus::Ok,
            }],
            cost_cents: 42,
            replans_used: 1,
            current_worker: None,
            pending_failure: None,
        };
        db.update_auto_run(&updated).unwrap();

        let reloaded = db.get_auto_run("auto-1").unwrap().unwrap();
        assert_eq!(reloaded.status, AutoStatus::Done);
        assert_eq!(reloaded.cost_cents, 42);
        assert_eq!(reloaded.replans_used, 1);
        assert_eq!(reloaded.plan_versions.len(), 1);
        let plan = &reloaded.plan_versions[0];
        assert_eq!(plan.version, 1);
        assert_eq!(plan.workers.len(), 1);
        assert_eq!(plan.workers[0].id, "w1");
        assert!(matches!(plan.workers[0].see_prior, SeePrior::Keyword(SeePriorKeyword::None)));
        assert_eq!(reloaded.worker_results.len(), 1);
        let result = &reloaded.worker_results[0];
        assert_eq!(result.summary, "README describes a Rust project");
        assert_eq!(result.status, WorkerStatus::Ok);

        // Unknown id returns None, not error.
        assert!(db.get_auto_run("does-not-exist").unwrap().is_none());

        // list_auto_runs_for_session returns this run; insert a second one for
        // a different session to confirm filtering.
        let other = AutoRun {
            id: "auto-2".into(),
            session_id: "sess-2".into(),
            status: AutoStatus::Failed,
            plan_versions: Vec::new(),
            worker_results: Vec::new(),
            cost_cents: 0,
            replans_used: 0,
            current_worker: None,
            pending_failure: None,
        };
        db.insert_auto_run(&other).unwrap();
        let runs = db.list_auto_runs_for_session("sess-1").unwrap();
        assert_eq!(runs.len(), 1);
        assert_eq!(runs[0].id, "auto-1");
        let other_runs = db.list_auto_runs_for_session("sess-2").unwrap();
        assert_eq!(other_runs.len(), 1);
        assert_eq!(other_runs[0].status, AutoStatus::Failed);
    }

    /// Every `AutoStatus` variant must round-trip through the DB column. If
    /// the snake_case serde tag drifts, this catches it before users see a
    /// silent rename.
    #[test]
    fn test_auto_runs_status_all_variants_roundtrip() {
        use agent::auto::{AutoRun, AutoStatus};
        let (_d, db) = make_db();
        for (i, status) in [
            AutoStatus::Planning,
            AutoStatus::AwaitingApproval,
            AutoStatus::Running,
            AutoStatus::AwaitingFailureDecision,
            AutoStatus::Done,
            AutoStatus::Failed,
            AutoStatus::Cancelled,
        ]
        .into_iter()
        .enumerate()
        {
            let id = format!("auto-{i}");
            let run = AutoRun {
                id: id.clone(),
                session_id: "sess".into(),
                status,
                plan_versions: Vec::new(),
                worker_results: Vec::new(),
                cost_cents: 0,
                replans_used: 0,
                current_worker: None,
                pending_failure: None,
            };
            db.insert_auto_run(&run).unwrap();
            let loaded = db.get_auto_run(&id).unwrap().unwrap();
            assert_eq!(loaded.status, status, "{status:?} did not round-trip");
        }
    }
}
