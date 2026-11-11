//! Trusted operator recovery of one interrupted, journal-authorized VGR file write.
//!
//! Does NOT replay a model/tool call, authorize a fresh write or grant delivery.
//! Caller supplies exact persisted identities; neither provider output nor the
//! untrusted stdio protocol exposes this operation. This is an abort/reconcile
//! transaction and must acquire the SAME daemon OS lock as Server::open.
use crate::{journal::Journal, session::State, DaemonError, Result};
use std::{
    fs::{self, File, OpenOptions},
    path::Path,
};
use voder_core::types::RevisionId;
use voder_protocol::json::Value;
use voder_runtime::{
    artifact_store::{sha256, ArtifactStore},
    durable_write::{self, RecoveryOutcome},
    receipt::WorkspaceSnapshot,
};

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum Reconciliation {
    RolledBack,
    AlreadyRestored,
}

fn text(value: impl Into<String>) -> Value {
    Value::String(value.into())
}

/// Recover one explicitly identified interrupted file mutation in an owned
/// worktree. Fail closed if the recorded phase or workspace is ambiguous.
/// The state path and repository tree must not overlap. Successful return
/// means original bytes are independently measured AND a recovery event has
/// been fsynced. It does NOT imply the task was successfully completed.
pub fn reconcile_interrupted_write(
    state_dir: &Path,
    session_id: &str,
    run_id: &str,
    action_id: &str,
) -> Result<Reconciliation> {
    if !state_dir.is_absolute()
        || session_id.is_empty()
        || run_id.is_empty()
        || action_id.is_empty()
    {
        return Err(DaemonError::Invalid(
            "absolute state path and explicit identities required",
        ));
    }
    let root_meta = fs::symlink_metadata(state_dir)?;
    if !root_meta.is_dir() || root_meta.file_type().is_symlink() {
        return Err(DaemonError::Invalid("unsafe daemon state directory"));
    }
    let state_dir = state_dir.canonicalize()?;
    let lock_path = state_dir.join("daemon.lock");
    if let Ok(meta) = fs::symlink_metadata(&lock_path) {
        if !meta.is_file() || meta.file_type().is_symlink() {
            return Err(DaemonError::Invalid("unsafe daemon writer lock"));
        }
    }
    let mut opts = OpenOptions::new();
    opts.write(true).create(true).truncate(false);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        opts.mode(0o600);
        opts.custom_flags(0o400000); // Linux O_NOFOLLOW
    }
    let lock: File = opts.open(&lock_path)?;
    lock.try_lock()
        .map_err(|_| DaemonError::Conflict("daemon writer already running"))?;
    let journal_path = state_dir.join("events.jsonl");
    let mut journal = Journal::open(&journal_path)?;
    // Unlike normal startup, recovery must reconstruct but may not reject an
    // interrupted execution phase. All identities come from durable journal.
    let mut state = State::reconstruct(journal.entries())?;
    let session = state.get(session_id)?;
    let run = session
        .runs
        .get(run_id)
        .ok_or(DaemonError::Invalid("unknown recovery run"))?;
    if !matches!(
        run.status.as_str(),
        "VGR_DISPATCHED"
            | "VGR_OBSERVED"
            | "VGR_BUDGET_SETTLED"
            | "VGR_EVIDENCE_ADMITTED"
            | "VGR_META_AUDITED"
    ) || run.native_action_id.as_deref() != Some(action_id)
        || run.native_contract_digest.is_none()
        || run.native_reservation_id.is_none()
    {
        return Err(DaemonError::Conflict(
            "run not eligible for interrupted-write rollback",
        ));
    }
    // Before touching any files, independently check the matching durable
    // DISPATCH event, not only the reconstructed cached run status.
    if !journal.entries().iter().any(|entry| {
        let body = &entry.body;
        body.get("type").and_then(Value::as_str) == Some("vgr.action_dispatched")
            && body.get("session").and_then(Value::as_str) == Some(session_id)
            && body.get("run_id").and_then(Value::as_str) == Some(run_id)
            && body.get("action_id").and_then(Value::as_str) == Some(action_id)
            && body.get("contract_digest").and_then(Value::as_str)
                == run.native_contract_digest.as_deref()
    }) {
        return Err(DaemonError::Integrity(
            "missing authoritative matching dispatch".into(),
        ));
    }
    let workspace = Path::new(&session.project_root);
    if !workspace.is_absolute() || !workspace.is_dir() {
        return Err(DaemonError::Invalid("invalid journaled worktree"));
    }
    let worktree = workspace.canonicalize()?;
    if workspace != worktree || worktree.starts_with(&state_dir) || state_dir.starts_with(&worktree)
    {
        return Err(DaemonError::Conflict(
            "unsafe or overlapping recovery roots",
        ));
    }
    let expected = run
        .native_workspace_digest
        .as_ref()
        .ok_or(DaemonError::Invalid("missing prepared workspace digest"))?
        .clone();
    let store = ArtifactStore::open(&state_dir.join("artifacts"))
        .map_err(|e| DaemonError::Integrity(format!("recovery artifact store rejected: {e}")))?;
    let wal_root = state_dir.join("voder-write-transactions");
    let action_key = format!("{session_id}:{run_id}:{action_id}");
    durable_write::verify_recovery_scope(&worktree, &wal_root, &action_key, &expected).map_err(
        |e| DaemonError::Integrity(format!("full-worktree recovery preflight rejected: {e}")),
    )?;
    let status = durable_write::recover_uncommitted(&worktree, &wal_root, &action_key, &store)
        .map_err(|e| DaemonError::Integrity(format!("write recovery rejected: {e}")))?;
    let revision = RevisionId::try_from("vgr-write-recovery")
        .map_err(|_| DaemonError::Invalid("invalid recovery revision"))?;
    let restored = WorkspaceSnapshot::measure(&worktree)
        .and_then(|s| s.workspace(revision))
        .map_err(|e| DaemonError::Integrity(format!("restored snapshot rejected: {e}")))?;
    if restored.digest.as_str() != expected {
        return Err(DaemonError::Conflict(
            "restored worktree differs from prepared full snapshot",
        ));
    }
    let label = match status {
        RecoveryOutcome::RolledBack => "ROLLED_BACK",
        RecoveryOutcome::AlreadyAtPreimage => "ALREADY_RESTORED",
    };
    let fingerprint = sha256(format!(
        "voder-interrupted-write-reconcile-v1\0{session_id}\0{run_id}\0{action_id}\0{expected}\0{label}"
    ).as_bytes()).map_err(|e| DaemonError::Integrity(e.to_string()))?;
    let body = Value::object([
        ("type".into(), text("vgr.write_reconciled")),
        ("session".into(), text(session_id)),
        (
            "key".into(),
            text(format!("sys-vgr-write-reconciled-{run_id}-{action_id}")),
        ),
        ("fingerprint".into(), text(fingerprint)),
        (
            "response".into(),
            Value::object([
                ("run_id".into(), text(run_id)),
                ("status".into(), text("VGR_RECOVERED_ROLLBACK")),
                ("execution_authorized".into(), Value::Bool(false)),
                ("delivery_allowed".into(), Value::Bool(false)),
            ]),
        ),
        ("run_id".into(), text(run_id)),
        ("action_id".into(), text(action_id)),
        (
            "contract_digest".into(),
            text(run.native_contract_digest.clone().unwrap_or_default()),
        ),
        ("restored_workspace_digest".into(), text(expected)),
        ("recovery_outcome".into(), text(label)),
    ]);
    // Journal fsync is the reconciliation commit. If this crashes after file
    // restoration but before event append, rerun observes ALREADY_RESTORED.
    let event = journal.append(body)?;
    state.apply(&event)?;
    // Reconstruct a second time so this operation never returns success for an
    // invalid event. Other sessions may still need separate reconciliation.
    let recovered = State::reconstruct(journal.entries())?;
    if recovered
        .get(session_id)?
        .runs
        .get(run_id)
        .map(|r| r.status.as_str())
        != Some("VGR_RECOVERED_ROLLBACK")
    {
        return Err(DaemonError::Integrity(
            "recovery replay did not converge".into(),
        ));
    }
    drop(lock);
    Ok(match status {
        RecoveryOutcome::RolledBack => Reconciliation::RolledBack,
        RecoveryOutcome::AlreadyAtPreimage => Reconciliation::AlreadyRestored,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{
        path::PathBuf,
        time::{SystemTime, UNIX_EPOCH},
    };
    use voder_runtime::{
        artifact_store::sha256,
        patch::{FilePatch, PatchOperation},
    };

    struct Fixture {
        state: PathBuf,
        worktree: PathBuf,
        before: String,
    }
    impl Fixture {
        fn new() -> Self {
            let n = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos();
            let base =
                std::env::temp_dir().join(format!("voder-reconcile-{}-{n}", std::process::id()));
            let state = base.join("state");
            let worktree = base.join("worktree");
            fs::create_dir_all(&state).unwrap();
            fs::create_dir_all(&worktree).unwrap();
            fs::write(worktree.join("a.rs"), b"original").unwrap();
            let before = WorkspaceSnapshot::measure(&worktree)
                .unwrap()
                .workspace(RevisionId::try_from("test-revision").unwrap())
                .unwrap()
                .digest
                .to_string();
            let mut j = Journal::open(&state.join("events.jsonl")).unwrap();
            let event = |kind: &str, key: &str, fields: Vec<(&str, Value)>| {
                let mut data = vec![
                    ("type".into(), text(kind)),
                    ("session".into(), text("s-1")),
                    ("key".into(), text(key)),
                    ("fingerprint".into(), text("test-pinned")),
                    (
                        "response".into(),
                        Value::object([
                            (
                                "execution_authorized".into(),
                                Value::Bool(kind == "vgr.action_dispatched"),
                            ),
                            ("delivery_allowed".into(), Value::Bool(false)),
                        ]),
                    ),
                ];
                data.extend(fields.into_iter().map(|(k, v)| (k.to_owned(), v)));
                Value::object(data)
            };
            j.append(event(
                "session.created",
                "s",
                vec![("project_root", text(worktree.to_string_lossy().to_string()))],
            ))
            .unwrap();
            j.append(event(
                "task.model_requested",
                "request",
                vec![
                    ("run_id", text("r-1")),
                    ("reason", text("test")),
                    ("instruction_digest", text("input")),
                ],
            ))
            .unwrap();
            j.append(event(
                "task.model_proposed",
                "propose",
                vec![
                    ("run_id", text("r-1")),
                    ("reason", text("test")),
                    ("proposal_digest", text("proposal")),
                ],
            ))
            .unwrap();
            j.append(event(
                "vgr.action_prepared",
                "prep",
                vec![
                    ("run_id", text("r-1")),
                    ("action_id", text("a-1")),
                    ("reservation_id", text("reserve")),
                    ("proposal_digest", text("proposal")),
                    ("contract_digest", text("contract")),
                    ("workspace_digest", text(&before)),
                ],
            ))
            .unwrap();
            j.append(event(
                "vgr.action_dispatched",
                "dispatch",
                vec![
                    ("run_id", text("r-1")),
                    ("action_id", text("a-1")),
                    ("reservation_id", text("reserve")),
                    ("contract_digest", text("contract")),
                    ("workspace_digest", text(&before)),
                ],
            ))
            .unwrap();
            State::reconstruct(j.entries()).unwrap();
            drop(j);
            Self {
                state,
                worktree,
                before,
            }
        }
        fn mutate(&self) {
            let store = ArtifactStore::open(&self.state.join("artifacts")).unwrap();
            durable_write::apply(
                &self.worktree,
                &self.state.join("voder-write-transactions"),
                "s-1:r-1:a-1",
                &FilePatch {
                    relative: "a.rs".into(),
                    operation: PatchOperation::Replace {
                        old_sha256: sha256(b"original").unwrap(),
                        new_bytes: b"patched".to_vec(),
                    },
                },
                &store,
            )
            .unwrap();
        }
    }
    impl Drop for Fixture {
        fn drop(&mut self) {
            if let Some(base) = self.state.parent() {
                let _ = fs::remove_dir_all(base);
            }
        }
    }
    #[test]
    fn recovered_interrupted_write_is_replayable_and_terminal() {
        let f = Fixture::new();
        f.mutate();
        assert_eq!(fs::read(f.worktree.join("a.rs")).unwrap(), b"patched");
        assert_eq!(
            reconcile_interrupted_write(&f.state, "s-1", "r-1", "a-1").unwrap(),
            Reconciliation::RolledBack
        );
        assert_eq!(fs::read(f.worktree.join("a.rs")).unwrap(), b"original");
        let j = Journal::open(&f.state.join("events.jsonl")).unwrap();
        assert_eq!(
            crate::recovery::recover(&j)
                .unwrap()
                .get("s-1")
                .unwrap()
                .runs["r-1"]
                .status,
            "VGR_RECOVERED_ROLLBACK"
        );
        assert_eq!(
            WorkspaceSnapshot::measure(&f.worktree)
                .unwrap()
                .workspace(RevisionId::try_from("test-revision").unwrap())
                .unwrap()
                .digest
                .to_string(),
            f.before
        );
        assert!(reconcile_interrupted_write(&f.state, "s-1", "r-1", "a-1").is_err());
    }
    #[test]
    fn recovery_preflight_rejects_changes_elsewhere_before_touching_patch() {
        let f = Fixture::new();
        f.mutate();
        fs::write(f.worktree.join("unrelated"), b"not-authorized").unwrap();
        assert!(reconcile_interrupted_write(&f.state, "s-1", "r-1", "a-1").is_err());
        assert_eq!(fs::read(f.worktree.join("a.rs")).unwrap(), b"patched");
        assert_eq!(
            fs::read(f.worktree.join("unrelated")).unwrap(),
            b"not-authorized"
        );
        let j = Journal::open(&f.state.join("events.jsonl")).unwrap();
        assert_eq!(j.entries().len(), 5);
    }
    #[test]
    fn recovery_never_overwrites_divergent_workspace() {
        let f = Fixture::new();
        f.mutate();
        fs::write(f.worktree.join("a.rs"), b"third-party-change").unwrap();
        assert!(reconcile_interrupted_write(&f.state, "s-1", "r-1", "a-1").is_err());
        assert_eq!(
            fs::read(f.worktree.join("a.rs")).unwrap(),
            b"third-party-change"
        );
        let j = Journal::open(&f.state.join("events.jsonl")).unwrap();
        assert_eq!(j.entries().len(), 5);
    }
    #[test]
    fn committed_write_cannot_be_reconciled_as_an_interruption() {
        let f = Fixture::new();
        f.mutate();
        let mut j = Journal::open(&f.state.join("events.jsonl")).unwrap();
        // Invalid final commit must not become a recovery bypass: the replay
        // validator fails closed instead of accepting synthetic authority.
        let forged = Value::object([
            ("type".into(), text("vgr.final_committed")),
            ("session".into(), text("s-1")),
            ("key".into(), text("forged-final")),
            ("fingerprint".into(), text("forged")),
            (
                "response".into(),
                Value::object([("delivery_allowed".into(), Value::Bool(true))]),
            ),
            ("run_id".into(), text("r-1")),
            ("action_id".into(), text("a-1")),
        ]);
        j.append(forged).unwrap();
        drop(j);
        assert!(reconcile_interrupted_write(&f.state, "s-1", "r-1", "a-1").is_err());
        assert_eq!(fs::read(f.worktree.join("a.rs")).unwrap(), b"patched");
    }

    #[test]
    fn recovery_refuses_missing_wal_and_wrong_identity() {
        let f = Fixture::new();
        assert!(reconcile_interrupted_write(&f.state, "s-1", "r-1", "a-1").is_err());
        f.mutate();
        assert!(reconcile_interrupted_write(&f.state, "s-1", "r-1", "wrong").is_err());
        assert_eq!(fs::read(f.worktree.join("a.rs")).unwrap(), b"patched");
    }
    /// Invoked only by the parent fault-injection test in a separate OS process.
    /// It deliberately remains alive after the filesystem mutation so that
    /// Child::kill simulates an ungraceful SIGKILL before journal reconciliation.
    #[cfg(unix)]
    #[test]
    fn write_child_for_sigkill_fault_injection() {
        let Some(base) = std::env::var_os("VODER_SIGKILL_FAULT_FIXTURE") else {
            return;
        };
        let base = PathBuf::from(base);
        let state = base.join("state");
        let worktree = base.join("worktree");
        let store = ArtifactStore::open(&state.join("artifacts")).unwrap();
        durable_write::apply(
            &worktree,
            &state.join("voder-write-transactions"),
            "s-1:r-1:a-1",
            &FilePatch {
                relative: "a.rs".into(),
                operation: PatchOperation::Replace {
                    old_sha256: sha256(b"original").unwrap(),
                    new_bytes: b"patched".to_vec(),
                },
            },
            &store,
        )
        .unwrap();
        fs::write(base.join("write-observed.ready"), b"written").unwrap();
        loop {
            std::thread::sleep(std::time::Duration::from_secs(60));
        }
    }

    #[cfg(unix)]
    #[test]
    fn sigkill_after_durable_write_reconciles_without_reexecution() {
        use std::{
            process::Command,
            thread,
            time::{Duration, Instant},
        };
        let fixture = Fixture::new();
        let base = fixture.state.parent().unwrap().to_owned();
        let mut child = Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "native_write_recovery::tests::write_child_for_sigkill_fault_injection",
                "--nocapture",
            ])
            .env("VODER_SIGKILL_FAULT_FIXTURE", &base)
            .spawn()
            .unwrap();
        let marker = base.join("write-observed.ready");
        let start = Instant::now();
        while !marker.is_file() {
            if let Some(exit) = child.try_wait().unwrap() {
                panic!("fault child unexpectedly exited: {exit}");
            }
            if start.elapsed() > Duration::from_secs(15) {
                let _ = child.kill();
                let _ = child.wait();
                panic!("fault child did not reach durable-write boundary");
            }
            thread::sleep(Duration::from_millis(25));
        }
        child.kill().unwrap();
        assert!(!child.wait().unwrap().success());
        assert_eq!(fs::read(fixture.worktree.join("a.rs")).unwrap(), b"patched");
        assert_eq!(
            reconcile_interrupted_write(&fixture.state, "s-1", "r-1", "a-1").unwrap(),
            Reconciliation::RolledBack,
        );
        assert_eq!(
            fs::read(fixture.worktree.join("a.rs")).unwrap(),
            b"original"
        );
        let state = Journal::open(&fixture.state.join("events.jsonl")).unwrap();
        assert_eq!(
            crate::recovery::recover(&state)
                .unwrap()
                .get("s-1")
                .unwrap()
                .runs["r-1"]
                .status,
            "VGR_RECOVERED_ROLLBACK",
        );
        assert!(reconcile_interrupted_write(&fixture.state, "s-1", "r-1", "a-1").is_err());
    }

    #[test]
    fn recovery_requires_exclusive_daemon_writer_lock() {
        let f = Fixture::new();
        f.mutate();
        let lock = OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(false)
            .open(f.state.join("daemon.lock"))
            .unwrap();
        lock.try_lock().unwrap();
        assert!(reconcile_interrupted_write(&f.state, "s-1", "r-1", "a-1").is_err());
        assert_eq!(fs::read(f.worktree.join("a.rs")).unwrap(), b"patched");
    }
}
