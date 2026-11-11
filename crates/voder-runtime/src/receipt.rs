//! Real workspace observations bind run/action, provenance and measured before/after tree state.
//! A measured receipt is NOT admitted evidence: Core admission and VEP durability remain separate.
use crate::{
    artifact_store::{sha256, ArtifactStore},
    process::ProcessResult,
    repository, Result, RuntimeError,
};
use std::{
    collections::BTreeMap,
    path::{Path, PathBuf},
};
use voder_core::{
    action_commit::MutationKind,
    consequence::{ExecutionCompletion, ExecutionReport, ObservedChange},
    contract::RepoScope,
    types::{ActionId, ContractIdentity, Digest, RevisionId, RunId, WorkspaceIdentity},
};
#[derive(Debug, Clone)]
pub struct WorkspaceSnapshot {
    root: PathBuf,
    files: BTreeMap<String, String>,
    fingerprint: String,
}
impl WorkspaceSnapshot {
    pub fn measure(root: &Path) -> Result<Self> {
        let root = root.canonicalize()?;
        let files = repository::snapshot_files(&root)?;
        let fingerprint = repository::hash_snapshot(&files)?;
        Ok(Self {
            root,
            files,
            fingerprint,
        })
    }
    pub fn workspace(&self, revision: RevisionId) -> Result<WorkspaceIdentity> {
        Ok(WorkspaceIdentity {
            revision,
            digest: Digest::parse(self.fingerprint.clone())
                .map_err(|_| RuntimeError::Integrity("invalid sha256 digest".into()))?,
        })
    }
    pub fn root(&self) -> &Path {
        &self.root
    }
}
#[derive(Debug, Clone)]
pub struct RuntimeReceipt {
    pub report: ExecutionReport,
    pub stdout_artifact: String,
    pub stderr_artifact: String,
    pub exit_code: Option<i32>,
    pub elapsed_millis: u128,
}
pub struct CaptureContext<'a> {
    pub run: RunId,
    pub action: ActionId,
    pub contract: ContractIdentity,
    pub prepared_before: WorkspaceIdentity,
    pub snapshot: &'a WorkspaceSnapshot,
    pub process: &'a ProcessResult,
    pub store: &'a ArtifactStore,
}
/// Detect actual observed changes; refuses a claimed preimage that does not match measured bytes.
pub fn capture(ctx: CaptureContext<'_>) -> Result<RuntimeReceipt> {
    let CaptureContext {
        run,
        action,
        contract,
        prepared_before,
        snapshot,
        process,
        store,
    } = ctx;
    if snapshot.fingerprint != prepared_before.digest.as_str() {
        return Err(RuntimeError::Integrity(
            "prepared workspace does not match measured snapshot".into(),
        ));
    }
    let now = WorkspaceSnapshot::measure(snapshot.root())?;
    let after = now.workspace(prepared_before.revision.clone())?;
    let mut observed_changes = Vec::new();
    let mut forbidden_side_effects = Vec::new();
    let all: std::collections::BTreeSet<_> =
        snapshot.files.keys().chain(now.files.keys()).collect();
    for name in all {
        let old = snapshot.files.get(name);
        let new = now.files.get(name);
        if old == new {
            continue;
        }
        if old.is_some_and(|v| v.starts_with("L:")) || new.is_some_and(|v| v.starts_with("L:")) {
            forbidden_side_effects.push(format!("symlink changed: {name}"));
        }
        let kind = match (old, new) {
            (None, Some(_)) => MutationKind::Create,
            (Some(_), None) => MutationKind::Delete,
            _ => MutationKind::Modify,
        };
        match crate::paths::relative_path(name).and_then(|_| {
            RepoScope::parse(name.to_string()).map_err(|_| RuntimeError::UnsafePath(name.into()))
        }) {
            Ok(path) => observed_changes.push(ObservedChange { path, kind }),
            Err(_) => forbidden_side_effects.push(format!("invalid modified path: {name}")),
        }
    }
    if observed_changes.len() > 256 {
        return Err(RuntimeError::Invalid("too many observed changes".into()));
    }
    let stdout_artifact = store.save(&process.stdout)?;
    let stderr_artifact = store.save(&process.stderr)?;
    let completion = if process.timed_out {
        ExecutionCompletion::TimedOut
    } else if process.exit == Some(0) && !process.output_limited {
        ExecutionCompletion::Completed
    } else {
        ExecutionCompletion::Failed
    };
    let canonical = format!(
        "{}\0{}\0{}\0{}\0{}\0{}\0{:?}\0{}\0{}\0{:?}",
        run,
        action,
        contract.digest,
        prepared_before.digest,
        after.digest,
        stdout_artifact,
        completion,
        stderr_artifact,
        process.exit.unwrap_or(-1),
        observed_changes
    );
    let receipt_digest = Digest::parse(sha256(canonical.as_bytes())?)
        .map_err(|_| RuntimeError::Integrity("bad receipt digest".into()))?;
    let report = ExecutionReport {
        run_id: run,
        action_id: action,
        contract,
        before: prepared_before,
        after,
        completion,
        observed_changes,
        forbidden_side_effects,
        receipt_digest,
    };
    Ok(RuntimeReceipt {
        report,
        stdout_artifact,
        stderr_artifact,
        exit_code: process.exit,
        elapsed_millis: process.elapsed.as_millis(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{fs, time::Duration};
    use voder_core::types::Version;
    fn id<T: TryFrom<&'static str>>() -> T
    where
        <T as TryFrom<&'static str>>::Error: std::fmt::Debug,
    {
        T::try_from("id").unwrap()
    }
    fn contract() -> ContractIdentity {
        ContractIdentity {
            contract_id: id(),
            version: Version::new(1).unwrap(),
            digest: id(),
            toolchain: id(),
            environment: id(),
        }
    }
    fn tempfile(label: &str) -> PathBuf {
        let p = std::env::temp_dir().join(format!(
            "voder-receipt-{label}-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir(&p).unwrap();
        p
    }
    #[test]
    fn failed_process_records_real_partial_mutations() {
        let root = tempfile("mutation");
        let store_root = tempfile("logs");
        let store = ArtifactStore::open(&store_root).unwrap();
        fs::write(root.join("main.rs"), "original").unwrap();
        let snapshot = WorkspaceSnapshot::measure(&root).unwrap();
        let before = snapshot.workspace(id()).unwrap();
        fs::write(root.join("main.rs"), "broken").unwrap();
        fs::write(root.join("new.rs"), "new file").unwrap();
        let process = ProcessResult {
            exit: Some(1),
            timed_out: false,
            output_limited: false,
            stdout: vec![],
            stderr: b"failed unit test".to_vec(),
            elapsed: Duration::from_millis(1),
        };
        let report = capture(CaptureContext {
            run: id(),
            action: id(),
            contract: contract(),
            prepared_before: before.clone(),
            snapshot: &snapshot,
            process: &process,
            store: &store,
        })
        .unwrap();
        assert_eq!(report.report.completion, ExecutionCompletion::Failed);
        assert_eq!(report.report.observed_changes.len(), 2);
        assert_eq!(report.report.observed_changes[0].kind, MutationKind::Modify);
        assert_eq!(report.report.observed_changes[1].kind, MutationKind::Create);
        assert_ne!(report.report.after.digest, before.digest);
        assert_eq!(
            store.read(&report.stderr_artifact).unwrap(),
            b"failed unit test"
        );
        fs::remove_dir_all(root).unwrap();
        fs::remove_dir_all(store_root).unwrap();
    }
    #[test]
    fn forged_prepared_workspace_is_rejected() {
        let root = tempfile("preimage");
        let store_root = tempfile("store");
        let store = ArtifactStore::open(&store_root).unwrap();
        let snapshot = WorkspaceSnapshot::measure(&root).unwrap();
        let mut before = snapshot.workspace(id()).unwrap();
        before.digest = Digest::parse("incorrect-digest").unwrap();
        let process = ProcessResult {
            exit: Some(0),
            timed_out: false,
            output_limited: false,
            stdout: vec![],
            stderr: vec![],
            elapsed: Duration::ZERO,
        };
        assert!(capture(CaptureContext {
            run: id(),
            action: id(),
            contract: contract(),
            prepared_before: before,
            snapshot: &snapshot,
            process: &process,
            store: &store
        })
        .is_err());
        fs::remove_dir_all(root).unwrap();
        fs::remove_dir_all(store_root).unwrap();
    }
    #[test]
    fn changed_symlink_is_reported_as_forbidden_effect() {
        let root = tempfile("symlink");
        let store_root = tempfile("store");
        let store = ArtifactStore::open(&store_root).unwrap();
        let snapshot = WorkspaceSnapshot::measure(&root).unwrap();
        let before = snapshot.workspace(id()).unwrap();
        #[cfg(unix)]
        std::os::unix::fs::symlink("/etc/passwd", root.join("link")).unwrap();
        let process = ProcessResult {
            exit: Some(0),
            timed_out: false,
            output_limited: false,
            stdout: vec![],
            stderr: vec![],
            elapsed: Duration::ZERO,
        };
        let report = capture(CaptureContext {
            run: id(),
            action: id(),
            contract: contract(),
            prepared_before: before,
            snapshot: &snapshot,
            process: &process,
            store: &store,
        })
        .unwrap();
        assert!(!report.report.forbidden_side_effects.is_empty());
        fs::remove_dir_all(root).unwrap();
        fs::remove_dir_all(store_root).unwrap();
    }
}
