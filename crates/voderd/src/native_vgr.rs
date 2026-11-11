//! Native VGR pre-dispatch admission. The MODEL never obtains this interface.
//! A prepared action is a journaled intent, NOT a tool permit, verified receipt,
//! committed change or delivery certificate. A crash after PREPARED requires
//! explicit reconciliation before further consequential work.
use crate::{journal::Journal, session::State, DaemonError, Result};
use std::collections::BTreeSet;
use voder_core::{
    action_commit::{ActionIntent, ApprovalGrant, CommitGate, PreparedAction},
    budget::BudgetLedger,
    contract::FrozenContract,
    types::{ActionId, InputId, RunId, WorkspaceIdentity},
};
use voder_protocol::json::Value;

/// Typed, trusted-only inputs for Core VGR pre-dispatch preparation.
/// This bundle contains no capability to execute tools or authorize delivery.
/// The ownership boundary stays with Server's authoritative journal and state.
pub struct NativePrepareRequest<'a> {
    pub session_id: &'a str,
    pub contract: &'a FrozenContract,
    pub workspace: &'a WorkspaceIdentity,
    pub verified_inputs: &'a [InputId],
    pub budget: &'a BudgetLedger,
    pub intent: ActionIntent,
    pub approval: Option<ApprovalGrant>,
}

/// In-memory replay fence is derived from durable journal; no silent reuse after crash.
pub struct NativePreparation {
    commit_gate: CommitGate,
    used_actions: BTreeSet<(RunId, ActionId)>,
    used_reservations: BTreeSet<String>,
}
impl NativePreparation {
    pub fn recover(journal: &Journal) -> Result<Self> {
        let mut used_actions = BTreeSet::new();
        let mut used_reservations = BTreeSet::new();
        for entry in journal.entries() {
            if entry.body.get("type").and_then(Value::as_str) != Some("vgr.action_prepared") {
                continue;
            }
            let run_value = entry
                .body
                .get("run_id")
                .and_then(Value::as_str)
                .ok_or(DaemonError::Invalid("missing native VGR run"))?;
            let action_value = entry
                .body
                .get("action_id")
                .and_then(Value::as_str)
                .ok_or(DaemonError::Invalid("missing native VGR action"))?;
            let reservation = entry
                .body
                .get("reservation_id")
                .and_then(Value::as_str)
                .ok_or(DaemonError::Invalid("missing native VGR reservation"))?
                .to_owned();
            let run =
                RunId::try_from(run_value).map_err(|_| DaemonError::Invalid("invalid run id"))?;
            let action = ActionId::try_from(action_value)
                .map_err(|_| DaemonError::Invalid("invalid action id"))?;
            if reservation.is_empty()
                || !used_actions.insert((run, action))
                || !used_reservations.insert(reservation)
            {
                return Err(DaemonError::Integrity(
                    "duplicate durable native VGR preparation".into(),
                ));
            }
        }
        Ok(Self {
            commit_gate: CommitGate::new(),
            used_actions,
            used_reservations,
        })
    }

    /// Single writer only. Caller must own daemon's OS lock and validate the
    /// session has an inert PROPOSED model receipt before calling this method.
    /// All writes must be recorded with the same authoritative Journal as State.
    /// Persist the dispatch decision BEFORE the Runtime is allowed to execute.
    /// PreparedAction is sealed by Core; the Daemon must already hold the OS
    /// single-writer lock and must reject any expired/unreconciled preparation.
    pub fn dispatch(
        &mut self,
        journal: &mut Journal,
        state: &mut State,
        session_id: &str,
        action: &PreparedAction,
    ) -> Result<()> {
        let intent = action.intent();
        let run_id = intent.run_id.to_string();
        let action_id = intent.action_id.to_string();
        let run = state
            .get(session_id)?
            .runs
            .get(&run_id)
            .ok_or(DaemonError::Invalid("unknown prepared VGR run"))?;
        if run.status != "VGR_PREPARED"
            || run.native_action_id.as_deref() != Some(action_id.as_str())
            || run.native_reservation_id.as_deref() != Some(intent.reservation_id.as_str())
            || run.native_contract_digest.as_deref() != Some(intent.contract.digest.as_str())
            || run.native_workspace_digest.as_deref() != Some(intent.workspace.digest.as_str())
        {
            return Err(DaemonError::Conflict(
                "VGR prepared identities do not match",
            ));
        }
        if !self
            .used_actions
            .contains(&(intent.run_id.clone(), intent.action_id.clone()))
        {
            return Err(DaemonError::Conflict(
                "VGR action not in replay-fenced registry",
            ));
        }
        let response = Value::object([
            ("run_id".into(), Value::String(run_id.clone())),
            ("status".into(), Value::String("VGR_DISPATCHED".into())),
            ("execution_authorized".into(), Value::Bool(true)),
            ("delivery_allowed".into(), Value::Bool(false)),
        ]);
        let body = Value::object([
            ("type".into(), Value::String("vgr.action_dispatched".into())),
            ("session".into(), Value::String(session_id.into())),
            (
                "key".into(),
                Value::String(format!("sys-vgr-dispatch-{run_id}-{action_id}")),
            ),
            (
                "fingerprint".into(),
                Value::String(format!("native-vgr-dispatch:{action_id}")),
            ),
            ("response".into(), response),
            ("run_id".into(), Value::String(run_id)),
            ("action_id".into(), Value::String(action_id)),
            (
                "reservation_id".into(),
                Value::String(intent.reservation_id.to_string()),
            ),
            (
                "contract_digest".into(),
                Value::String(intent.contract.digest.to_string()),
            ),
            (
                "workspace_digest".into(),
                Value::String(intent.workspace.digest.to_string()),
            ),
        ]);
        let record = journal.append(body)?;
        state.apply(&record)?;
        Ok(())
    }

    /// Persist a measured runtime receipt, without converting a test exit code
    /// into trusted Core evidence or a delivery verdict. No model input accepted.
    pub fn observe(
        &mut self,
        journal: &mut Journal,
        state: &mut State,
        session_id: &str,
        action: &PreparedAction,
        receipt: &voder_runtime::receipt::RuntimeReceipt,
        store: &voder_runtime::artifact_store::ArtifactStore,
    ) -> Result<voder_core::consequence::ConsequenceAssessment> {
        let intent = action.intent();
        let run_id = intent.run_id.to_string();
        let action_id = intent.action_id.to_string();
        let run = state
            .get(session_id)?
            .runs
            .get(&run_id)
            .ok_or(DaemonError::Invalid("unknown dispatched VGR run"))?;
        if run.status != "VGR_DISPATCHED"
            || run.native_action_id.as_deref() != Some(action_id.as_str())
            || run.native_contract_digest.as_deref() != Some(intent.contract.digest.as_str())
            || run.native_workspace_digest.as_deref() != Some(intent.workspace.digest.as_str())
        {
            return Err(DaemonError::Conflict(
                "VGR observation lacks matching dispatch",
            ));
        }
        let report = &receipt.report;
        if report.run_id != intent.run_id
            || report.action_id != intent.action_id
            || report.contract != intent.contract
            || report.before != intent.workspace
        {
            return Err(DaemonError::Integrity(
                "runtime receipt identity mismatch".into(),
            ));
        }
        // Independently re-read both content-addressed artifacts: a text PASS
        // or a digest supplied by a model is never accepted as evidence.
        store
            .read(&receipt.stdout_artifact)
            .map_err(|e| DaemonError::Integrity(e.to_string()))?;
        store
            .read(&receipt.stderr_artifact)
            .map_err(|e| DaemonError::Integrity(e.to_string()))?;
        // Recompute Runtime's measured report fingerprint from the actual
        // content-addressed stream artifacts and recorded workspace diff.
        let canonical = format!(
            "{}\0{}\0{}\0{}\0{}\0{}\0{:?}\0{}\0{}\0{:?}",
            report.run_id,
            report.action_id,
            report.contract.digest,
            report.before.digest,
            report.after.digest,
            receipt.stdout_artifact,
            report.completion,
            receipt.stderr_artifact,
            receipt.exit_code.unwrap_or(-1),
            report.observed_changes
        );
        let observed_digest = voder_runtime::artifact_store::sha256(canonical.as_bytes())
            .map_err(|e| DaemonError::Integrity(e.to_string()))?;
        if observed_digest != report.receipt_digest.as_str() {
            return Err(DaemonError::Integrity(
                "runtime report digest mismatch".into(),
            ));
        }
        let assessment = voder_core::consequence::compare_consequences(action, report);
        let response = Value::object([
            ("run_id".into(), Value::String(run_id.clone())),
            ("status".into(), Value::String("VGR_OBSERVED".into())),
            ("execution_authorized".into(), Value::Bool(false)),
            ("evidence_admitted".into(), Value::Bool(false)),
            ("delivery_allowed".into(), Value::Bool(false)),
        ]);
        let body = Value::object([
            ("type".into(), Value::String("vgr.action_observed".into())),
            ("session".into(), Value::String(session_id.into())),
            (
                "key".into(),
                Value::String(format!("sys-vgr-observed-{run_id}-{action_id}")),
            ),
            (
                "fingerprint".into(),
                Value::String(format!("runtime-observation:{}", report.receipt_digest)),
            ),
            ("response".into(), response),
            ("run_id".into(), Value::String(run_id)),
            ("action_id".into(), Value::String(action_id)),
            (
                "contract_digest".into(),
                Value::String(intent.contract.digest.to_string()),
            ),
            (
                "workspace_digest".into(),
                Value::String(intent.workspace.digest.to_string()),
            ),
            (
                "receipt_digest".into(),
                Value::String(report.receipt_digest.to_string()),
            ),
            (
                "observed_workspace_digest".into(),
                Value::String(report.after.digest.to_string()),
            ),
            (
                "runtime_completed".into(),
                Value::Bool(matches!(
                    report.completion,
                    voder_core::consequence::ExecutionCompletion::Completed
                )),
            ),
            (
                "consequence".into(),
                Value::String(format!("{:?}", assessment.next)),
            ),
        ]);
        let record = journal.append(body)?;
        state.apply(&record)?;
        Ok(assessment)
    }

    pub fn prepare(
        &mut self,
        journal: &mut Journal,
        state: &mut State,
        request: NativePrepareRequest<'_>,
    ) -> Result<PreparedAction> {
        let NativePrepareRequest {
            session_id,
            contract,
            workspace,
            verified_inputs,
            budget,
            intent,
            approval,
        } = request;
        let run_id = intent.run_id.to_string();
        let action_id = intent.action_id.to_string();
        let reservation_id = intent.reservation_id.to_string();
        let run = state
            .get(session_id)?
            .runs
            .get(&run_id)
            .ok_or(DaemonError::Invalid("unknown native VGR run"))?;
        if run.status != "PROPOSED" {
            return Err(DaemonError::Conflict(
                "native VGR requires a stored inert proposal",
            ));
        }
        let proposal_digest = run
            .proposal_digest
            .as_ref()
            .ok_or(DaemonError::Invalid("native VGR proposal digest missing"))?
            .clone();
        if self
            .used_actions
            .contains(&(intent.run_id.clone(), intent.action_id.clone()))
            || self.used_reservations.contains(&reservation_id)
        {
            return Err(DaemonError::Conflict(
                "native VGR action or reservation already prepared",
            ));
        }
        let prepared = self
            .commit_gate
            .prepare(
                contract,
                workspace,
                verified_inputs,
                budget,
                intent,
                approval,
            )
            .map_err(|_| DaemonError::Invalid("Core VGR preparation rejected"))?;
        let response = Value::object([
            ("run_id".into(), Value::String(run_id.clone())),
            ("status".into(), Value::String("VGR_PREPARED".into())),
            ("execution_authorized".into(), Value::Bool(false)),
            ("delivery_allowed".into(), Value::Bool(false)),
        ]);
        let body = Value::object([
            ("type".into(), Value::String("vgr.action_prepared".into())),
            ("session".into(), Value::String(session_id.into())),
            (
                "key".into(),
                Value::String(format!("sys-vgr-prep-{run_id}-{action_id}")),
            ),
            (
                "fingerprint".into(),
                Value::String(format!("native-vgr-intent:{action_id}")),
            ),
            ("response".into(), response),
            ("run_id".into(), Value::String(run_id.clone())),
            ("action_id".into(), Value::String(action_id.clone())),
            (
                "reservation_id".into(),
                Value::String(reservation_id.clone()),
            ),
            ("proposal_digest".into(), Value::String(proposal_digest)),
            (
                "contract_digest".into(),
                Value::String(contract.identity().digest.to_string()),
            ),
            (
                "workspace_digest".into(),
                Value::String(workspace.digest.to_string()),
            ),
            (
                "reason".into(),
                Value::String("Core prepared; no execution permit minted".into()),
            ),
        ]);
        // Journal append fsyncs BEFORE the authority state is observed.
        let record = journal.append(body)?;
        // An unexpected apply error after fsync makes caller stop permanently;
        // on restart replay validates the persisted event and refuses dispatch.
        state.apply(&record)?;
        self.used_actions.insert((
            prepared.intent().run_id.clone(),
            prepared.intent().action_id.clone(),
        ));
        self.used_reservations.insert(reservation_id);
        Ok(prepared)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{
        fs,
        path::PathBuf,
        time::{SystemTime, UNIX_EPOCH},
    };

    fn private_dir() -> PathBuf {
        let n = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir =
            std::env::temp_dir().join(format!("voder-native-vgr-test-{}-{n}", std::process::id()));
        fs::create_dir(&dir).unwrap();
        dir
    }
    fn event(kind: &str, key: &str, others: Vec<(&str, &str)>, response: Value) -> Value {
        let mut entries = vec![
            ("type".into(), Value::String(kind.into())),
            ("session".into(), Value::String("s-1".into())),
            ("key".into(), Value::String(key.into())),
            ("fingerprint".into(), Value::String("source-test".into())),
            ("response".into(), response),
        ];
        for (key, value) in others {
            entries.push((key.into(), Value::String(value.into())));
        }
        Value::object(entries)
    }
    fn append_fixture(journal: &mut Journal) {
        let blank = Value::object([]);
        journal
            .append(event(
                "session.created",
                "create",
                vec![("project_root", "/tmp/code")],
                blank.clone(),
            ))
            .unwrap();
        journal
            .append(event(
                "task.model_requested",
                "request",
                vec![
                    ("run_id", "r-1"),
                    ("reason", "provider"),
                    ("instruction_digest", "sha256:work"),
                ],
                blank.clone(),
            ))
            .unwrap();
        journal
            .append(event(
                "task.model_proposed",
                "proposed",
                vec![
                    ("run_id", "r-1"),
                    ("reason", "inert"),
                    ("proposal_digest", "sha256:proposal"),
                ],
                blank,
            ))
            .unwrap();
    }
    fn prepared(authorized: bool) -> Value {
        let response = Value::object([
            ("execution_authorized".into(), Value::Bool(authorized)),
            ("delivery_allowed".into(), Value::Bool(false)),
        ]);
        event(
            "vgr.action_prepared",
            "native",
            vec![
                ("run_id", "r-1"),
                ("action_id", "action-1"),
                ("reservation_id", "reservation-1"),
                ("proposal_digest", "sha256:proposal"),
                ("contract_digest", "sha256:contract"),
                ("workspace_digest", "sha256:workspace"),
            ],
            response,
        )
    }
    #[test]
    fn durable_prepare_has_no_authority_and_blocks_unreconciled_restart() {
        let dir = private_dir();
        let mut journal = Journal::open(&dir.join("events.jsonl")).unwrap();
        append_fixture(&mut journal);
        journal.append(prepared(false)).unwrap();
        let state = State::reconstruct(journal.entries()).unwrap();
        assert_eq!(state.get("s-1").unwrap().runs["r-1"].status, "VGR_PREPARED");
        assert!(crate::recovery::recover(&journal).is_err());
        assert!(NativePreparation::recover(&journal).is_ok());
        drop(journal);
        fs::remove_dir_all(dir).unwrap();
    }
    #[test]
    fn model_claimed_authority_is_not_a_native_prepared_event() {
        let dir = private_dir();
        let mut journal = Journal::open(&dir.join("events.jsonl")).unwrap();
        append_fixture(&mut journal);
        journal.append(prepared(true)).unwrap();
        assert!(State::reconstruct(journal.entries()).is_err());
        drop(journal);
        fs::remove_dir_all(dir).unwrap();
    }
    #[test]
    fn journal_replay_fences_duplicate_action_and_reservation() {
        let dir = private_dir();
        let mut journal = Journal::open(&dir.join("events.jsonl")).unwrap();
        append_fixture(&mut journal);
        journal.append(prepared(false)).unwrap();
        journal
            .append(event(
                "vgr.action_prepared",
                "native-2",
                vec![
                    ("run_id", "r-1"),
                    ("action_id", "action-2"),
                    ("reservation_id", "reservation-1"),
                ],
                Value::object([]),
            ))
            .unwrap();
        assert!(NativePreparation::recover(&journal).is_err());
        drop(journal);
        fs::remove_dir_all(dir).unwrap();
    }
    fn dispatched(authorized: bool) -> Value {
        let response = Value::object([
            ("execution_authorized".into(), Value::Bool(authorized)),
            ("delivery_allowed".into(), Value::Bool(false)),
        ]);
        event(
            "vgr.action_dispatched",
            "dispatch",
            vec![
                ("run_id", "r-1"),
                ("action_id", "action-1"),
                ("reservation_id", "reservation-1"),
                ("contract_digest", "sha256:contract"),
                ("workspace_digest", "sha256:workspace"),
            ],
            response,
        )
    }
    #[test]
    fn journal_dispatch_is_replayable_but_not_automatically_reexecutable() {
        let dir = private_dir();
        let mut journal = Journal::open(&dir.join("events.jsonl")).unwrap();
        append_fixture(&mut journal);
        journal.append(prepared(false)).unwrap();
        journal.append(dispatched(true)).unwrap();
        let state = State::reconstruct(journal.entries()).unwrap();
        assert_eq!(
            state.get("s-1").unwrap().runs["r-1"].status,
            "VGR_DISPATCHED"
        );
        assert!(crate::recovery::recover(&journal).is_err());
        journal.append(dispatched(true)).unwrap();
        assert!(State::reconstruct(journal.entries()).is_err());
        drop(journal);
        fs::remove_dir_all(dir).unwrap();
    }
    #[test]
    fn forged_dispatch_flag_and_inconsistent_identity_fail_replay() {
        let dir = private_dir();
        let mut journal = Journal::open(&dir.join("events.jsonl")).unwrap();
        append_fixture(&mut journal);
        journal.append(prepared(false)).unwrap();
        journal.append(dispatched(false)).unwrap();
        assert!(State::reconstruct(journal.entries()).is_err());
        drop(journal);
        fs::remove_dir_all(&dir).unwrap();
        let dir = private_dir();
        let mut journal = Journal::open(&dir.join("events.jsonl")).unwrap();
        append_fixture(&mut journal);
        journal.append(prepared(false)).unwrap();
        let mut bad = dispatched(true);
        if let Value::Object(ref mut values) = bad {
            values.insert(
                "workspace_digest".into(),
                Value::String("other-workspace".into()),
            );
        }
        journal.append(bad).unwrap();
        assert!(State::reconstruct(journal.entries()).is_err());
        drop(journal);
        fs::remove_dir_all(dir).unwrap();
    }
    #[test]
    fn observed_without_durable_dispatch_is_rejected() {
        let dir = private_dir();
        let mut journal = Journal::open(&dir.join("events.jsonl")).unwrap();
        append_fixture(&mut journal);
        journal.append(prepared(false)).unwrap();
        let response = Value::object([
            ("execution_authorized".into(), Value::Bool(false)),
            ("delivery_allowed".into(), Value::Bool(false)),
            ("evidence_admitted".into(), Value::Bool(false)),
        ]);
        journal
            .append(event(
                "vgr.action_observed",
                "observed",
                vec![
                    ("run_id", "r-1"),
                    ("action_id", "action-1"),
                    ("contract_digest", "sha256:contract"),
                    ("workspace_digest", "sha256:workspace"),
                    ("receipt_digest", "sha256:receipt"),
                ],
                response,
            ))
            .unwrap();
        assert!(State::reconstruct(journal.entries()).is_err());
        drop(journal);
        fs::remove_dir_all(dir).unwrap();
    }
    #[test]
    fn orphaned_preparation_cannot_be_replayed_as_a_successful_task() {
        let dir = private_dir();
        let mut journal = Journal::open(&dir.join("events.jsonl")).unwrap();
        journal.append(prepared(false)).unwrap();
        assert!(State::reconstruct(journal.entries()).is_err());
        drop(journal);
        fs::remove_dir_all(dir).unwrap();
    }
}
