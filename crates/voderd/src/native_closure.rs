//! Phase 4: Rust-only, independently re-executed validator meta-audit and
//! crash-durable final commit. Neither model output nor stdio protocol can
//! call these operations. The audit event does NOT grant delivery authority.
//! No validator's textual PASS or arbitrary user-supplied bool is sufficient.
use crate::{journal::Journal, session::State, DaemonError, Result};
use std::{collections::BTreeSet, fs, path::Path};
use voder_core::{
    action_commit::PreparedAction,
    contract::FrozenContract,
    evidence::EvidenceLedger,
    obligation::ObligationRegistry,
    types::{RequirementId, RuntimeStatus, WorkspaceIdentity},
    verdict::{self, DeliveryAssessment, DeliveryContext},
};
use voder_protocol::json::Value;
use voder_runtime::{
    artifact_store::{sha256, ArtifactStore},
    receipt::WorkspaceSnapshot,
    sandbox::{SandboxBackend, SandboxPolicy},
    validator::ValidatorRegistry,
};

fn txt(s: impl Into<String>) -> Value {
    Value::String(s.into())
}
fn no_delivery(status: &str, run: &str) -> Value {
    Value::object([
        ("run_id".into(), txt(run)),
        ("status".into(), txt(status)),
        ("execution_authorized".into(), Value::Bool(false)),
        ("delivery_allowed".into(), Value::Bool(false)),
    ])
}

fn validate_context(
    state: &State,
    context: &ClosureContext<'_>,
    expected_status: &str,
) -> Result<()> {
    let sid = context.session_id;
    let action = context.action;
    let contract = context.contract;
    let workspace = context.workspace;
    let obligations = context.obligations;
    let evidence = context.evidence;
    let hard_evidence = context.hard_evidence;
    let intent = action.intent();
    let session = state.get(sid)?;
    let run = session
        .runs
        .get(intent.run_id.as_str())
        .ok_or(DaemonError::Invalid("unknown native closure run"))?;
    if run.status != expected_status
        || run.native_action_id.as_deref() != Some(intent.action_id.as_str())
        || run.native_contract_digest.as_deref() != Some(contract.identity().digest.as_str())
        || intent.contract != *contract.identity()
        || run.native_observed_workspace_digest.as_deref() != Some(workspace.digest.as_str())
        || run.native_runtime_completed != Some(true)
        || run.native_budget_digest.is_none()
        || run.native_evidence_ids.is_empty()
        || run.native_receipt_digest.is_none()
        || run.native_workspace_digest.as_deref() != Some(intent.workspace.digest.as_str())
        || run.native_reservation_id.as_deref() != Some(intent.reservation_id.as_str())
    {
        return Err(DaemonError::Conflict(
            "native closure identity or phase mismatch",
        ));
    }
    // A journaled evidence ID by itself is not proof; the authoritative Core
    // ledger must contain the actual sealed, admitted evidence before audit.
    if evidence.len() != run.native_evidence_ids.len()
        || hard_evidence
            .iter()
            .any(|(_, id)| !run.native_evidence_ids.contains(id.as_str()))
    {
        return Err(DaemonError::Conflict(
            "durable and Core evidence count differs",
        ));
    }
    for id in &run.native_evidence_ids {
        let identity = voder_core::types::EvidenceId::try_from(id.as_str())
            .map_err(|_| DaemonError::Integrity("invalid durable evidence id".into()))?;
        let admission = evidence.get(&identity).ok_or(DaemonError::Conflict(
            "journal evidence missing from Core ledger",
        ))?;
        if admission.run_id != intent.run_id
            || admission.action_id != intent.action_id
            || admission.contract != *contract.identity()
            || admission.workspace != *workspace
        {
            return Err(DaemonError::Conflict(
                "Core admitted evidence identity mismatch",
            ));
        }
    }
    let check = verdict::evaluate_delivery(DeliveryContext {
        contract,
        workspace,
        run: &intent.run_id,
        runtime: RuntimeStatus::Completed,
        pending_transaction: None,
        obligations,
        admitted: evidence,
        evidence_for_hard_requirements: hard_evidence,
    })
    .map_err(|e| DaemonError::Integrity(format!("Core delivery preflight failed: {e:?}")))?;
    if check != DeliveryAssessment::EligibleForMetaAudit {
        return Err(DaemonError::Conflict(
            "Core closure obligations not satisfied",
        ));
    }
    Ok(())
}

pub struct ClosureContext<'a> {
    pub session_id: &'a str,
    pub action: &'a PreparedAction,
    pub contract: &'a FrozenContract,
    pub workspace: &'a WorkspaceIdentity,
    pub obligations: &'a ObligationRegistry,
    pub evidence: &'a EvidenceLedger,
    pub hard_evidence: &'a [(RequirementId, voder_core::types::EvidenceId)],
    pub workspace_root: &'a Path,
}

fn check_workspace(state: &State, context: &ClosureContext<'_>) -> Result<()> {
    let root = context.workspace_root.canonicalize()?;
    let session = state.get(context.session_id)?;
    if root.to_string_lossy().as_ref() != session.project_root.as_str() {
        return Err(DaemonError::Conflict(
            "audited root differs from session root",
        ));
    }
    let now = WorkspaceSnapshot::measure(&root)
        .and_then(|s| s.workspace(context.workspace.revision.clone()))
        .map_err(|e| {
            DaemonError::Integrity(format!("closure workspace measurement failed: {e}"))
        })?;
    if now != *context.workspace {
        return Err(DaemonError::Conflict(
            "workspace changed since verified evidence",
        ));
    }
    Ok(())
}

// Bind all meta-audit digest inputs explicitly. This preserves the v1
// journal wire format and its field order without passing nine loose values.
struct AuditDigestInputs<'a> {
    session_id: &'a str,
    run_id: &'a str,
    action_id: &'a str,
    contract_digest: &'a str,
    workspace_digest: &'a str,
    parent_receipt_digest: &'a str,
    budget_digest: &'a str,
    evidence_ids: &'a Value,
    validator_proofs: &'a Value,
}

fn audit_digest(inputs: &AuditDigestInputs<'_>) -> Result<String> {
    let AuditDigestInputs {
        session_id: sid,
        run_id: run,
        action_id: action,
        contract_digest: contract,
        workspace_digest: workspace,
        parent_receipt_digest: parent,
        budget_digest: budget,
        evidence_ids: evidences,
        validator_proofs: validators,
    } = inputs;
    sha256(format!("voder-vgr4-meta-v1\0{sid}\0{run}\0{action}\0{contract}\0{workspace}\0{parent}\0{budget}\0{}\0{}",
        evidences.encode(), validators.encode()).as_bytes())
        .map_err(|e| DaemonError::Integrity(e.to_string()))
}

// The five runtime dependencies form one trusted audit environment. The
// observed Core contract remains a separate immutable ClosureContext.
pub struct MetaAuditResources<'a, B: SandboxBackend> {
    pub registry: &'a ValidatorRegistry,
    pub backend: &'a B,
    pub policy: &'a SandboxPolicy,
    pub store: &'a ArtifactStore,
    pub capture_root: &'a Path,
}

/// Fresh, separate validator execution AFTER Core evidence admission. All
/// validators bound to any hard requirement must pass, not just one validator.
/// ValidatorRegistry owns the pinned commands; SandboxBackend is sealed by Runtime.
/// If any tool fails, no audit event is written and no final permit exists.
pub fn meta_audit<B: SandboxBackend>(
    journal: &mut Journal,
    state: &mut State,
    context: &ClosureContext<'_>,
    resources: &MetaAuditResources<'_, B>,
) -> Result<String> {
    let registry = resources.registry;
    let backend = resources.backend;
    let policy = resources.policy;
    let store = resources.store;
    let capture_root = resources.capture_root;
    let intent = context.action.intent();
    if capture_root
        .canonicalize()?
        .starts_with(context.workspace_root.canonicalize()?)
    {
        return Err(DaemonError::Conflict(
            "audit captures must be outside the observed worktree",
        ));
    }
    validate_context(state, context, "VGR_EVIDENCE_ADMITTED")?;
    check_workspace(state, context)?;
    if policy.worktree.canonicalize()? != context.workspace_root.canonicalize()? {
        return Err(DaemonError::Conflict(
            "audit sandbox root differs from verified root",
        ));
    }
    let run = &state.get(context.session_id)?.runs[intent.run_id.as_str()];
    let parent = run
        .native_receipt_digest
        .clone()
        .ok_or(DaemonError::Invalid("missing runtime receipt"))?;
    let budget = run
        .native_budget_digest
        .clone()
        .ok_or(DaemonError::Invalid("missing budget digest"))?;
    let evidence_ids = Value::Array(
        run.native_evidence_ids
            .iter()
            .map(|x| txt(x.as_str()))
            .collect(),
    );
    let mut seen = BTreeSet::new();
    let mut outcomes = Vec::new();
    for requirement in context.contract.requirements().iter().filter(|r| r.hard) {
        for binding in context
            .contract
            .validators()
            .iter()
            .filter(|v| v.requirement_ids.contains(&requirement.id))
        {
            if !seen.insert(binding.id.clone()) {
                continue;
            }
            let index = outcomes.len();
            let individual = capture_root.join(format!("vgr4-meta-{index:04}"));
            fs::create_dir(&individual)?; // existing capture directory fails closed
            let observation = registry
                .run(
                    context.contract,
                    &binding.id,
                    policy,
                    backend,
                    store,
                    &individual,
                )
                .map_err(|e| {
                    DaemonError::Integrity(format!("meta-validator failed to execute: {e}"))
                })?;
            if !observation.passed
                || observation.validator != binding.id
                || observation.version != binding.version
                || observation.executable_digest.is_empty()
            {
                return Err(DaemonError::Conflict(
                    "meta-validator failed or identity mismatch",
                ));
            }
            store
                .read(&observation.stdout_digest)
                .map_err(|e| DaemonError::Integrity(format!("meta stdout integrity: {e}")))?;
            store
                .read(&observation.stderr_digest)
                .map_err(|e| DaemonError::Integrity(format!("meta stderr integrity: {e}")))?;
            // Detect any validator that mutated the evaluated workspace.
            check_workspace(state, context)?;
            outcomes.push(Value::object([
                (
                    "validator_id".into(),
                    txt(observation.validator.to_string()),
                ),
                (
                    "validator_version".into(),
                    txt(observation.version.to_string()),
                ),
                ("stdout_digest".into(), txt(observation.stdout_digest)),
                ("stderr_digest".into(), txt(observation.stderr_digest)),
                ("command_digest".into(), txt(observation.executable_digest)),
                ("passed".into(), Value::Bool(true)),
            ]));
        }
    }
    if outcomes.is_empty() {
        return Err(DaemonError::Conflict("meta audit requires validators"));
    }
    let validators = Value::Array(outcomes);
    let digest = audit_digest(&AuditDigestInputs {
        session_id: context.session_id,
        run_id: intent.run_id.as_str(),
        action_id: intent.action_id.as_str(),
        contract_digest: intent.contract.digest.as_str(),
        workspace_digest: context.workspace.digest.as_str(),
        parent_receipt_digest: &parent,
        budget_digest: &budget,
        evidence_ids: &evidence_ids,
        validator_proofs: &validators,
    })?;
    let body = Value::object([
        ("type".into(), txt("vgr.meta_audited")),
        ("session".into(), txt(context.session_id)),
        (
            "key".into(),
            txt(format!(
                "sys-vgr4-audit-{}-{}",
                intent.run_id, intent.action_id
            )),
        ),
        ("fingerprint".into(), txt(format!("vgr4-meta:{digest}"))),
        (
            "response".into(),
            no_delivery("VGR_META_AUDITED", intent.run_id.as_str()),
        ),
        ("run_id".into(), txt(intent.run_id.to_string())),
        ("action_id".into(), txt(intent.action_id.to_string())),
        (
            "contract_digest".into(),
            txt(intent.contract.digest.to_string()),
        ),
        (
            "audited_workspace_digest".into(),
            txt(context.workspace.digest.to_string()),
        ),
        ("parent_receipt_digest".into(), txt(parent)),
        ("budget_digest".into(), txt(budget)),
        ("evidence_ids".into(), evidence_ids),
        ("validator_proofs".into(), validators),
        ("audit_digest".into(), txt(digest.clone())),
    ]);
    let entry = journal.append(body)?;
    state.apply(&entry)?;
    Ok(digest)
}

/// Commit is a separate fsynced event. No tool is re-executed, no new model
/// output is accepted. A crash between audit and commit remains fail-closed.
pub fn final_commit(
    journal: &mut Journal,
    state: &mut State,
    context: &ClosureContext<'_>,
) -> Result<String> {
    let intent = context.action.intent();
    validate_context(state, context, "VGR_META_AUDITED")?;
    check_workspace(state, context)?;
    let run = &state.get(context.session_id)?.runs[intent.run_id.as_str()];
    let audit = run
        .native_meta_audit_digest
        .clone()
        .ok_or(DaemonError::Invalid("missing durable independent audit"))?;
    let commit = sha256(
        format!(
            "voder-vgr4-commit-v1\0{}\0{}\0{}\0{}\0{}\0{}",
            context.session_id,
            intent.run_id,
            intent.action_id,
            intent.contract.digest,
            context.workspace.digest,
            audit
        )
        .as_bytes(),
    )
    .map_err(|e| DaemonError::Integrity(e.to_string()))?;
    let body = Value::object([
        ("type".into(), txt("vgr.final_committed")),
        ("session".into(), txt(context.session_id)),
        (
            "key".into(),
            txt(format!(
                "sys-vgr4-final-{}-{}",
                intent.run_id, intent.action_id
            )),
        ),
        ("fingerprint".into(), txt(format!("vgr4-final:{commit}"))),
        (
            "response".into(),
            Value::object([
                ("run_id".into(), txt(intent.run_id.to_string())),
                ("status".into(), txt("VGR_COMMITTED")),
                ("execution_authorized".into(), Value::Bool(false)),
                ("delivery_allowed".into(), Value::Bool(true)),
            ]),
        ),
        ("run_id".into(), txt(intent.run_id.to_string())),
        ("action_id".into(), txt(intent.action_id.to_string())),
        (
            "contract_digest".into(),
            txt(intent.contract.digest.to_string()),
        ),
        (
            "committed_workspace_digest".into(),
            txt(context.workspace.digest.to_string()),
        ),
        ("audit_digest".into(), txt(audit)),
        ("commit_digest".into(), txt(commit.clone())),
    ]);
    let entry = journal.append(body)?;
    state.apply(&entry)?;
    Ok(commit)
}

/// Recomputed by journal replay; caller-controlled digest strings are never trusted.
pub(crate) fn verify_meta_event(
    sid: &str,
    body: &Value,
    expected_evidence: &BTreeSet<String>,
) -> Result<String> {
    let field = |k| {
        body.get(k)
            .and_then(Value::as_str)
            .ok_or(DaemonError::Invalid("missing meta field"))
    };
    let raw_evidence = body
        .get("evidence_ids")
        .ok_or(DaemonError::Invalid("missing meta evidence"))?;
    let raw_validators = body
        .get("validator_proofs")
        .ok_or(DaemonError::Invalid("missing meta validators"))?;
    let Value::Array(evidence) = raw_evidence else {
        return Err(DaemonError::Integrity("bad meta evidence array".into()));
    };
    let Value::Array(validators) = raw_validators else {
        return Err(DaemonError::Integrity("bad meta validators array".into()));
    };
    if evidence.len() != expected_evidence.len() || validators.is_empty() || validators.len() > 256
    {
        return Err(DaemonError::Integrity("meta audit count mismatch".into()));
    }
    let actual: Vec<_> = evidence.iter().map(|v| v.as_str()).collect();
    if actual.iter().any(|v| v.is_none())
        || actual.into_iter().flatten().collect::<Vec<_>>()
            != expected_evidence
                .iter()
                .map(String::as_str)
                .collect::<Vec<_>>()
    {
        return Err(DaemonError::Integrity(
            "meta evidence IDs differ from journal".into(),
        ));
    }
    let mut seen = BTreeSet::new();
    for item in validators {
        let get = |k| {
            item.get(k)
                .and_then(Value::as_str)
                .ok_or(DaemonError::Invalid("bad meta validator field"))
        };
        let id = get("validator_id")?;
        if id.is_empty()
            || !seen.insert(id.to_owned())
            || get("validator_version")?.is_empty()
            || item.get("passed") != Some(&Value::Bool(true))
        {
            return Err(DaemonError::Integrity(
                "meta validator failure/duplicate".into(),
            ));
        }
        for key in ["stdout_digest", "stderr_digest", "command_digest"] {
            let hash = get(key)?;
            if hash.len() != 64 || !hash.bytes().all(|c| c.is_ascii_hexdigit()) {
                return Err(DaemonError::Integrity(
                    "bad validator artifact digest".into(),
                ));
            }
        }
    }
    let digest = audit_digest(&AuditDigestInputs {
        session_id: sid,
        run_id: field("run_id")?,
        action_id: field("action_id")?,
        contract_digest: field("contract_digest")?,
        workspace_digest: field("audited_workspace_digest")?,
        parent_receipt_digest: field("parent_receipt_digest")?,
        budget_digest: field("budget_digest")?,
        evidence_ids: raw_evidence,
        validator_proofs: raw_validators,
    })?;
    if digest != field("audit_digest")? {
        return Err(DaemonError::Integrity("meta audit digest mismatch".into()));
    }
    Ok(digest)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};
    fn temp() -> std::path::PathBuf {
        let n = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let p = std::env::temp_dir().join(format!("voder-vgr4-test-{}-{n}", std::process::id()));
        fs::create_dir(&p).unwrap();
        p
    }
    fn event(kind: &str, key: &str, fields: Vec<(&str, Value)>, response: Value) -> Value {
        let mut v = vec![
            ("type".to_string(), txt(kind)),
            ("session".to_string(), txt("s-1")),
            ("key".to_string(), txt(key)),
            ("fingerprint".to_string(), txt("rust-only-test")),
            ("response".to_string(), response),
        ];
        v.extend(fields.into_iter().map(|(k, val)| (k.to_owned(), val)));
        Value::object(v)
    }
    fn f(k: &'static str, v: &str) -> (&'static str, Value) {
        (k, txt(v))
    }
    fn deny() -> Value {
        no_delivery("NONE", "r-1")
    }
    fn prefix(j: &mut Journal) {
        j.append(event(
            "session.created",
            "create",
            vec![f("project_root", "/tmp/repo")],
            deny(),
        ))
        .unwrap();
        j.append(event(
            "task.model_requested",
            "request",
            vec![
                f("run_id", "r-1"),
                f("instruction_digest", "input"),
                f("reason", "started"),
            ],
            deny(),
        ))
        .unwrap();
        j.append(event(
            "task.model_proposed",
            "proposal",
            vec![
                f("run_id", "r-1"),
                f("reason", "proposed"),
                f("proposal_digest", "model"),
            ],
            deny(),
        ))
        .unwrap();
        j.append(event(
            "vgr.action_prepared",
            "prep",
            vec![
                f("run_id", "r-1"),
                f("action_id", "a-1"),
                f("reservation_id", "reserved"),
                f("proposal_digest", "model"),
                f("contract_digest", "contract"),
                f("workspace_digest", "before"),
            ],
            deny(),
        ))
        .unwrap();
        j.append(event(
            "vgr.action_dispatched",
            "dispatch",
            vec![
                f("run_id", "r-1"),
                f("action_id", "a-1"),
                f("reservation_id", "reserved"),
                f("contract_digest", "contract"),
                f("workspace_digest", "before"),
            ],
            Value::object([
                ("execution_authorized".into(), Value::Bool(true)),
                ("delivery_allowed".into(), Value::Bool(false)),
            ]),
        ))
        .unwrap();
        j.append(event(
            "vgr.action_observed",
            "observe",
            vec![
                f("run_id", "r-1"),
                f("action_id", "a-1"),
                f("contract_digest", "contract"),
                f("workspace_digest", "before"),
                f("observed_workspace_digest", "after"),
                f("receipt_digest", "receipt"),
                ("runtime_completed", Value::Bool(true)),
            ],
            Value::object([
                ("execution_authorized".into(), Value::Bool(false)),
                ("delivery_allowed".into(), Value::Bool(false)),
                ("evidence_admitted".into(), Value::Bool(false)),
            ]),
        ))
        .unwrap();
        let fields = [0u64; 8];
        let budget = sha256(
            format!(
                "vgr-budget-v1\0s-1\0r-1\0a-1\0receipt\0{:?}\0{:?}\0{:?}\0{:?}",
                fields, fields, fields, fields
            )
            .as_bytes(),
        )
        .unwrap();
        let amounts = Value::object(
            [
                "steps",
                "tokens",
                "model_calls",
                "tool_calls",
                "wall_time_ms",
                "risk_microunits",
                "context_tokens",
                "api_cost_microusd",
            ]
            .into_iter()
            .map(|k| (k.to_owned(), Value::Number("0".into()))),
        );
        j.append(event(
            "vgr.budget_settled",
            "settle",
            vec![
                f("run_id", "r-1"),
                f("action_id", "a-1"),
                f("reservation_id", "reserved"),
                f("contract_digest", "contract"),
                f("workspace_digest", "before"),
                f("receipt_digest", "receipt"),
                f("budget_digest", &budget),
                ("reserved_upper", amounts.clone()),
                ("actual", amounts.clone()),
                ("budget_spent", amounts.clone()),
                ("budget_reserved", amounts),
            ],
            Value::object([
                ("execution_authorized".into(), Value::Bool(false)),
                ("delivery_allowed".into(), Value::Bool(false)),
                ("evidence_admitted".into(), Value::Bool(false)),
            ]),
        ))
        .unwrap();
        j.append(event(
            "vgr.evidence_admitted",
            "evidence",
            vec![
                f("run_id", "r-1"),
                f("action_id", "a-1"),
                f("parent_receipt_digest", "receipt"),
                f("contract_digest", "contract"),
                f("evidence_workspace_digest", "after"),
                f("evidence_id", "e-1"),
                f("content_digest", "content"),
            ],
            Value::object([
                ("execution_authorized".into(), Value::Bool(false)),
                ("delivery_allowed".into(), Value::Bool(false)),
                ("evidence_admitted".into(), Value::Bool(true)),
            ]),
        ))
        .unwrap();
    }
    fn audited(j: &mut Journal, corrupt: bool) -> String {
        let budget = j
            .entries()
            .iter()
            .find(|e| e.body.get("type").and_then(Value::as_str) == Some("vgr.budget_settled"))
            .unwrap()
            .body
            .get("budget_digest")
            .unwrap()
            .as_str()
            .unwrap()
            .to_owned();
        let evidences = Value::Array(vec![txt("e-1")]);
        let validators = Value::Array(vec![Value::object([
            ("validator_id".into(), txt("v-1")),
            ("validator_version".into(), txt("1")),
            ("stdout_digest".into(), txt("a".repeat(64))),
            ("stderr_digest".into(), txt("b".repeat(64))),
            ("command_digest".into(), txt("c".repeat(64))),
            ("passed".into(), Value::Bool(!corrupt)),
        ])]);
        let digest = audit_digest(&AuditDigestInputs {
            session_id: "s-1",
            run_id: "r-1",
            action_id: "a-1",
            contract_digest: "contract",
            workspace_digest: "after",
            parent_receipt_digest: "receipt",
            budget_digest: &budget,
            evidence_ids: &evidences,
            validator_proofs: &validators,
        })
        .unwrap();
        j.append(event(
            "vgr.meta_audited",
            "meta",
            vec![
                f("run_id", "r-1"),
                f("action_id", "a-1"),
                f("contract_digest", "contract"),
                f("audited_workspace_digest", "after"),
                f("parent_receipt_digest", "receipt"),
                f("budget_digest", &budget),
                f("audit_digest", &digest),
                ("evidence_ids", evidences),
                ("validator_proofs", validators),
            ],
            no_delivery("VGR_META_AUDITED", "r-1"),
        ))
        .unwrap();
        digest
    }
    fn committed(j: &mut Journal, audit: &str, corrupt: bool) {
        let digest = sha256(
            format!("voder-vgr4-commit-v1\0s-1\0r-1\0a-1\0contract\0after\0{audit}").as_bytes(),
        )
        .unwrap();
        j.append(event(
            "vgr.final_committed",
            "final",
            vec![
                f("run_id", "r-1"),
                f("action_id", "a-1"),
                f("contract_digest", "contract"),
                f("committed_workspace_digest", "after"),
                f("audit_digest", audit),
                f("commit_digest", if corrupt { "bad" } else { &digest }),
            ],
            Value::object([
                ("run_id".into(), txt("r-1")),
                ("status".into(), txt("VGR_COMMITTED")),
                ("execution_authorized".into(), Value::Bool(false)),
                ("delivery_allowed".into(), Value::Bool(true)),
            ]),
        ))
        .unwrap();
    }
    #[test]
    fn final_replay_is_terminal_and_idempotency_is_not_repeatable() {
        let dir = temp();
        let mut journal = Journal::open(&dir.join("events.jsonl")).unwrap();
        prefix(&mut journal);
        let digest = audited(&mut journal, false);
        assert!(crate::recovery::recover(&journal).is_err()); // never auto-finalize after crash
        committed(&mut journal, &digest, false);
        let state = crate::recovery::recover(&journal).unwrap();
        let run = &state.get("s-1").unwrap().runs["r-1"];
        assert_eq!(run.status, "VGR_COMMITTED");
        assert!(run.native_final_commit_digest.is_some());
        committed(&mut journal, &digest, false);
        assert!(State::reconstruct(journal.entries()).is_err());
        drop(journal);
        fs::remove_dir_all(dir).unwrap();
    }
    #[test]
    fn reject_fake_validator_and_tampered_commit() {
        for failure in ["validator", "commit"] {
            let dir = temp();
            let mut j = Journal::open(&dir.join("events.jsonl")).unwrap();
            prefix(&mut j);
            let digest = audited(&mut j, failure == "validator");
            if failure == "commit" {
                committed(&mut j, &digest, true);
            }
            assert!(State::reconstruct(j.entries()).is_err());
            drop(j);
            fs::remove_dir_all(dir).unwrap();
        }
    }
    #[test]
    fn missing_final_response_identity_fails_closed() {
        let dir = temp();
        let mut j = Journal::open(&dir.join("events.jsonl")).unwrap();
        prefix(&mut j);
        let digest = audited(&mut j, false);
        let commit_digest = sha256(
            format!("voder-vgr4-commit-v1\0s-1\0r-1\0a-1\0contract\0after\0{digest}").as_bytes(),
        )
        .unwrap();
        j.append(event(
            "vgr.final_committed",
            "final-incomplete",
            vec![
                f("run_id", "r-1"),
                f("action_id", "a-1"),
                f("contract_digest", "contract"),
                f("committed_workspace_digest", "after"),
                f("audit_digest", &digest),
                f("commit_digest", &commit_digest),
            ],
            Value::object([
                ("execution_authorized".into(), Value::Bool(false)),
                ("delivery_allowed".into(), Value::Bool(true)),
            ]),
        ))
        .unwrap();
        assert!(State::reconstruct(j.entries()).is_err());
        drop(j);
        fs::remove_dir_all(dir).unwrap();
    }
    #[test]
    fn final_without_audit_never_grants_delivery() {
        let dir = temp();
        let mut j = Journal::open(&dir.join("events.jsonl")).unwrap();
        prefix(&mut j);
        committed(&mut j, "fake-audit", false);
        assert!(State::reconstruct(j.entries()).is_err());
        drop(j);
        fs::remove_dir_all(dir).unwrap();
    }
}
