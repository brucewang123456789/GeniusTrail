//! Third-stage crash-durable accounting gate. This module **never** issues
//! delivery authority, admits model output as evidence, or replays a tool call.
//! Budget settlement is a trusted Daemon-only API, not a wire command.
use crate::{journal::Journal, session::State, DaemonError, Result};
use voder_core::{
    action_commit::PreparedAction,
    budget::{BudgetAmount, BudgetLedger},
};
use voder_protocol::json::Value;
use voder_runtime::artifact_store::sha256;

const AXES: [&str; 8] = [
    "steps",
    "tokens",
    "model_calls",
    "tool_calls",
    "wall_time_ms",
    "risk_microunits",
    "context_tokens",
    "api_cost_microusd",
];
fn fields(v: BudgetAmount) -> [u64; 8] {
    [
        v.steps,
        v.tokens,
        v.model_calls,
        v.tool_calls,
        v.wall_time_ms,
        v.risk_microunits,
        v.context_tokens,
        v.api_cost_microusd,
    ]
}
fn encode(v: BudgetAmount) -> Value {
    Value::object(
        AXES.into_iter()
            .zip(fields(v))
            .map(|(key, amount)| (key.to_owned(), Value::Number(amount.to_string()))),
    )
}

/// The calling Daemon must own the single-writer lock. The supplied `actual`
/// must originate in the measured Runtime/provider receipts, never from a
/// model instruction or untrusted JSON. This transition intentionally marks
/// `evidence_admitted=false` and cannot finalize delivery.
pub fn settle_observed_action(
    journal: &mut Journal,
    state: &mut State,
    session_id: &str,
    action: &PreparedAction,
    ledger: &mut BudgetLedger,
    actual: BudgetAmount,
) -> Result<()> {
    let intent = action.intent();
    let run_id = intent.run_id.to_string();
    let action_id = intent.action_id.to_string();
    let run = state
        .get(session_id)?
        .runs
        .get(&run_id)
        .ok_or(DaemonError::Invalid("unknown observed VGR action"))?;
    if run.status != "VGR_OBSERVED"
        || run.native_action_id.as_deref() != Some(action_id.as_str())
        || run.native_reservation_id.as_deref() != Some(intent.reservation_id.as_str())
        || run.native_contract_digest.as_deref() != Some(intent.contract.digest.as_str())
        || run.native_workspace_digest.as_deref() != Some(intent.workspace.digest.as_str())
    {
        return Err(DaemonError::Conflict(
            "budget settlement requires matching observed VGR action",
        ));
    }
    let receipt_digest = run
        .native_receipt_digest
        .as_ref()
        .ok_or(DaemonError::Invalid("observed receipt identity missing"))?
        .clone();
    let reservation = ledger
        .record(&intent.reservation_id)
        .ok_or(DaemonError::Invalid("missing Core budget reservation"))?;
    if reservation.upper_bound != action.budget_upper_bound() {
        return Err(DaemonError::Conflict(
            "prepared action and reservation budget mismatch",
        ));
    }
    // Simulate BOTH transitions before the journal append. A failed settle
    // must leave original budget, journal and daemon state unchanged.
    let mut next = ledger.clone();
    next.mark_dispatched(&intent.reservation_id)
        .map_err(|_| DaemonError::Conflict("budget reservation is not held"))?;
    next.settle(&intent.reservation_id, actual).map_err(|_| {
        DaemonError::Conflict("actual usage exceeds reservation or phase is invalid")
    })?;
    let upper = reservation.upper_bound;
    let snapshot = next.snapshot();
    let canonical = format!(
        "vgr-budget-v1\0{}\0{}\0{}\0{}\0{:?}\0{:?}\0{:?}\0{:?}",
        session_id,
        run_id,
        action_id,
        receipt_digest,
        fields(upper),
        fields(actual),
        fields(snapshot.spent),
        fields(snapshot.reserved),
    );
    let digest = sha256(canonical.as_bytes()).map_err(|e| DaemonError::Integrity(e.to_string()))?;
    let response = Value::object([
        ("run_id".into(), Value::String(run_id.clone())),
        ("status".into(), Value::String("VGR_BUDGET_SETTLED".into())),
        ("execution_authorized".into(), Value::Bool(false)),
        ("evidence_admitted".into(), Value::Bool(false)),
        ("delivery_allowed".into(), Value::Bool(false)),
    ]);
    let body = Value::object([
        ("type".into(), Value::String("vgr.budget_settled".into())),
        ("session".into(), Value::String(session_id.into())),
        (
            "key".into(),
            Value::String(format!("sys-vgr-settle-{run_id}-{action_id}")),
        ),
        (
            "fingerprint".into(),
            Value::String(format!("budget-settlement:{digest}")),
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
        ("receipt_digest".into(), Value::String(receipt_digest)),
        ("reserved_upper".into(), encode(upper)),
        ("actual".into(), encode(actual)),
        ("budget_spent".into(), encode(snapshot.spent)),
        ("budget_reserved".into(), encode(snapshot.reserved)),
        ("budget_digest".into(), Value::String(digest)),
    ]);
    // Journal append fsync precedes both in-memory mutations. Any failure
    // to apply after append must poison the owning Daemon and force replay.
    let entry = journal.append(body)?;
    state.apply(&entry)?;
    *ledger = next;
    Ok(())
}

/// Core admission is only reachable with its sealed VerifiedReceipt type.
/// This method cannot promote a model PASS string into evidence, and does not
/// permit delivery merely because an admission was recorded.
pub struct NativeEvidenceRequest<'a> {
    pub session_id: &'a str,
    pub action: &'a PreparedAction,
    pub contract: &'a voder_core::contract::FrozenContract,
    pub observed_workspace: &'a voder_core::types::WorkspaceIdentity,
    pub obligations: &'a mut voder_core::obligation::ObligationRegistry,
    pub ledger: &'a mut voder_core::evidence::EvidenceLedger,
    pub proposal: voder_core::evidence::EvidenceProposal,
    pub verified: &'a voder_core::evidence::VerifiedReceipt,
}

pub fn admit_core_verified_receipt(
    journal: &mut Journal,
    state: &mut State,
    request: NativeEvidenceRequest<'_>,
) -> Result<voder_core::evidence::EvidenceAdmission> {
    let NativeEvidenceRequest {
        session_id,
        action,
        contract,
        observed_workspace,
        obligations,
        ledger,
        proposal,
        verified,
    } = request;
    let intent = action.intent();
    let run_id = intent.run_id.to_string();
    let action_id = intent.action_id.to_string();
    let run = state
        .get(session_id)?
        .runs
        .get(&run_id)
        .ok_or(DaemonError::Invalid("unknown native evidence run"))?;
    if !matches!(
        run.status.as_str(),
        "VGR_BUDGET_SETTLED" | "VGR_EVIDENCE_ADMITTED"
    ) || run.native_action_id.as_deref() != Some(action_id.as_str())
        || run.native_contract_digest.as_deref() != Some(contract.identity().digest.as_str())
        || run.native_observed_workspace_digest.as_deref()
            != Some(observed_workspace.digest.as_str())
        || proposal.run_id != intent.run_id
        || proposal.action_id != intent.action_id
    {
        return Err(DaemonError::Conflict(
            "trusted admission requires settled observation bound to Core action",
        ));
    }
    let parent_receipt = run
        .native_receipt_digest
        .as_ref()
        .ok_or(DaemonError::Invalid("original runtime receipt missing"))?
        .clone();
    // Clone-validate-commit makes ordinary Core admission errors atomic.
    let mut new_ledger = ledger.clone();
    let mut new_obligations = obligations.clone();
    let admission = new_ledger
        .admit_and_apply(
            contract,
            observed_workspace,
            &mut new_obligations,
            proposal,
            verified,
        )
        .map_err(|e| DaemonError::Integrity(format!("Core evidence rejected: {e}")))?;
    let response = Value::object([
        ("run_id".into(), Value::String(run_id.clone())),
        (
            "status".into(),
            Value::String("VGR_EVIDENCE_ADMITTED".into()),
        ),
        ("execution_authorized".into(), Value::Bool(false)),
        ("evidence_admitted".into(), Value::Bool(true)),
        ("delivery_allowed".into(), Value::Bool(false)),
    ]);
    let evidence_id = admission.evidence_id.to_string();
    let body = Value::object([
        ("type".into(), Value::String("vgr.evidence_admitted".into())),
        ("session".into(), Value::String(session_id.into())),
        (
            "key".into(),
            Value::String(format!("sys-vgr-evidence-{run_id}-{evidence_id}")),
        ),
        (
            "fingerprint".into(),
            Value::String(format!("core-verified:{}", admission.receipt_digest)),
        ),
        ("response".into(), response),
        ("run_id".into(), Value::String(run_id)),
        ("action_id".into(), Value::String(action_id)),
        (
            "parent_receipt_digest".into(),
            Value::String(parent_receipt),
        ),
        (
            "contract_digest".into(),
            Value::String(intent.contract.digest.to_string()),
        ),
        (
            "evidence_workspace_digest".into(),
            Value::String(observed_workspace.digest.to_string()),
        ),
        ("evidence_id".into(), Value::String(evidence_id)),
        (
            "receipt_digest".into(),
            Value::String(admission.receipt_digest.to_string()),
        ),
        (
            "content_digest".into(),
            Value::String(admission.content_digest.to_string()),
        ),
        (
            "validator_id".into(),
            Value::String(admission.validator_id.to_string()),
        ),
        (
            "validator_version".into(),
            Value::String(admission.validator_version.to_string()),
        ),
        (
            "effect".into(),
            Value::String(format!("{:?}", admission.effect)),
        ),
    ]);
    let entry = journal.append(body)?;
    state.apply(&entry)?;
    *ledger = new_ledger;
    *obligations = new_obligations;
    Ok(admission)
}

/// Read-only Core verdict: positive result remains EligibleForMetaAudit,
/// deliberately NOT final COMMITTED. A separate independent validator and
/// durable finalization transaction are required before delivery.
pub fn review_delivery(
    state: &State,
    session_id: &str,
    context: voder_core::verdict::DeliveryContext<'_>,
) -> Result<voder_core::verdict::DeliveryAssessment> {
    let run = state
        .get(session_id)?
        .runs
        .get(&context.run.to_string())
        .ok_or(DaemonError::Invalid("unknown delivery run"))?;
    if run.status != "VGR_EVIDENCE_ADMITTED"
        || run.native_runtime_completed != Some(true)
        || run.native_observed_workspace_digest.as_deref()
            != Some(context.workspace.digest.as_str())
        || run.native_contract_digest.as_deref()
            != Some(context.contract.identity().digest.as_str())
    {
        return Err(DaemonError::Conflict(
            "delivery review requires completed observed run and admitted evidence",
        ));
    }
    voder_core::verdict::evaluate_delivery(context)
        .map_err(|e| DaemonError::Integrity(format!("Core delivery gate rejected: {e:?}")))
}

/// Record a rollback ONLY AFTER independently measuring that all files have
/// returned to the prepared workspace digest. It does not perform rollback.
pub fn record_measured_rollback(
    journal: &mut Journal,
    state: &mut State,
    session_id: &str,
    action: &PreparedAction,
    workspace_root: &std::path::Path,
) -> Result<()> {
    let intent = action.intent();
    let run_id = intent.run_id.to_string();
    let run = state
        .get(session_id)?
        .runs
        .get(&run_id)
        .ok_or(DaemonError::Invalid("unknown rollback run"))?;
    if !matches!(
        run.status.as_str(),
        "VGR_BUDGET_SETTLED" | "VGR_EVIDENCE_ADMITTED"
    ) || run.native_action_id.as_deref() != Some(intent.action_id.as_str())
        || run.native_contract_digest.as_deref() != Some(intent.contract.digest.as_str())
    {
        return Err(DaemonError::Conflict("rollback lacks settled VGR action"));
    }
    let current = voder_runtime::receipt::WorkspaceSnapshot::measure(workspace_root)
        .and_then(|s| s.workspace(intent.workspace.revision.clone()))
        .map_err(|e| DaemonError::Integrity(format!("rollback measurement failed: {e}")))?;
    if current != intent.workspace {
        return Err(DaemonError::Conflict(
            "rollback workspace digest differs from frozen preimage",
        ));
    }
    let response = Value::object([
        ("run_id".into(), Value::String(run_id.clone())),
        ("status".into(), Value::String("VGR_ROLLED_BACK".into())),
        ("execution_authorized".into(), Value::Bool(false)),
        ("delivery_allowed".into(), Value::Bool(false)),
    ]);
    let body = Value::object([
        (
            "type".into(),
            Value::String("vgr.action_rolled_back".into()),
        ),
        ("session".into(), Value::String(session_id.into())),
        (
            "key".into(),
            Value::String(format!("sys-vgr-rollback-{run_id}-{}", intent.action_id)),
        ),
        (
            "fingerprint".into(),
            Value::String(format!("rollback-measured:{}", current.digest)),
        ),
        ("response".into(), response),
        ("run_id".into(), Value::String(run_id)),
        (
            "action_id".into(),
            Value::String(intent.action_id.to_string()),
        ),
        (
            "contract_digest".into(),
            Value::String(intent.contract.digest.to_string()),
        ),
        (
            "restored_workspace_digest".into(),
            Value::String(current.digest.to_string()),
        ),
    ]);
    let entry = journal.append(body)?;
    state.apply(&entry)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn budget_wire_dimensions_are_complete_and_ordered() {
        let a = BudgetAmount {
            steps: 1,
            tokens: 2,
            model_calls: 3,
            tool_calls: 4,
            wall_time_ms: 5,
            risk_microunits: 6,
            context_tokens: 7,
            api_cost_microusd: 8,
        };
        let v = encode(a);
        for (i, name) in AXES.iter().enumerate() {
            assert_eq!(v.get(name).and_then(Value::as_u64), Some((i + 1) as u64));
        }
    }
    #[test]
    fn incomplete_or_overbudget_usage_fails_in_core() {
        use voder_core::types::ReservationId;
        let id = ReservationId::try_from("reserve-1").unwrap();
        let upper = BudgetAmount {
            steps: 1,
            tool_calls: 1,
            ..BudgetAmount::default()
        };
        let mut b = BudgetLedger::new(upper);
        b.reserve(id.clone(), upper).unwrap();
        b.mark_dispatched(&id).unwrap();
        assert!(b.settle(&id, BudgetAmount { steps: 2, ..upper }).is_err());
        assert!(b.settle(&id, upper).is_ok());
        assert!(b.settle(&id, upper).is_err());
    }
}
