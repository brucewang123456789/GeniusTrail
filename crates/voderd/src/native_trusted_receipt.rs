//! Phase 5: the trusted, non-wire validation receipt factory.
//!
//! Trust boundary: only the daemon's pinned ValidatorRegistry and sealed
//! SandboxBackend may execute the validation. A user/model PASS string, JSON
//! observation, or caller-provided receipt digest is NEVER considered proof.
//! A workspace change, missing artifact, missing pin, failed validator, or
//! incomplete budget settlement fails closed BEFORE any Core admission.
use crate::{
    journal::Journal,
    native_finalization::{admit_core_verified_receipt, NativeEvidenceRequest},
    session::State,
    DaemonError, Result,
};
use std::{fs, path::Path};
use voder_core::{
    action_commit::PreparedAction,
    contract::FrozenContract,
    evidence::{
        EvidenceAdmission, EvidenceLedger, EvidenceProposal, TrustedValidatorAttestation,
        VerifiedReceipt,
    },
    obligation::ObligationRegistry,
    types::{Digest, EvidenceId, ObligationId, ObservationId, WorkspaceIdentity},
};
use voder_runtime::{
    artifact_store::{sha256, ArtifactStore},
    receipt::WorkspaceSnapshot,
    sandbox::{SandboxBackend, SandboxPolicy},
    validator::ValidatorRegistry,
};

/// Inputs are held by the trusted daemon and are NOT accepted over stdio.
/// The validator command digest must be pinned in deployment configuration
/// independently of the command registry and any model-produced proposal.
pub struct TrustedReceiptRequest<'a> {
    pub session_id: &'a str,
    pub action: &'a PreparedAction,
    pub contract: &'a FrozenContract,
    pub observed_workspace: &'a WorkspaceIdentity,
    pub obligation_id: ObligationId,
    pub observation_id: ObservationId,
    pub evidence_id: EvidenceId,
    pub obligations: &'a mut ObligationRegistry,
    pub ledger: &'a mut EvidenceLedger,
    pub expected_validator_command_sha256: &'a str,
}

pub struct TrustedReceiptResources<'a, B: SandboxBackend> {
    pub validators: &'a ValidatorRegistry,
    pub backend: &'a B,
    pub policy: &'a SandboxPolicy,
    pub store: &'a ArtifactStore,
    pub capture_dir: &'a Path,
}

fn workspace_matches(root: &Path, expected: &WorkspaceIdentity) -> Result<()> {
    let observed = WorkspaceSnapshot::measure(root)
        .and_then(|snapshot| snapshot.workspace(expected.revision.clone()))
        .map_err(|e| {
            DaemonError::Integrity(format!("trusted receipt workspace measurement: {e}"))
        })?;
    if observed != *expected {
        return Err(DaemonError::Conflict(
            "trusted validator changed or mismatched workspace",
        ));
    }
    Ok(())
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|b| b.is_ascii_hexdigit())
}

/// Execute the actual pinned validator, verify its output bytes and measured
/// workspace, issue a Core-sealed receipt, and atomically journal its admission.
/// The narrow Core minting API trusts this daemon function as the host TCB.
/// It must NEVER be invoked with validator metadata supplied by an LLM.
pub fn execute_verify_and_admit<B: SandboxBackend>(
    journal: &mut Journal,
    state: &mut State,
    request: TrustedReceiptRequest<'_>,
    resources: &TrustedReceiptResources<'_, B>,
) -> Result<EvidenceAdmission> {
    let intent = request.action.intent();
    let session = state.get(request.session_id)?;
    let run = session
        .runs
        .get(intent.run_id.as_str())
        .ok_or(DaemonError::Invalid("unknown trusted receipt run"))?;
    if !matches!(
        run.status.as_str(),
        "VGR_BUDGET_SETTLED" | "VGR_EVIDENCE_ADMITTED"
    ) || run.native_action_id.as_deref() != Some(intent.action_id.as_str())
        || run.native_contract_digest.as_deref()
            != Some(request.contract.identity().digest.as_str())
        || run.native_observed_workspace_digest.as_deref()
            != Some(request.observed_workspace.digest.as_str())
        || run.native_runtime_completed != Some(true)
        || run.native_budget_digest.is_none()
        || intent.contract != *request.contract.identity()
    {
        return Err(DaemonError::Conflict(
            "trusted receipt requires settled matching run",
        ));
    }
    let parent_receipt = run
        .native_receipt_digest
        .clone()
        .ok_or(DaemonError::Invalid("original execution receipt missing"))?;
    let root = resources.policy.worktree.canonicalize()?;
    if root.to_string_lossy().as_ref() != session.project_root.as_str()
        || resources
            .capture_dir
            .canonicalize()
            .ok()
            .is_some_and(|p| p.starts_with(&root))
    {
        return Err(DaemonError::Conflict(
            "untrusted validator workspace or capture directory",
        ));
    }
    if !is_sha256(request.expected_validator_command_sha256) {
        return Err(DaemonError::Invalid(
            "validator command must have independent SHA256 pin",
        ));
    }
    if resources.capture_dir.exists() || resources.capture_dir.is_symlink() {
        return Err(DaemonError::Conflict(
            "validator capture directory must be fresh",
        ));
    }
    // Reject lexical paths inside the worktree even before the capture folder
    // exists; no evidence logs may alter the measured source tree.
    let capture_parent = resources
        .capture_dir
        .parent()
        .ok_or(DaemonError::Invalid("capture parent required"))?
        .canonicalize()?;
    if capture_parent.starts_with(&root) {
        return Err(DaemonError::Conflict(
            "validator capture belongs outside worktree",
        ));
    }
    let spec = request
        .obligations
        .spec(&request.obligation_id)
        .ok_or(DaemonError::Invalid("unknown evidence obligation"))?;
    if spec.admission_rule_id != "voder-validator-exit-zero-v1"
        || spec.expected_observation_type != "validator-process-exit"
    {
        return Err(DaemonError::Conflict(
            "unsupported trusted evidence interpretation rule",
        ));
    }
    let requirement = spec.requirement_id.as_ref().ok_or(DaemonError::Invalid(
        "validator requires mapped requirement",
    ))?;
    let binding = request
        .contract
        .validators()
        .iter()
        .find(|v| v.requirement_ids.contains(requirement))
        .ok_or(DaemonError::Invalid(
            "no frozen validator covers obligation",
        ))?;
    let source = spec.source.clone();
    let claim = spec.claim_id.clone();
    workspace_matches(&root, request.observed_workspace)?;
    fs::create_dir(resources.capture_dir)?;
    let observed = resources
        .validators
        .run(
            request.contract,
            &binding.id,
            resources.policy,
            resources.backend,
            resources.store,
            resources.capture_dir,
        )
        .map_err(|e| DaemonError::Integrity(format!("trusted validator execution: {e}")))?;
    if observed.validator != binding.id
        || observed.version != binding.version
        || !observed.passed
        || observed.executable_digest != request.expected_validator_command_sha256
    {
        return Err(DaemonError::Conflict(
            "trusted validator failed or command pin changed",
        ));
    }
    // ArtifactStore::read rehashes each stored object's real bytes; hashes in
    // an arbitrary textual validator response are never accepted directly.
    let stdout = resources
        .store
        .read(&observed.stdout_digest)
        .map_err(|e| DaemonError::Integrity(format!("stdout artifact verification: {e}")))?;
    let stderr = resources
        .store
        .read(&observed.stderr_digest)
        .map_err(|e| DaemonError::Integrity(format!("stderr artifact verification: {e}")))?;
    if sha256(&stdout).map_err(|e| DaemonError::Integrity(e.to_string()))? != observed.stdout_digest
        || sha256(&stderr).map_err(|e| DaemonError::Integrity(e.to_string()))?
            != observed.stderr_digest
    {
        return Err(DaemonError::Integrity(
            "validator artifact digest mismatch".into(),
        ));
    }
    workspace_matches(&root, request.observed_workspace)?;
    // Domain-separated host transcript, bound to the original execution,
    // frozen contract, measured workspace, typed obligation and current action.
    let canonical =
        format!(
        "voder-vgr5-trusted-validator-v1\0{}\0{}\0{}\0{}\0{}\0{}\0{}\0{}\0{}\0{}\0{}\0{}\0{}\0{}",
        request.session_id, intent.run_id, intent.action_id,
        request.contract.identity().digest, request.observed_workspace.digest,
        parent_receipt, request.obligation_id, request.observation_id,
        binding.id, binding.version, source,
        observed.executable_digest, observed.stdout_digest, observed.stderr_digest,
    );
    let stored_digest = resources
        .store
        .save(canonical.as_bytes())
        .map_err(|e| DaemonError::Integrity(format!("receipt transcript persistence: {e}")))?;
    if resources
        .store
        .read(&stored_digest)
        .map_err(|e| DaemonError::Integrity(format!("receipt transcript verification: {e}")))?
        != canonical.as_bytes()
    {
        return Err(DaemonError::Integrity(
            "receipt transcript not durable and identical".into(),
        ));
    }
    let digest = Digest::parse(stored_digest)
        .map_err(|_| DaemonError::Integrity("invalid host transcript digest".into()))?;
    let content_digest = Digest::parse(observed.stdout_digest)
        .map_err(|_| DaemonError::Integrity("invalid stdout digest".into()))?;
    let proposal = EvidenceProposal {
        evidence_id: request.evidence_id,
        run_id: intent.run_id.clone(),
        action_id: intent.action_id.clone(),
        observation_id: request.observation_id,
        obligation_id: request.obligation_id,
        claim_id: claim,
        expected_receipt_digest: digest.clone(),
    };
    let verified = VerifiedReceipt::from_trusted_validator(
        request.contract,
        request.observed_workspace,
        request.obligations,
        &proposal,
        TrustedValidatorAttestation {
            validator_id: observed.validator,
            validator_version: observed.version,
            source,
            content_digest,
            receipt_digest: digest,
        },
    )
    .map_err(|e| DaemonError::Integrity(format!("Core seal refused: {e}")))?;
    admit_core_verified_receipt(
        journal,
        state,
        NativeEvidenceRequest {
            session_id: request.session_id,
            action: request.action,
            contract: request.contract,
            observed_workspace: request.observed_workspace,
            obligations: request.obligations,
            ledger: request.ledger,
            proposal,
            verified: &verified,
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn malformed_command_pins_never_authorize_trusted_evidence() {
        for pin in ["", "pass", "0"] {
            assert!(!is_sha256(pin));
        }
        assert!(!is_sha256(&"x".repeat(64)));
        assert!(is_sha256(&"a".repeat(64)));
    }
}
