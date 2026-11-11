//! Frozen-contract validation *metadata* gate.
//!
//! A matching PASS string is not proof: the trusted Runtime must independently
//! attest the validator invocation, exit outcome, artifact bytes and provenance
//! before `evidence::VerifiedReceipt` can be constructed. This pure module only
//! checks whether a reported result COULD qualify for that later admission.

use crate::contract::FrozenContract;
use crate::types::{
    ActionId, ContractIdentity, Digest, EvidenceRelation, ObservationId, RequirementId, RunId,
    SourceId, ToolchainId, ValidatorId, ValidatorVersion, WorkspaceIdentity,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ValidationOutcome {
    Passed,
    Failed,
    Inconclusive,
}

/// Untrusted metadata, not a trusted execution attestation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ValidationReport {
    pub run_id: RunId,
    pub action_id: ActionId,
    pub observation_id: ObservationId,
    pub validator_id: ValidatorId,
    pub validator_version: ValidatorVersion,
    pub requirement_id: RequirementId,
    pub contract: ContractIdentity,
    pub workspace: WorkspaceIdentity,
    pub toolchain: ToolchainId,
    pub source: SourceId,
    pub receipt_digest: Digest,
    pub artifact_digest: Digest,
    pub outcome: ValidationOutcome,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ValidationError {
    WrongRun,
    WrongAction,
    ContractMismatch,
    WorkspaceMismatch,
    ToolchainMismatch,
    UnknownRequirement,
    UnboundValidator,
    WrongValidatorVersion,
    UntrustedSource,
    DuplicateObservation,
    DuplicateReceipt,
    BatchTooLarge,
}

/// Metadata is consistent, but authenticity is still UNVERIFIED.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ValidationMetadata {
    CandidateSupports,
    CandidateRefutes,
    CandidateUnresolved,
}

impl ValidationMetadata {
    pub const fn proposed_relation(self) -> EvidenceRelation {
        match self {
            Self::CandidateSupports => EvidenceRelation::Supports,
            Self::CandidateRefutes => EvidenceRelation::Refutes,
            Self::CandidateUnresolved => EvidenceRelation::Unresolved,
        }
    }
}

/// Does not confer evidence qualification or delivery authority.
pub fn check_validation_metadata(
    contract: &FrozenContract,
    workspace: &WorkspaceIdentity,
    run_id: &RunId,
    action_id: &ActionId,
    report: &ValidationReport,
) -> Result<ValidationMetadata, ValidationError> {
    if &report.run_id != run_id {
        return Err(ValidationError::WrongRun);
    }
    if &report.action_id != action_id {
        return Err(ValidationError::WrongAction);
    }
    if &report.contract != contract.identity() {
        return Err(ValidationError::ContractMismatch);
    }
    if &report.workspace != workspace {
        return Err(ValidationError::WorkspaceMismatch);
    }
    if report.toolchain != contract.identity().toolchain {
        return Err(ValidationError::ToolchainMismatch);
    }
    if !contract
        .requirements()
        .iter()
        .any(|r| r.id == report.requirement_id)
    {
        return Err(ValidationError::UnknownRequirement);
    }
    let validator = contract
        .validators()
        .iter()
        .find(|v| v.id == report.validator_id)
        .ok_or(ValidationError::UnboundValidator)?;
    if validator.version != report.validator_version {
        return Err(ValidationError::WrongValidatorVersion);
    }
    if !validator.requirement_ids.contains(&report.requirement_id) {
        return Err(ValidationError::UnboundValidator);
    }
    if !contract.trusted_sources().contains(&report.source) {
        return Err(ValidationError::UntrustedSource);
    }
    Ok(match report.outcome {
        ValidationOutcome::Passed => ValidationMetadata::CandidateSupports,
        ValidationOutcome::Failed => ValidationMetadata::CandidateRefutes,
        ValidationOutcome::Inconclusive => ValidationMetadata::CandidateUnresolved,
    })
}

/// Checks an entire batch before reporting metadata suitability. A repeated
/// receipt/observation is invalid rather than extra independent evidence.
pub fn check_validation_batch(
    contract: &FrozenContract,
    workspace: &WorkspaceIdentity,
    run: &RunId,
    action: &ActionId,
    reports: &[ValidationReport],
) -> Result<Vec<ValidationMetadata>, ValidationError> {
    use std::collections::BTreeSet;
    if reports.len() > 4096 {
        return Err(ValidationError::BatchTooLarge);
    }
    let mut observations = BTreeSet::new();
    let mut receipts = BTreeSet::new();
    let mut out = Vec::with_capacity(reports.len());
    for r in reports {
        if !observations.insert(&r.observation_id) {
            return Err(ValidationError::DuplicateObservation);
        }
        if !receipts.insert(&r.receipt_digest) {
            return Err(ValidationError::DuplicateReceipt);
        }
        out.push(check_validation_metadata(
            contract, workspace, run, action, r,
        )?);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::contract::{ContractDraft, RepoScope, Requirement, TaskAnchor, ValidatorBinding};
    use crate::types::{Severity, Version};
    fn id<T: TryFrom<&'static str>>(s: &'static str) -> T
    where
        <T as TryFrom<&'static str>>::Error: std::fmt::Debug,
    {
        T::try_from(s).unwrap()
    }
    fn fixture() -> (
        FrozenContract,
        WorkspaceIdentity,
        RunId,
        ActionId,
        ValidationReport,
    ) {
        let identity = ContractIdentity {
            contract_id: id("contract"),
            version: Version::new(1).unwrap(),
            digest: id("contract-digest"),
            toolchain: id("rust-1.99"),
            environment: id("env"),
        };
        let workspace = WorkspaceIdentity {
            revision: id("rev"),
            digest: id("world-digest"),
        };
        let run: RunId = id("run");
        let action: ActionId = id("action");
        let requirement: RequirementId = id("req");
        let validator: ValidatorId = id("cargo-test");
        let source: SourceId = id("trusted");
        let contract = FrozenContract::compile(ContractDraft {
            anchor: TaskAnchor {
                task_id: id("task"),
                identity: "task-ident".into(),
                goal: "compile".into(),
                version: Version::new(1).unwrap(),
                accepted_facts: vec![],
                environment_constraints: vec![],
                evaluation_boundary: vec![],
            },
            identity: identity.clone(),
            requirements: vec![Requirement {
                id: requirement.clone(),
                description: "tests pass".into(),
                hard: true,
                severity: Severity::Critical,
            }],
            validators: vec![ValidatorBinding {
                id: validator.clone(),
                version: id("v1"),
                requirement_ids: vec![requirement.clone()],
            }],
            trusted_sources: vec![source.clone()],
            tools: vec![],
            read_scopes: vec![RepoScope::parse("src").unwrap()],
            write_scopes: vec![],
            forbidden_effects: vec![],
            required_inputs: vec![],
            sampling_rules: vec![],
        })
        .unwrap();
        let report = ValidationReport {
            run_id: run.clone(),
            action_id: action.clone(),
            observation_id: id("obs"),
            validator_id: validator,
            validator_version: id("v1"),
            requirement_id: requirement,
            contract: identity,
            workspace: workspace.clone(),
            toolchain: id("rust-1.99"),
            source,
            receipt_digest: id("receipt"),
            artifact_digest: id("artifact"),
            outcome: ValidationOutcome::Passed,
        };
        (contract, workspace, run, action, report)
    }
    #[test]
    fn pass_is_unverified_candidate_only() {
        let (c, w, r, a, v) = fixture();
        assert_eq!(
            check_validation_metadata(&c, &w, &r, &a, &v),
            Ok(ValidationMetadata::CandidateSupports)
        );
    }
    #[test]
    fn failure_is_not_success() {
        let (c, w, r, a, mut v) = fixture();
        v.outcome = ValidationOutcome::Failed;
        assert_eq!(
            check_validation_metadata(&c, &w, &r, &a, &v),
            Ok(ValidationMetadata::CandidateRefutes)
        );
    }
    #[test]
    fn inconclusive_is_not_success() {
        let (c, w, r, a, mut v) = fixture();
        v.outcome = ValidationOutcome::Inconclusive;
        assert_eq!(
            check_validation_metadata(&c, &w, &r, &a, &v),
            Ok(ValidationMetadata::CandidateUnresolved)
        );
    }
    #[test]
    fn stale_world_rejected() {
        let (c, w, r, a, mut v) = fixture();
        v.workspace.digest = id("stale");
        assert_eq!(
            check_validation_metadata(&c, &w, &r, &a, &v),
            Err(ValidationError::WorkspaceMismatch)
        );
    }
    #[test]
    fn wrong_validator_version_rejected() {
        let (c, w, r, a, mut v) = fixture();
        v.validator_version = id("v2");
        assert_eq!(
            check_validation_metadata(&c, &w, &r, &a, &v),
            Err(ValidationError::WrongValidatorVersion)
        );
    }
    #[test]
    fn mismatched_toolchain_rejected() {
        let (c, w, r, a, mut v) = fixture();
        v.toolchain = id("other");
        assert_eq!(
            check_validation_metadata(&c, &w, &r, &a, &v),
            Err(ValidationError::ToolchainMismatch)
        );
    }
    #[test]
    fn unknown_validator_rejected() {
        let (c, w, r, a, mut v) = fixture();
        v.validator_id = id("fake");
        assert_eq!(
            check_validation_metadata(&c, &w, &r, &a, &v),
            Err(ValidationError::UnboundValidator)
        );
    }
    #[test]
    fn untrusted_source_rejected() {
        let (c, w, r, a, mut v) = fixture();
        v.source = id("model");
        assert_eq!(
            check_validation_metadata(&c, &w, &r, &a, &v),
            Err(ValidationError::UntrustedSource)
        );
    }
    #[test]
    fn changed_action_rejected() {
        let (c, w, r, a, mut v) = fixture();
        v.action_id = id("wrong");
        assert_eq!(
            check_validation_metadata(&c, &w, &r, &a, &v),
            Err(ValidationError::WrongAction)
        );
    }
    #[test]
    fn duplicate_receipt_rejected_batch() {
        let (c, w, r, a, v) = fixture();
        let mut second = v.clone();
        second.observation_id = id("obs2");
        assert_eq!(
            check_validation_batch(&c, &w, &r, &a, &[v, second]),
            Err(ValidationError::DuplicateReceipt)
        );
    }
    #[test]
    fn duplicate_observation_rejected_batch() {
        let (c, w, r, a, v) = fixture();
        let mut second = v.clone();
        second.receipt_digest = id("receipt2");
        assert_eq!(
            check_validation_batch(&c, &w, &r, &a, &[v, second]),
            Err(ValidationError::DuplicateObservation)
        );
    }
    #[test]
    fn empty_batch_is_not_a_pass_claim() {
        let (c, w, r, a, _) = fixture();
        assert!(check_validation_batch(&c, &w, &r, &a, &[])
            .unwrap()
            .is_empty());
    }
}
