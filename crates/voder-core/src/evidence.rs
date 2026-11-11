//! Evidence admission: untrusted claims are never promoted to proof by assertion.
//!
//! This module is pure Rust. It consumes a receipt already verified by a trusted
//! executor/validator, binds it to the frozen contract and current workspace,
//! admits its factual relation, and applies it to the obligation registry.
//!
//! `VerifiedReceipt` is deliberately sealed: model output and external crate
//! callers cannot construct it. A future trusted `validation.rs` adapter in
//! this crate must verify the ORIGINAL runtime receipt, artifact integrity,
//! validator identity and rule execution before minting one. Until that exists,
//! production admission is intentionally unavailable (fail closed).
//!
//! In-memory atomicity is NOT crash-durable VEP commit. The future daemon must
//! journal the admission + obligation transition as one durable transaction.
//!
//! External callers cannot mint a verified receipt:
//! ```compile_fail
//! use voder_core::evidence::VerifiedReceipt;
//! let forged = VerifiedReceipt::from_verified_runtime(todo!());
//! ```

use crate::contract::{FrozenContract, RepoScope};
use crate::obligation::{AdmittedEvidence, EvidenceEffect, ObligationError, ObligationRegistry};
use crate::retry::RelevantEvidenceFact;
use crate::types::{
    ActionId, ContractIdentity, Digest, EvidenceId, EvidenceRelation, ObligationId, ObservationId,
    RunId, SourceId, ValidatorId, ValidatorVersion, WorkspaceIdentity,
};
use std::collections::{BTreeMap, BTreeSet};

pub const MAX_ADMISSIONS: usize = 65_536;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReceiptCompletion {
    Completed,
    TimedOut,
    Cancelled,
    ExecutionFailed,
    RecoveryRequired,
}

/// A proposal is merely a request to associate evidence with an obligation.
/// It does not carry a trusted result, validator verdict, or authority token.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EvidenceProposal {
    pub evidence_id: EvidenceId,
    pub run_id: RunId,
    pub action_id: ActionId,
    pub observation_id: ObservationId,
    pub obligation_id: ObligationId,
    pub claim_id: String,
    pub expected_receipt_digest: Digest,
}

/// Facts only the trusted execution/validation boundary may populate.
/// The digest fields are *identifiers*, not proofs of cryptographic integrity.
/// The future adapter MUST check real artifact bytes and receipt authenticity.
/// The constructor is intentionally unused before the trusted adapter exists.
#[allow(dead_code)]
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct VerifiedReceiptFields {
    pub run_id: RunId,
    pub action_id: ActionId,
    pub observation_id: ObservationId,
    pub obligation_id: ObligationId,
    pub claim_id: String,
    pub contract: ContractIdentity,
    pub workspace: WorkspaceIdentity,
    pub source: SourceId,
    pub scope: RepoScope,
    pub observation_type: String,
    pub admission_rule_id: String,
    pub validator_id: ValidatorId,
    pub validator_version: ValidatorVersion,
    pub content_digest: Digest,
    pub receipt_digest: Digest,
    pub relation: EvidenceRelation,
    pub completion: ReceiptCompletion,
    /// Whether the trusted adapter actually evaluated this named rule.
    pub rule_verified: bool,
    /// Whether the trusted adapter actually verified receipt + artifact bytes.
    pub integrity_verified: bool,
}

/// A host-owned validator transcript, supplied ONLY by trusted in-process
/// Rust code after an actual sandbox execution and byte-for-byte verification
/// of the captured artifacts. This is NOT a remote JSON/LLM input type.
///
/// Its digest values are evidence identities, not authentication. The daemon
/// is part of the trusted computing base: only its sealed SandboxBackend and
/// frozen ValidatorRegistry may produce this value. Do not expose this type
/// as a wire deserialization target or allow third-party plugin code to call
/// the Core minting function.
#[derive(Debug, Clone)]
pub struct TrustedValidatorAttestation {
    pub validator_id: ValidatorId,
    pub validator_version: ValidatorVersion,
    pub source: SourceId,
    pub content_digest: Digest,
    pub receipt_digest: Digest,
}

/// Intentionally cannot be created with a struct literal or a public `new()`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VerifiedReceipt {
    fields: VerifiedReceiptFields,
}

impl VerifiedReceipt {
    /// Only an in-crate trusted runtime/validator adapter may invoke this.
    /// NEVER pass model-provided JSON directly to this constructor.
    #[allow(dead_code)]
    pub(crate) fn from_verified_runtime(fields: VerifiedReceiptFields) -> Self {
        Self { fields }
    }

    /// Narrow trusted-daemon bridge. The *caller* must be trusted host code
    /// that actually ran the pinned validator and re-read artifact bytes.
    /// This method does not authenticate the host; Core cannot do that from
    /// mere digest strings. It binds the verified host transcript to the
    /// immutable Core contract and obligation, and never accepts a PASS flag.
    /// It is intentionally not exposed by the stdio protocol.
    pub fn from_trusted_validator(
        contract: &FrozenContract,
        workspace: &WorkspaceIdentity,
        registry: &ObligationRegistry,
        proposal: &EvidenceProposal,
        attestation: TrustedValidatorAttestation,
    ) -> Result<Self, EvidenceError> {
        let spec = registry
            .spec(&proposal.obligation_id)
            .ok_or(EvidenceError::UnknownObligation)?;
        registry
            .status(contract, workspace, &proposal.obligation_id)
            .map_err(EvidenceError::Registry)?;
        if proposal.expected_receipt_digest != attestation.receipt_digest {
            return Err(EvidenceError::ReceiptDigestMismatch);
        }
        if spec.claim_id != proposal.claim_id {
            return Err(EvidenceError::ClaimMismatch);
        }
        if spec.source != attestation.source
            || !contract.trusted_sources().contains(&attestation.source)
        {
            return Err(EvidenceError::UntrustedSource);
        }
        if !contract
            .read_scopes()
            .iter()
            .any(|s| s.contains(&spec.scope))
        {
            return Err(EvidenceError::OutOfScope);
        }
        // Only the narrowly defined exit-zero rule can currently be issued.
        // Arbitrary admission-rule strings require a separately audited
        // evidence interpreter rather than treating process success as proof.
        if spec.admission_rule_id != "voder-validator-exit-zero-v1"
            || spec.expected_observation_type != "validator-process-exit"
        {
            return Err(EvidenceError::AdmissionRuleMismatch);
        }
        if !contract.validators().iter().any(|v| {
            v.id == attestation.validator_id
                && v.version == attestation.validator_version
                && spec
                    .requirement_id
                    .as_ref()
                    .is_some_and(|r| v.requirement_ids.contains(r))
        }) {
            return Err(EvidenceError::ValidatorMismatch);
        }
        Ok(Self::from_verified_runtime(VerifiedReceiptFields {
            run_id: proposal.run_id.clone(),
            action_id: proposal.action_id.clone(),
            observation_id: proposal.observation_id.clone(),
            obligation_id: proposal.obligation_id.clone(),
            claim_id: proposal.claim_id.clone(),
            contract: contract.identity().clone(),
            workspace: workspace.clone(),
            source: attestation.source,
            scope: spec.scope.clone(),
            observation_type: spec.expected_observation_type.clone(),
            admission_rule_id: spec.admission_rule_id.clone(),
            validator_id: attestation.validator_id,
            validator_version: attestation.validator_version,
            content_digest: attestation.content_digest,
            receipt_digest: attestation.receipt_digest,
            relation: EvidenceRelation::Supports,
            completion: ReceiptCompletion::Completed,
            rule_verified: true,
            integrity_verified: true,
        }))
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EvidenceError {
    UnknownObligation,
    ContractMismatch,
    WorkspaceMismatch,
    ObservationMismatch,
    RunMismatch,
    ActionMismatch,
    ObligationMismatch,
    ClaimMismatch,
    ReceiptDigestMismatch,
    UntrustedSource,
    OutOfScope,
    ObservationTypeMismatch,
    AdmissionRuleMismatch,
    ValidatorMismatch,
    UnverifiedIntegrity,
    UnverifiedRule,
    IncompleteReceipt,
    NonAdmissibleRelation,
    DuplicateEvidenceId,
    DuplicateObservation,
    DuplicateReceipt,
    LedgerFull,
    Registry(ObligationError),
}

impl std::fmt::Display for EvidenceError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self:?}")
    }
}
impl std::error::Error for EvidenceError {}

/// Admitted *fact*, not a claim that the task is complete.
/// `effect` is computed by the obligation registry, not by the model.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EvidenceAdmission {
    pub evidence_id: EvidenceId,
    pub run_id: RunId,
    pub action_id: ActionId,
    pub observation_id: ObservationId,
    pub receipt_digest: Digest,
    pub content_digest: Digest,
    pub contract: ContractIdentity,
    pub workspace: WorkspaceIdentity,
    pub source: SourceId,
    pub validator_id: ValidatorId,
    pub validator_version: ValidatorVersion,
    pub scope: RepoScope,
    pub admission_rule_id: String,
    pub relation: EvidenceRelation,
    pub effect: EvidenceEffect,
    admitted: AdmittedEvidence,
    fact: RelevantEvidenceFact,
}

impl EvidenceAdmission {
    pub fn admitted_evidence(&self) -> &AdmittedEvidence {
        &self.admitted
    }
    pub fn relevant_fact(&self) -> &RelevantEvidenceFact {
        &self.fact
    }
}

/// Append-only in-memory reference ledger. Its public surface does not allow
/// inserting arbitrary `AdmittedEvidence` or resetting replay-protection sets.
#[derive(Debug, Clone, Default)]
pub struct EvidenceLedger {
    records: BTreeMap<EvidenceId, EvidenceAdmission>,
    observations: BTreeSet<ObservationId>,
    receipts: BTreeSet<Digest>,
}

impl EvidenceLedger {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn len(&self) -> usize {
        self.records.len()
    }

    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }

    pub fn get(&self, id: &EvidenceId) -> Option<&EvidenceAdmission> {
        self.records.get(id)
    }

    /// Use ONLY this projection (not fabricated model facts) as input to retry.rs.
    pub fn relevant_facts(&self) -> Vec<RelevantEvidenceFact> {
        self.records.values().map(|r| r.fact.clone()).collect()
    }

    /// Admit and apply together; validation failures leave BOTH ledgers intact.
    /// This is atomic with respect to ordinary Result errors, not crashes.
    pub fn admit_and_apply(
        &mut self,
        contract: &FrozenContract,
        workspace: &WorkspaceIdentity,
        registry: &mut ObligationRegistry,
        proposal: EvidenceProposal,
        receipt: &VerifiedReceipt,
    ) -> Result<EvidenceAdmission, EvidenceError> {
        if self.records.len() >= MAX_ADMISSIONS {
            return Err(EvidenceError::LedgerFull);
        }
        if self.records.contains_key(&proposal.evidence_id) {
            return Err(EvidenceError::DuplicateEvidenceId);
        }
        if self.observations.contains(&proposal.observation_id) {
            return Err(EvidenceError::DuplicateObservation);
        }
        if self.receipts.contains(&proposal.expected_receipt_digest) {
            return Err(EvidenceError::DuplicateReceipt);
        }
        // The registry has to be compiled from the same frozen contract.
        let spec = registry
            .spec(&proposal.obligation_id)
            .ok_or(EvidenceError::UnknownObligation)?;
        registry
            .status(contract, workspace, &proposal.obligation_id)
            .map_err(EvidenceError::Registry)?;

        let r = &receipt.fields;
        if r.contract != *contract.identity() {
            return Err(EvidenceError::ContractMismatch);
        }
        if r.workspace != *workspace {
            return Err(EvidenceError::WorkspaceMismatch);
        }
        if r.observation_id != proposal.observation_id {
            return Err(EvidenceError::ObservationMismatch);
        }
        if r.run_id != proposal.run_id {
            return Err(EvidenceError::RunMismatch);
        }
        if r.action_id != proposal.action_id {
            return Err(EvidenceError::ActionMismatch);
        }
        if r.obligation_id != proposal.obligation_id {
            return Err(EvidenceError::ObligationMismatch);
        }
        if r.claim_id != proposal.claim_id || r.claim_id != spec.claim_id {
            return Err(EvidenceError::ClaimMismatch);
        }
        if r.receipt_digest != proposal.expected_receipt_digest {
            return Err(EvidenceError::ReceiptDigestMismatch);
        }
        if !r.integrity_verified {
            return Err(EvidenceError::UnverifiedIntegrity);
        }
        if r.completion != ReceiptCompletion::Completed {
            return Err(EvidenceError::IncompleteReceipt);
        }
        if !contract.trusted_sources().contains(&r.source) || r.source != spec.source {
            return Err(EvidenceError::UntrustedSource);
        }
        if !spec.scope.contains(&r.scope)
            || !contract
                .read_scopes()
                .iter()
                .any(|scope| scope.contains(&r.scope))
        {
            return Err(EvidenceError::OutOfScope);
        }
        if r.observation_type != spec.expected_observation_type {
            return Err(EvidenceError::ObservationTypeMismatch);
        }
        if r.admission_rule_id != spec.admission_rule_id {
            return Err(EvidenceError::AdmissionRuleMismatch);
        }
        if !r.rule_verified {
            return Err(EvidenceError::UnverifiedRule);
        }
        // A validator name without its frozen version AND requirement coverage
        // is not evidence for the protected acceptance contract.
        let validator_matches = contract.validators().iter().any(|v| {
            v.id == r.validator_id
                && v.version == r.validator_version
                && spec
                    .requirement_id
                    .as_ref()
                    .is_none_or(|req| v.requirement_ids.contains(req))
        });
        if !validator_matches {
            return Err(EvidenceError::ValidatorMismatch);
        }
        if r.relation == EvidenceRelation::OutOfScope {
            return Err(EvidenceError::NonAdmissibleRelation);
        }
        // A completed validator may REFUTE the claim. Such evidence is useful
        // for the next repair decision but cannot close a SUPPORTS obligation.
        let admitted = AdmittedEvidence::from_trusted_admission(
            proposal.evidence_id.clone(),
            proposal.obligation_id.clone(),
            r.claim_id.clone(),
            r.relation,
            r.scope.clone(),
            r.observation_type.clone(),
            r.admission_rule_id.clone(),
            r.source.clone(),
            r.contract.digest.clone(),
            r.workspace.digest.clone(),
            r.contract.environment.clone(),
        );
        let fact = RelevantEvidenceFact {
            obligation_id: proposal.obligation_id.clone(),
            claim_id: r.claim_id.clone(),
            relation: r.relation,
            content_digest: r.content_digest.clone(),
            contract_digest: r.contract.digest.clone(),
            workspace_digest: r.workspace.digest.clone(),
            environment: r.contract.environment.clone(),
        };
        // Registry mutation is last possible fallible operation. No evidence
        // ledger indices are written unless the registry accepted this record.
        let effect = registry
            .apply_admitted_evidence(contract, workspace, &admitted)
            .map_err(EvidenceError::Registry)?;
        let record = EvidenceAdmission {
            evidence_id: proposal.evidence_id.clone(),
            run_id: r.run_id.clone(),
            action_id: r.action_id.clone(),
            observation_id: proposal.observation_id.clone(),
            receipt_digest: r.receipt_digest.clone(),
            content_digest: r.content_digest.clone(),
            contract: r.contract.clone(),
            workspace: r.workspace.clone(),
            source: r.source.clone(),
            validator_id: r.validator_id.clone(),
            validator_version: r.validator_version.clone(),
            scope: r.scope.clone(),
            admission_rule_id: r.admission_rule_id.clone(),
            relation: r.relation,
            effect,
            admitted,
            fact,
        };
        self.observations.insert(proposal.observation_id);
        self.receipts.insert(r.receipt_digest.clone());
        self.records.insert(proposal.evidence_id, record.clone());
        Ok(record)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::contract::{ContractDraft, Requirement, TaskAnchor, ValidatorBinding};
    use crate::obligation::ObligationSpec;
    use crate::types::{
        ContractId, EnvironmentId, ObligationStatus, RequirementId, RevisionId, Severity, TaskId,
        ToolchainId, ValidatorId, Version,
    };

    fn id<T: TryFrom<&'static str>>(s: &'static str) -> T
    where
        <T as TryFrom<&'static str>>::Error: std::fmt::Debug,
    {
        T::try_from(s).unwrap()
    }

    fn fixture() -> (
        FrozenContract,
        WorkspaceIdentity,
        ObligationRegistry,
        EvidenceProposal,
        VerifiedReceiptFields,
    ) {
        let contract = FrozenContract::compile(ContractDraft {
            anchor: TaskAnchor {
                task_id: id::<TaskId>("task"),
                identity: "repair-task".into(),
                goal: "Test the patch".into(),
                version: Version::new(1).unwrap(),
                accepted_facts: vec![],
                environment_constraints: vec![],
                evaluation_boundary: vec![],
            },
            identity: ContractIdentity {
                contract_id: id::<ContractId>("contract"),
                version: Version::new(1).unwrap(),
                digest: id("contract-hash"),
                toolchain: id::<ToolchainId>("rust"),
                environment: id::<EnvironmentId>("env-a"),
            },
            requirements: vec![Requirement {
                id: id::<RequirementId>("req"),
                description: "Tests must pass".into(),
                hard: true,
                severity: Severity::Critical,
            }],
            validators: vec![ValidatorBinding {
                id: id::<ValidatorId>("validator"),
                version: id("v1"),
                requirement_ids: vec![id("req")],
            }],
            trusted_sources: vec![id("runner")],
            tools: vec![],
            read_scopes: vec![RepoScope::parse("src").unwrap()],
            write_scopes: vec![],
            forbidden_effects: vec![],
            required_inputs: vec![],
            sampling_rules: vec![],
        })
        .unwrap();
        let workspace = WorkspaceIdentity {
            revision: id::<RevisionId>("revision-1"),
            digest: id("workspace-a"),
        };
        let spec = ObligationSpec {
            id: id("obligation"),
            claim_id: "patch-tests-pass".into(),
            requirement_id: Some(id("req")),
            threatened_judgement: "delivery".into(),
            expected_observation_type: "cargo-test".into(),
            admission_rule_id: "rule-test-v1".into(),
            scope: RepoScope::parse("src").unwrap(),
            priority: 10,
            stop_condition: "tests pass".into(),
            source: id("runner"),
            critical: true,
            closure_relation: EvidenceRelation::Supports,
            dependencies: vec![],
        };
        let registry = ObligationRegistry::compile(&contract, vec![spec]).unwrap();
        let proposed = EvidenceProposal {
            evidence_id: id("e-1"),
            run_id: id("run-1"),
            action_id: id("action-1"),
            observation_id: id("obs-1"),
            obligation_id: id("obligation"),
            claim_id: "patch-tests-pass".into(),
            expected_receipt_digest: id("receipt-1"),
        };
        let receipt = VerifiedReceiptFields {
            run_id: proposed.run_id.clone(),
            action_id: proposed.action_id.clone(),
            observation_id: proposed.observation_id.clone(),
            obligation_id: proposed.obligation_id.clone(),
            claim_id: proposed.claim_id.clone(),
            contract: contract.identity().clone(),
            workspace: workspace.clone(),
            source: id("runner"),
            scope: RepoScope::parse("src/lib.rs").unwrap(),
            observation_type: "cargo-test".into(),
            admission_rule_id: "rule-test-v1".into(),
            validator_id: id("validator"),
            validator_version: id("v1"),
            content_digest: id("contents-pass"),
            receipt_digest: proposed.expected_receipt_digest.clone(),
            relation: EvidenceRelation::Supports,
            completion: ReceiptCompletion::Completed,
            rule_verified: true,
            integrity_verified: true,
        };
        (contract, workspace, registry, proposed, receipt)
    }

    fn submit(
        ledger: &mut EvidenceLedger,
        contract: &FrozenContract,
        workspace: &WorkspaceIdentity,
        registry: &mut ObligationRegistry,
        proposal: EvidenceProposal,
        fields: VerifiedReceiptFields,
    ) -> Result<EvidenceAdmission, EvidenceError> {
        ledger.admit_and_apply(
            contract,
            workspace,
            registry,
            proposal,
            &VerifiedReceipt::from_verified_runtime(fields),
        )
    }

    fn rejected_with(
        mutation: impl FnOnce(&mut EvidenceProposal, &mut VerifiedReceiptFields),
        expected: EvidenceError,
    ) {
        let (contract, workspace, mut registry, mut proposal, mut fields) = fixture();
        let mut ledger = EvidenceLedger::new();
        mutation(&mut proposal, &mut fields);
        assert_eq!(
            submit(
                &mut ledger,
                &contract,
                &workspace,
                &mut registry,
                proposal,
                fields
            ),
            Err(expected)
        );
        assert!(ledger.is_empty(), "rejection must not commit evidence");
        assert!(!registry.all_critical_closed(&contract, &workspace).unwrap());
    }

    #[test]
    fn verified_support_closes_hard_obligation() {
        let (contract, workspace, mut registry, proposal, fields) = fixture();
        let mut ledger = EvidenceLedger::new();
        let record = submit(
            &mut ledger,
            &contract,
            &workspace,
            &mut registry,
            proposal,
            fields,
        )
        .unwrap();
        assert_eq!(record.effect, EvidenceEffect::Closed);
        assert_eq!(record.relation, EvidenceRelation::Supports);
        assert_eq!(record.run_id, id::<RunId>("run-1"));
        assert_eq!(record.action_id, id::<ActionId>("action-1"));
        assert_eq!(record.validator_version, id::<ValidatorVersion>("v1"));
        assert_eq!(record.content_digest, id::<Digest>("contents-pass"));
        assert_eq!(ledger.len(), 1);
        assert_eq!(ledger.relevant_facts().len(), 1);
        assert!(registry.all_critical_closed(&contract, &workspace).unwrap());
    }

    #[test]
    fn completed_failed_test_is_admitted_but_does_not_close() {
        let (contract, workspace, mut registry, proposal, mut fields) = fixture();
        fields.relation = EvidenceRelation::Refutes;
        fields.content_digest = id("contents-failed");
        let mut ledger = EvidenceLedger::new();
        let record = submit(
            &mut ledger,
            &contract,
            &workspace,
            &mut registry,
            proposal,
            fields,
        )
        .unwrap();
        assert_eq!(record.effect, EvidenceEffect::DidNotSatisfyClosure);
        assert_eq!(record.relation, EvidenceRelation::Refutes);
        assert_eq!(
            registry
                .status(&contract, &workspace, &id("obligation"))
                .unwrap(),
            ObligationStatus::Open
        );
        assert_eq!(
            ledger.relevant_facts()[0].content_digest,
            id::<Digest>("contents-failed")
        );
    }

    #[test]
    fn unknown_relation_does_not_close() {
        let (contract, workspace, mut registry, proposal, mut fields) = fixture();
        fields.relation = EvidenceRelation::Unresolved;
        let record = submit(
            &mut EvidenceLedger::new(),
            &contract,
            &workspace,
            &mut registry,
            proposal,
            fields,
        )
        .unwrap();
        assert_eq!(record.effect, EvidenceEffect::DidNotSatisfyClosure);
    }

    #[test]
    fn out_of_scope_relation_is_not_admitted() {
        rejected_with(
            |_, f| f.relation = EvidenceRelation::OutOfScope,
            EvidenceError::NonAdmissibleRelation,
        );
    }

    #[test]
    fn forged_proposal_cannot_replace_receipt_observation() {
        rejected_with(
            |p, _| p.observation_id = id("other-observation"),
            EvidenceError::ObservationMismatch,
        );
    }

    #[test]
    fn forged_proposal_cannot_retarget_obligation() {
        rejected_with(
            |p, _| p.obligation_id = id("other-obligation"),
            EvidenceError::UnknownObligation,
        );
    }

    #[test]
    fn receipt_from_another_run_is_rejected() {
        rejected_with(|p, _| p.run_id = id("run-2"), EvidenceError::RunMismatch);
    }

    #[test]
    fn receipt_from_another_action_is_rejected() {
        rejected_with(
            |p, _| p.action_id = id("action-2"),
            EvidenceError::ActionMismatch,
        );
    }

    #[test]
    fn forged_proposal_cannot_retarget_claim() {
        rejected_with(
            |p, _| p.claim_id = "other-claim".into(),
            EvidenceError::ClaimMismatch,
        );
    }

    #[test]
    fn forged_proposal_cannot_retarget_receipt() {
        rejected_with(
            |p, _| p.expected_receipt_digest = id("unknown-receipt"),
            EvidenceError::ReceiptDigestMismatch,
        );
    }

    #[test]
    fn wrong_frozen_contract_is_rejected() {
        rejected_with(
            |_, f| f.contract.digest = id("older-contract"),
            EvidenceError::ContractMismatch,
        );
    }

    #[test]
    fn stale_workspace_bytes_are_rejected() {
        rejected_with(
            |_, f| f.workspace.digest = id("older-tree"),
            EvidenceError::WorkspaceMismatch,
        );
    }

    #[test]
    fn stale_workspace_revision_is_rejected() {
        rejected_with(
            |_, f| f.workspace.revision = id("older-revision"),
            EvidenceError::WorkspaceMismatch,
        );
    }

    #[test]
    fn trusted_source_must_match_obligation() {
        rejected_with(
            |_, f| f.source = id("model"),
            EvidenceError::UntrustedSource,
        );
    }

    #[test]
    fn directory_scope_must_be_contained() {
        rejected_with(
            |_, f| f.scope = RepoScope::parse("other").unwrap(),
            EvidenceError::OutOfScope,
        );
    }

    #[test]
    fn observation_type_must_match() {
        rejected_with(
            |_, f| f.observation_type = "model-guess".into(),
            EvidenceError::ObservationTypeMismatch,
        );
    }

    #[test]
    fn policy_id_must_match() {
        rejected_with(
            |_, f| f.admission_rule_id = "weaker-rule".into(),
            EvidenceError::AdmissionRuleMismatch,
        );
    }

    #[test]
    fn policy_execution_must_be_verified() {
        rejected_with(
            |_, f| f.rule_verified = false,
            EvidenceError::UnverifiedRule,
        );
    }

    #[test]
    fn receipt_integrity_must_be_verified() {
        rejected_with(
            |_, f| f.integrity_verified = false,
            EvidenceError::UnverifiedIntegrity,
        );
    }

    #[test]
    fn pinned_validator_version_must_match() {
        rejected_with(
            |_, f| f.validator_version = id("v0"),
            EvidenceError::ValidatorMismatch,
        );
    }

    #[test]
    fn validator_identity_must_match() {
        rejected_with(
            |_, f| f.validator_id = id("other-validator"),
            EvidenceError::ValidatorMismatch,
        );
    }

    #[test]
    fn incomplete_execution_results_cannot_enter_ledger() {
        for completion in [
            ReceiptCompletion::TimedOut,
            ReceiptCompletion::Cancelled,
            ReceiptCompletion::ExecutionFailed,
            ReceiptCompletion::RecoveryRequired,
        ] {
            rejected_with(
                |_, f| f.completion = completion,
                EvidenceError::IncompleteReceipt,
            );
        }
    }

    #[test]
    fn replayed_evidence_id_is_rejected_without_registry_change() {
        let (contract, workspace, mut registry, proposal, fields) = fixture();
        let mut ledger = EvidenceLedger::new();
        submit(
            &mut ledger,
            &contract,
            &workspace,
            &mut registry,
            proposal.clone(),
            fields.clone(),
        )
        .unwrap();
        let error = submit(
            &mut ledger,
            &contract,
            &workspace,
            &mut registry,
            proposal,
            fields,
        )
        .unwrap_err();
        assert_eq!(error, EvidenceError::DuplicateEvidenceId);
        assert_eq!(ledger.len(), 1);
    }

    #[test]
    fn replayed_observation_with_new_id_is_rejected() {
        let (contract, workspace, mut registry, proposal, fields) = fixture();
        let mut ledger = EvidenceLedger::new();
        submit(
            &mut ledger,
            &contract,
            &workspace,
            &mut registry,
            proposal.clone(),
            fields.clone(),
        )
        .unwrap();
        let mut other = proposal;
        other.evidence_id = id("e-2");
        assert_eq!(
            submit(
                &mut ledger,
                &contract,
                &workspace,
                &mut registry,
                other,
                fields
            ),
            Err(EvidenceError::DuplicateObservation)
        );
    }

    #[test]
    fn replayed_receipt_with_new_observation_id_is_rejected() {
        let (contract, workspace, mut registry, proposal, fields) = fixture();
        let mut ledger = EvidenceLedger::new();
        submit(
            &mut ledger,
            &contract,
            &workspace,
            &mut registry,
            proposal.clone(),
            fields.clone(),
        )
        .unwrap();
        let mut other = proposal;
        other.evidence_id = id("e-2");
        other.observation_id = id("obs-2");
        let mut other_receipt = fields;
        other_receipt.observation_id = other.observation_id.clone();
        assert_eq!(
            submit(
                &mut ledger,
                &contract,
                &workspace,
                &mut registry,
                other,
                other_receipt
            ),
            Err(EvidenceError::DuplicateReceipt)
        );
    }

    #[test]
    fn invalid_registry_application_does_not_write_evidence_ledger() {
        let (contract, workspace, mut registry, proposal, fields) = fixture();
        let mut ledger = EvidenceLedger::new();
        let mut changed = contract.identity().clone();
        changed.digest = id("other-contract");
        let changed_contract = FrozenContract::compile(ContractDraft {
            anchor: contract.anchor().clone(),
            identity: changed,
            requirements: contract.requirements().to_vec(),
            validators: contract.validators().to_vec(),
            trusted_sources: contract.trusted_sources().to_vec(),
            tools: contract.tools().to_vec(),
            read_scopes: contract.read_scopes().to_vec(),
            write_scopes: contract.write_scopes().to_vec(),
            forbidden_effects: contract.forbidden_effects().to_vec(),
            required_inputs: contract.required_inputs().to_vec(),
            sampling_rules: contract.sampling_rules().to_vec(),
        })
        .unwrap();
        assert_eq!(
            submit(
                &mut ledger,
                &changed_contract,
                &workspace,
                &mut registry,
                proposal,
                fields
            ),
            Err(EvidenceError::Registry(ObligationError::ContractMismatch))
        );
        assert!(ledger.is_empty());
    }

    #[test]
    fn current_workspace_closure_does_not_survive_workspace_change() {
        let (contract, workspace, mut registry, proposal, fields) = fixture();
        let mut ledger = EvidenceLedger::new();
        submit(
            &mut ledger,
            &contract,
            &workspace,
            &mut registry,
            proposal,
            fields,
        )
        .unwrap();
        let changed = WorkspaceIdentity {
            revision: id("revision-2"),
            digest: id("workspace-b"),
        };
        assert!(!registry.all_critical_closed(&contract, &changed).unwrap());
    }

    #[test]
    fn preserved_semantic_digest_supports_retry_gate() {
        let (contract, workspace, mut registry, proposal, mut fields) = fixture();
        fields.relation = EvidenceRelation::Refutes;
        fields.content_digest = id("test-failed-with-error-1");
        let mut ledger = EvidenceLedger::new();
        submit(
            &mut ledger,
            &contract,
            &workspace,
            &mut registry,
            proposal,
            fields,
        )
        .unwrap();
        let facts = ledger.relevant_facts();
        let key = crate::retry::relevant_evidence_key(
            &facts,
            Some(&id("obligation")),
            contract.identity(),
            &workspace,
        );
        assert_eq!(key.triples().len(), 1);
        assert_eq!(key.triples()[0].2, id::<Digest>("test-failed-with-error-1"));
    }
    #[test]
    fn trusted_validator_factory_only_seals_bound_exit_zero_rule() {
        let (contract, workspace, registry, proposal, _) = fixture();
        let mut spec = registry.spec(&proposal.obligation_id).unwrap().clone();
        let make = || TrustedValidatorAttestation {
            validator_id: id("validator"),
            validator_version: id("v1"),
            source: id("runner"),
            content_digest: id("sha-of-stdout"),
            receipt_digest: proposal.expected_receipt_digest.clone(),
        };
        assert_eq!(
            VerifiedReceipt::from_trusted_validator(
                &contract,
                &workspace,
                &registry,
                &proposal,
                make(),
            )
            .unwrap_err(),
            EvidenceError::AdmissionRuleMismatch,
        );
        spec.admission_rule_id = "voder-validator-exit-zero-v1".into();
        spec.expected_observation_type = "validator-process-exit".into();
        let mut obligations = ObligationRegistry::compile(&contract, vec![spec]).unwrap();
        let mut ledger = EvidenceLedger::new();
        let wrong = TrustedValidatorAttestation {
            validator_id: id("validator"),
            validator_version: id("other"),
            source: id("runner"),
            content_digest: id("sha-of-stdout"),
            receipt_digest: proposal.expected_receipt_digest.clone(),
        };
        assert_eq!(
            VerifiedReceipt::from_trusted_validator(
                &contract,
                &workspace,
                &obligations,
                &proposal,
                wrong,
            )
            .unwrap_err(),
            EvidenceError::ValidatorMismatch,
        );
        let receipt = VerifiedReceipt::from_trusted_validator(
            &contract,
            &workspace,
            &obligations,
            &proposal,
            make(),
        )
        .unwrap();
        let admitted = ledger
            .admit_and_apply(&contract, &workspace, &mut obligations, proposal, &receipt)
            .unwrap();
        assert_eq!(admitted.relation, EvidenceRelation::Supports);
        assert_eq!(ledger.len(), 1);
    }
}
