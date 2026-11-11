//! Evidence-governed delivery preflight. Qualification is not final delivery:
//! a fresh independent meta-audit and durable finalization belong to `voderd`.
//! No model/CLI/plugin boolean can mint a completion certificate here.
use crate::contract::FrozenContract;
use crate::evidence::EvidenceLedger;
use crate::obligation::EvidenceEffect;
use crate::obligation::{ObligationError, ObligationRegistry};
use crate::types::{
    EvidenceId, EvidenceRelation, ObligationStatus, RequirementId, RunId, RuntimeStatus,
    TransactionPhase, WorkspaceIdentity,
};
use std::collections::BTreeSet;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeliveryBlocker {
    UnsafeRuntime,
    PendingTransaction,
    RecoveryRequired,
    HardObligationOpen,
    HardEvidenceMissing,
    InvalidValidatorBinding,
}
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DeliveryAssessment {
    /// Not permission to mark COMPLETED. Daemon must still run final meta-audit.
    EligibleForMetaAudit,
    Blocked(DeliveryBlocker),
    RollbackRequired,
}
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VerdictError {
    Obligation(ObligationError),
}

/// Input is a registry and append-only admitted ledger, not validation metadata
/// or a list of model-declared PASS results. Hard requirements must each have
/// actual matching, current, trusted, SUPPORTS admissions from pinned validators.
/// The set of accepted EvidenceIds comes from the ledger itself.
/// Explicitly named, borrowed admission context. Keeping the eight independent
/// identities and safety gates visible prevents positional-argument confusion.
/// This is a read-only preflight input, not a delivery authorization.
pub struct DeliveryContext<'a> {
    pub contract: &'a FrozenContract,
    pub workspace: &'a WorkspaceIdentity,
    pub run: &'a RunId,
    pub runtime: RuntimeStatus,
    pub pending_transaction: Option<TransactionPhase>,
    pub obligations: &'a ObligationRegistry,
    pub admitted: &'a EvidenceLedger,
    pub evidence_for_hard_requirements: &'a [(RequirementId, EvidenceId)],
}

pub fn evaluate_delivery(context: DeliveryContext<'_>) -> Result<DeliveryAssessment, VerdictError> {
    let DeliveryContext {
        contract,
        workspace,
        run,
        runtime,
        pending_transaction,
        obligations,
        admitted,
        evidence_for_hard_requirements,
    } = context;
    if runtime == RuntimeStatus::RecoveryRequired
        || pending_transaction == Some(TransactionPhase::RecoveryRequired)
    {
        return Ok(DeliveryAssessment::Blocked(
            DeliveryBlocker::RecoveryRequired,
        ));
    }
    if runtime == RuntimeStatus::RolledBack {
        return Ok(DeliveryAssessment::RollbackRequired);
    }
    if !matches!(runtime, RuntimeStatus::Active | RuntimeStatus::Completed) {
        return Ok(DeliveryAssessment::Blocked(DeliveryBlocker::UnsafeRuntime));
    }
    if matches!(
        pending_transaction,
        Some(TransactionPhase::Prepared | TransactionPhase::Executing | TransactionPhase::Observed)
    ) {
        return Ok(DeliveryAssessment::Blocked(
            DeliveryBlocker::PendingTransaction,
        ));
    }
    if !obligations
        .all_critical_closed(contract, workspace)
        .map_err(VerdictError::Obligation)?
    {
        return Ok(DeliveryAssessment::Blocked(
            DeliveryBlocker::HardObligationOpen,
        ));
    }
    let hard: Vec<_> = contract.requirements().iter().filter(|r| r.hard).collect();
    if hard.is_empty() || admitted.is_empty() {
        return Ok(DeliveryAssessment::Blocked(
            DeliveryBlocker::HardEvidenceMissing,
        ));
    }
    let mut used = BTreeSet::new();
    for (req, id) in evidence_for_hard_requirements {
        if !used.insert((req.clone(), id.clone())) {
            return Ok(DeliveryAssessment::Blocked(
                DeliveryBlocker::HardEvidenceMissing,
            ));
        }
    }
    for requirement in hard {
        let bindings: Vec<_> = contract
            .validators()
            .iter()
            .filter(|v| v.requirement_ids.contains(&requirement.id))
            .collect();
        if bindings.is_empty() {
            return Ok(DeliveryAssessment::Blocked(
                DeliveryBlocker::InvalidValidatorBinding,
            ));
        }
        // Every validator the frozen contract binds to this hard requirement
        // needs one fresh admitted supportive record; evidence IDs alone do not pass.
        for binding in bindings {
            let mut satisfied = false;
            for (req, eid) in evidence_for_hard_requirements {
                if req != &requirement.id {
                    continue;
                }
                let Some(a) = admitted.get(eid) else {
                    continue;
                };
                if a.run_id != *run
                    || a.contract != *contract.identity()
                    || a.workspace != *workspace
                    || a.relation != EvidenceRelation::Supports
                    || a.effect != EvidenceEffect::Closed
                    || a.validator_id != binding.id
                    || a.validator_version != binding.version
                {
                    continue;
                }
                let obligation_id = &a.relevant_fact().obligation_id;
                let Some(spec) = obligations.spec(obligation_id) else {
                    continue;
                };
                if spec.requirement_id.as_ref() != Some(&requirement.id)
                    || !spec.critical
                    || spec.closure_relation != EvidenceRelation::Supports
                    || !contract.trusted_sources().contains(&a.source)
                    || a.source != spec.source
                    || obligations
                        .status(contract, workspace, obligation_id)
                        .map_err(VerdictError::Obligation)?
                        != ObligationStatus::Closed
                {
                    continue;
                }
                satisfied = true;
                break;
            }
            if !satisfied {
                return Ok(DeliveryAssessment::Blocked(
                    DeliveryBlocker::HardEvidenceMissing,
                ));
            }
        }
    }
    // This output is purposely weaker than DELIVERY_READY. The independent
    // evaluator may discover new obligations; Voder MUST NOT auto-finalize here.
    Ok(DeliveryAssessment::EligibleForMetaAudit)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::contract::{ContractDraft, RepoScope, Requirement, TaskAnchor, ValidatorBinding};
    use crate::evidence::{
        EvidenceLedger, EvidenceProposal, ReceiptCompletion, VerifiedReceipt, VerifiedReceiptFields,
    };
    use crate::obligation::ObligationSpec;
    use crate::types::{ActionId, ContractIdentity, EvidenceRelation, Severity, Version};

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
        ObligationRegistry,
        EvidenceLedger,
        Vec<(RequirementId, EvidenceId)>,
    ) {
        let contract = FrozenContract::compile(ContractDraft {
            anchor: TaskAnchor {
                task_id: id("t"),
                identity: "task".into(),
                goal: "test".into(),
                version: Version::new(1).unwrap(),
                accepted_facts: vec![],
                environment_constraints: vec![],
                evaluation_boundary: vec![],
            },
            identity: ContractIdentity {
                contract_id: id("c"),
                version: Version::new(1).unwrap(),
                digest: id("digest"),
                toolchain: id("rust"),
                environment: id("env"),
            },
            requirements: vec![Requirement {
                id: id("req"),
                description: "tests pass".into(),
                hard: true,
                severity: Severity::Critical,
            }],
            validators: vec![ValidatorBinding {
                id: id("test"),
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
            revision: id("rev"),
            digest: id("world"),
        };
        let ob = ObligationSpec {
            id: id("ob"),
            claim_id: "passes".into(),
            requirement_id: Some(id("req")),
            threatened_judgement: "delivery".into(),
            expected_observation_type: "cargo-test".into(),
            admission_rule_id: "rule1".into(),
            scope: RepoScope::parse("src").unwrap(),
            priority: 10,
            stop_condition: "pass".into(),
            source: id("runner"),
            critical: true,
            closure_relation: EvidenceRelation::Supports,
            dependencies: vec![],
        };
        let mut registry = ObligationRegistry::compile(&contract, vec![ob]).unwrap();
        let run: RunId = id("run");
        let action: ActionId = id("action");
        let proposal = EvidenceProposal {
            evidence_id: id("e1"),
            run_id: run.clone(),
            action_id: action.clone(),
            observation_id: id("obs"),
            obligation_id: id("ob"),
            claim_id: "passes".into(),
            expected_receipt_digest: id("receipt"),
        };
        let fields = VerifiedReceiptFields {
            run_id: run.clone(),
            action_id: action,
            observation_id: proposal.observation_id.clone(),
            obligation_id: proposal.obligation_id.clone(),
            claim_id: proposal.claim_id.clone(),
            contract: contract.identity().clone(),
            workspace: workspace.clone(),
            source: id("runner"),
            scope: RepoScope::parse("src/lib.rs").unwrap(),
            observation_type: "cargo-test".into(),
            admission_rule_id: "rule1".into(),
            validator_id: id("test"),
            validator_version: id("v1"),
            content_digest: id("content"),
            receipt_digest: proposal.expected_receipt_digest.clone(),
            relation: EvidenceRelation::Supports,
            completion: ReceiptCompletion::Completed,
            rule_verified: true,
            integrity_verified: true,
        };
        let mut ledger = EvidenceLedger::new();
        ledger
            .admit_and_apply(
                &contract,
                &workspace,
                &mut registry,
                proposal,
                &VerifiedReceipt::from_verified_runtime(fields),
            )
            .unwrap();
        (
            contract,
            workspace,
            run,
            registry,
            ledger,
            vec![(id("req"), id("e1"))],
        )
    }
    #[test]
    fn admitted_current_support_is_only_eligible_for_independent_meta_audit() {
        let (c, w, r, ob, ledger, refs) = fixture();
        assert_eq!(
            evaluate_delivery(DeliveryContext {
                contract: &c,
                workspace: &w,
                run: &r,
                runtime: RuntimeStatus::Active,
                pending_transaction: None,
                obligations: &ob,
                admitted: &ledger,
                evidence_for_hard_requirements: &refs,
            })
            .unwrap(),
            DeliveryAssessment::EligibleForMetaAudit
        );
    }
    #[test]
    fn stale_workspace_invalidates_previous_positive_proof() {
        let (c, mut w, r, ob, ledger, refs) = fixture();
        w.digest = id("next-world");
        assert_eq!(
            evaluate_delivery(DeliveryContext {
                contract: &c,
                workspace: &w,
                run: &r,
                runtime: RuntimeStatus::Active,
                pending_transaction: None,
                obligations: &ob,
                admitted: &ledger,
                evidence_for_hard_requirements: &refs,
            })
            .unwrap(),
            DeliveryAssessment::Blocked(DeliveryBlocker::HardObligationOpen)
        );
    }
    #[test]
    fn forged_evidence_identifier_does_not_satisfy_hard_validation() {
        let (c, w, r, ob, ledger, _) = fixture();
        let refs = vec![(id("req"), id("unknown"))];
        assert_eq!(
            evaluate_delivery(DeliveryContext {
                contract: &c,
                workspace: &w,
                run: &r,
                runtime: RuntimeStatus::Active,
                pending_transaction: None,
                obligations: &ob,
                admitted: &ledger,
                evidence_for_hard_requirements: &refs,
            })
            .unwrap(),
            DeliveryAssessment::Blocked(DeliveryBlocker::HardEvidenceMissing)
        );
    }
    #[test]
    fn a_supporting_receipt_from_another_run_cannot_be_reused() {
        let (c, w, _, ob, ledger, refs) = fixture();
        let run = id("other-run");
        assert_eq!(
            evaluate_delivery(DeliveryContext {
                contract: &c,
                workspace: &w,
                run: &run,
                runtime: RuntimeStatus::Active,
                pending_transaction: None,
                obligations: &ob,
                admitted: &ledger,
                evidence_for_hard_requirements: &refs,
            })
            .unwrap(),
            DeliveryAssessment::Blocked(DeliveryBlocker::HardEvidenceMissing)
        );
    }
    #[test]
    fn claimed_wrong_requirement_does_not_count_for_hard_contract() {
        let (c, w, r, ob, ledger, _) = fixture();
        let refs = vec![(id("other-req"), id("e1"))];
        assert_eq!(
            evaluate_delivery(DeliveryContext {
                contract: &c,
                workspace: &w,
                run: &r,
                runtime: RuntimeStatus::Active,
                pending_transaction: None,
                obligations: &ob,
                admitted: &ledger,
                evidence_for_hard_requirements: &refs,
            })
            .unwrap(),
            DeliveryAssessment::Blocked(DeliveryBlocker::HardEvidenceMissing)
        );
    }
    #[test]
    fn duplicate_claimed_validation_is_not_multiple_independent_proofs() {
        let (c, w, r, ob, ledger, _) = fixture();
        let refs = vec![(id("req"), id("e1")), (id("req"), id("e1"))];
        assert_eq!(
            evaluate_delivery(DeliveryContext {
                contract: &c,
                workspace: &w,
                run: &r,
                runtime: RuntimeStatus::Active,
                pending_transaction: None,
                obligations: &ob,
                admitted: &ledger,
                evidence_for_hard_requirements: &refs,
            })
            .unwrap(),
            DeliveryAssessment::Blocked(DeliveryBlocker::HardEvidenceMissing)
        );
    }
    #[test]
    fn valid_support_during_pending_transaction_still_blocks_delivery() {
        let (c, w, r, ob, ledger, refs) = fixture();
        assert_eq!(
            evaluate_delivery(DeliveryContext {
                contract: &c,
                workspace: &w,
                run: &r,
                runtime: RuntimeStatus::Active,
                pending_transaction: Some(TransactionPhase::Executing),
                obligations: &ob,
                admitted: &ledger,
                evidence_for_hard_requirements: &refs,
            })
            .unwrap(),
            DeliveryAssessment::Blocked(DeliveryBlocker::PendingTransaction)
        );
    }
}
