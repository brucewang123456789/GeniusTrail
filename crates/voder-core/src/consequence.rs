//! Compare a prepared one-step intent with an actual *reported* execution.
//!
//! No observed output proves itself: trusted runtime must authenticate and
//! persist the receipt before this judgement may affect authoritative state.
//! A PASS from this comparator is not a delivery verdict or evidence admission.

use crate::action_commit::{MutationKind, PreparedAction};
use crate::contract::RepoScope;
use crate::types::{
    ActionId, ConsequenceClass, ContractIdentity, Digest, RunId, ToolEffect, WorkspaceIdentity,
};
use std::collections::BTreeSet;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExecutionCompletion {
    Completed,
    Failed,
    TimedOut,
    Cancelled,
    RecoveryRequired,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ObservedChange {
    pub path: RepoScope,
    pub kind: MutationKind,
}

/// Untrusted until `voder-runtime` authenticates process identity, file diff,
/// exit code, and artifact bytes; no public accessor here declares this trusted.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExecutionReport {
    pub run_id: RunId,
    pub action_id: ActionId,
    pub contract: ContractIdentity,
    pub before: WorkspaceIdentity,
    pub after: WorkspaceIdentity,
    pub completion: ExecutionCompletion,
    pub observed_changes: Vec<ObservedChange>,
    pub forbidden_side_effects: Vec<String>,
    pub receipt_digest: Digest,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NextSafetyStep {
    ValidateEvidence,
    InvestigateDeviation,
    RollbackBeforeRetry,
    SafeStop,
    RecoveryReconcile,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ConsequenceAssessment {
    pub class: ConsequenceClass,
    pub next: NextSafetyStep,
}

/// Pure comparison of *reported* facts, not certification of runtime integrity.
/// Any unexpected file mutation is a forbidden side effect; failed commands
/// with partial changes require rollback rather than silent rerun.
pub fn compare_consequences(
    prepared: &PreparedAction,
    report: &ExecutionReport,
) -> ConsequenceAssessment {
    let intent = prepared.intent();
    if report.run_id != intent.run_id
        || report.action_id != intent.action_id
        || report.contract != intent.contract
        || report.before != intent.workspace
    {
        return ConsequenceAssessment {
            class: ConsequenceClass::ForbiddenSideEffect,
            next: NextSafetyStep::SafeStop,
        };
    }
    if report.completion == ExecutionCompletion::RecoveryRequired {
        return ConsequenceAssessment {
            class: ConsequenceClass::DecisionRelevantDeviation,
            next: NextSafetyStep::RecoveryReconcile,
        };
    }
    if report.observed_changes.len() > 256 {
        return ConsequenceAssessment {
            class: ConsequenceClass::ForbiddenSideEffect,
            next: NextSafetyStep::SafeStop,
        };
    }
    let expected: BTreeSet<_> = intent
        .expected_changes
        .iter()
        .map(|e| (&e.path, e.kind))
        .collect();
    let observed: BTreeSet<_> = report
        .observed_changes
        .iter()
        .map(|e| (&e.path, e.kind))
        .collect();
    let unknown_change = observed.iter().any(|e| !expected.contains(e))
        || observed.len() != report.observed_changes.len();
    if !report.forbidden_side_effects.is_empty()
        || unknown_change
        || (prepared.effect() == ToolEffect::Read && report.before.digest != report.after.digest)
    {
        return ConsequenceAssessment {
            class: ConsequenceClass::ForbiddenSideEffect,
            next: NextSafetyStep::SafeStop,
        };
    }
    if report.completion != ExecutionCompletion::Completed {
        return ConsequenceAssessment {
            class: ConsequenceClass::DecisionRelevantDeviation,
            next: if observed.is_empty() {
                NextSafetyStep::InvestigateDeviation
            } else {
                NextSafetyStep::RollbackBeforeRetry
            },
        };
    }
    // A reported mutation with an unchanged content fingerprint is inconsistent.
    // It cannot be accepted as MATCH, even if the tool exited successfully.
    if !observed.is_empty() && report.before.digest == report.after.digest {
        return ConsequenceAssessment {
            class: ConsequenceClass::DecisionRelevantDeviation,
            next: NextSafetyStep::SafeStop,
        };
    }
    if prepared.effect() == ToolEffect::Write && observed.is_empty() {
        return ConsequenceAssessment {
            class: ConsequenceClass::DecisionRelevantDeviation,
            next: NextSafetyStep::InvestigateDeviation,
        };
    }
    if observed == expected {
        ConsequenceAssessment {
            class: ConsequenceClass::Match,
            next: NextSafetyStep::ValidateEvidence,
        }
    } else {
        ConsequenceAssessment {
            class: ConsequenceClass::ExpectedVariation,
            next: NextSafetyStep::InvestigateDeviation,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::action_commit::{ActionIntent, CommitGate, ExpectedChange};
    use crate::budget::{BudgetAmount, BudgetLedger};
    use crate::contract::{
        ContractDraft, FrozenContract, RepoScope, Requirement, TaskAnchor, ToolRule,
        ValidatorBinding,
    };
    use crate::retry::ActionDescriptor;
    use crate::types::{ContractIdentity, Severity, ToolEffect, Version};
    fn id<T: TryFrom<&'static str>>(s: &'static str) -> T
    where
        <T as TryFrom<&'static str>>::Error: std::fmt::Debug,
    {
        T::try_from(s).unwrap()
    }
    fn fixture() -> (PreparedAction, ExecutionReport) {
        let identity = ContractIdentity {
            contract_id: id("c"),
            version: Version::new(1).unwrap(),
            digest: id("contract"),
            toolchain: id("rust"),
            environment: id("env"),
        };
        let world = WorkspaceIdentity {
            revision: id("rev1"),
            digest: id("world1"),
        };
        let contract = FrozenContract::compile(ContractDraft {
            anchor: TaskAnchor {
                task_id: id("t"),
                identity: "task".into(),
                goal: "repair".into(),
                version: Version::new(1).unwrap(),
                accepted_facts: vec![],
                environment_constraints: vec![],
                evaluation_boundary: vec![],
            },
            identity: identity.clone(),
            requirements: vec![Requirement {
                id: id("req"),
                description: "tests".into(),
                hard: true,
                severity: Severity::Critical,
            }],
            validators: vec![ValidatorBinding {
                id: id("test"),
                version: id("v1"),
                requirement_ids: vec![id("req")],
            }],
            trusted_sources: vec![id("runtime")],
            tools: vec![ToolRule {
                id: id("write"),
                effect: ToolEffect::Write,
                requires_approval: false,
            }],
            read_scopes: vec![RepoScope::parse("src").unwrap()],
            write_scopes: vec![RepoScope::parse("src").unwrap()],
            forbidden_effects: vec![],
            required_inputs: vec![],
            sampling_rules: vec![],
        })
        .unwrap();
        let mut budget = BudgetLedger::new(BudgetAmount {
            steps: 2,
            tool_calls: 2,
            ..BudgetAmount::default()
        });
        budget
            .reserve(
                id("res"),
                BudgetAmount {
                    steps: 1,
                    tool_calls: 1,
                    ..BudgetAmount::default()
                },
            )
            .unwrap();
        let change = ExpectedChange {
            path: RepoScope::parse("src/main.rs").unwrap(),
            kind: MutationKind::Modify,
        };
        let intent = ActionIntent {
            run_id: id("run"),
            action_id: id("action"),
            descriptor: ActionDescriptor {
                kind: "patch".into(),
                target_obligation_id: Some(id("ob")),
                hypothesis_id: None,
                tool_id: id("write"),
                parameter_key: id("params"),
                scope: RepoScope::parse("src").unwrap(),
            },
            contract: identity.clone(),
            workspace: world.clone(),
            reservation_id: id("res"),
            expected_changes: vec![change.clone()],
            required_validators: vec![id("test")],
            rollback_ref: Some("snapshot".into()),
        };
        let prepared = CommitGate::new()
            .prepare(&contract, &world, &[], &budget, intent, None)
            .unwrap();
        let report = ExecutionReport {
            run_id: id("run"),
            action_id: id("action"),
            contract: identity,
            before: world,
            after: WorkspaceIdentity {
                revision: id("rev2"),
                digest: id("world2"),
            },
            completion: ExecutionCompletion::Completed,
            observed_changes: vec![ObservedChange {
                path: change.path,
                kind: change.kind,
            }],
            forbidden_side_effects: vec![],
            receipt_digest: id("receipt"),
        };
        (prepared, report)
    }
    #[test]
    fn exact_report_match_only_enters_validation() {
        let (p, r) = fixture();
        assert_eq!(
            compare_consequences(&p, &r),
            ConsequenceAssessment {
                class: ConsequenceClass::Match,
                next: NextSafetyStep::ValidateEvidence
            }
        );
    }
    #[test]
    fn failed_write_with_mutation_must_rollback() {
        let (p, mut r) = fixture();
        r.completion = ExecutionCompletion::Failed;
        assert_eq!(
            compare_consequences(&p, &r).next,
            NextSafetyStep::RollbackBeforeRetry
        );
    }
    #[test]
    fn timeout_with_mutation_must_rollback() {
        let (p, mut r) = fixture();
        r.completion = ExecutionCompletion::TimedOut;
        assert_eq!(
            compare_consequences(&p, &r).next,
            NextSafetyStep::RollbackBeforeRetry
        );
    }
    #[test]
    fn failed_without_mutation_requires_investigation() {
        let (p, mut r) = fixture();
        r.completion = ExecutionCompletion::Failed;
        r.observed_changes.clear();
        assert_eq!(
            compare_consequences(&p, &r).next,
            NextSafetyStep::InvestigateDeviation
        );
    }
    #[test]
    fn recovery_uncertainty_never_auto_retries() {
        let (p, mut r) = fixture();
        r.completion = ExecutionCompletion::RecoveryRequired;
        assert_eq!(
            compare_consequences(&p, &r).next,
            NextSafetyStep::RecoveryReconcile
        );
    }
    #[test]
    fn wrong_action_stops() {
        let (p, mut r) = fixture();
        r.action_id = id("another");
        assert_eq!(
            compare_consequences(&p, &r).class,
            ConsequenceClass::ForbiddenSideEffect
        );
    }
    #[test]
    fn wrong_world_stops() {
        let (p, mut r) = fixture();
        r.before.digest = id("different");
        assert_eq!(compare_consequences(&p, &r).next, NextSafetyStep::SafeStop);
    }
    #[test]
    fn outside_write_stops() {
        let (p, mut r) = fixture();
        r.observed_changes.push(ObservedChange {
            path: RepoScope::parse("secrets/token").unwrap(),
            kind: MutationKind::Modify,
        });
        assert_eq!(
            compare_consequences(&p, &r).class,
            ConsequenceClass::ForbiddenSideEffect
        );
    }
    #[test]
    fn unplanned_write_inside_scope_stops() {
        let (p, mut r) = fixture();
        r.observed_changes.push(ObservedChange {
            path: RepoScope::parse("src/other.rs").unwrap(),
            kind: MutationKind::Modify,
        });
        assert_eq!(compare_consequences(&p, &r).next, NextSafetyStep::SafeStop);
    }
    #[test]
    fn forbidden_effect_stops() {
        let (p, mut r) = fixture();
        r.forbidden_side_effects.push("network".into());
        assert_eq!(compare_consequences(&p, &r).next, NextSafetyStep::SafeStop);
    }
    #[test]
    fn empty_success_is_not_match() {
        let (p, mut r) = fixture();
        r.observed_changes.clear();
        assert_ne!(compare_consequences(&p, &r).class, ConsequenceClass::Match);
    }
    #[test]
    fn failed_partial_write_with_unchanged_digest_still_requires_rollback() {
        let (p, mut r) = fixture();
        r.completion = ExecutionCompletion::Failed;
        r.after = r.before.clone();
        assert_eq!(
            compare_consequences(&p, &r).next,
            NextSafetyStep::RollbackBeforeRetry
        );
    }
    #[test]
    fn world_unchanged_cannot_match_write() {
        let (p, mut r) = fixture();
        r.after = r.before.clone();
        assert_ne!(compare_consequences(&p, &r).class, ConsequenceClass::Match);
    }
    #[test]
    fn partially_completed_expected_effect_not_match() {
        let (p, mut r) = fixture();
        r.observed_changes.clear();
        assert_eq!(
            compare_consequences(&p, &r).class,
            ConsequenceClass::DecisionRelevantDeviation
        );
    }
}
