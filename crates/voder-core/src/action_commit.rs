//! One-step, bounded intent authorization. Never runs a command.
//!
//! This module is a pure pre-dispatch eligibility gate. The future single-writer
//! daemon MUST durably journal the returned intent, retry admission and budget
//! dispatch before any consequential I/O. `RepoScope` is lexical only; runtime
//! must separately enforce symlink-safe containment and process/network sandbox.

use crate::budget::{BudgetAmount, BudgetLedger, ReservationPhase};
use crate::contract::{FrozenContract, RepoScope};
use crate::retry::{ActionDescriptor, ActionSignature, RetryError};
use crate::types::{
    ActionId, ContractIdentity, InputId, ReservationId, RunId, ToolEffect, ValidatorId,
    WorkspaceIdentity,
};
use std::collections::BTreeSet;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum MutationKind {
    Create,
    Modify,
    Delete,
}
impl MutationKind {
    pub const fn as_policy_name(self) -> &'static str {
        match self {
            Self::Create => "CREATE",
            Self::Modify => "MODIFY",
            Self::Delete => "DELETE",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct ExpectedChange {
    pub path: RepoScope,
    pub kind: MutationKind,
}

#[derive(Debug, Clone)]
pub struct ActionIntent {
    pub run_id: RunId,
    pub action_id: ActionId,
    pub descriptor: ActionDescriptor,
    pub contract: ContractIdentity,
    pub workspace: WorkspaceIdentity,
    pub reservation_id: ReservationId,
    pub expected_changes: Vec<ExpectedChange>,
    pub required_validators: Vec<ValidatorId>,
    /// For any consequential action a known rollback strategy is mandatory.
    /// This is a *reference*, not proof that rollback can succeed.
    pub rollback_ref: Option<String>,
}

/// Approval is a separately scoped, trusted-user/policy credential. Cannot be
/// instantiated by external crates or modeled as `approved: bool`.
#[derive(Debug)]
pub struct ApprovalGrant {
    run_id: RunId,
    action_id: ActionId,
    signature: ActionSignature,
    contract: ContractIdentity,
    workspace: WorkspaceIdentity,
}
impl ApprovalGrant {
    /// To be called only by a future in-crate trusted approval adapter.
    /// Until one exists, production approval-required writes fail closed.
    #[allow(dead_code)]
    pub(crate) fn from_trusted_authority(
        run_id: RunId,
        action_id: ActionId,
        signature: ActionSignature,
        contract: ContractIdentity,
        workspace: WorkspaceIdentity,
    ) -> Self {
        Self {
            run_id,
            action_id,
            signature,
            contract,
            workspace,
        }
    }
}

/// An authorized *intent*, not a certificate of execution or durable commit.
#[derive(Debug, Clone)]
pub struct PreparedAction {
    intent: ActionIntent,
    signature: ActionSignature,
    budget_upper_bound: BudgetAmount,
    effect: ToolEffect,
}
impl PreparedAction {
    pub fn intent(&self) -> &ActionIntent {
        &self.intent
    }
    pub fn signature(&self) -> &ActionSignature {
        &self.signature
    }
    pub fn budget_upper_bound(&self) -> BudgetAmount {
        self.budget_upper_bound
    }
    pub fn effect(&self) -> ToolEffect {
        self.effect
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CommitError {
    ContractMismatch,
    WorkspaceMismatch,
    MissingRequiredInputs,
    UnknownTool,
    ReadScopeViolation,
    WriteScopeViolation,
    ReadOnlyToolMutation,
    MissingExpectedChanges,
    MissingTargetObligation,
    MissingRequiredValidators,
    UnknownRequiredValidator,
    DuplicateRequiredValidator,
    DuplicateExpectedChange,
    TooManyExpectedChanges,
    ForbiddenEffect,
    MissingRollback,
    ApprovalRequired,
    ApprovalMismatch,
    BudgetNotReserved,
    ReservationAlreadyPrepared,
    ActionAlreadyPrepared,
    BudgetInsufficientForAction,
    InvalidRetryDescriptor(RetryError),
}

/// Stage preconditions only: this does NOT reserve or consume budget, does NOT
/// mark a retry attempt, and does NOT issue OS permission. The single writer
/// must compose all three decisions with a durable PREPARED journal record.
fn prepare_action(
    contract: &FrozenContract,
    current_workspace: &WorkspaceIdentity,
    verified_present_inputs: &[InputId],
    budget: &BudgetLedger,
    intent: ActionIntent,
    approval: Option<ApprovalGrant>,
) -> Result<PreparedAction, CommitError> {
    if &intent.contract != contract.identity() {
        return Err(CommitError::ContractMismatch);
    }
    if &intent.workspace != current_workspace {
        return Err(CommitError::WorkspaceMismatch);
    }
    if !contract.missing_inputs(verified_present_inputs).is_empty() {
        return Err(CommitError::MissingRequiredInputs);
    }
    let tool = contract
        .tools()
        .iter()
        .find(|t| t.id == intent.descriptor.tool_id)
        .ok_or(CommitError::UnknownTool)?;
    if !contract
        .read_scopes()
        .iter()
        .any(|s| s.contains(&intent.descriptor.scope))
    {
        return Err(CommitError::ReadScopeViolation);
    }
    let signature = intent
        .descriptor
        .signature()
        .map_err(CommitError::InvalidRetryDescriptor)?;
    let reservation = budget
        .record(&intent.reservation_id)
        .ok_or(CommitError::BudgetNotReserved)?;
    if reservation.phase != ReservationPhase::Held {
        return Err(CommitError::BudgetNotReserved);
    }
    if reservation.upper_bound.steps == 0 || reservation.upper_bound.tool_calls == 0 {
        return Err(CommitError::BudgetInsufficientForAction);
    }
    if tool.requires_approval {
        let grant = approval.ok_or(CommitError::ApprovalRequired)?;
        if grant.run_id != intent.run_id
            || grant.action_id != intent.action_id
            || grant.signature != signature
            || grant.contract != intent.contract
            || grant.workspace != intent.workspace
        {
            return Err(CommitError::ApprovalMismatch);
        }
    }
    let validator_set: BTreeSet<_> = intent.required_validators.iter().collect();
    if validator_set.len() != intent.required_validators.len() {
        return Err(CommitError::DuplicateRequiredValidator);
    }
    if intent
        .required_validators
        .iter()
        .any(|id| !contract.validators().iter().any(|v| v.id == *id))
    {
        return Err(CommitError::UnknownRequiredValidator);
    }
    if intent.expected_changes.len() > 256 {
        return Err(CommitError::TooManyExpectedChanges);
    }
    let mut seen = BTreeSet::new();
    for change in &intent.expected_changes {
        if !seen.insert(change.path.clone()) {
            return Err(CommitError::DuplicateExpectedChange);
        }
        if !intent.descriptor.scope.contains(&change.path)
            || !contract
                .write_scopes()
                .iter()
                .any(|s| s.contains(&change.path))
        {
            return Err(CommitError::WriteScopeViolation);
        }
        if contract
            .forbidden_effects()
            .iter()
            .any(|f| f == change.kind.as_policy_name())
        {
            return Err(CommitError::ForbiddenEffect);
        }
    }
    match tool.effect {
        ToolEffect::Read if !intent.expected_changes.is_empty() => {
            return Err(CommitError::ReadOnlyToolMutation)
        }
        ToolEffect::Write => {
            if intent.descriptor.target_obligation_id.is_none() {
                return Err(CommitError::MissingTargetObligation);
            }
            if intent.required_validators.is_empty() {
                return Err(CommitError::MissingRequiredValidators);
            }
            if !contract
                .write_scopes()
                .iter()
                .any(|s| s.contains(&intent.descriptor.scope))
            {
                return Err(CommitError::WriteScopeViolation);
            }
            if intent.expected_changes.is_empty() {
                return Err(CommitError::MissingExpectedChanges);
            }
            if intent
                .rollback_ref
                .as_deref()
                .is_none_or(|r| r.trim().is_empty() || r.chars().any(char::is_control))
            {
                return Err(CommitError::MissingRollback);
            }
        }
        ToolEffect::Read => {}
    }
    Ok(PreparedAction {
        intent,
        signature,
        budget_upper_bound: reservation.upper_bound,
        effect: tool.effect,
    })
}

/// Volatile single-writer guard: a reservation or action identity cannot be
/// prepared twice. The daemon must persist the same identity in VEP before
/// dispatch and recover it after crash; this is not crash-durable by itself.
#[derive(Debug, Default)]
pub struct CommitGate {
    used_reservations: BTreeSet<ReservationId>,
    used_actions: BTreeSet<(RunId, ActionId)>,
}
impl CommitGate {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn prepare(
        &mut self,
        contract: &FrozenContract,
        workspace: &WorkspaceIdentity,
        verified_inputs: &[InputId],
        budget: &BudgetLedger,
        intent: ActionIntent,
        approval: Option<ApprovalGrant>,
    ) -> Result<PreparedAction, CommitError> {
        if self.used_reservations.contains(&intent.reservation_id) {
            return Err(CommitError::ReservationAlreadyPrepared);
        }
        let key = (intent.run_id.clone(), intent.action_id.clone());
        if self.used_actions.contains(&key) {
            return Err(CommitError::ActionAlreadyPrepared);
        }
        let prepared = prepare_action(
            contract,
            workspace,
            verified_inputs,
            budget,
            intent,
            approval,
        )?;
        self.used_reservations
            .insert(prepared.intent.reservation_id.clone());
        self.used_actions.insert(key);
        Ok(prepared)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::budget::{BudgetAmount, BudgetLedger};
    use crate::contract::{
        ContractDraft, InputKind, RepoScope, RequiredInput, Requirement, TaskAnchor, ToolRule,
        ValidatorBinding,
    };
    use crate::types::{InputId, Severity, Version};
    fn id<T: TryFrom<&'static str>>(s: &'static str) -> T
    where
        <T as TryFrom<&'static str>>::Error: std::fmt::Debug,
    {
        T::try_from(s).unwrap()
    }
    fn prepare_action(
        contract: &FrozenContract,
        workspace: &WorkspaceIdentity,
        inputs: &[InputId],
        budget: &BudgetLedger,
        intent: ActionIntent,
        approval: Option<ApprovalGrant>,
    ) -> Result<PreparedAction, CommitError> {
        CommitGate::new().prepare(contract, workspace, inputs, budget, intent, approval)
    }
    fn fixture(
        effect: ToolEffect,
        approval: bool,
    ) -> (
        FrozenContract,
        WorkspaceIdentity,
        BudgetLedger,
        ActionIntent,
        Vec<InputId>,
    ) {
        let identity = ContractIdentity {
            contract_id: id("contract"),
            version: Version::new(1).unwrap(),
            digest: id("contract-hash"),
            toolchain: id("rust"),
            environment: id("env"),
        };
        let workspace = WorkspaceIdentity {
            revision: id("revision"),
            digest: id("world"),
        };
        let required: InputId = id("src-file");
        let contract = FrozenContract::compile(ContractDraft {
            anchor: TaskAnchor {
                task_id: id("task"),
                identity: "task".into(),
                goal: "fix test".into(),
                version: Version::new(1).unwrap(),
                accepted_facts: vec![],
                environment_constraints: vec![],
                evaluation_boundary: vec![],
            },
            identity: identity.clone(),
            requirements: vec![Requirement {
                id: id("req"),
                description: "passing tests".into(),
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
                id: id("tool"),
                effect,
                requires_approval: approval,
            }],
            read_scopes: vec![RepoScope::parse("src").unwrap()],
            write_scopes: if effect == ToolEffect::Write {
                vec![RepoScope::parse("src").unwrap()]
            } else {
                vec![]
            },
            forbidden_effects: vec!["DELETE".into()],
            required_inputs: vec![RequiredInput {
                id: required.clone(),
                kind: InputKind::File,
                description: "src/file".into(),
                why_required: "required".into(),
            }],
            sampling_rules: vec![],
        })
        .unwrap();
        let mut budget = BudgetLedger::new(BudgetAmount {
            steps: 5,
            tool_calls: 5,
            ..BudgetAmount::default()
        });
        budget
            .reserve(
                id("reservation"),
                BudgetAmount {
                    steps: 1,
                    tool_calls: 1,
                    ..BudgetAmount::default()
                },
            )
            .unwrap();
        let intent = ActionIntent {
            run_id: id("run"),
            action_id: id("action"),
            descriptor: ActionDescriptor {
                kind: "patch".into(),
                target_obligation_id: if effect == ToolEffect::Write {
                    Some(id("ob"))
                } else {
                    None
                },
                hypothesis_id: None,
                tool_id: id("tool"),
                parameter_key: id("params"),
                scope: RepoScope::parse("src").unwrap(),
            },
            contract: identity,
            workspace: workspace.clone(),
            reservation_id: id("reservation"),
            expected_changes: if effect == ToolEffect::Write {
                vec![ExpectedChange {
                    path: RepoScope::parse("src/file.rs").unwrap(),
                    kind: MutationKind::Modify,
                }]
            } else {
                vec![]
            },
            required_validators: if effect == ToolEffect::Write {
                vec![id("test")]
            } else {
                vec![]
            },
            rollback_ref: if effect == ToolEffect::Write {
                Some("worktree-snapshot".into())
            } else {
                None
            },
        };
        (contract, workspace, budget, intent, vec![required])
    }
    #[test]
    fn read_with_held_budget_allowed_without_approval() {
        let (c, w, b, a, inputs) = fixture(ToolEffect::Read, false);
        assert!(prepare_action(&c, &w, &inputs, &b, a, None).is_ok());
    }
    #[test]
    fn write_with_held_budget_allowed_without_required_approval() {
        let (c, w, b, a, inputs) = fixture(ToolEffect::Write, false);
        assert!(prepare_action(&c, &w, &inputs, &b, a, None).is_ok());
    }
    #[test]
    fn missing_input_stops_action() {
        let (c, w, b, a, _) = fixture(ToolEffect::Write, false);
        assert!(matches!(
            prepare_action(&c, &w, &[], &b, a, None),
            Err(CommitError::MissingRequiredInputs)
        ));
    }
    #[test]
    fn write_cannot_claim_empty_effects() {
        let (c, w, b, mut a, inputs) = fixture(ToolEffect::Write, false);
        a.expected_changes.clear();
        assert!(matches!(
            prepare_action(&c, &w, &inputs, &b, a, None),
            Err(CommitError::MissingExpectedChanges)
        ));
    }
    #[test]
    fn write_cannot_omit_rollback() {
        let (c, w, b, mut a, inputs) = fixture(ToolEffect::Write, false);
        a.rollback_ref = None;
        assert!(matches!(
            prepare_action(&c, &w, &inputs, &b, a, None),
            Err(CommitError::MissingRollback)
        ));
    }
    #[test]
    fn user_approval_cannot_be_replaced_with_bool() {
        let (c, w, b, a, inputs) = fixture(ToolEffect::Write, true);
        assert!(matches!(
            prepare_action(&c, &w, &inputs, &b, a, None),
            Err(CommitError::ApprovalRequired)
        ));
    }
    #[test]
    fn grant_bound_to_different_action_is_rejected() {
        let (c, w, b, a, inputs) = fixture(ToolEffect::Write, true);
        let grant = ApprovalGrant::from_trusted_authority(
            id("run"),
            id("action"),
            ActionDescriptor {
                kind: "other".into(),
                ..a.descriptor.clone()
            }
            .signature()
            .unwrap(),
            c.identity().clone(),
            w.clone(),
        );
        assert!(matches!(
            prepare_action(&c, &w, &inputs, &b, a, Some(grant)),
            Err(CommitError::ApprovalMismatch)
        ));
    }
    #[test]
    fn matching_grant_can_prepare_intent() {
        let (c, w, b, a, inputs) = fixture(ToolEffect::Write, true);
        let grant = ApprovalGrant::from_trusted_authority(
            id("run"),
            id("action"),
            a.descriptor.signature().unwrap(),
            c.identity().clone(),
            w.clone(),
        );
        assert!(prepare_action(&c, &w, &inputs, &b, a, Some(grant)).is_ok());
    }
    #[test]
    fn stale_workspace_rejected() {
        let (c, w, b, mut a, inputs) = fixture(ToolEffect::Write, false);
        a.workspace.digest = id("old");
        assert!(matches!(
            prepare_action(&c, &w, &inputs, &b, a, None),
            Err(CommitError::WorkspaceMismatch)
        ));
    }
    #[test]
    fn forbidden_delete_rejected() {
        let (c, w, b, mut a, inputs) = fixture(ToolEffect::Write, false);
        a.expected_changes[0].kind = MutationKind::Delete;
        assert!(matches!(
            prepare_action(&c, &w, &inputs, &b, a, None),
            Err(CommitError::ForbiddenEffect)
        ));
    }
    #[test]
    fn out_of_scope_write_rejected() {
        let (c, w, b, mut a, inputs) = fixture(ToolEffect::Write, false);
        a.expected_changes[0].path = RepoScope::parse("private/secrets").unwrap();
        assert!(matches!(
            prepare_action(&c, &w, &inputs, &b, a, None),
            Err(CommitError::WriteScopeViolation)
        ));
    }
    #[test]
    fn read_tool_cannot_become_write() {
        let (c, w, b, mut a, inputs) = fixture(ToolEffect::Read, false);
        a.expected_changes.push(ExpectedChange {
            path: RepoScope::parse("src/file").unwrap(),
            kind: MutationKind::Modify,
        });
        assert!(prepare_action(&c, &w, &inputs, &b, a, None).is_err());
    }
    #[test]
    fn missing_reservation_rejected() {
        let (c, w, b, mut a, inputs) = fixture(ToolEffect::Write, false);
        a.reservation_id = id("unknown");
        assert!(matches!(
            prepare_action(&c, &w, &inputs, &b, a, None),
            Err(CommitError::BudgetNotReserved)
        ));
    }
    #[test]
    fn dispatched_reservation_not_reusable() {
        let (c, w, mut b, a, inputs) = fixture(ToolEffect::Write, false);
        b.mark_dispatched(&id("reservation")).unwrap();
        assert!(matches!(
            prepare_action(&c, &w, &inputs, &b, a, None),
            Err(CommitError::BudgetNotReserved)
        ));
    }
    #[test]
    fn reservation_cannot_prepare_two_different_actions() {
        let (c, w, b, a, inputs) = fixture(ToolEffect::Write, false);
        let mut gate = CommitGate::new();
        gate.prepare(&c, &w, &inputs, &b, a.clone(), None).unwrap();
        let mut changed = a;
        changed.action_id = id("action-2");
        assert!(matches!(
            gate.prepare(&c, &w, &inputs, &b, changed, None),
            Err(CommitError::ReservationAlreadyPrepared)
        ));
    }
    #[test]
    fn action_id_cannot_move_between_reservations() {
        let (c, w, mut b, a, inputs) = fixture(ToolEffect::Write, false);
        b.reserve(
            id("reservation2"),
            BudgetAmount {
                steps: 1,
                tool_calls: 1,
                ..BudgetAmount::default()
            },
        )
        .unwrap();
        let mut gate = CommitGate::new();
        gate.prepare(&c, &w, &inputs, &b, a.clone(), None).unwrap();
        let mut changed = a;
        changed.reservation_id = id("reservation2");
        assert!(matches!(
            gate.prepare(&c, &w, &inputs, &b, changed, None),
            Err(CommitError::ActionAlreadyPrepared)
        ));
    }
    #[test]
    fn write_requires_a_target_evidence_obligation() {
        let (c, w, b, mut a, inputs) = fixture(ToolEffect::Write, false);
        a.descriptor.target_obligation_id = None;
        assert!(matches!(
            prepare_action(&c, &w, &inputs, &b, a, None),
            Err(CommitError::MissingTargetObligation)
        ));
    }
    #[test]
    fn write_requires_frozen_validator() {
        let (c, w, b, mut a, inputs) = fixture(ToolEffect::Write, false);
        a.required_validators.clear();
        assert!(matches!(
            prepare_action(&c, &w, &inputs, &b, a, None),
            Err(CommitError::MissingRequiredValidators)
        ));
    }
    #[test]
    fn arbitrary_validator_cannot_be_added() {
        let (c, w, b, mut a, inputs) = fixture(ToolEffect::Write, false);
        a.required_validators.push(id("fake"));
        assert!(matches!(
            prepare_action(&c, &w, &inputs, &b, a, None),
            Err(CommitError::UnknownRequiredValidator)
        ));
    }
    #[test]
    fn preflight_does_not_mutate_budget() {
        let (c, w, b, a, inputs) = fixture(ToolEffect::Write, false);
        let before = b.snapshot();
        let _ = prepare_action(&c, &w, &inputs, &b, a, None).unwrap();
        assert_eq!(before, b.snapshot());
    }
}
