//! The only pure state-transition planner for the eventual single-writer `voderd`.
//!
//! Produces a proposed next state; neither its events nor its return values
//! prove a runtime receipt is genuine or that a VEP event was durably committed.
//! After a crash the daemon MUST reconcile the journal and actual sandbox state
//! before issuing another consequential action. No public `mark_done` escape hatch.
use crate::action_commit::PreparedAction;
use crate::consequence::{compare_consequences, ExecutionReport, NextSafetyStep};
use crate::types::{
    ActionId, ContractIdentity, EpistemicStatus, RunId, RuntimeStatus, TransactionPhase,
    WorkspaceIdentity,
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ControlState {
    run: RunId,
    contract: ContractIdentity,
    workspace: WorkspaceIdentity,
    runtime: RuntimeStatus,
    epistemic: EpistemicStatus,
    phase: Option<TransactionPhase>,
    pending_action: Option<ActionId>,
    sequence: u64,
}
impl ControlState {
    pub fn new(run: RunId, contract: ContractIdentity, workspace: WorkspaceIdentity) -> Self {
        Self {
            run,
            contract,
            workspace,
            runtime: RuntimeStatus::Active,
            epistemic: EpistemicStatus::Active,
            phase: None,
            pending_action: None,
            sequence: 0,
        }
    }
    pub fn runtime(&self) -> RuntimeStatus {
        self.runtime
    }
    pub fn epistemic(&self) -> EpistemicStatus {
        self.epistemic
    }
    pub fn phase(&self) -> Option<TransactionPhase> {
        self.phase
    }
    pub fn workspace(&self) -> &WorkspaceIdentity {
        &self.workspace
    }
    pub fn sequence(&self) -> u64 {
        self.sequence
    }
    pub fn pending_action(&self) -> Option<&ActionId> {
        self.pending_action.as_ref()
    }
}

/// These requests are NOT authority tokens. Caller must use the VEP single-writer
/// adapter; only it can apply returned state to an authoritative persisted run.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TerminalReason {
    BudgetExhausted,
    MissingInput,
    NoEligibleAction,
    IrrecoverableViolation,
}

#[derive(Debug)]
pub enum TransitionRequest<'a> {
    Prepare(&'a PreparedAction),
    Dispatch {
        action_id: &'a ActionId,
    },
    Observe {
        action_id: &'a ActionId,
    },
    Adjudicate {
        prepared: &'a PreparedAction,
        report: &'a ExecutionReport,
    },
    Stop(TerminalReason),
    /// Indeterminate transaction MUST NOT auto-retry.
    RequireRecovery,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransitionError {
    InvalidPhase,
    InactiveRun,
    WrongRun,
    WrongContract,
    WrongWorkspace,
    WrongAction,
    SequenceOverflow,
}

/// Validate the complete transition first, returning a new immutable value.
/// Never mutates original state on error. Complete/deliver is intentionally
/// unavailable until meta-audit, trusted authorization and durable VEP exist.
pub fn plan_transition(
    state: &ControlState,
    request: TransitionRequest<'_>,
) -> Result<ControlState, TransitionError> {
    let mut next = state.clone();
    match request {
        TransitionRequest::Prepare(prepared) => {
            if state.runtime != RuntimeStatus::Active {
                return Err(TransitionError::InactiveRun);
            }
            if state.phase.is_some() {
                return Err(TransitionError::InvalidPhase);
            }
            let intent = prepared.intent();
            if intent.run_id != state.run {
                return Err(TransitionError::WrongRun);
            }
            if intent.contract != state.contract {
                return Err(TransitionError::WrongContract);
            }
            if intent.workspace != state.workspace {
                return Err(TransitionError::WrongWorkspace);
            }
            next.phase = Some(TransactionPhase::Prepared);
            next.pending_action = Some(intent.action_id.clone());
        }
        TransitionRequest::Dispatch { action_id } => {
            if state.runtime != RuntimeStatus::Active {
                return Err(TransitionError::InactiveRun);
            }
            if state.phase != Some(TransactionPhase::Prepared) {
                return Err(TransitionError::InvalidPhase);
            }
            if state.pending_action.as_ref() != Some(action_id) {
                return Err(TransitionError::WrongAction);
            }
            next.phase = Some(TransactionPhase::Executing);
        }
        TransitionRequest::Observe { action_id } => {
            if state.runtime != RuntimeStatus::Active {
                return Err(TransitionError::InactiveRun);
            }
            if state.phase != Some(TransactionPhase::Executing) {
                return Err(TransitionError::InvalidPhase);
            }
            if state.pending_action.as_ref() != Some(action_id) {
                return Err(TransitionError::WrongAction);
            }
            next.phase = Some(TransactionPhase::Observed);
        }
        TransitionRequest::Adjudicate { prepared, report } => {
            if state.runtime != RuntimeStatus::Active {
                return Err(TransitionError::InactiveRun);
            }
            if state.phase != Some(TransactionPhase::Observed) {
                return Err(TransitionError::InvalidPhase);
            }
            if state.pending_action.as_ref() != Some(&prepared.intent().action_id)
                || report.action_id != prepared.intent().action_id
            {
                return Err(TransitionError::WrongAction);
            }
            if prepared.intent().run_id != state.run || report.run_id != state.run {
                return Err(TransitionError::WrongRun);
            }
            if prepared.intent().contract != state.contract || report.contract != state.contract {
                return Err(TransitionError::WrongContract);
            }
            if prepared.intent().workspace != state.workspace || report.before != state.workspace {
                return Err(TransitionError::WrongWorkspace);
            }
            // Pure reported-effects comparison, NOT runtime authenticity attestation.
            let assessment = compare_consequences(prepared, report);
            match assessment.next {
                NextSafetyStep::ValidateEvidence | NextSafetyStep::InvestigateDeviation => {
                    next.phase = None;
                    next.pending_action = None;
                    next.workspace = report.after.clone();
                }
                NextSafetyStep::RollbackBeforeRetry | NextSafetyStep::SafeStop => {
                    next.runtime = RuntimeStatus::SafeStopped;
                    next.epistemic = EpistemicStatus::Unresolved;
                    next.phase = Some(TransactionPhase::RecoveryRequired);
                    // Do not overwrite known workspace with uncertain post-state.
                }
                NextSafetyStep::RecoveryReconcile => {
                    next.runtime = RuntimeStatus::RecoveryRequired;
                    next.epistemic = EpistemicStatus::Unresolved;
                    next.phase = Some(TransactionPhase::RecoveryRequired);
                }
            }
        }
        TransitionRequest::Stop(reason) => {
            if state.runtime != RuntimeStatus::Active || state.phase.is_some() {
                return Err(TransitionError::InvalidPhase);
            }
            next.runtime = match reason {
                TerminalReason::BudgetExhausted => RuntimeStatus::Exhausted,
                TerminalReason::MissingInput => RuntimeStatus::Blocked,
                TerminalReason::NoEligibleAction => RuntimeStatus::Stalled,
                TerminalReason::IrrecoverableViolation => RuntimeStatus::SafeStopped,
            };
            next.epistemic = EpistemicStatus::Unresolved;
        }
        TransitionRequest::RequireRecovery => {
            if state.phase.is_none() || state.runtime != RuntimeStatus::Active {
                return Err(TransitionError::InvalidPhase);
            }
            next.runtime = RuntimeStatus::RecoveryRequired;
            next.epistemic = EpistemicStatus::Unresolved;
            next.phase = Some(TransactionPhase::RecoveryRequired);
        }
    }
    next.sequence = state
        .sequence
        .checked_add(1)
        .ok_or(TransitionError::SequenceOverflow)?;
    Ok(next)
}
