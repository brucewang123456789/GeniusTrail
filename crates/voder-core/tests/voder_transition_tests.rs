mod stage5_support;
use stage5_support::*;
use voder_core::action_commit::{CommitGate, PreparedAction};
use voder_core::consequence::{ExecutionCompletion, ExecutionReport, ObservedChange};
use voder_core::transition::*;
use voder_core::types::*;

fn fixture() -> (ControlState, PreparedAction, ExecutionReport) {
    let c = contract(false, false);
    let w = world();
    let action = candidate(&c, "write1", true);
    let mut b = budget();
    b.reserve(action.intent.reservation_id.clone(), cost())
        .unwrap();
    let prepared = CommitGate::new()
        .prepare(&c, &w, &[], &b, action.intent.clone(), None)
        .unwrap();
    let after = WorkspaceIdentity {
        revision: id("r2"),
        digest: id("w2"),
    };
    let report = ExecutionReport {
        run_id: id("run"),
        action_id: id("write1"),
        contract: c.identity().clone(),
        before: w.clone(),
        after,
        completion: ExecutionCompletion::Completed,
        observed_changes: vec![ObservedChange {
            path: prepared.intent().expected_changes[0].path.clone(),
            kind: prepared.intent().expected_changes[0].kind,
        }],
        forbidden_side_effects: vec![],
        receipt_digest: id("receipt"),
    };
    (
        ControlState::new(id("run"), c.identity().clone(), w),
        prepared,
        report,
    )
}
fn executing() -> (ControlState, PreparedAction, ExecutionReport) {
    let (s, p, r) = fixture();
    let s = plan_transition(&s, TransitionRequest::Prepare(&p)).unwrap();
    let s = plan_transition(
        &s,
        TransitionRequest::Dispatch {
            action_id: &p.intent().action_id,
        },
    )
    .unwrap();
    (s, p, r)
}
fn observed() -> (ControlState, PreparedAction, ExecutionReport) {
    let (s, p, r) = executing();
    let s = plan_transition(
        &s,
        TransitionRequest::Observe {
            action_id: &p.intent().action_id,
        },
    )
    .unwrap();
    (s, p, r)
}
#[test]
fn initial_state_is_active_without_pending_transaction() {
    let (s, _, _) = fixture();
    assert_eq!(s.runtime(), RuntimeStatus::Active);
    assert_eq!(s.phase(), None);
    assert_eq!(s.sequence(), 0);
}
#[test]
fn prepare_requires_valid_prepared_action() {
    let (s, p, _) = fixture();
    let next = plan_transition(&s, TransitionRequest::Prepare(&p)).unwrap();
    assert_eq!(next.phase(), Some(TransactionPhase::Prepared));
    assert_eq!(next.sequence(), 1);
    assert_eq!(s.phase(), None);
}
#[test]
fn dispatch_without_preparation_rejected() {
    let (s, p, _) = fixture();
    assert_eq!(
        plan_transition(
            &s,
            TransitionRequest::Dispatch {
                action_id: &p.intent().action_id
            }
        ),
        Err(TransitionError::InvalidPhase)
    );
}
#[test]
fn observe_without_dispatch_rejected() {
    let (s, p, _) = fixture();
    assert_eq!(
        plan_transition(
            &s,
            TransitionRequest::Observe {
                action_id: &p.intent().action_id
            }
        ),
        Err(TransitionError::InvalidPhase)
    );
}
#[test]
fn second_preparation_without_adjudication_rejected() {
    let (s, p, _) = fixture();
    let next = plan_transition(&s, TransitionRequest::Prepare(&p)).unwrap();
    assert_eq!(
        plan_transition(&next, TransitionRequest::Prepare(&p)),
        Err(TransitionError::InvalidPhase)
    );
}
#[test]
fn wrong_action_identity_cannot_dispatch() {
    let (s, p, _) = fixture();
    let s = plan_transition(&s, TransitionRequest::Prepare(&p)).unwrap();
    assert_eq!(
        plan_transition(
            &s,
            TransitionRequest::Dispatch {
                action_id: &id("wrong")
            }
        ),
        Err(TransitionError::WrongAction)
    );
}
#[test]
fn cannot_adjudicate_before_observation() {
    let (s, p, r) = executing();
    assert_eq!(
        plan_transition(
            &s,
            TransitionRequest::Adjudicate {
                prepared: &p,
                report: &r
            }
        ),
        Err(TransitionError::InvalidPhase)
    );
}
#[test]
fn verified_shaped_match_moves_to_current_workspace_but_never_completes_run() {
    let (s, p, r) = observed();
    let next = plan_transition(
        &s,
        TransitionRequest::Adjudicate {
            prepared: &p,
            report: &r,
        },
    )
    .unwrap();
    assert_eq!(next.workspace(), &r.after);
    assert_eq!(next.runtime(), RuntimeStatus::Active);
    assert_eq!(next.phase(), None);
    assert_eq!(next.sequence(), 4);
}
#[test]
fn failed_write_with_partial_mutation_safestops_in_recovery_phase() {
    let (s, p, mut r) = observed();
    r.completion = ExecutionCompletion::Failed;
    let next = plan_transition(
        &s,
        TransitionRequest::Adjudicate {
            prepared: &p,
            report: &r,
        },
    )
    .unwrap();
    assert_eq!(next.runtime(), RuntimeStatus::SafeStopped);
    assert_eq!(next.phase(), Some(TransactionPhase::RecoveryRequired));
    assert_eq!(next.workspace(), s.workspace());
}
#[test]
fn unauthorized_report_from_other_action_rejected() {
    let (s, p, mut r) = observed();
    r.action_id = id("other");
    assert_eq!(
        plan_transition(
            &s,
            TransitionRequest::Adjudicate {
                prepared: &p,
                report: &r
            }
        ),
        Err(TransitionError::WrongAction)
    );
}
#[test]
fn unauthorized_report_from_other_contract_rejected() {
    let (s, p, mut r) = observed();
    r.contract.digest = id("altered");
    assert_eq!(
        plan_transition(
            &s,
            TransitionRequest::Adjudicate {
                prepared: &p,
                report: &r
            }
        ),
        Err(TransitionError::WrongContract)
    );
}
#[test]
fn stale_report_before_workspace_rejected() {
    let (s, p, mut r) = observed();
    r.before.digest = id("altered");
    assert_eq!(
        plan_transition(
            &s,
            TransitionRequest::Adjudicate {
                prepared: &p,
                report: &r
            }
        ),
        Err(TransitionError::WrongWorkspace)
    );
}
#[test]
fn prepare_cross_run_rejected() {
    let (s, p, _) = fixture();
    let new = ControlState::new(
        id("other"),
        p.intent().contract.clone(),
        s.workspace().clone(),
    );
    assert_eq!(
        plan_transition(&new, TransitionRequest::Prepare(&p)),
        Err(TransitionError::WrongRun)
    );
}
#[test]
fn stopping_while_an_action_is_pending_rejected() {
    let (s, p, _) = fixture();
    let s = plan_transition(&s, TransitionRequest::Prepare(&p)).unwrap();
    assert_eq!(
        plan_transition(&s, TransitionRequest::Stop(TerminalReason::MissingInput)),
        Err(TransitionError::InvalidPhase)
    );
}
#[test]
fn user_missing_input_stops_without_false_success() {
    let (s, _, _) = fixture();
    let next = plan_transition(&s, TransitionRequest::Stop(TerminalReason::MissingInput)).unwrap();
    assert_eq!(next.runtime(), RuntimeStatus::Blocked);
    assert_eq!(next.epistemic(), EpistemicStatus::Unresolved);
}
#[test]
fn recovery_request_during_execution_fails_closed() {
    let (s, _, _) = executing();
    let next = plan_transition(&s, TransitionRequest::RequireRecovery).unwrap();
    assert_eq!(next.runtime(), RuntimeStatus::RecoveryRequired);
    assert_eq!(next.phase(), Some(TransactionPhase::RecoveryRequired));
    assert_eq!(
        plan_transition(
            &next,
            TransitionRequest::Stop(TerminalReason::NoEligibleAction)
        ),
        Err(TransitionError::InvalidPhase)
    );
}
#[test]
fn recovery_request_without_pending_transaction_is_invalid() {
    let (s, _, _) = fixture();
    assert_eq!(
        plan_transition(&s, TransitionRequest::RequireRecovery),
        Err(TransitionError::InvalidPhase)
    );
}
#[test]
fn no_public_recovery_resume_or_complete_event_exists_by_design() {
    // Compile-time API surface deliberately exposes no Reconciled/MarkCompleted;
    // this runtime check verifies recovery is terminal until trusted VEP exists.
    let (s, _, _) = executing();
    let s = plan_transition(&s, TransitionRequest::RequireRecovery).unwrap();
    assert_eq!(
        plan_transition(&s, TransitionRequest::RequireRecovery),
        Err(TransitionError::InvalidPhase)
    );
}
#[test]
fn read_only_initial_state_is_not_considered_delivered() {
    let (s, _, _) = fixture();
    assert_ne!(s.runtime(), RuntimeStatus::Completed);
}
