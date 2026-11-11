mod stage5_support;
use stage5_support::*;
use voder_core::evidence::EvidenceLedger;
use voder_core::types::*;
use voder_core::verdict::*;

fn check(runtime: RuntimeStatus, phase: Option<TransactionPhase>) -> DeliveryAssessment {
    let c = contract(false, false);
    let o = obligations(&c);
    let e = EvidenceLedger::new();
    evaluate_delivery(DeliveryContext {
        contract: &c,
        workspace: &world(),
        run: &id("run"),
        runtime,
        pending_transaction: phase,
        obligations: &o,
        admitted: &e,
        evidence_for_hard_requirements: &[],
    })
    .unwrap()
}
#[test]
fn unresolved_critical_obligation_blocks_delivery() {
    assert_eq!(
        check(RuntimeStatus::Active, None),
        DeliveryAssessment::Blocked(DeliveryBlocker::HardObligationOpen)
    );
}
#[test]
fn completed_status_alone_cannot_override_obligations() {
    assert_eq!(
        check(RuntimeStatus::Completed, None),
        DeliveryAssessment::Blocked(DeliveryBlocker::HardObligationOpen)
    );
}
#[test]
fn block_when_validator_call_is_still_executing() {
    assert_eq!(
        check(RuntimeStatus::Active, Some(TransactionPhase::Executing)),
        DeliveryAssessment::Blocked(DeliveryBlocker::PendingTransaction)
    );
}
#[test]
fn block_when_observation_not_yet_adjudicated() {
    assert_eq!(
        check(RuntimeStatus::Active, Some(TransactionPhase::Observed)),
        DeliveryAssessment::Blocked(DeliveryBlocker::PendingTransaction)
    );
}
#[test]
fn block_when_action_not_yet_dispatched() {
    assert_eq!(
        check(RuntimeStatus::Active, Some(TransactionPhase::Prepared)),
        DeliveryAssessment::Blocked(DeliveryBlocker::PendingTransaction)
    );
}
#[test]
fn uncertain_recovery_cannot_deliver() {
    assert_eq!(
        check(RuntimeStatus::RecoveryRequired, None),
        DeliveryAssessment::Blocked(DeliveryBlocker::RecoveryRequired)
    );
}
#[test]
fn recovered_status_cannot_override_pending_uncertain_transaction() {
    assert_eq!(
        check(
            RuntimeStatus::Active,
            Some(TransactionPhase::RecoveryRequired)
        ),
        DeliveryAssessment::Blocked(DeliveryBlocker::RecoveryRequired)
    );
}
#[test]
fn rolled_back_is_not_delivery() {
    assert_eq!(
        check(RuntimeStatus::RolledBack, None),
        DeliveryAssessment::RollbackRequired
    );
}
#[test]
fn blocked_runtime_is_not_delivery() {
    assert_eq!(
        check(RuntimeStatus::Blocked, None),
        DeliveryAssessment::Blocked(DeliveryBlocker::UnsafeRuntime)
    );
}
#[test]
fn exhausted_budget_does_not_imply_success() {
    assert_eq!(
        check(RuntimeStatus::Exhausted, None),
        DeliveryAssessment::Blocked(DeliveryBlocker::UnsafeRuntime)
    );
}
#[test]
fn unsafe_stop_is_not_delivery() {
    assert_eq!(
        check(RuntimeStatus::SafeStopped, None),
        DeliveryAssessment::Blocked(DeliveryBlocker::UnsafeRuntime)
    );
}
#[test]
fn fabricated_requirement_evidence_pair_does_not_close_unresolved_obligation() {
    let c = contract(false, false);
    let o = obligations(&c);
    let e = EvidenceLedger::new();
    let pairs = vec![(id("req"), id("forged"))];
    assert_eq!(
        evaluate_delivery(DeliveryContext {
            contract: &c,
            workspace: &world(),
            run: &id("run"),
            runtime: RuntimeStatus::Active,
            pending_transaction: None,
            obligations: &o,
            admitted: &e,
            evidence_for_hard_requirements: &pairs,
        })
        .unwrap(),
        DeliveryAssessment::Blocked(DeliveryBlocker::HardObligationOpen)
    );
}
