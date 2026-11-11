use voder_core::budget::*;
use voder_core::types::ReservationId;

fn id(s: &str) -> ReservationId {
    ReservationId::parse(s).unwrap()
}
fn cost(steps: u64, tokens: u64, dollars_micros: u64) -> BudgetAmount {
    BudgetAmount {
        steps,
        tokens,
        api_cost_microusd: dollars_micros,
        ..BudgetAmount::default()
    }
}

#[test]
fn reservation_then_dispatch_then_settlement_is_conservative() {
    let mut l = BudgetLedger::new(cost(4, 1000, 100_000));
    assert_eq!(
        l.reserve(id("api1"), cost(1, 400, 30_000)).unwrap(),
        ReserveOutcome::Created
    );
    assert_eq!(l.available(), cost(3, 600, 70_000));
    l.mark_dispatched(&id("api1")).unwrap();
    l.settle(&id("api1"), cost(1, 240, 18_000)).unwrap();
    assert_eq!(l.spent(), cost(1, 240, 18_000));
    assert_eq!(l.reserved(), BudgetAmount::default());
    assert_eq!(l.available(), cost(3, 760, 82_000));
}

#[test]
fn retrying_reserve_before_dispatch_is_idempotent() {
    let mut l = BudgetLedger::new(cost(2, 600, 100));
    assert_eq!(
        l.reserve(id("r"), cost(1, 100, 10)).unwrap(),
        ReserveOutcome::Created
    );
    assert_eq!(
        l.reserve(id("r"), cost(1, 100, 10)).unwrap(),
        ReserveOutcome::AlreadyHeld
    );
    assert_eq!(l.reserved(), cost(1, 100, 10));
    assert_eq!(
        l.reserve(id("r"), cost(1, 101, 10)),
        Err(BudgetError::ReusedReservationId)
    );
}

#[test]
fn dispatch_cannot_be_repeated_as_a_second_permission_to_call() {
    let mut l = BudgetLedger::new(cost(2, 600, 100));
    l.reserve(id("r"), cost(1, 100, 10)).unwrap();
    l.mark_dispatched(&id("r")).unwrap();
    assert_eq!(
        l.mark_dispatched(&id("r")),
        Err(BudgetError::IncorrectPhase)
    );
    assert_eq!(
        l.cancel_before_dispatch(&id("r")),
        Err(BudgetError::IncorrectPhase)
    );
    assert_eq!(l.available(), cost(1, 500, 90));
}

#[test]
fn cannot_settle_without_external_call_phase() {
    let mut l = BudgetLedger::new(cost(2, 600, 100));
    l.reserve(id("r"), cost(1, 100, 10)).unwrap();
    assert_eq!(
        l.settle(&id("r"), cost(1, 10, 2)),
        Err(BudgetError::IncorrectPhase)
    );
}

#[test]
fn missing_reservation_fails_closed() {
    let mut l = BudgetLedger::new(cost(2, 600, 100));
    assert_eq!(
        l.mark_dispatched(&id("missing")),
        Err(BudgetError::UnknownReservation)
    );
    assert_eq!(
        l.settle(&id("missing"), cost(1, 1, 1)),
        Err(BudgetError::UnknownReservation)
    );
    assert_eq!(
        l.cancel_before_dispatch(&id("missing")),
        Err(BudgetError::UnknownReservation)
    );
}

#[test]
fn actual_overage_does_not_spend_or_release_reservation() {
    let mut l = BudgetLedger::new(cost(2, 600, 100));
    l.reserve(id("r"), cost(1, 100, 10)).unwrap();
    l.mark_dispatched(&id("r")).unwrap();
    let before = l.snapshot();
    assert_eq!(
        l.settle(&id("r"), cost(1, 101, 10)),
        Err(BudgetError::ActualExceedsReservation)
    );
    assert_eq!(l.snapshot(), before);
}

#[test]
fn cannot_reuse_settled_or_cancelled_request_id() {
    let mut l = BudgetLedger::new(cost(5, 500, 100));
    l.reserve(id("r"), cost(1, 100, 10)).unwrap();
    l.mark_dispatched(&id("r")).unwrap();
    l.settle(&id("r"), cost(1, 90, 8)).unwrap();
    assert_eq!(
        l.reserve(id("r"), cost(1, 100, 10)),
        Err(BudgetError::ReusedReservationId)
    );
    l.reserve(id("s"), cost(1, 100, 10)).unwrap();
    l.cancel_before_dispatch(&id("s")).unwrap();
    assert_eq!(
        l.reserve(id("s"), cost(1, 100, 10)),
        Err(BudgetError::ReusedReservationId)
    );
}

#[test]
fn fail_without_partial_state_mutation_when_out_of_budget() {
    let mut l = BudgetLedger::new(cost(1, 200, 100));
    let before = l.snapshot();
    assert_eq!(
        l.reserve(id("big"), cost(2, 100, 20)),
        Err(BudgetError::InsufficientResources)
    );
    assert_eq!(l.snapshot(), before);
    assert_eq!(
        l.reserve(id("zero"), BudgetAmount::default()),
        Err(BudgetError::ZeroReservation)
    );
}

#[test]
fn all_dimensions_are_hard_limits() {
    let budget = BudgetAmount {
        steps: 2,
        tokens: 200,
        model_calls: 2,
        tool_calls: 2,
        wall_time_ms: 500,
        risk_microunits: 50,
        context_tokens: 80,
        api_cost_microusd: 1_000,
    };
    let mut l = BudgetLedger::new(budget);
    let fields = [
        BudgetAmount {
            steps: 3,
            ..BudgetAmount::default()
        },
        BudgetAmount {
            tokens: 201,
            ..BudgetAmount::default()
        },
        BudgetAmount {
            model_calls: 3,
            ..BudgetAmount::default()
        },
        BudgetAmount {
            tool_calls: 3,
            ..BudgetAmount::default()
        },
        BudgetAmount {
            wall_time_ms: 501,
            ..BudgetAmount::default()
        },
        BudgetAmount {
            risk_microunits: 51,
            ..BudgetAmount::default()
        },
        BudgetAmount {
            context_tokens: 81,
            ..BudgetAmount::default()
        },
        BudgetAmount {
            api_cost_microusd: 1001,
            ..BudgetAmount::default()
        },
    ];
    for (i, value) in fields.into_iter().enumerate() {
        assert_eq!(
            l.reserve(id(&format!("probe{i}")), value),
            Err(BudgetError::InsufficientResources)
        );
        assert_eq!(l.available(), budget);
    }
}

#[test]
fn snapshot_restore_keeps_dispatched_hold_until_explicit_reconciliation() {
    let mut l = BudgetLedger::new(cost(3, 500, 100));
    l.reserve(id("api"), cost(1, 200, 50)).unwrap();
    l.mark_dispatched(&id("api")).unwrap();
    let snap = l.snapshot();
    let mut restored = BudgetLedger::restore(snap.clone()).unwrap();
    assert_eq!(restored.snapshot(), snap);
    assert_eq!(
        restored.cancel_before_dispatch(&id("api")),
        Err(BudgetError::IncorrectPhase)
    );
    assert_eq!(
        restored.mark_dispatched(&id("api")),
        Err(BudgetError::IncorrectPhase)
    );
    restored.settle(&id("api"), cost(1, 190, 45)).unwrap();
    assert_eq!(restored.available(), cost(2, 310, 55));
}

#[test]
fn corrupt_snapshot_cannot_be_loaded() {
    let mut l = BudgetLedger::new(cost(3, 500, 100));
    l.reserve(id("r"), cost(1, 200, 50)).unwrap();
    let mut snapshot = l.snapshot();
    snapshot.reserved.tokens = 1;
    assert!(matches!(
        BudgetLedger::restore(snapshot),
        Err(BudgetError::InvalidSnapshot)
    ));
    let mut snapshot = l.snapshot();
    snapshot.records.push(snapshot.records[0].clone());
    assert!(matches!(
        BudgetLedger::restore(snapshot),
        Err(BudgetError::InvalidSnapshot)
    ));
    let mut snapshot = l.snapshot();
    snapshot.records[0].actual = Some(cost(1, 10, 5));
    assert!(matches!(
        BudgetLedger::restore(snapshot),
        Err(BudgetError::InvalidSnapshot)
    ));
}

#[test]
fn checked_arithmetic_refuses_overflow_or_underflow() {
    assert!(cost(u64::MAX, 0, 0).checked_add(cost(1, 0, 0)).is_none());
    assert!(cost(0, 0, 0).checked_sub(cost(1, 0, 0)).is_none());
    assert_eq!(
        cost(1, 2, 3).checked_add(cost(4, 5, 6)),
        Some(cost(5, 7, 9))
    );
}

#[test]
fn sequential_multi_request_budget_never_exceeds_total() {
    let mut l = BudgetLedger::new(cost(200, 20_000, 20_000));
    let mut seed = 0xA5A5_1234_u64;
    for index in 0..400 {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        let upper = cost(1, 10 + (seed % 20), 1 + (seed % 10));
        let key = id(&format!("r{index}"));
        if l.reserve(key.clone(), upper).is_ok() {
            if index % 5 == 0 {
                l.cancel_before_dispatch(&key).unwrap();
            } else {
                l.mark_dispatched(&key).unwrap();
                l.settle(&key, cost(1, upper.tokens - 1, upper.api_cost_microusd))
                    .unwrap();
            }
        }
        assert!(l
            .spent()
            .checked_add(l.reserved())
            .unwrap()
            .fits_within(l.total()));
        assert!(BudgetLedger::restore(l.snapshot()).is_ok());
    }
}
