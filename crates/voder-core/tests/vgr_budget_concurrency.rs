//! F09: bounded shared-ledger contention; synchronization belongs to the daemon.
use std::sync::{Arc, Barrier, Mutex};
use std::thread;
use voder_core::{
    budget::{BudgetAmount, BudgetError, BudgetLedger},
    types::ReservationId,
};

#[test]
fn concurrent_session_reservations_never_overspend_and_replay_is_consistent() {
    const WORKERS: usize = 48;
    const CAPACITY: usize = 32;
    let amount = BudgetAmount {
        steps: 1,
        tool_calls: 1,
        wall_time_ms: 1000,
        api_cost_microusd: 7,
        ..BudgetAmount::default()
    };
    let budget = Arc::new(Mutex::new(BudgetLedger::new(BudgetAmount {
        steps: CAPACITY as u64,
        tool_calls: CAPACITY as u64,
        wall_time_ms: (CAPACITY * 1000) as u64,
        api_cost_microusd: (CAPACITY * 7) as u64,
        ..BudgetAmount::default()
    })));
    let start = Arc::new(Barrier::new(WORKERS));
    let mut jobs = Vec::new();
    for n in 0..WORKERS {
        let budget = Arc::clone(&budget);
        let start = Arc::clone(&start);
        jobs.push(thread::spawn(move || {
            let id = ReservationId::parse(format!("session-{n}-action")).unwrap();
            start.wait();
            let mut ledger = budget.lock().unwrap();
            match ledger.reserve(id.clone(), amount) {
                Ok(_) => {
                    ledger.mark_dispatched(&id).unwrap();
                    ledger.settle(&id, amount).unwrap();
                    assert_eq!(
                        ledger.mark_dispatched(&id),
                        Err(BudgetError::IncorrectPhase)
                    );
                    true
                }
                Err(BudgetError::InsufficientResources) => false,
                Err(other) => panic!("unexpected admission result: {other:?}"),
            }
        }));
    }
    let accepted = jobs
        .into_iter()
        .map(|j| j.join().unwrap())
        .filter(|accepted| *accepted)
        .count();
    assert_eq!(accepted, CAPACITY);
    let ledger = budget.lock().unwrap();
    assert_eq!(ledger.spent().steps, CAPACITY as u64);
    assert_eq!(ledger.spent().api_cost_microusd, (CAPACITY * 7) as u64);
    assert_eq!(ledger.reserved(), BudgetAmount::default());
    assert!(ledger.available().is_zero());
    let recovered = BudgetLedger::restore(ledger.snapshot()).unwrap();
    assert_eq!(recovered.spent(), ledger.spent());
    assert_eq!(recovered.reserved(), ledger.reserved());
}
