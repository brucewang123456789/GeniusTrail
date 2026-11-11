mod stage5_support;
use stage5_support::*;
use voder_core::budget::BudgetAmount;
use voder_core::decision::*;
use voder_core::retry::RetryError;
use voder_core::types::{InputId, RuntimeStatus};

fn run(
    candidates: &[DecisionCandidate],
    present: &[InputId],
    file: bool,
    approval: bool,
) -> Result<DecisionPlan, DecisionError> {
    let c = contract(file, approval);
    let (ob, b, r, e) = engine(&c);
    select_next_action(
        &c,
        &world(),
        RuntimeStatus::Active,
        present,
        &b,
        &r,
        &ob,
        &e,
        candidates,
        ChoicePolicy::default(),
    )
}
#[test]
fn cheapest_inspection_eligible_without_reserving_anything() {
    let c = contract(false, false);
    let (ob, b, r, e) = engine(&c);
    let x = candidate(&c, "read1", false);
    let result = select_next_action(
        &c,
        &world(),
        RuntimeStatus::Active,
        &[],
        &b,
        &r,
        &ob,
        &e,
        &[x],
        ChoicePolicy::default(),
    )
    .unwrap();
    assert_eq!(result.selected.unwrap().index, 0);
    assert_eq!(b.reserved(), BudgetAmount::default());
}
#[test]
fn missing_files_block_before_model_or_tool_action() {
    let c = contract(true, false);
    let x = candidate(&c, "act", false);
    assert!(matches!(
        run(&[x], &[], true, false),
        Err(DecisionError::MissingInputs(_))
    ));
}
#[test]
fn verified_file_satisfies_missing_input_gate() {
    let c = contract(true, false);
    let x = candidate(&c, "act", false);
    assert!(run(&[x], &[id("file")], true, false)
        .unwrap()
        .selected
        .is_some());
}
#[test]
fn optimistic_score_cannot_escape_read_scope() {
    let c = contract(false, false);
    let mut x = candidate(&c, "act", false);
    x.intent.descriptor.scope = voder_core::contract::RepoScope::parse("private").unwrap();
    x.estimate = choice_value(100);
    let r = run(&[x], &[], false, false).unwrap();
    assert!(r.selected.is_none());
    assert_eq!(r.rejected.len(), 1);
}
#[test]
fn unauthorized_write_cannot_win_any_ranking() {
    let c = contract(false, true);
    let x = candidate(&c, "write1", true);
    let result = run(&[x], &[], false, true).unwrap();
    assert!(result.selected.is_none());
    assert!(matches!(
        result.rejected[0].1,
        CandidateRejection::Contract(_)
    ));
}
#[test]
fn low_value_candidate_loses_to_information_gain() {
    let c = contract(false, false);
    let mut a = candidate(&c, "a", false);
    let mut b = candidate(&c, "b", false);
    a.estimate = choice_value(10);
    b.estimate = choice_value(90);
    let r = run(&[a, b], &[], false, false).unwrap();
    assert_eq!(r.selected.unwrap().action_id, id("b"));
}
#[test]
fn equal_scoring_actions_have_deterministic_tie_break() {
    let c = contract(false, false);
    let a = candidate(&c, "z", false);
    let b = candidate(&c, "a", false);
    assert_eq!(
        run(&[a, b], &[], false, false)
            .unwrap()
            .selected
            .unwrap()
            .action_id,
        id("a")
    );
}
#[test]
fn duplicate_action_ids_are_rejected_globally() {
    let c = contract(false, false);
    let x = candidate(&c, "same", false);
    assert_eq!(
        run(&[x.clone(), x], &[], false, false),
        Err(DecisionError::DuplicateActionIdentity)
    );
}
#[test]
fn over_budget_cannot_win_with_max_value() {
    let c = contract(false, false);
    let mut x = candidate(&c, "large", false);
    x.estimate = choice_value(100);
    x.max_cost.tokens = 100_001;
    let r = run(&[x], &[], false, false).unwrap();
    assert!(r.selected.is_none());
    assert!(matches!(r.rejected[0].1, CandidateRejection::Budget(_)));
}
#[test]
fn invalid_normalized_score_fails_closed() {
    let c = contract(false, false);
    let mut x = candidate(&c, "badscore", false);
    x.estimate.risk = 101;
    let r = run(&[x], &[], false, false).unwrap();
    assert_eq!(r.rejected[0].1, CandidateRejection::InvalidEstimate);
}
#[test]
fn no_candidates_does_not_mean_task_completed() {
    let r = run(&[], &[], false, false).unwrap();
    assert_eq!(r.selected, None);
}
#[test]
fn threshold_cannot_turn_negative_utility_into_execution() {
    let c = contract(false, false);
    let mut x = candidate(&c, "low", false);
    // Nonzero estimated information makes this a candidate with *real* value,
    // but its explicit costs outweigh that small value: utility = 6 - 1400.
    x.estimate = ChoiceEstimate {
        closure: 0,
        information: 1,
        progress: 0,
        reversibility: 0,
        token_cost: 100,
        latency_cost: 100,
        risk: 100,
        context_cost: 100,
    };
    let r = run(&[x], &[], false, false).unwrap();
    assert!(r.selected.is_none());
    assert_eq!(r.rejected[0].1, CandidateRejection::BelowUtilityThreshold);
}
#[test]
fn reversibility_alone_is_not_a_reason_to_execute() {
    let c = contract(false, false);
    let mut x = candidate(&c, "no-benefit", false);
    // Before this fix, zero gain + 70 reversibility yielded utility +140.
    x.estimate = choice_value(0);
    let r = run(&[x], &[], false, false).unwrap();
    assert!(r.selected.is_none());
    assert_eq!(r.rejected[0].1, CandidateRejection::NoExpectedBenefit);
}
#[test]
fn negative_minimum_utility_policy_is_invalid() {
    let c = contract(false, false);
    let x = candidate(&c, "x", false);
    let (ob, b, r, e) = engine(&c);
    let policy = ChoicePolicy {
        min_utility: -1,
        ..ChoicePolicy::default()
    };
    assert_eq!(
        select_next_action(
            &c,
            &world(),
            RuntimeStatus::Active,
            &[],
            &b,
            &r,
            &ob,
            &e,
            &[x],
            policy
        ),
        Err(DecisionError::InvalidPolicy)
    );
}
#[test]
fn inactive_status_blocks_all_selections() {
    let c = contract(false, false);
    let x = candidate(&c, "a", false);
    let (ob, b, r, e) = engine(&c);
    assert_eq!(
        select_next_action(
            &c,
            &world(),
            RuntimeStatus::RecoveryRequired,
            &[],
            &b,
            &r,
            &ob,
            &e,
            &[x],
            ChoicePolicy::default()
        ),
        Err(DecisionError::NotActive)
    );
}
#[test]
fn reject_more_than_max_candidates_before_allocating() {
    let c = contract(false, false);
    let (ob, b, r, e) = engine(&c);
    let xs = (0..=MAX_CANDIDATES)
        .map(|_| candidate(&c, "duplicate", false))
        .collect::<Vec<_>>();
    assert_eq!(
        select_next_action(
            &c,
            &world(),
            RuntimeStatus::Active,
            &[],
            &b,
            &r,
            &ob,
            &e,
            &xs,
            ChoicePolicy::default()
        ),
        Err(DecisionError::TooManyCandidates)
    );
}
#[test]
fn unbound_nonempty_write_expected_effect_is_still_a_hard_gate() {
    let c = contract(false, false);
    let mut x = candidate(&c, "write", true);
    x.intent.expected_changes.clear();
    x.estimate = choice_value(100);
    let r = run(&[x], &[], false, false).unwrap();
    assert!(matches!(r.rejected[0].1, CandidateRejection::Contract(_)));
}
#[test]
fn retry_gate_is_read_only_and_rejects_repeat_without_new_evidence() {
    let c = contract(false, false);
    let (ob, b, mut retry, e) = engine(&c);
    let x = candidate(&c, "repeat", false);
    retry
        .try_record(&c, &world(), &x.intent.descriptor, &[], None)
        .unwrap();
    let before = retry.records().len();
    let p = select_next_action(
        &c,
        &world(),
        RuntimeStatus::Active,
        &[],
        &b,
        &retry,
        &ob,
        &e,
        &[x],
        ChoicePolicy::default(),
    )
    .unwrap();
    assert!(matches!(
        p.rejected[0].1,
        CandidateRejection::Retry(RetryError::UnchangedAction)
    ));
    assert_eq!(retry.records().len(), before);
}
#[test]
fn invalid_policy_is_not_interpreted_as_better_ranking() {
    let c = contract(false, false);
    let x = candidate(&c, "x", false);
    let (ob, b, r, e) = engine(&c);
    let weights = ChoicePolicy {
        closure: 1001,
        ..ChoicePolicy::default()
    };
    assert_eq!(
        select_next_action(
            &c,
            &world(),
            RuntimeStatus::Active,
            &[],
            &b,
            &r,
            &ob,
            &e,
            &[x],
            weights
        ),
        Err(DecisionError::InvalidPolicy)
    );
}
