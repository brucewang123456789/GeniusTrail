//! Choice before compute: hard authorization gates BEFORE bounded heuristic ranking.
//!
//! Selection is advisory, never a dispatch token. Only action_commit can prepare;
//! only the future single-writer daemon can journal and dispatch. Estimated utility
//! is NOT evidence and is never allowed to override a hard rejection.
use crate::action_commit::{ActionIntent, CommitError, CommitGate};
use crate::budget::{BudgetAmount, BudgetError, BudgetLedger};
use crate::contract::{FrozenContract, MissingInput};
use crate::evidence::EvidenceLedger;
use crate::obligation::{ObligationError, ObligationRegistry};
use crate::retry::{RetryError, RetryLedger};
use crate::types::InputId;
use crate::types::{ActionId, ObligationStatus, RuntimeStatus, WorkspaceIdentity};
use std::collections::BTreeSet;
use std::num::NonZeroU32;

pub const MAX_CANDIDATES: usize = 128;

/// All eight policy heuristics are normalized to [0,100]. These are ranking
/// hints only, NOT authorization, validated test outcomes, or measured savings.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ChoiceEstimate {
    pub closure: u8,
    pub information: u8,
    pub progress: u8,
    pub reversibility: u8,
    pub token_cost: u8,
    pub latency_cost: u8,
    pub risk: u8,
    pub context_cost: u8,
}
impl ChoiceEstimate {
    pub fn valid(self) -> bool {
        [
            self.closure,
            self.information,
            self.progress,
            self.reversibility,
            self.token_cost,
            self.latency_cost,
            self.risk,
            self.context_cost,
        ]
        .iter()
        .all(|&x| x <= 100)
    }
}

/// Immutable controller-defined weights; providers never choose these weights.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ChoicePolicy {
    pub closure: u16,
    pub information: u16,
    pub progress: u16,
    pub reversibility: u16,
    pub token_penalty: u16,
    pub latency_penalty: u16,
    pub risk_penalty: u16,
    pub context_penalty: u16,
    pub min_utility: i64,
}
impl Default for ChoicePolicy {
    fn default() -> Self {
        Self {
            closure: 8,
            information: 6,
            progress: 4,
            reversibility: 3,
            token_penalty: 4,
            latency_penalty: 2,
            risk_penalty: 6,
            context_penalty: 2,
            min_utility: 0,
        }
    }
}
impl ChoicePolicy {
    fn valid(self) -> bool {
        let all = [
            self.closure,
            self.information,
            self.progress,
            self.reversibility,
            self.token_penalty,
            self.latency_penalty,
            self.risk_penalty,
            self.context_penalty,
        ];
        all.iter().all(|&x| x <= 1000) && all.iter().any(|&x| x > 0) && self.min_utility >= 0
    }
    fn utility(self, e: ChoiceEstimate) -> i64 {
        let gain = i64::from(self.closure) * i64::from(e.closure)
            + i64::from(self.information) * i64::from(e.information)
            + i64::from(self.progress) * i64::from(e.progress)
            + i64::from(self.reversibility) * i64::from(e.reversibility);
        let cost = i64::from(self.token_penalty) * i64::from(e.token_cost)
            + i64::from(self.latency_penalty) * i64::from(e.latency_cost)
            + i64::from(self.risk_penalty) * i64::from(e.risk)
            + i64::from(self.context_penalty) * i64::from(e.context_cost);
        gain - cost
    }
}

#[derive(Debug, Clone)]
pub struct DecisionCandidate {
    pub intent: ActionIntent,
    pub estimate: ChoiceEstimate,
    /// Upper-bound estimate. The real reservation is made ONLY after selection.
    pub max_cost: BudgetAmount,
    pub sample_ordinal: Option<NonZeroU32>,
}
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CandidateRejection {
    InvalidEstimate,
    ReservationIdInUse,
    Budget(BudgetError),
    TargetClosedOrBlocked,
    DependencyNotClosed,
    NoSuchObligation,
    Contract(CommitError),
    Retry(RetryError),
    BelowUtilityThreshold,
    NoExpectedBenefit,
}
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DecisionError {
    NotActive,
    MissingInputs(Vec<MissingInput>),
    InvalidPolicy,
    TooManyCandidates,
    DuplicateActionIdentity,
    Obligation(ObligationError),
}
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RankedAction {
    pub index: usize,
    pub action_id: ActionId,
    pub utility: i64,
}
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DecisionPlan {
    /// No selection means stop or ask for new information, NOT silently rerun.
    pub selected: Option<RankedAction>,
    pub rejected: Vec<(ActionId, CandidateRejection)>,
}

/// Side-effect-free preflight. `EvidenceLedger` supplies relevant facts rather
/// than model-provided evidence; the retry check is performed on a clone.
/// A selected candidate MUST still pass the durable commit/dispatch gates.
#[allow(clippy::too_many_arguments)]
pub fn select_next_action(
    contract: &FrozenContract,
    workspace: &WorkspaceIdentity,
    runtime: RuntimeStatus,
    verified_inputs: &[InputId],
    budget: &BudgetLedger,
    retries: &RetryLedger,
    obligations: &ObligationRegistry,
    evidence: &EvidenceLedger,
    candidates: &[DecisionCandidate],
    policy: ChoicePolicy,
) -> Result<DecisionPlan, DecisionError> {
    if runtime != RuntimeStatus::Active {
        return Err(DecisionError::NotActive);
    }
    if !policy.valid() {
        return Err(DecisionError::InvalidPolicy);
    }
    if candidates.len() > MAX_CANDIDATES {
        return Err(DecisionError::TooManyCandidates);
    }
    let missing = contract.missing_inputs(verified_inputs);
    if !missing.is_empty() {
        return Err(DecisionError::MissingInputs(missing));
    }
    // Check the contract binding even when there are no candidates.
    obligations
        .highest_priority_open(contract, workspace)
        .map_err(DecisionError::Obligation)?;
    let mut ids = BTreeSet::new();
    if candidates
        .iter()
        .any(|c| !ids.insert((c.intent.run_id.clone(), c.intent.action_id.clone())))
    {
        return Err(DecisionError::DuplicateActionIdentity);
    }
    let facts = evidence.relevant_facts();
    let mut plan = DecisionPlan {
        selected: None,
        rejected: Vec::new(),
    };
    for (index, c) in candidates.iter().enumerate() {
        let id = c.intent.action_id.clone();
        let fail = (|| -> Result<i64, CandidateRejection> {
            if !c.estimate.valid() {
                return Err(CandidateRejection::InvalidEstimate);
            }
            // Explore on a shadow ledger; never hold resources for rejected
            // alternatives. This is the literal Choice-before-Compute boundary.
            if budget.record(&c.intent.reservation_id).is_some() {
                return Err(CandidateRejection::ReservationIdInUse);
            }
            if !c.max_cost.fits_within(budget.available()) {
                return Err(CandidateRejection::Budget(
                    BudgetError::InsufficientResources,
                ));
            }
            // One tiny shadow ledger per candidate; no clone of all historic
            // reservations and no expensive real budget mutation at ranking time.
            let mut shadow = BudgetLedger::new(c.max_cost);
            shadow
                .reserve(c.intent.reservation_id.clone(), c.max_cost)
                .map_err(CandidateRejection::Budget)?;
            if let Some(target) = &c.intent.descriptor.target_obligation_id {
                let spec = obligations
                    .spec(target)
                    .ok_or(CandidateRejection::NoSuchObligation)?;
                if obligations
                    .status(contract, workspace, target)
                    .map_err(|_| CandidateRejection::NoSuchObligation)?
                    != ObligationStatus::Open
                {
                    return Err(CandidateRejection::TargetClosedOrBlocked);
                }
                for dep in &spec.dependencies {
                    if obligations
                        .status(contract, workspace, dep)
                        .map_err(|_| CandidateRejection::DependencyNotClosed)?
                        != ObligationStatus::Closed
                    {
                        return Err(CandidateRejection::DependencyNotClosed);
                    }
                }
            }
            // Do not allow a write's target to be absent or imply READ by a model claim:
            // CommitGate resolves actual tool effects from the frozen contract.
            let mut gate = CommitGate::new();
            gate.prepare(
                contract,
                workspace,
                verified_inputs,
                &shadow,
                c.intent.clone(),
                None,
            )
            .map_err(CandidateRejection::Contract)?;
            // Recheck in side-effect-free cloned ledger; actual attempt registration
            // happens only at authoritative durable dispatch.
            let mut probe = retries.clone();
            probe
                .try_record(
                    contract,
                    workspace,
                    &c.intent.descriptor,
                    &facts,
                    c.sample_ordinal,
                )
                .map_err(CandidateRejection::Retry)?;
            // Reversibility is a safety property, not useful progress by itself.
            // Reject no-benefit actions before heuristic scoring can reward them.
            if c.estimate.closure == 0 && c.estimate.information == 0 && c.estimate.progress == 0 {
                return Err(CandidateRejection::NoExpectedBenefit);
            }
            let score = policy.utility(c.estimate);
            if score < policy.min_utility {
                return Err(CandidateRejection::BelowUtilityThreshold);
            }
            Ok(score)
        })();
        match fail {
            Ok(utility) => {
                let candidate = RankedAction {
                    index,
                    action_id: id,
                    utility,
                };
                let better = match &plan.selected {
                    None => true,
                    Some(best) => {
                        utility > best.utility
                            || (utility == best.utility && candidate.action_id < best.action_id)
                    }
                };
                if better {
                    plan.selected = Some(candidate);
                }
            }
            Err(reason) => plan.rejected.push((id, reason)),
        }
    }
    Ok(plan)
}
