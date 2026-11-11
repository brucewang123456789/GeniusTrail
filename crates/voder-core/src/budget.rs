//! Bounded, idempotent resource reservation for model and tool actions.
//!
//! Pure state only: the future daemon must durably journal reservation and
//! dispatch transitions BEFORE making an external call. Timeouts, process
//! limits, and cost reconciliation require runtime enforcement.

use crate::types::ReservationId;
use std::collections::{BTreeMap, BTreeSet};

/// All resource amounts use unsigned integers, never floating point.
/// `risk_microunits` is a policy score (NOT a probability).
/// `api_cost_microusd` represents estimated/charged USD in units of 10^-6 USD.
/// `context_tokens` counts active context token budget.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct BudgetAmount {
    pub steps: u64,
    pub tokens: u64,
    pub model_calls: u64,
    pub tool_calls: u64,
    pub wall_time_ms: u64,
    pub risk_microunits: u64,
    pub context_tokens: u64,
    pub api_cost_microusd: u64,
}

impl BudgetAmount {
    pub fn is_zero(self) -> bool {
        self == Self::default()
    }
    pub fn fits_within(self, limit: Self) -> bool {
        self.steps <= limit.steps
            && self.tokens <= limit.tokens
            && self.model_calls <= limit.model_calls
            && self.tool_calls <= limit.tool_calls
            && self.wall_time_ms <= limit.wall_time_ms
            && self.risk_microunits <= limit.risk_microunits
            && self.context_tokens <= limit.context_tokens
            && self.api_cost_microusd <= limit.api_cost_microusd
    }
    pub fn checked_add(self, b: Self) -> Option<Self> {
        Some(Self {
            steps: self.steps.checked_add(b.steps)?,
            tokens: self.tokens.checked_add(b.tokens)?,
            model_calls: self.model_calls.checked_add(b.model_calls)?,
            tool_calls: self.tool_calls.checked_add(b.tool_calls)?,
            wall_time_ms: self.wall_time_ms.checked_add(b.wall_time_ms)?,
            risk_microunits: self.risk_microunits.checked_add(b.risk_microunits)?,
            context_tokens: self.context_tokens.checked_add(b.context_tokens)?,
            api_cost_microusd: self.api_cost_microusd.checked_add(b.api_cost_microusd)?,
        })
    }
    pub fn checked_sub(self, b: Self) -> Option<Self> {
        Some(Self {
            steps: self.steps.checked_sub(b.steps)?,
            tokens: self.tokens.checked_sub(b.tokens)?,
            model_calls: self.model_calls.checked_sub(b.model_calls)?,
            tool_calls: self.tool_calls.checked_sub(b.tool_calls)?,
            wall_time_ms: self.wall_time_ms.checked_sub(b.wall_time_ms)?,
            risk_microunits: self.risk_microunits.checked_sub(b.risk_microunits)?,
            context_tokens: self.context_tokens.checked_sub(b.context_tokens)?,
            api_cost_microusd: self.api_cost_microusd.checked_sub(b.api_cost_microusd)?,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BudgetError {
    ZeroReservation,
    InsufficientResources,
    UnknownReservation,
    ReusedReservationId,
    IncorrectPhase,
    ActualExceedsReservation,
    InvalidSnapshot,
    ArithmeticOverflow,
}
impl std::fmt::Display for BudgetError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self:?}")
    }
}
impl std::error::Error for BudgetError {}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReservationPhase {
    Held,
    Dispatched,
    Settled,
    Cancelled,
}
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReservationRecord {
    pub id: ReservationId,
    pub upper_bound: BudgetAmount,
    pub phase: ReservationPhase,
    pub actual: Option<BudgetAmount>,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReserveOutcome {
    Created,
    AlreadyHeld,
}

/// Snapshot serialization and integrity verification are responsibilities of VEP.
/// Rehydration strictly checks conservation but is not a cryptographic validator.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BudgetSnapshot {
    pub total: BudgetAmount,
    pub spent: BudgetAmount,
    pub reserved: BudgetAmount,
    pub records: Vec<ReservationRecord>,
}

/// Single-authority ledger; callers must serialize mutation through voderd.
#[derive(Debug, Clone)]
pub struct BudgetLedger {
    total: BudgetAmount,
    spent: BudgetAmount,
    reserved: BudgetAmount,
    records: BTreeMap<ReservationId, ReservationRecord>,
}

impl BudgetLedger {
    pub fn new(total: BudgetAmount) -> Self {
        Self {
            total,
            spent: BudgetAmount::default(),
            reserved: BudgetAmount::default(),
            records: BTreeMap::new(),
        }
    }
    pub fn total(&self) -> BudgetAmount {
        self.total
    }
    pub fn spent(&self) -> BudgetAmount {
        self.spent
    }
    pub fn reserved(&self) -> BudgetAmount {
        self.reserved
    }
    pub fn available(&self) -> BudgetAmount {
        // Ledger invariant guarantees subtraction always succeeds.
        self.total
            .checked_sub(self.spent)
            .and_then(|x| x.checked_sub(self.reserved))
            .expect("validated budget invariant")
    }
    pub fn record(&self, id: &ReservationId) -> Option<&ReservationRecord> {
        self.records.get(id)
    }
    pub fn reserve(
        &mut self,
        id: ReservationId,
        upper: BudgetAmount,
    ) -> Result<ReserveOutcome, BudgetError> {
        if upper.is_zero() {
            return Err(BudgetError::ZeroReservation);
        }
        if let Some(existing) = self.records.get(&id) {
            return if existing.upper_bound == upper && existing.phase == ReservationPhase::Held {
                Ok(ReserveOutcome::AlreadyHeld)
            } else {
                Err(BudgetError::ReusedReservationId)
            };
        }
        if !upper.fits_within(self.available()) {
            return Err(BudgetError::InsufficientResources);
        }
        let new_reserved = self
            .reserved
            .checked_add(upper)
            .ok_or(BudgetError::ArithmeticOverflow)?;
        self.records.insert(
            id.clone(),
            ReservationRecord {
                id,
                upper_bound: upper,
                phase: ReservationPhase::Held,
                actual: None,
            },
        );
        self.reserved = new_reserved;
        Ok(ReserveOutcome::Created)
    }
    /// Transition must be journaled durably before sending a network/tool request.
    /// Repeating dispatch is an error rather than permission to call twice.
    pub fn mark_dispatched(&mut self, id: &ReservationId) -> Result<(), BudgetError> {
        let record = self
            .records
            .get_mut(id)
            .ok_or(BudgetError::UnknownReservation)?;
        if record.phase != ReservationPhase::Held {
            return Err(BudgetError::IncorrectPhase);
        }
        record.phase = ReservationPhase::Dispatched;
        Ok(())
    }
    /// Settle with verified usage. Excess charges are rejected and held for
    /// operator reconciliation; never silently clamp or exceed a limit.
    pub fn settle(&mut self, id: &ReservationId, actual: BudgetAmount) -> Result<(), BudgetError> {
        let record = self
            .records
            .get(id)
            .ok_or(BudgetError::UnknownReservation)?;
        if record.phase != ReservationPhase::Dispatched {
            return Err(BudgetError::IncorrectPhase);
        }
        if !actual.fits_within(record.upper_bound) {
            return Err(BudgetError::ActualExceedsReservation);
        }
        let next_spent = self
            .spent
            .checked_add(actual)
            .ok_or(BudgetError::ArithmeticOverflow)?;
        let next_reserved = self
            .reserved
            .checked_sub(record.upper_bound)
            .ok_or(BudgetError::InvalidSnapshot)?;
        let combined = next_spent
            .checked_add(next_reserved)
            .ok_or(BudgetError::ArithmeticOverflow)?;
        if !combined.fits_within(self.total) {
            return Err(BudgetError::InvalidSnapshot);
        }
        let entry = self
            .records
            .get_mut(id)
            .ok_or(BudgetError::UnknownReservation)?;
        entry.phase = ReservationPhase::Settled;
        entry.actual = Some(actual);
        self.spent = next_spent;
        self.reserved = next_reserved;
        Ok(())
    }
    /// Only an action proven not dispatched may release its reservation.
    pub fn cancel_before_dispatch(&mut self, id: &ReservationId) -> Result<(), BudgetError> {
        let record = self
            .records
            .get(id)
            .ok_or(BudgetError::UnknownReservation)?;
        if record.phase != ReservationPhase::Held {
            return Err(BudgetError::IncorrectPhase);
        }
        let next_reserved = self
            .reserved
            .checked_sub(record.upper_bound)
            .ok_or(BudgetError::InvalidSnapshot)?;
        let entry = self
            .records
            .get_mut(id)
            .ok_or(BudgetError::UnknownReservation)?;
        entry.phase = ReservationPhase::Cancelled;
        self.reserved = next_reserved;
        Ok(())
    }
    pub fn snapshot(&self) -> BudgetSnapshot {
        BudgetSnapshot {
            total: self.total,
            spent: self.spent,
            reserved: self.reserved,
            records: self.records.values().cloned().collect(),
        }
    }
    /// Fail closed on malformed or inconsistent persistent snapshots.
    pub fn restore(snapshot: BudgetSnapshot) -> Result<Self, BudgetError> {
        let mut seen = BTreeSet::new();
        let mut spent = BudgetAmount::default();
        let mut reserved = BudgetAmount::default();
        let mut records = BTreeMap::new();
        for record in snapshot.records {
            if !seen.insert(record.id.clone()) || record.upper_bound.is_zero() {
                return Err(BudgetError::InvalidSnapshot);
            }
            match record.phase {
                ReservationPhase::Held | ReservationPhase::Dispatched => {
                    if record.actual.is_some() {
                        return Err(BudgetError::InvalidSnapshot);
                    }
                    reserved = reserved
                        .checked_add(record.upper_bound)
                        .ok_or(BudgetError::InvalidSnapshot)?;
                }
                ReservationPhase::Settled => {
                    let actual = record.actual.ok_or(BudgetError::InvalidSnapshot)?;
                    if !actual.fits_within(record.upper_bound) {
                        return Err(BudgetError::InvalidSnapshot);
                    }
                    spent = spent
                        .checked_add(actual)
                        .ok_or(BudgetError::InvalidSnapshot)?;
                }
                ReservationPhase::Cancelled => {
                    if record.actual.is_some() {
                        return Err(BudgetError::InvalidSnapshot);
                    }
                }
            }
            records.insert(record.id.clone(), record);
        }
        let combined = spent
            .checked_add(reserved)
            .ok_or(BudgetError::InvalidSnapshot)?;
        if spent != snapshot.spent
            || reserved != snapshot.reserved
            || !combined.fits_within(snapshot.total)
        {
            return Err(BudgetError::InvalidSnapshot);
        }
        Ok(Self {
            total: snapshot.total,
            spent,
            reserved,
            records,
        })
    }
}
