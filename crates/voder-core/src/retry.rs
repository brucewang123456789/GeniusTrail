//! Semantic retry eligibility, independent of event counts and observation IDs.
//!
//! This module is pure and does not authorize execution. `try_record` must be
//! called by the single trusted controller, and its result must be durably
//! journaled BEFORE dispatch. Inputs called `admitted` must come from the
//! trusted evidence ledger, never directly from a model or tool payload.

use crate::contract::{FrozenContract, RepoScope};
use crate::types::{
    ContractIdentity, Digest, EnvironmentId, EvidenceRelation, ObligationId, ToolId,
    WorkspaceIdentity,
};
use std::collections::BTreeSet;
use std::num::NonZeroU32;

const MAX_LABEL_BYTES: usize = 256;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RetryError {
    InvalidActionField,
    SamplingNotAuthorized,
    SamplingOrdinalOutOfSequence,
    UnchangedAction,
}
impl std::fmt::Display for RetryError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self:?}")
    }
}
impl std::error::Error for RetryError {}

fn valid_label(s: &str) -> bool {
    !s.is_empty() && s == s.trim() && s.len() <= MAX_LABEL_BYTES && !s.chars().any(char::is_control)
}

/// The caller MUST derive `parameter_key` from trusted canonical tool arguments;
/// arbitrary model-supplied keys must never be accepted as fresh identities.
/// The action ID itself is intentionally excluded from retry identity.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ActionDescriptor {
    pub kind: String,
    pub target_obligation_id: Option<ObligationId>,
    pub hypothesis_id: Option<String>,
    pub tool_id: ToolId,
    pub parameter_key: Digest,
    pub scope: RepoScope,
}

/// Collision-free length-prefixed representation, not a cryptographic hash.
/// Encoding intentionally differs from the TypeScript reference JSON encoding;
/// equality semantics are the same and should be checked with golden vectors.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ActionSignature(String);
impl ActionSignature {
    pub fn as_str(&self) -> &str {
        &self.0
    }
}
fn append_field(out: &mut String, value: &str) {
    out.push_str(&value.len().to_string());
    out.push(':');
    out.push_str(value);
}
impl ActionDescriptor {
    pub fn signature(&self) -> Result<ActionSignature, RetryError> {
        if !valid_label(&self.kind)
            || self
                .hypothesis_id
                .as_deref()
                .is_some_and(|h| !valid_label(h))
        {
            return Err(RetryError::InvalidActionField);
        }
        let mut s = "voder-action-v1|".to_owned();
        append_field(&mut s, &self.kind);
        append_field(
            &mut s,
            self.target_obligation_id
                .as_ref()
                .map_or("", |id| id.as_str()),
        );
        append_field(&mut s, self.hypothesis_id.as_deref().unwrap_or(""));
        append_field(&mut s, self.tool_id.as_str());
        append_field(&mut s, self.parameter_key.as_str());
        append_field(&mut s, self.scope.as_str());
        Ok(ActionSignature(s))
    }
}

/// A projection of an ALREADY ADMITTED evidence record. The trusted ledger must
/// supply this; this type alone does not authenticate an admission.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RelevantEvidenceFact {
    pub obligation_id: ObligationId,
    pub claim_id: String,
    pub relation: EvidenceRelation,
    pub content_digest: Digest,
    pub contract_digest: Digest,
    pub workspace_digest: Digest,
    pub environment: EnvironmentId,
}

/// Canonically sorted, de-duplicated semantic triples. IDs, step counters and
/// timestamps are excluded, so irrelevant evidence cannot reset a retry gate.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RelevantEvidenceKey(Vec<(String, String, Digest)>);
impl RelevantEvidenceKey {
    pub fn triples(&self) -> &[(String, String, Digest)] {
        &self.0
    }
}
pub fn relevant_evidence_key(
    facts: &[RelevantEvidenceFact],
    obligation: Option<&ObligationId>,
    contract: &ContractIdentity,
    workspace: &WorkspaceIdentity,
) -> RelevantEvidenceKey {
    let set: BTreeSet<_> = facts
        .iter()
        .filter(|e| {
            Some(&e.obligation_id) == obligation
                && e.contract_digest == contract.digest
                && e.workspace_digest == workspace.digest
                && e.environment == contract.environment
        })
        .map(|e| {
            (
                e.claim_id.clone(),
                e.relation.as_wire().to_owned(),
                e.content_digest.clone(),
            )
        })
        .collect();
    RelevantEvidenceKey(set.into_iter().collect())
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AttemptRecord {
    pub signature: ActionSignature,
    pub contract: ContractIdentity,
    pub workspace_digest: Digest,
    pub relevant_evidence: RelevantEvidenceKey,
    /// 0 = ordinary attempt; n>=1 = pre-authorized sample n.
    pub sample_ordinal: u32,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RetryPermit {
    FirstAttempt,
    ChangedRelevantState,
    AuthorizedSample,
}

/// Volatile reference gate. The authoritative daemon must append a durable
/// intent before the corresponding side effect; process restart cannot rebuild
/// this ledger from model conversation text.
#[derive(Debug, Clone, Default)]
pub struct RetryLedger {
    records: Vec<AttemptRecord>,
}
impl RetryLedger {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn records(&self) -> &[AttemptRecord] {
        &self.records
    }

    /// Atomic in-memory check-and-record. No allowance is issued for an identical
    /// attempt at unchanged world/contract/relevant evidence unless the frozen
    /// acceptance contract pre-authorized a bounded statistical sample.
    pub fn try_record(
        &mut self,
        contract: &FrozenContract,
        workspace: &WorkspaceIdentity,
        action: &ActionDescriptor,
        admitted: &[RelevantEvidenceFact],
        sampling_ordinal: Option<NonZeroU32>,
    ) -> Result<RetryPermit, RetryError> {
        let signature = action.signature()?;
        let identity = contract.identity();
        let evidence = relevant_evidence_key(
            admitted,
            action.target_obligation_id.as_ref(),
            identity,
            workspace,
        );
        let same: Vec<_> = self
            .records
            .iter()
            .filter(|r| {
                r.signature == signature
                    && &r.contract == identity
                    && r.workspace_digest == workspace.digest
                    && r.relevant_evidence == evidence
            })
            .collect();
        let ordinal = if let Some(n) = sampling_ordinal {
            let target = action
                .target_obligation_id
                .as_ref()
                .ok_or(RetryError::SamplingNotAuthorized)?;
            let rule = contract
                .sampling_rules()
                .iter()
                .find(|r| r.tool_id == action.tool_id && r.obligation_id == *target)
                .ok_or(RetryError::SamplingNotAuthorized)?;
            if n.get() > rule.max_samples
                || same.iter().any(|r| r.sample_ordinal == 0)
                || Some(n.get())
                    != u32::try_from(same.len())
                        .ok()
                        .and_then(|x| x.checked_add(1))
            {
                return Err(RetryError::SamplingOrdinalOutOfSequence);
            }
            n.get()
        } else {
            if !same.is_empty() {
                return Err(RetryError::UnchangedAction);
            }
            0
        };
        let permit = if ordinal != 0 {
            RetryPermit::AuthorizedSample
        } else if self.records.iter().any(|r| r.signature == signature) {
            RetryPermit::ChangedRelevantState
        } else {
            RetryPermit::FirstAttempt
        };
        self.records.push(AttemptRecord {
            signature,
            contract: identity.clone(),
            workspace_digest: workspace.digest.clone(),
            relevant_evidence: evidence,
            sample_ordinal: ordinal,
        });
        Ok(permit)
    }
}
