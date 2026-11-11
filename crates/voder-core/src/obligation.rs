//! Frozen-contract-bound evidence obligations and their closure eligibility.
//!
//! A `CLOSED` state is effective only for the CURRENT contract and workspace.
//! Evidence admission is a separate trusted module: this file never promotes
//! arbitrary observations or model claims to admitted evidence.

use crate::contract::{FrozenContract, RepoScope};
use crate::types::{
    ContractIdentity, Digest, EnvironmentId, EvidenceId, EvidenceRelation, ObligationId,
    ObligationStatus, RequirementId, SourceId, WorkspaceIdentity,
};
use std::collections::{BTreeMap, BTreeSet};

pub const MAX_OBLIGATIONS: usize = 4096;
const MAX_DEPENDENCIES: usize = 256;
const MAX_TEXT_BYTES: usize = 8192;
const MAX_ID_TEXT_BYTES: usize = 256;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ObligationError {
    EmptyObligations,
    TooManyObligations,
    InvalidField,
    DuplicateId,
    UnknownRequirement,
    MissingHardCoverage,
    HardRequirementMustBeCriticalAndSupported,
    CriticalNeedsRequirement,
    OutOfContractScope,
    UntrustedSource,
    InvalidClosureRelation,
    InvalidDependency,
    DependencyCycle,
    ContractMismatch,
    WorkspaceMismatch,
    UnknownObligation,
    EvidenceMismatch,
    DuplicateEvidence,
}
impl std::fmt::Display for ObligationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self:?}")
    }
}
impl std::error::Error for ObligationError {}

fn valid_text(s: &str) -> bool {
    !s.trim().is_empty()
        && s == s.trim()
        && s.len() <= MAX_TEXT_BYTES
        && !s.chars().any(char::is_control)
}

/// Authoring data; this is validated against `FrozenContract` on registration.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ObligationSpec {
    pub id: ObligationId,
    pub claim_id: String,
    pub requirement_id: Option<RequirementId>,
    pub threatened_judgement: String,
    pub expected_observation_type: String,
    pub admission_rule_id: String,
    pub scope: RepoScope,
    pub priority: u32,
    pub stop_condition: String,
    pub source: SourceId,
    pub critical: bool,
    pub closure_relation: EvidenceRelation,
    pub dependencies: Vec<ObligationId>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct Closure {
    evidence_id: EvidenceId,
    contract_digest: Digest,
    workspace_digest: Digest,
    environment: EnvironmentId,
}
#[derive(Debug, Clone, PartialEq, Eq)]
struct ObligationEntry {
    spec: ObligationSpec,
    status: ObligationStatus,
    closure: Option<Closure>,
}

/// Trusted evidence.rs is the ONLY intended producer of these values.
/// All fields are private to prevent public callers from fabricating an
/// "admitted" receipt with just a struct literal.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AdmittedEvidence {
    id: EvidenceId,
    obligation_id: ObligationId,
    claim_id: String,
    relation: EvidenceRelation,
    scope: RepoScope,
    observation_type: String,
    admission_rule_id: String,
    source: SourceId,
    contract_digest: Digest,
    workspace_digest: Digest,
    environment: EnvironmentId,
}
impl AdmittedEvidence {
    /// Not exposed outside the trusted crate. The future evidence module must
    /// call this only AFTER checking provenance, integrity, permissions and
    /// freshness against original observation and execution receipt.
    #[allow(dead_code, clippy::too_many_arguments)]
    pub(crate) fn from_trusted_admission(
        id: EvidenceId,
        obligation_id: ObligationId,
        claim_id: String,
        relation: EvidenceRelation,
        scope: RepoScope,
        observation_type: String,
        admission_rule_id: String,
        source: SourceId,
        contract_digest: Digest,
        workspace_digest: Digest,
        environment: EnvironmentId,
    ) -> Self {
        Self {
            id,
            obligation_id,
            claim_id,
            relation,
            scope,
            observation_type,
            admission_rule_id,
            source,
            contract_digest,
            workspace_digest,
            environment,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EvidenceEffect {
    Closed,
    DidNotSatisfyClosure,
    PendingDependencies,
    AlreadyBlocked,
}

/// Only the authoritative controller owns mutation; callers cannot directly
/// mutate an obligation, its status, or its closing evidence identity.
#[derive(Debug, Clone)]
pub struct ObligationRegistry {
    contract: ContractIdentity,
    entries: BTreeMap<ObligationId, ObligationEntry>,
    seen_evidence: BTreeSet<EvidenceId>,
}

impl ObligationRegistry {
    pub fn compile(
        contract: &FrozenContract,
        specs: Vec<ObligationSpec>,
    ) -> Result<Self, ObligationError> {
        if specs.is_empty() {
            return Err(ObligationError::EmptyObligations);
        }
        if specs.len() > MAX_OBLIGATIONS {
            return Err(ObligationError::TooManyObligations);
        }
        let mut entries = BTreeMap::new();
        for spec in specs {
            if !valid_text(&spec.claim_id)
                || spec.claim_id.len() > MAX_ID_TEXT_BYTES
                || !valid_text(&spec.threatened_judgement)
                || !valid_text(&spec.expected_observation_type)
                || spec.expected_observation_type.len() > MAX_ID_TEXT_BYTES
                || !valid_text(&spec.admission_rule_id)
                || spec.admission_rule_id.len() > MAX_ID_TEXT_BYTES
                || !valid_text(&spec.stop_condition)
            {
                return Err(ObligationError::InvalidField);
            }
            if !contract
                .read_scopes()
                .iter()
                .any(|s| s.contains(&spec.scope))
            {
                return Err(ObligationError::OutOfContractScope);
            }
            if !contract.trusted_sources().contains(&spec.source) {
                return Err(ObligationError::UntrustedSource);
            }
            if !matches!(
                spec.closure_relation,
                EvidenceRelation::Supports | EvidenceRelation::Refutes
            ) {
                return Err(ObligationError::InvalidClosureRelation);
            }
            if spec.critical && spec.requirement_id.is_none() {
                return Err(ObligationError::CriticalNeedsRequirement);
            }
            if let Some(req_id) = &spec.requirement_id {
                let req = contract
                    .requirements()
                    .iter()
                    .find(|r| &r.id == req_id)
                    .ok_or(ObligationError::UnknownRequirement)?;
                if req.hard
                    && (!spec.critical || spec.closure_relation != EvidenceRelation::Supports)
                {
                    return Err(ObligationError::HardRequirementMustBeCriticalAndSupported);
                }
            }
            if spec.critical && spec.closure_relation != EvidenceRelation::Supports {
                return Err(ObligationError::InvalidClosureRelation);
            }
            if entries
                .insert(
                    spec.id.clone(),
                    ObligationEntry {
                        spec,
                        status: ObligationStatus::Open,
                        closure: None,
                    },
                )
                .is_some()
            {
                return Err(ObligationError::DuplicateId);
            }
        }
        for req in contract.requirements().iter().filter(|r| r.hard) {
            if !entries.values().any(|e| {
                e.spec.requirement_id.as_ref() == Some(&req.id)
                    && e.spec.critical
                    && e.spec.closure_relation == EvidenceRelation::Supports
            }) {
                return Err(ObligationError::MissingHardCoverage);
            }
        }
        for (id, e) in &entries {
            let deps: BTreeSet<_> = e.spec.dependencies.iter().collect();
            if e.spec.dependencies.len() > MAX_DEPENDENCIES
                || deps.len() != e.spec.dependencies.len()
                || deps.iter().any(|d| *d == id)
                || deps.iter().any(|d| !entries.contains_key(*d))
            {
                return Err(ObligationError::InvalidDependency);
            }
        }
        // Detect indirect cycles without recursion and without external crates.
        for id in entries.keys() {
            let mut visited = BTreeSet::new();
            let mut stack = entries[id].spec.dependencies.clone();
            while let Some(dep) = stack.pop() {
                if &dep == id {
                    return Err(ObligationError::DependencyCycle);
                }
                if visited.insert(dep.clone()) {
                    stack.extend(entries[&dep].spec.dependencies.iter().cloned());
                }
            }
        }
        Ok(Self {
            contract: contract.identity().clone(),
            entries,
            seen_evidence: BTreeSet::new(),
        })
    }

    pub fn spec(&self, id: &ObligationId) -> Option<&ObligationSpec> {
        self.entries.get(id).map(|e| &e.spec)
    }
    fn check_contract(&self, contract: &FrozenContract) -> Result<(), ObligationError> {
        if self.contract != *contract.identity() {
            return Err(ObligationError::ContractMismatch);
        }
        Ok(())
    }
    fn effective_status_of(
        &self,
        entry: &ObligationEntry,
        workspace: &WorkspaceIdentity,
    ) -> ObligationStatus {
        if entry.status != ObligationStatus::Closed {
            return entry.status;
        }
        match &entry.closure {
            Some(c)
                if c.workspace_digest == workspace.digest
                    && c.contract_digest == self.contract.digest
                    && c.environment == self.contract.environment =>
            {
                ObligationStatus::Closed
            }
            _ => ObligationStatus::Open,
        }
    }
    pub fn status(
        &self,
        contract: &FrozenContract,
        workspace: &WorkspaceIdentity,
        id: &ObligationId,
    ) -> Result<ObligationStatus, ObligationError> {
        self.check_contract(contract)?;
        let entry = self
            .entries
            .get(id)
            .ok_or(ObligationError::UnknownObligation)?;
        Ok(self.effective_status_of(entry, workspace))
    }
    pub fn highest_priority_open<'a>(
        &'a self,
        contract: &FrozenContract,
        workspace: &WorkspaceIdentity,
    ) -> Result<Option<&'a ObligationSpec>, ObligationError> {
        self.check_contract(contract)?;
        let mut ready: Vec<_> = self
            .entries
            .values()
            .filter(|entry| self.effective_status_of(entry, workspace) == ObligationStatus::Open)
            .map(|entry| &entry.spec)
            .collect();
        ready.sort_by(|a, b| {
            b.critical
                .cmp(&a.critical)
                .then_with(|| b.priority.cmp(&a.priority))
                .then_with(|| a.id.cmp(&b.id))
        });
        Ok(ready.first().copied())
    }
    /// Ready for a candidate action: open AND every prerequisite closed in
    /// the current workspace. Prevents repeatedly choosing a blocked branch.
    pub fn highest_priority_actionable<'a>(
        &'a self,
        contract: &FrozenContract,
        workspace: &WorkspaceIdentity,
    ) -> Result<Option<&'a ObligationSpec>, ObligationError> {
        self.check_contract(contract)?;
        let mut ready: Vec<_> = self
            .entries
            .values()
            .filter(|entry| {
                self.effective_status_of(entry, workspace) == ObligationStatus::Open
                    && entry.spec.dependencies.iter().all(|id| {
                        self.entries.get(id).is_some_and(|parent| {
                            self.effective_status_of(parent, workspace) == ObligationStatus::Closed
                        })
                    })
            })
            .map(|entry| &entry.spec)
            .collect();
        ready.sort_by(|a, b| {
            b.critical
                .cmp(&a.critical)
                .then_with(|| b.priority.cmp(&a.priority))
                .then_with(|| a.id.cmp(&b.id))
        });
        Ok(ready.first().copied())
    }
    pub fn all_critical_closed(
        &self,
        contract: &FrozenContract,
        workspace: &WorkspaceIdentity,
    ) -> Result<bool, ObligationError> {
        self.check_contract(contract)?;
        Ok(self
            .entries
            .values()
            .filter(|e| e.spec.critical)
            .all(|e| self.effective_status_of(e, workspace) == ObligationStatus::Closed))
    }

    /// An accepted record is *not* necessarily a closing record. A REFUTES or
    /// UNRESOLVED for a SUPPORTS obligation never closes it. An already BLOCKED
    /// obligation cannot be silently reopened by a matching observation.
    pub fn apply_admitted_evidence(
        &mut self,
        contract: &FrozenContract,
        workspace: &WorkspaceIdentity,
        evidence: &AdmittedEvidence,
    ) -> Result<EvidenceEffect, ObligationError> {
        self.check_contract(contract)?;
        let entry = self
            .entries
            .get(&evidence.obligation_id)
            .ok_or(ObligationError::UnknownObligation)?;
        if self.seen_evidence.contains(&evidence.id) {
            return Err(ObligationError::DuplicateEvidence);
        }
        if evidence.contract_digest != self.contract.digest
            || evidence.environment != self.contract.environment
        {
            return Err(ObligationError::ContractMismatch);
        }
        if evidence.workspace_digest != workspace.digest {
            return Err(ObligationError::WorkspaceMismatch);
        }
        if evidence.claim_id != entry.spec.claim_id
            || evidence.observation_type != entry.spec.expected_observation_type
            || evidence.admission_rule_id != entry.spec.admission_rule_id
            || evidence.source != entry.spec.source
            || !entry.spec.scope.contains(&evidence.scope)
        {
            return Err(ObligationError::EvidenceMismatch);
        }
        let effect = if entry.status == ObligationStatus::Blocked {
            EvidenceEffect::AlreadyBlocked
        } else if evidence.relation != entry.spec.closure_relation
            || self.effective_status_of(entry, workspace) == ObligationStatus::Closed
        {
            EvidenceEffect::DidNotSatisfyClosure
        } else if !entry.spec.dependencies.iter().all(|id| {
            self.entries
                .get(id)
                .is_some_and(|e| self.effective_status_of(e, workspace) == ObligationStatus::Closed)
        }) {
            EvidenceEffect::PendingDependencies
        } else {
            EvidenceEffect::Closed
        };
        // Duplicate evidence IDs cannot be replayed as a new admission.
        self.seen_evidence.insert(evidence.id.clone());
        if effect == EvidenceEffect::Closed {
            let target = self
                .entries
                .get_mut(&evidence.obligation_id)
                .ok_or(ObligationError::UnknownObligation)?;
            target.status = ObligationStatus::Closed;
            target.closure = Some(Closure {
                evidence_id: evidence.id.clone(),
                contract_digest: evidence.contract_digest.clone(),
                workspace_digest: evidence.workspace_digest.clone(),
                environment: evidence.environment.clone(),
            });
        }
        Ok(effect)
    }
}

#[cfg(test)]
mod admission_tests {
    use super::*;
    use crate::contract::{ContractDraft, Requirement, TaskAnchor, ValidatorBinding};
    use crate::types::{Severity, Version};

    fn id<T: TryFrom<&'static str>>(s: &'static str) -> T
    where
        <T as TryFrom<&'static str>>::Error: std::fmt::Debug,
    {
        T::try_from(s).unwrap()
    }
    fn fixture() -> (FrozenContract, WorkspaceIdentity, ObligationSpec) {
        let c = FrozenContract::compile(ContractDraft {
            anchor: TaskAnchor {
                task_id: id("task"),
                identity: "test".into(),
                goal: "validate".into(),
                version: Version::new(1).unwrap(),
                accepted_facts: vec![],
                environment_constraints: vec![],
                evaluation_boundary: vec![],
            },
            identity: ContractIdentity {
                contract_id: id("contract"),
                version: Version::new(1).unwrap(),
                digest: id("contract-digest"),
                toolchain: id("rust"),
                environment: id("env"),
            },
            requirements: vec![Requirement {
                id: id("req"),
                description: "required".into(),
                hard: true,
                severity: Severity::Critical,
            }],
            validators: vec![ValidatorBinding {
                id: id("validator"),
                version: id("v1"),
                requirement_ids: vec![id("req")],
            }],
            trusted_sources: vec![id("trusted")],
            tools: vec![],
            read_scopes: vec![RepoScope::parse("src").unwrap()],
            write_scopes: vec![],
            forbidden_effects: vec![],
            required_inputs: vec![],
            sampling_rules: vec![],
        })
        .unwrap();
        let w = WorkspaceIdentity {
            revision: id("r1"),
            digest: id("workspace-a"),
        };
        let o = ObligationSpec {
            id: id("o1"),
            claim_id: "claim".into(),
            requirement_id: Some(id("req")),
            threatened_judgement: "deliver".into(),
            expected_observation_type: "test".into(),
            admission_rule_id: "rule".into(),
            scope: RepoScope::parse("src").unwrap(),
            priority: 5,
            stop_condition: "passed".into(),
            source: id("trusted"),
            critical: true,
            closure_relation: EvidenceRelation::Supports,
            dependencies: vec![],
        };
        (c, w, o)
    }
    fn ev(
        c: &FrozenContract,
        w: &WorkspaceIdentity,
        relation: EvidenceRelation,
        id_: &'static str,
    ) -> AdmittedEvidence {
        AdmittedEvidence::from_trusted_admission(
            id(id_),
            id("o1"),
            "claim".into(),
            relation,
            RepoScope::parse("src/lib.rs").unwrap(),
            "test".into(),
            "rule".into(),
            id("trusted"),
            c.identity().digest.clone(),
            w.digest.clone(),
            c.identity().environment.clone(),
        )
    }
    #[test]
    fn refutes_cannot_close_hard_obligation() {
        let (c, w, o) = fixture();
        let mut registry = ObligationRegistry::compile(&c, vec![o]).unwrap();
        assert_eq!(
            registry
                .apply_admitted_evidence(&c, &w, &ev(&c, &w, EvidenceRelation::Refutes, "e1"))
                .unwrap(),
            EvidenceEffect::DidNotSatisfyClosure
        );
        assert!(!registry.all_critical_closed(&c, &w).unwrap());
        assert_eq!(
            registry
                .apply_admitted_evidence(&c, &w, &ev(&c, &w, EvidenceRelation::Supports, "e2"))
                .unwrap(),
            EvidenceEffect::Closed
        );
        assert!(registry.all_critical_closed(&c, &w).unwrap());
    }
    #[test]
    fn stale_close_and_replay_are_rejected() {
        let (c, w, o) = fixture();
        let mut registry = ObligationRegistry::compile(&c, vec![o]).unwrap();
        let evidence = ev(&c, &w, EvidenceRelation::Supports, "e1");
        registry.apply_admitted_evidence(&c, &w, &evidence).unwrap();
        assert_eq!(
            registry.apply_admitted_evidence(&c, &w, &evidence),
            Err(ObligationError::DuplicateEvidence)
        );
        let mut changed = w.clone();
        changed.digest = id("workspace-b");
        assert_eq!(
            registry.status(&c, &changed, &id("o1")).unwrap(),
            ObligationStatus::Open
        );
        assert!(!registry.all_critical_closed(&c, &changed).unwrap());
        assert_eq!(
            registry.apply_admitted_evidence(
                &c,
                &changed,
                &ev(&c, &w, EvidenceRelation::Supports, "e2")
            ),
            Err(ObligationError::WorkspaceMismatch)
        );
        assert_eq!(
            registry
                .apply_admitted_evidence(
                    &c,
                    &changed,
                    &ev(&c, &changed, EvidenceRelation::Supports, "e3")
                )
                .unwrap(),
            EvidenceEffect::Closed
        );
    }
    #[test]
    fn dependencies_require_new_current_admitted_evidence() {
        let (c, w, o) = fixture();
        let mut dep = o.clone();
        dep.id = id("dep");
        dep.critical = false;
        dep.requirement_id = None;
        let mut target = o;
        target.dependencies = vec![id("dep")];
        let mut registry = ObligationRegistry::compile(&c, vec![target, dep]).unwrap();
        assert_eq!(
            registry
                .apply_admitted_evidence(&c, &w, &ev(&c, &w, EvidenceRelation::Supports, "e1"))
                .unwrap(),
            EvidenceEffect::PendingDependencies
        );
        let mut dep_ev = ev(&c, &w, EvidenceRelation::Supports, "e2");
        dep_ev.obligation_id = id("dep");
        assert_eq!(
            registry.apply_admitted_evidence(&c, &w, &dep_ev).unwrap(),
            EvidenceEffect::Closed
        );
        // Already admitted e1 cannot be replayed; new current evidence is required.
        assert_eq!(
            registry.apply_admitted_evidence(&c, &w, &ev(&c, &w, EvidenceRelation::Supports, "e1")),
            Err(ObligationError::DuplicateEvidence)
        );
        assert_eq!(
            registry
                .apply_admitted_evidence(&c, &w, &ev(&c, &w, EvidenceRelation::Supports, "e3"))
                .unwrap(),
            EvidenceEffect::Closed
        );
    }
}
