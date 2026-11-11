//! Task anchor, acceptance contract, and required-input readiness.
//!
//! A contract can be compiled only from a complete, internally consistent draft.
//! Compiled fields are private; model/plugin code must not mutate hard acceptance.
//! Digest authenticity, actual file existence, and symlink/OS path containment
//! require trusted runtime adapters; a string digest is not proof of integrity.

use crate::types::{
    ContractIdentity, InputId, ObligationId, RequirementId, Severity, SourceId, TaskId, ToolEffect,
    ToolId, ValidatorId, ValidatorVersion, Version,
};
use std::collections::BTreeSet;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ContractError {
    Empty(&'static str),
    Duplicate(&'static str),
    InvalidScope,
    MissingHardRequirement,
    MissingValidationRoots,
    UnknownRequirement,
    HardRequirementWithoutValidator,
    WriteToolWithoutScope,
    WriteScopeOutsideReadScope,
    InvalidSamplingPolicy,
}
impl std::fmt::Display for ContractError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self:?}")
    }
}
impl std::error::Error for ContractError {}

fn valid_text(s: &str) -> bool {
    !s.trim().is_empty() && s == s.trim() && !s.chars().any(char::is_control)
}

/// A declared repository-relative lexical scope (NOT a filesystem sandbox).
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct RepoScope(String);
impl RepoScope {
    pub fn parse(s: impl Into<String>) -> Result<Self, ContractError> {
        let s = s.into();
        if s == "@repository" { return Ok(Self(s)); }
        if s.is_empty()
            || s.starts_with('/')
            || s.starts_with('~')
            || s.contains('\\')
            || s.contains(':')
            || s.chars().any(char::is_control)
        {
            return Err(ContractError::InvalidScope);
        }
        for segment in s.split('/') {
            if segment.is_empty()
                || segment == "."
                || segment == ".."
                || segment.ends_with(' ')
                || segment.ends_with('.')
            {
                return Err(ContractError::InvalidScope);
            }
            let stem = segment.split('.').next().unwrap_or("").to_ascii_lowercase();
            let windows_device = matches!(stem.as_str(), "con" | "prn" | "aux" | "nul")
                || (stem.len() == 4
                    && (stem.starts_with("com") || stem.starts_with("lpt"))
                    && matches!(stem.as_bytes()[3], b'1'..=b'9'));
            if windows_device {
                return Err(ContractError::InvalidScope);
            }
        }
        Ok(Self(s))
    }
    pub fn as_str(&self) -> &str {
        &self.0
    }
    /// Lexical comparison only; symlink containment must be checked in voder-runtime.
    pub fn contains(&self, other: &Self) -> bool {
        self.0 == "@repository" || other.0 == self.0 || other.0.starts_with(&format!("{}/", self.0))
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TaskAnchor {
    pub task_id: TaskId,
    pub identity: String,
    pub goal: String,
    pub version: Version,
    pub accepted_facts: Vec<String>,
    pub environment_constraints: Vec<String>,
    pub evaluation_boundary: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Requirement {
    pub id: RequirementId,
    pub description: String,
    pub hard: bool,
    pub severity: Severity,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ValidatorBinding {
    pub id: ValidatorId,
    pub version: ValidatorVersion,
    pub requirement_ids: Vec<RequirementId>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolRule {
    pub id: ToolId,
    pub effect: ToolEffect,
    pub requires_approval: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SamplingRule {
    pub tool_id: ToolId,
    pub obligation_id: ObligationId,
    pub max_samples: u32,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum InputKind {
    File,
    Context,
    Configuration,
    SecretReference,
}
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RequiredInput {
    pub id: InputId,
    pub kind: InputKind,
    pub description: String,
    pub why_required: String,
}

/// Editable only before compile. All external/model-generated fields are untrusted.
#[derive(Debug, Clone)]
pub struct ContractDraft {
    pub anchor: TaskAnchor,
    pub identity: ContractIdentity,
    pub requirements: Vec<Requirement>,
    pub validators: Vec<ValidatorBinding>,
    pub trusted_sources: Vec<SourceId>,
    pub tools: Vec<ToolRule>,
    pub read_scopes: Vec<RepoScope>,
    pub write_scopes: Vec<RepoScope>,
    pub forbidden_effects: Vec<String>,
    pub required_inputs: Vec<RequiredInput>,
    pub sampling_rules: Vec<SamplingRule>,
}

/// Stable, private, compiled acceptance. No setter or weakening API is exposed.
#[derive(Debug, Clone)]
pub struct FrozenContract {
    anchor: TaskAnchor,
    identity: ContractIdentity,
    requirements: Vec<Requirement>,
    validators: Vec<ValidatorBinding>,
    trusted_sources: Vec<SourceId>,
    tools: Vec<ToolRule>,
    read_scopes: Vec<RepoScope>,
    write_scopes: Vec<RepoScope>,
    forbidden_effects: Vec<String>,
    required_inputs: Vec<RequiredInput>,
    sampling_rules: Vec<SamplingRule>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MissingInput {
    pub id: InputId,
    pub kind: InputKind,
    pub description: String,
    pub why_required: String,
}

impl FrozenContract {
    /// Reject inconsistent contracts and report all detected errors together.
    /// No partial contract escapes on failure.
    pub fn compile(draft: ContractDraft) -> Result<Self, Vec<ContractError>> {
        let mut errors = Vec::new();
        if !valid_text(&draft.anchor.goal) {
            errors.push(ContractError::Empty("goal"));
        }
        if !valid_text(&draft.anchor.identity) {
            errors.push(ContractError::Empty("task_identity"));
        }
        for text in draft
            .anchor
            .accepted_facts
            .iter()
            .chain(draft.anchor.environment_constraints.iter())
            .chain(draft.anchor.evaluation_boundary.iter())
        {
            if !valid_text(text) {
                errors.push(ContractError::Empty("anchor_field"));
            }
        }
        if draft.requirements.is_empty() || !draft.requirements.iter().any(|r| r.hard) {
            errors.push(ContractError::MissingHardRequirement);
        }
        let req_ids: BTreeSet<_> = draft.requirements.iter().map(|r| &r.id).collect();
        if req_ids.len() != draft.requirements.len() {
            errors.push(ContractError::Duplicate("requirement"));
        }
        for r in &draft.requirements {
            if !valid_text(&r.description) {
                errors.push(ContractError::Empty("requirement_description"));
            }
        }
        if draft.validators.is_empty() || draft.trusted_sources.is_empty() {
            errors.push(ContractError::MissingValidationRoots);
        }
        let validator_ids: BTreeSet<_> = draft.validators.iter().map(|v| &v.id).collect();
        if validator_ids.len() != draft.validators.len() {
            errors.push(ContractError::Duplicate("validator"));
        }
        let source_ids: BTreeSet<_> = draft.trusted_sources.iter().collect();
        if source_ids.len() != draft.trusted_sources.len() {
            errors.push(ContractError::Duplicate("source"));
        }
        for validator in &draft.validators {
            let targets: BTreeSet<_> = validator.requirement_ids.iter().collect();
            if targets.len() != validator.requirement_ids.len() {
                errors.push(ContractError::Duplicate("validator_target"));
            }
            if targets.is_empty() {
                errors.push(ContractError::Empty("validator_targets"));
            }
            if validator
                .requirement_ids
                .iter()
                .any(|id| !draft.requirements.iter().any(|r| r.id == *id))
            {
                errors.push(ContractError::UnknownRequirement);
            }
        }
        for r in draft.requirements.iter().filter(|r| r.hard) {
            if !draft
                .validators
                .iter()
                .any(|v| v.requirement_ids.contains(&r.id))
            {
                errors.push(ContractError::HardRequirementWithoutValidator);
            }
        }
        let tool_ids: BTreeSet<_> = draft.tools.iter().map(|t| &t.id).collect();
        if tool_ids.len() != draft.tools.len() {
            errors.push(ContractError::Duplicate("tool"));
        }
        if draft.tools.iter().any(|t| t.effect == ToolEffect::Write)
            && draft.write_scopes.is_empty()
        {
            errors.push(ContractError::WriteToolWithoutScope);
        }
        if draft.read_scopes.is_empty() {
            errors.push(ContractError::Empty("read_scopes"));
        }
        if draft
            .write_scopes
            .iter()
            .any(|w| !draft.read_scopes.iter().any(|r| r.contains(w)))
        {
            errors.push(ContractError::WriteScopeOutsideReadScope);
        }
        for (label, scopes) in [
            ("read_scope", &draft.read_scopes),
            ("write_scope", &draft.write_scopes),
        ] {
            let unique: BTreeSet<_> = scopes.iter().collect();
            if unique.len() != scopes.len() {
                errors.push(ContractError::Duplicate(label));
            }
        }
        if draft.forbidden_effects.iter().any(|s| !valid_text(s)) {
            errors.push(ContractError::Empty("forbidden_effect"));
        }
        let effects: BTreeSet<_> = draft.forbidden_effects.iter().collect();
        if effects.len() != draft.forbidden_effects.len() {
            errors.push(ContractError::Duplicate("forbidden_effect"));
        }
        let inputs: BTreeSet<_> = draft.required_inputs.iter().map(|i| &i.id).collect();
        if inputs.len() != draft.required_inputs.len() {
            errors.push(ContractError::Duplicate("required_input"));
        }
        for input in &draft.required_inputs {
            if !valid_text(&input.description) || !valid_text(&input.why_required) {
                errors.push(ContractError::Empty("input_description_or_reason"));
            }
        }
        let samples: BTreeSet<_> = draft
            .sampling_rules
            .iter()
            .map(|r| (&r.tool_id, &r.obligation_id))
            .collect();
        if samples.len() != draft.sampling_rules.len() {
            errors.push(ContractError::Duplicate("sampling_policy"));
        }
        for sample in &draft.sampling_rules {
            if sample.max_samples == 0 || !draft.tools.iter().any(|t| t.id == sample.tool_id) {
                errors.push(ContractError::InvalidSamplingPolicy);
            }
        }
        if !errors.is_empty() {
            return Err(errors);
        }
        Ok(Self {
            anchor: draft.anchor,
            identity: draft.identity,
            requirements: draft.requirements,
            validators: draft.validators,
            trusted_sources: draft.trusted_sources,
            tools: draft.tools,
            read_scopes: draft.read_scopes,
            write_scopes: draft.write_scopes,
            forbidden_effects: draft.forbidden_effects,
            required_inputs: draft.required_inputs,
            sampling_rules: draft.sampling_rules,
        })
    }

    pub fn anchor(&self) -> &TaskAnchor {
        &self.anchor
    }
    pub fn identity(&self) -> &ContractIdentity {
        &self.identity
    }
    pub fn requirements(&self) -> &[Requirement] {
        &self.requirements
    }
    pub fn validators(&self) -> &[ValidatorBinding] {
        &self.validators
    }
    pub fn trusted_sources(&self) -> &[SourceId] {
        &self.trusted_sources
    }
    pub fn tools(&self) -> &[ToolRule] {
        &self.tools
    }
    pub fn read_scopes(&self) -> &[RepoScope] {
        &self.read_scopes
    }
    pub fn write_scopes(&self) -> &[RepoScope] {
        &self.write_scopes
    }
    pub fn forbidden_effects(&self) -> &[String] {
        &self.forbidden_effects
    }
    pub fn required_inputs(&self) -> &[RequiredInput] {
        &self.required_inputs
    }
    pub fn sampling_rules(&self) -> &[SamplingRule] {
        &self.sampling_rules
    }

    /// Return missing prerequisites for an ask-user-and-pause response.
    /// `verified_present` MUST come from a trusted inventory adapter that has
    /// verified actual content/ownership. This is not its own evidence admission.
    pub fn missing_inputs(&self, verified_present: &[InputId]) -> Vec<MissingInput> {
        let present: BTreeSet<_> = verified_present.iter().collect();
        self.required_inputs
            .iter()
            .filter(|i| !present.contains(&i.id))
            .map(|i| MissingInput {
                id: i.id.clone(),
                kind: i.kind,
                description: i.description.clone(),
                why_required: i.why_required.clone(),
            })
            .collect()
    }
}

#[cfg(test)] mod repository_scope_tests {use super::*;#[test]fn repository_scope_is_explicit_and_paths_still_validate(){let root=RepoScope::parse("@repository").unwrap();assert!(root.contains(&RepoScope::parse("src/a").unwrap()));assert!(root.contains(&RepoScope::parse("tests/a").unwrap()));assert!(!RepoScope::parse("src").unwrap().contains(&RepoScope::parse("tests/a").unwrap()));assert!(RepoScope::parse("../escape").is_err());}}
