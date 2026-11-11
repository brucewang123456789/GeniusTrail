//! Stable domain identities and finite vocabularies for the Voder trusted core.
//!
//! Only identities and vocabularies live here: no I/O, model, scoring, or
//! authorization policy. An opaque digest is *not* a cryptographic verification.

use std::fmt;
use std::num::NonZeroU64;

pub const MAX_ID_BYTES: usize = 256;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TypeError {
    Empty {
        field: &'static str,
    },
    SurroundedByWhitespace {
        field: &'static str,
    },
    ControlCharacter {
        field: &'static str,
    },
    TooLong {
        field: &'static str,
        maximum_bytes: usize,
    },
    ZeroVersion,
}

impl fmt::Display for TypeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Empty { field } => write!(f, "{field}: value must not be empty"),
            Self::SurroundedByWhitespace { field } => {
                write!(f, "{field}: surrounding whitespace is forbidden")
            }
            Self::ControlCharacter { field } => {
                write!(f, "{field}: control characters are forbidden")
            }
            Self::TooLong {
                field,
                maximum_bytes,
            } => {
                write!(f, "{field}: exceeds {maximum_bytes} UTF-8 bytes")
            }
            Self::ZeroVersion => f.write_str("version must be positive"),
        }
    }
}
impl std::error::Error for TypeError {}

fn validate_identifier(field: &'static str, value: &str) -> Result<(), TypeError> {
    if value.trim().is_empty() {
        return Err(TypeError::Empty { field });
    }
    if value != value.trim() {
        return Err(TypeError::SurroundedByWhitespace { field });
    }
    if value.chars().any(char::is_control) {
        return Err(TypeError::ControlCharacter { field });
    }
    if value.len() > MAX_ID_BYTES {
        return Err(TypeError::TooLong {
            field,
            maximum_bytes: MAX_ID_BYTES,
        });
    }
    Ok(())
}

// One type per identity class prevents cross-category substitution at compile time.
// This macro is only boilerplate removal, not another runtime abstraction layer.
macro_rules! stable_identifier {
    ($name:ident, $field:literal) => {
        #[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
        pub struct $name(String);
        impl $name {
            pub fn parse(value: impl Into<String>) -> Result<Self, TypeError> {
                let value = value.into();
                validate_identifier($field, &value)?;
                Ok(Self(value))
            }
            pub fn as_str(&self) -> &str {
                &self.0
            }
        }
        impl fmt::Display for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.write_str(&self.0)
            }
        }
        impl TryFrom<&str> for $name {
            type Error = TypeError;
            fn try_from(value: &str) -> Result<Self, Self::Error> {
                Self::parse(value)
            }
        }
    };
}

stable_identifier!(RunId, "run_id");
stable_identifier!(SessionId, "session_id");
stable_identifier!(TaskId, "task_id");
stable_identifier!(ContractId, "contract_id");
stable_identifier!(RequirementId, "requirement_id");
stable_identifier!(InputId, "input_id");
stable_identifier!(ObligationId, "obligation_id");
stable_identifier!(ObservationId, "observation_id");
stable_identifier!(EvidenceId, "evidence_id");
stable_identifier!(ActionId, "action_id");
stable_identifier!(ToolId, "tool_id");
stable_identifier!(ValidatorId, "validator_id");
stable_identifier!(SourceId, "observation_source_id");
stable_identifier!(ReservationId, "budget_reservation_id");
stable_identifier!(RevisionId, "revision_id");
stable_identifier!(EnvironmentId, "environment_fingerprint");
stable_identifier!(ToolchainId, "toolchain_identity");
stable_identifier!(ValidatorVersion, "validator_version");
stable_identifier!(Digest, "digest");

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Version(NonZeroU64);
impl Version {
    pub fn new(value: u64) -> Result<Self, TypeError> {
        NonZeroU64::new(value)
            .map(Self)
            .ok_or(TypeError::ZeroVersion)
    }
    pub fn get(self) -> u64 {
        self.0.get()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct WorkspaceIdentity {
    pub revision: RevisionId,
    pub digest: Digest,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ContractIdentity {
    pub contract_id: ContractId,
    pub version: Version,
    pub digest: Digest,
    pub toolchain: ToolchainId,
    pub environment: EnvironmentId,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum EvidenceRelation {
    Supports,
    Refutes,
    Unresolved,
    OutOfScope,
}
impl EvidenceRelation {
    pub const fn as_wire(self) -> &'static str {
        match self {
            Self::Supports => "SUPPORTS",
            Self::Refutes => "REFUTES",
            Self::Unresolved => "UNRESOLVED",
            Self::OutOfScope => "OUT_OF_SCOPE",
        }
    }
}
impl TryFrom<&str> for EvidenceRelation {
    type Error = &'static str;
    fn try_from(value: &str) -> Result<Self, Self::Error> {
        match value {
            "SUPPORTS" => Ok(Self::Supports),
            "REFUTES" => Ok(Self::Refutes),
            "UNRESOLVED" => Ok(Self::Unresolved),
            "OUT_OF_SCOPE" => Ok(Self::OutOfScope),
            _ => Err("unknown evidence relation"),
        }
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum AdmissionDecision {
    Admitted,
    Rejected,
    OutOfScope,
    NeedsNewObligation,
}
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum ConsequenceClass {
    Match,
    ExpectedVariation,
    DecisionRelevantDeviation,
    ForbiddenSideEffect,
}
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum RuntimeStatus {
    Active,
    Completed,
    Blocked,
    Stalled,
    Exhausted,
    RolledBack,
    SafeStopped,
    RecoveryRequired,
}
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum EpistemicStatus {
    Active,
    Resolved,
    Saturated,
    Unresolved,
}
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum Verdict {
    DeliveryReady,
    DeliveryBlocked,
    RollbackRequired,
}
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum Severity {
    Low,
    Medium,
    High,
    Critical,
}
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum ToolEffect {
    Read,
    Write,
}
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum ObligationStatus {
    Open,
    Closed,
    Blocked,
}
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum TransactionPhase {
    Prepared,
    Executing,
    Observed,
    Adjudicated,
    RecoveryRequired,
}

// Compile-time evidence that IDs cannot be substituted by equal text:
// ```compile_fail
// use voder_core::types::{ActionId, TaskId};
// fn execute(_: ActionId) {}
// execute(TaskId::parse("same-text").unwrap());
// ```
