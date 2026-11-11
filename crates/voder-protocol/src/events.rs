//! Read-only presentation events are NOT Core evidence or journal authority.
use crate::{
    envelope::{Envelope, FrameKind},
    errors::{err, ErrorCode, Result},
    json::Value,
};
use std::collections::BTreeMap;

/// Frozen v1 presentation events, never admissible as incoming client commands.
pub const SERVER_EVENT_NAMES: &[&str] = &[
    "run.started",
    "contract.bound",
    "obligation.opened",
    "decision.committed",
    "tool.started",
    "tool.completed",
    "evidence.admitted",
    "validation.passed",
    "validation.failed",
    "rollback.executed",
    "approval.requested",
    "inputs.missing",
    "run.completed",
    "run.stopped",
];

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum EventType {
    RunStarted,
    ContractBound,
    ObligationOpened,
    DecisionCommitted,
    ToolStarted,
    ToolCompleted,
    EvidenceAdmitted,
    ValidationPassed,
    ValidationFailed,
    RollbackExecuted,
    ApprovalRequested,
    InputsMissing,
    RunCompleted,
    RunStopped,
}
impl EventType {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::RunStarted => "run.started",
            Self::ContractBound => "contract.bound",
            Self::ObligationOpened => "obligation.opened",
            Self::DecisionCommitted => "decision.committed",
            Self::ToolStarted => "tool.started",
            Self::ToolCompleted => "tool.completed",
            Self::EvidenceAdmitted => "evidence.admitted",
            Self::ValidationPassed => "validation.passed",
            Self::ValidationFailed => "validation.failed",
            Self::RollbackExecuted => "rollback.executed",
            Self::ApprovalRequested => "approval.requested",
            Self::InputsMissing => "inputs.missing",
            Self::RunCompleted => "run.completed",
            Self::RunStopped => "run.stopped",
        }
    }
    pub fn parse(kind: &str) -> Result<Self> {
        match kind {
            "run.started" => Ok(Self::RunStarted),
            "contract.bound" => Ok(Self::ContractBound),
            "obligation.opened" => Ok(Self::ObligationOpened),
            "decision.committed" => Ok(Self::DecisionCommitted),
            "tool.started" => Ok(Self::ToolStarted),
            "tool.completed" => Ok(Self::ToolCompleted),
            "evidence.admitted" => Ok(Self::EvidenceAdmitted),
            "validation.passed" => Ok(Self::ValidationPassed),
            "validation.failed" => Ok(Self::ValidationFailed),
            "rollback.executed" => Ok(Self::RollbackExecuted),
            "approval.requested" => Ok(Self::ApprovalRequested),
            "inputs.missing" => Ok(Self::InputsMissing),
            "run.completed" => Ok(Self::RunCompleted),
            "run.stopped" => Ok(Self::RunStopped),
            _ => Err(err(ErrorCode::UnsupportedMethod, "unknown server event")),
        }
    }
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ServerEvent {
    pub kind: EventType,
    pub session_id: String,
    pub sequence: u64,
    pub payload: Value,
}
impl ServerEvent {
    pub fn parse(frame: &Envelope) -> Result<Self> {
        frame.validate()?;
        if frame.frame_type != FrameKind::Event {
            return Err(err(ErrorCode::InvalidFrame, "expected server event"));
        }
        Ok(Self {
            kind: EventType::parse(&frame.kind)?,
            session_id: frame
                .session_id
                .clone()
                .ok_or_else(|| err(ErrorCode::InvalidFrame, "event session missing"))?,
            sequence: frame
                .sequence
                .ok_or_else(|| err(ErrorCode::InvalidFrame, "event sequence missing"))?,
            payload: frame.payload.clone(),
        })
    }
    pub fn envelope(&self) -> Envelope {
        Envelope {
            protocol_version: 1,
            frame_type: FrameKind::Event,
            request_id: None,
            session_id: Some(self.session_id.clone()),
            kind: self.kind.as_str().into(),
            idempotency_key: None,
            sequence: Some(self.sequence),
            payload: self.payload.clone(),
        }
    }
}
/// Per-session in-memory ordering check. Durable stream offsets belong to voderd.
#[derive(Default)]
pub struct EventCursor {
    last: BTreeMap<String, u64>,
}
impl EventCursor {
    pub fn start_at(&mut self, session: &str, last_committed: u64) {
        self.last.insert(session.to_string(), last_committed);
    }
    pub fn accept(&mut self, event: &ServerEvent) -> Result<()> {
        let previous = self.last.get(&event.session_id).copied().unwrap_or(0);
        if previous.checked_add(1) != Some(event.sequence) {
            return Err(err(
                ErrorCode::Conflict,
                "out-of-order, duplicate or missing event",
            ));
        }
        self.last.insert(event.session_id.clone(), event.sequence);
        Ok(())
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn event_roundtrip_and_gap() {
        let event = ServerEvent {
            kind: EventType::InputsMissing,
            session_id: "s".into(),
            sequence: 1,
            payload: Value::object([]),
        };
        let encoded = event.envelope().encode_line().unwrap();
        let decoded = ServerEvent::parse(&Envelope::decode(&encoded).unwrap()).unwrap();
        assert_eq!(decoded, event);
        let mut cursor = EventCursor::default();
        assert!(cursor.accept(&event).is_ok());
        assert_eq!(cursor.accept(&event).unwrap_err().code, ErrorCode::Conflict);
        cursor.start_at("s", 7);
        assert!(cursor
            .accept(&ServerEvent {
                sequence: 8,
                ..event
            })
            .is_ok());
    }
}
