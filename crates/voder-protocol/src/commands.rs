//! Client command allowlist and shape validation. Commands are UNTRUSTED INTENTS.
//! No command can manufacture a PreparedAction, Evidence, Verdict, or approval grant.
use crate::{
    capabilities::{ClientHello, Negotiated},
    envelope::{wire_id, Envelope, FrameKind},
    errors::{err, ErrorCode, Result},
    json::Value,
};
use std::collections::{BTreeSet, HashSet};

/// Frozen v1 client methods. No raw tool execution or state override endpoint.
pub const CLIENT_METHODS: &[&str] = &[
    "initialize",
    "session.create",
    "session.resume",
    "task.submit",
    "task.cancel",
    "approval.respond",
    "state.get",
    "artifact.read",
    "run.replay",
    "run.export",
];

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Command {
    Initialize(ClientHello),
    SessionCreate {
        project_root: String,
    },
    SessionResume,
    TaskSubmit {
        instruction: String,
        context_refs: Vec<String>,
    },
    TaskCancel {
        run_id: String,
    },
    ApprovalRespond {
        approval_id: String,
        approved: bool,
    },
    StateGet {
        run_id: Option<String>,
    },
    ArtifactRead {
        digest: String,
        offset: u64,
        limit: u64,
    },
    RunReplay {
        run_id: String,
        after_sequence: u64,
        limit: u64,
    },
    RunExport {
        run_id: String,
        format: ExportFormat,
    },
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExportFormat {
    Jsonl,
}
impl Command {
    pub fn parse(envelope: &Envelope) -> Result<Self> {
        envelope.validate()?;
        if envelope.frame_type != FrameKind::Request {
            return Err(err(ErrorCode::InvalidFrame, "expected client request"));
        }
        let p = &envelope.payload;
        let method = envelope.kind.as_str();
        let consequential = matches!(
            method,
            "session.create" | "task.submit" | "task.cancel" | "approval.respond" | "run.export"
        );
        if consequential && envelope.idempotency_key.is_none() {
            return Err(err(
                ErrorCode::InvalidField,
                "consequential request requires idempotency_key",
            ));
        }
        if !consequential && envelope.idempotency_key.is_some() {
            return Err(err(ErrorCode::InvalidField, "unexpected idempotency_key"));
        }
        if !matches!(method, "initialize" | "session.create") && envelope.session_id.is_none() {
            return Err(err(ErrorCode::InvalidField, "session_id required"));
        }
        let cmd = match method {
            "initialize" => {
                if envelope.session_id.is_some() {
                    return Err(err(
                        ErrorCode::InvalidField,
                        "initialize cannot have session_id",
                    ));
                }
                Self::Initialize(ClientHello::parse(p)?)
            }
            "session.create" => {
                if envelope.session_id.is_some() {
                    return Err(err(
                        ErrorCode::InvalidField,
                        "session.create must not claim server-assigned session_id",
                    ));
                }
                Self::SessionCreate {
                    project_root: nonempty(p, "project_root", 4096)?,
                }
            }
            "session.resume" => Self::SessionResume,
            "task.submit" => {
                let instruction = nonempty(p, "instruction", 32 * 1024)?;
                let refs = optional_array(p, "context_refs", 32)?;
                let mut seen = BTreeSet::new();
                for digest in &refs {
                    check_digest(digest)?;
                    if !seen.insert(digest) {
                        return Err(err(ErrorCode::InvalidPayload, "duplicate context digest"));
                    }
                }
                Self::TaskSubmit {
                    instruction,
                    context_refs: refs,
                }
            }
            "task.cancel" => Self::TaskCancel {
                run_id: identifier(p, "run_id")?,
            },
            "approval.respond" => {
                let decision = required_str(p, "decision")?;
                let approved = match decision {
                    "approve" => true,
                    "deny" => false,
                    _ => {
                        return Err(err(
                            ErrorCode::InvalidPayload,
                            "decision must be approve or deny",
                        ))
                    }
                };
                Self::ApprovalRespond {
                    approval_id: identifier(p, "approval_id")?,
                    approved,
                }
            }
            "state.get" => Self::StateGet {
                run_id: optional_identifier(p, "run_id")?,
            },
            "artifact.read" => {
                let digest = nonempty(p, "digest", 71)?;
                check_digest(&digest)?;
                let offset = number(p, "offset")?;
                let limit = number(p, "limit")?;
                if !(1..=65536).contains(&limit) {
                    return Err(err(
                        ErrorCode::InvalidPayload,
                        "artifact page size must be 1..65536",
                    ));
                }
                Self::ArtifactRead {
                    digest,
                    offset,
                    limit,
                }
            }
            "run.replay" => {
                let limit = number(p, "limit")?;
                if !(1..=1024).contains(&limit) {
                    return Err(err(
                        ErrorCode::InvalidPayload,
                        "replay page limit must be 1..1024",
                    ));
                }
                Self::RunReplay {
                    run_id: identifier(p, "run_id")?,
                    after_sequence: number(p, "after_sequence")?,
                    limit,
                }
            }
            "run.export" => {
                if required_str(p, "format")? != "jsonl" {
                    return Err(err(ErrorCode::InvalidPayload, "unsupported export format"));
                }
                Self::RunExport {
                    run_id: identifier(p, "run_id")?,
                    format: ExportFormat::Jsonl,
                }
            }
            _ => {
                return Err(err(
                    ErrorCode::UnsupportedMethod,
                    "unsupported or unauthorized client method",
                ))
            }
        };
        let allowed: &[&str] = match method {
            "initialize" => &[
                "minimum_version",
                "maximum_version",
                "client_name",
                "requested_features",
            ],
            "session.create" => &["project_root"],
            "session.resume" => &[],
            "task.submit" => &["instruction", "context_refs"],
            "task.cancel" => &["run_id"],
            "approval.respond" => &["approval_id", "decision"],
            "state.get" => &["run_id"],
            "artifact.read" => &["digest", "offset", "limit"],
            "run.replay" => &["run_id", "after_sequence", "limit"],
            "run.export" => &["run_id", "format"],
            _ => &[],
        };
        if let Value::Object(o) = p {
            if o.keys().any(|key| !allowed.contains(&key.as_str())) {
                return Err(err(
                    ErrorCode::InvalidPayload,
                    "unknown command payload field",
                ));
            }
        }
        Ok(cmd)
    }
}
fn required_str<'a>(payload: &'a Value, key: &str) -> Result<&'a str> {
    payload.get(key).and_then(Value::as_str).ok_or_else(|| {
        err(
            ErrorCode::InvalidPayload,
            "missing or non-string required field",
        )
    })
}
fn nonempty(payload: &Value, key: &str, max: usize) -> Result<String> {
    let s = required_str(payload, key)?;
    if s.trim().is_empty() || s.len() > max || s.contains('\0') {
        return Err(err(ErrorCode::InvalidPayload, "invalid text field"));
    }
    Ok(s.into())
}
fn identifier(payload: &Value, key: &str) -> Result<String> {
    let s = required_str(payload, key)?;
    wire_id(key, s)?;
    Ok(s.into())
}
fn optional_identifier(payload: &Value, key: &str) -> Result<Option<String>> {
    match payload.get(key) {
        None => Ok(None),
        Some(Value::String(s)) => {
            wire_id(key, s)?;
            Ok(Some(s.clone()))
        }
        _ => Err(err(
            ErrorCode::InvalidPayload,
            "invalid optional identifier",
        )),
    }
}
fn number(payload: &Value, key: &str) -> Result<u64> {
    payload.get(key).and_then(Value::as_u64).ok_or_else(|| {
        err(
            ErrorCode::InvalidPayload,
            "missing or invalid unsigned integer",
        )
    })
}
fn optional_array(payload: &Value, key: &str, max: usize) -> Result<Vec<String>> {
    match payload.get(key) {
        None => Ok(Vec::new()),
        Some(Value::Array(a)) if a.len() <= max => a
            .iter()
            .map(|v| {
                v.as_str()
                    .map(str::to_owned)
                    .ok_or_else(|| err(ErrorCode::InvalidPayload, "array items must be strings"))
            })
            .collect(),
        _ => Err(err(ErrorCode::InvalidPayload, "invalid array field")),
    }
}
pub fn check_digest(value: &str) -> Result<()> {
    let Some(hex) = value.strip_prefix("sha256:") else {
        return Err(err(
            ErrorCode::InvalidPayload,
            "digest must have sha256 prefix",
        ));
    };
    if hex.len() != 64
        || !hex
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
    {
        return Err(err(
            ErrorCode::InvalidPayload,
            "digest must be lowercase sha256 hex",
        ));
    }
    Ok(())
}
/// Connection-local admission only; **voderd must persist idempotency records before IO**.
/// This structure cannot guarantee exactly-once effects across reconnects or crashes.
#[derive(Default)]
pub struct ConnectionAdmission {
    negotiated: Option<Negotiated>,
    request_ids: HashSet<String>,
    operation_keys: HashSet<(String, String)>,
}
impl ConnectionAdmission {
    pub fn negotiated(&self) -> Option<&Negotiated> {
        self.negotiated.as_ref()
    }
    pub fn admit(&mut self, frame: &Envelope) -> Result<Command> {
        let cmd = Command::parse(frame)?;
        if matches!(cmd, Command::Initialize(_)) {
            if self.negotiated.is_some() {
                return Err(err(
                    ErrorCode::InvalidTransition,
                    "initialize already completed",
                ));
            }
        } else if self.negotiated.is_none() {
            return Err(err(
                ErrorCode::InvalidTransition,
                "initialize required before commands",
            ));
        }
        let req = frame
            .request_id
            .as_ref()
            .ok_or_else(|| err(ErrorCode::InvalidFrame, "request_id required"))?;
        if self.request_ids.contains(req) {
            return Err(err(ErrorCode::DuplicateOperation, "duplicate request_id"));
        }
        if let Some(key) = &frame.idempotency_key {
            let pair = (frame.session_id.clone().unwrap_or_default(), key.clone());
            if self.operation_keys.contains(&pair) {
                return Err(err(
                    ErrorCode::DuplicateOperation,
                    "duplicate connection operation key",
                ));
            }
        }
        // All validations complete; update connection-local state.
        if let Command::Initialize(ref hello) = cmd {
            self.negotiated = Some(hello.negotiate()?);
        }
        if self.request_ids.len() >= 16384 {
            return Err(err(
                ErrorCode::InvalidTransition,
                "connection admission window exhausted",
            ));
        }
        self.request_ids.insert(req.clone());
        if let Some(key) = &frame.idempotency_key {
            self.operation_keys
                .insert((frame.session_id.clone().unwrap_or_default(), key.clone()));
        }
        Ok(cmd)
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn block_privilege_escalation() {
        for method in [
            "state.set",
            "mark_done",
            "force_evidence_pass",
            "tool.request",
            "evidence.admitted",
            "action.commit",
            "approval.grant",
        ] {
            let mut frame = Envelope::request("id", method, Value::object([]));
            frame.session_id = Some("s".into());
            assert_eq!(
                Command::parse(&frame).unwrap_err().code,
                ErrorCode::UnsupportedMethod
            );
        }
    }
    #[test]
    fn mutation_must_have_operation_key() {
        let mut frame = Envelope::request(
            "r",
            "task.cancel",
            Value::object([("run_id".into(), Value::String("run".into()))]),
        );
        frame.session_id = Some("session".into());
        assert_eq!(
            Command::parse(&frame).unwrap_err().code,
            ErrorCode::InvalidField
        );
        frame.idempotency_key = Some("op".into());
        assert!(Command::parse(&frame).is_ok());
    }
    #[test]
    fn connection_replay_guard() {
        let mut conn = ConnectionAdmission::default();
        let hello = Envelope::request(
            "r1",
            "initialize",
            Value::object([
                ("minimum_version".into(), Value::Number("1".into())),
                ("maximum_version".into(), Value::Number("1".into())),
                ("client_name".into(), Value::String("cli".into())),
                ("requested_features".into(), Value::Array(vec![])),
            ]),
        );
        let mut mutation = Envelope::request(
            "r2",
            "task.cancel",
            Value::object([("run_id".into(), Value::String("run".into()))]),
        );
        mutation.session_id = Some("sess".into());
        mutation.idempotency_key = Some("op1".into());
        assert!(conn.admit(&mutation).is_err());
        assert!(conn.admit(&hello).is_ok());
        assert!(conn.admit(&mutation).is_ok());
        mutation.request_id = Some("r3".into());
        assert_eq!(
            conn.admit(&mutation).unwrap_err().code,
            ErrorCode::DuplicateOperation
        );
    }
    #[test]
    fn artifact_paging_and_context_ref_limits() {
        let mut frame = Envelope::request(
            "r",
            "artifact.read",
            Value::object([
                (
                    "digest".into(),
                    Value::String(format!("sha256:{}", "a".repeat(64))),
                ),
                ("offset".into(), Value::Number("0".into())),
                ("limit".into(), Value::Number("65537".into())),
            ]),
        );
        frame.session_id = Some("s".into());
        assert!(Command::parse(&frame).is_err());
    }
}
