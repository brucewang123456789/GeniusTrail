//! Bounded JSONL frame and identifiers. No transport IO, dispatch or authorization here.
use crate::{
    errors::{err, ErrorCode, Result},
    json::{self, Value},
};

pub const PROTOCOL_VERSION: u64 = 1;
pub const MAX_FRAME_BYTES: usize = 1024 * 1024;
pub const MAX_ID_BYTES: usize = 128;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FrameKind {
    Request,
    Response,
    Event,
    Error,
}
impl FrameKind {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Request => "request",
            Self::Response => "response",
            Self::Event => "event",
            Self::Error => "error",
        }
    }
    fn parse(s: &str) -> Result<Self> {
        match s {
            "request" => Ok(Self::Request),
            "response" => Ok(Self::Response),
            "event" => Ok(Self::Event),
            "error" => Ok(Self::Error),
            _ => Err(err(ErrorCode::InvalidFrame, "unknown frame type")),
        }
    }
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Envelope {
    pub protocol_version: u64,
    pub frame_type: FrameKind,
    pub request_id: Option<String>,
    pub session_id: Option<String>,
    pub kind: String,
    pub idempotency_key: Option<String>,
    pub sequence: Option<u64>,
    pub payload: Value,
}
impl Envelope {
    pub fn request(request_id: impl Into<String>, kind: impl Into<String>, payload: Value) -> Self {
        Self {
            protocol_version: PROTOCOL_VERSION,
            frame_type: FrameKind::Request,
            request_id: Some(request_id.into()),
            session_id: None,
            kind: kind.into(),
            idempotency_key: None,
            sequence: None,
            payload,
        }
    }
    pub fn decode(line: &[u8]) -> Result<Self> {
        if line.len() > MAX_FRAME_BYTES {
            return Err(err(ErrorCode::FrameTooLarge, "frame too large"));
        }
        // JSONL: one frame, optional final LF; embedded raw LF forbidden.
        let data = line.strip_suffix(b"\n").unwrap_or(line);
        if data.iter().any(|b| *b == b'\n' || *b == b'\r') {
            return Err(err(ErrorCode::InvalidFrame, "line separator inside frame"));
        }
        let value = json::parse(data)?;
        let obj = match value {
            Value::Object(m) => m,
            _ => return Err(err(ErrorCode::InvalidFrame, "frame must be an object")),
        };
        for name in obj.keys() {
            if !matches!(
                name.as_str(),
                "protocol_version"
                    | "frame_type"
                    | "request_id"
                    | "session_id"
                    | "kind"
                    | "idempotency_key"
                    | "sequence"
                    | "payload"
            ) {
                return Err(err(ErrorCode::InvalidField, "unknown envelope field"));
            }
        }
        let ver = obj
            .get("protocol_version")
            .and_then(Value::as_u64)
            .ok_or_else(|| {
                err(
                    ErrorCode::InvalidField,
                    "protocol_version must be unsigned integer",
                )
            })?;
        let ty = FrameKind::parse(
            obj.get("frame_type")
                .and_then(Value::as_str)
                .ok_or_else(|| err(ErrorCode::InvalidField, "frame_type missing"))?,
        )?;
        let id = optional_string(&obj, "request_id")?;
        let session = optional_string(&obj, "session_id")?;
        let key = optional_string(&obj, "idempotency_key")?;
        let sequence = match obj.get("sequence") {
            None => None,
            Some(v) => Some(
                v.as_u64()
                    .ok_or_else(|| err(ErrorCode::InvalidField, "sequence invalid"))?,
            ),
        };
        let kind = obj
            .get("kind")
            .and_then(Value::as_str)
            .ok_or_else(|| err(ErrorCode::InvalidField, "kind missing"))?
            .to_string();
        let payload = obj
            .get("payload")
            .ok_or_else(|| err(ErrorCode::InvalidField, "payload missing"))?
            .clone();
        let env = Self {
            protocol_version: ver,
            frame_type: ty,
            request_id: id,
            session_id: session,
            kind,
            idempotency_key: key,
            sequence,
            payload,
        };
        env.validate()?;
        Ok(env)
    }
    pub fn validate(&self) -> Result<()> {
        if self.protocol_version != PROTOCOL_VERSION {
            return Err(err(
                ErrorCode::UnsupportedVersion,
                "unsupported protocol version",
            ));
        }
        wire_id("kind", &self.kind)?;
        for id in [
            self.request_id.as_ref(),
            self.session_id.as_ref(),
            self.idempotency_key.as_ref(),
        ]
        .into_iter()
        .flatten()
        {
            wire_id("identifier", id)?;
        }
        if !matches!(self.payload, Value::Object(_)) {
            return Err(err(ErrorCode::InvalidPayload, "payload must be object"));
        }
        if self.payload.depth_and_nodes().is_none() {
            return Err(err(ErrorCode::InvalidPayload, "payload too complex"));
        }
        match self.frame_type {
            FrameKind::Request | FrameKind::Response | FrameKind::Error => {
                if self.request_id.is_none() || self.sequence.is_some() {
                    return Err(err(
                        ErrorCode::InvalidFrame,
                        "request/response/error requires request_id, no sequence",
                    ));
                }
            }
            FrameKind::Event => {
                if self.request_id.is_some()
                    || self.sequence.is_none_or(|n| n == 0)
                    || self.idempotency_key.is_some()
                {
                    return Err(err(
                        ErrorCode::InvalidFrame,
                        "event requires positive sequence, forbids request_id/idempotency key",
                    ));
                }
                if self.session_id.is_none() {
                    return Err(err(ErrorCode::InvalidFrame, "event requires session_id"));
                }
            }
        }
        if self.frame_type != FrameKind::Request && self.idempotency_key.is_some() {
            return Err(err(ErrorCode::InvalidFrame, "idempotency only on request"));
        }
        Ok(())
    }
    pub fn encode_line(&self) -> Result<Vec<u8>> {
        self.validate()?;
        let mut props = vec![
            (
                "protocol_version".to_string(),
                Value::Number(self.protocol_version.to_string()),
            ),
            (
                "frame_type".to_string(),
                Value::String(self.frame_type.as_str().to_owned()),
            ),
            ("kind".to_string(), Value::String(self.kind.clone())),
            ("payload".to_string(), self.payload.clone()),
        ];
        if let Some(v) = &self.request_id {
            props.push(("request_id".to_string(), Value::String(v.clone())));
        }
        if let Some(v) = &self.session_id {
            props.push(("session_id".to_string(), Value::String(v.clone())));
        }
        if let Some(v) = &self.idempotency_key {
            props.push(("idempotency_key".to_string(), Value::String(v.clone())));
        }
        if let Some(v) = self.sequence {
            props.push(("sequence".to_string(), Value::Number(v.to_string())));
        }
        let mut bytes = Value::object(props).encode().into_bytes();
        bytes.push(b'\n');
        if bytes.len() > MAX_FRAME_BYTES {
            return Err(err(ErrorCode::FrameTooLarge, "encoded frame too large"));
        }
        // The Value enum is deliberately constructible by other crates. Refuse to emit
        // invalid JSON created by malformed Number strings or forged nested structures.
        if json::parse(bytes.strip_suffix(b"\n").unwrap_or(&bytes))?
            .depth_and_nodes()
            .is_none()
        {
            return Err(err(ErrorCode::InvalidPayload, "outbound JSON too complex"));
        }
        Ok(bytes)
    }
    /// Server response to a validated request; no authority implied by a reply.
    pub fn respond_to(request: &Self, result: Value) -> Result<Self> {
        request.validate()?;
        if request.frame_type != FrameKind::Request {
            return Err(err(
                ErrorCode::InvalidFrame,
                "response target must be request",
            ));
        }
        let response = Self {
            protocol_version: PROTOCOL_VERSION,
            frame_type: FrameKind::Response,
            request_id: request.request_id.clone(),
            session_id: request.session_id.clone(),
            kind: request.kind.clone(),
            idempotency_key: None,
            sequence: None,
            payload: result,
        };
        response.validate()?;
        Ok(response)
    }
    /// Protocol failures stay a typed error reply; do not leak internal stack traces.
    pub fn reject(request: &Self, error: &crate::errors::ProtocolError) -> Result<Self> {
        let mut frame = Self::respond_to(
            request,
            Value::object([
                (
                    "code".to_string(),
                    Value::String(error.code.as_str().to_string()),
                ),
                ("message".to_string(), Value::String(error.message.clone())),
            ]),
        )?;
        frame.frame_type = FrameKind::Error;
        frame.validate()?;
        Ok(frame)
    }
}
fn optional_string(
    obj: &std::collections::BTreeMap<String, Value>,
    field: &str,
) -> Result<Option<String>> {
    match obj.get(field) {
        None => Ok(None),
        Some(Value::String(s)) => Ok(Some(s.clone())),
        _ => Err(err(
            ErrorCode::InvalidField,
            "optional identifier must be string",
        )),
    }
}
/// Identifiers are ASCII wire tokens. No path separators, Unicode confusables or whitespace.
pub fn wire_id(_field: &str, value: &str) -> Result<()> {
    if value.is_empty()
        || value.len() > MAX_ID_BYTES
        || !value
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'.' | b'_' | b'-' | b':'))
    {
        return Err(err(ErrorCode::InvalidField, "invalid wire identifier"));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn roundtrip_and_trailing_newline() {
        let msg = Envelope::request("r:1", "initialize", Value::object([]));
        let bytes = msg.encode_line().unwrap();
        assert_eq!(Envelope::decode(&bytes).unwrap(), msg);
    }
    #[test]
    fn unknown_fields_and_duplicates_fail() {
        for wire in [
            r#"{"protocol_version":1,"frame_type":"request","request_id":"x","kind":"initialize","payload":{},"evil":true}"#,
            r#"{"protocol_version":1,"protocol_version":1,"frame_type":"request","request_id":"x","kind":"initialize","payload":{}}"#,
            r#"{"protocol_version":1,"frame_type":"request","request_id":"x","kind":"initialize","payload":{},"sequence":1}"#,
            r#"{"protocol_version":2,"frame_type":"request","request_id":"x","kind":"initialize","payload":{}}"#,
        ] {
            assert!(Envelope::decode(wire.as_bytes()).is_err(), "{wire}");
        }
    }
    #[test]
    fn invalid_events_and_gigantic_frames_fail() {
        let mut env = Envelope::request("req", "foo", Value::object([]));
        env.frame_type = FrameKind::Event;
        assert!(env.encode_line().is_err());
        assert_eq!(
            Envelope::decode(&vec![b'a'; MAX_FRAME_BYTES + 1])
                .unwrap_err()
                .code,
            ErrorCode::FrameTooLarge
        );
    }
}
