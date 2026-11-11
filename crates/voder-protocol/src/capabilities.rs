//! Negotiation is feature discovery, never permission issuance or sandbox certification.
use crate::{
    envelope::PROTOCOL_VERSION,
    errors::{err, ErrorCode, Result},
    json::Value,
};
use std::collections::BTreeSet;

pub const MAX_FEATURES: usize = 32;
const FEATURES: &[&str] = &[
    "jsonl",
    "approval-events",
    "artifact-pages",
    "event-resume",
    "context-references",
];
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ClientHello {
    pub minimum_version: u64,
    pub maximum_version: u64,
    pub client_name: String,
    pub requested_features: BTreeSet<String>,
}
impl ClientHello {
    pub fn parse(payload: &Value) -> Result<Self> {
        let object = match payload {
            Value::Object(o) => o,
            _ => {
                return Err(err(
                    ErrorCode::InvalidPayload,
                    "initialize payload must be object",
                ))
            }
        };
        for field in object.keys() {
            if !matches!(
                field.as_str(),
                "minimum_version" | "maximum_version" | "client_name" | "requested_features"
            ) {
                return Err(err(ErrorCode::InvalidPayload, "unknown initialize field"));
            }
        }
        let minimum = object
            .get("minimum_version")
            .and_then(Value::as_u64)
            .ok_or_else(|| err(ErrorCode::InvalidPayload, "minimum_version missing"))?;
        let maximum = object
            .get("maximum_version")
            .and_then(Value::as_u64)
            .ok_or_else(|| err(ErrorCode::InvalidPayload, "maximum_version missing"))?;
        let client = object
            .get("client_name")
            .and_then(Value::as_str)
            .ok_or_else(|| err(ErrorCode::InvalidPayload, "client_name missing"))?;
        if client.is_empty() || client.len() > 64 || client.chars().any(char::is_control) {
            return Err(err(ErrorCode::InvalidPayload, "invalid client_name"));
        }
        let features = object
            .get("requested_features")
            .ok_or_else(|| err(ErrorCode::InvalidPayload, "requested_features missing"))?;
        let list = match features {
            Value::Array(a) => a,
            _ => {
                return Err(err(
                    ErrorCode::InvalidPayload,
                    "requested_features must be array",
                ))
            }
        };
        if list.len() > MAX_FEATURES {
            return Err(err(ErrorCode::InvalidPayload, "too many features"));
        }
        let mut requested = BTreeSet::new();
        for feature in list {
            let name = feature
                .as_str()
                .ok_or_else(|| err(ErrorCode::InvalidPayload, "feature must be string"))?;
            if !FEATURES.contains(&name) || !requested.insert(name.to_owned()) {
                return Err(err(
                    ErrorCode::InvalidPayload,
                    "unknown or duplicate feature",
                ));
            }
        }
        if minimum == 0 || minimum > maximum {
            return Err(err(ErrorCode::InvalidPayload, "invalid version interval"));
        }
        Ok(Self {
            minimum_version: minimum,
            maximum_version: maximum,
            client_name: client.to_owned(),
            requested_features: requested,
        })
    }
    pub fn negotiate(&self) -> Result<Negotiated> {
        if !(self.minimum_version..=self.maximum_version).contains(&PROTOCOL_VERSION) {
            return Err(err(
                ErrorCode::UnsupportedVersion,
                "incompatible protocol version interval",
            ));
        }
        Ok(Negotiated {
            protocol_version: PROTOCOL_VERSION,
            features: self.requested_features.clone(),
        })
    }
}
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Negotiated {
    pub protocol_version: u64,
    pub features: BTreeSet<String>,
}
impl Negotiated {
    pub fn as_payload(&self) -> Value {
        Value::object([
            (
                "protocol_version".to_owned(),
                Value::Number(self.protocol_version.to_string()),
            ),
            (
                "features".to_owned(),
                Value::Array(
                    self.features
                        .iter()
                        .map(|f| Value::String(f.clone()))
                        .collect(),
                ),
            ),
        ])
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn correct_negotiation() {
        let p = Value::object([
            ("minimum_version".into(), Value::Number("1".into())),
            ("maximum_version".into(), Value::Number("2".into())),
            ("client_name".into(), Value::String("voder-cli".into())),
            (
                "requested_features".into(),
                Value::Array(vec![Value::String("jsonl".into())]),
            ),
        ]);
        assert_eq!(
            ClientHello::parse(&p)
                .unwrap()
                .negotiate()
                .unwrap()
                .protocol_version,
            1
        );
    }
    #[test]
    fn version_rejection() {
        let hello = ClientHello {
            minimum_version: 2,
            maximum_version: 4,
            client_name: "x".into(),
            requested_features: BTreeSet::new(),
        };
        assert_eq!(
            hello.negotiate().unwrap_err().code,
            ErrorCode::UnsupportedVersion
        );
    }
}
