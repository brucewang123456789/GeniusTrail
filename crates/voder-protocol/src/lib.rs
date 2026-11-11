//! Voder's single versioned, offline-testable wire contract.
//!
//! Trust boundary: all incoming commands are untrusted intents, events are
//! read-only projections, no transport message creates Core authority.
//! Durable deduplication, authorization, journal ordering and model orchestration
//! belong exclusively to `voderd`/`voder-core`, never this crate.
pub mod capabilities;
pub mod commands;
pub mod envelope;
pub mod errors;
pub mod events;
pub mod json;

/// Machine-readable protocol v1 manifest generated from the Rust-defined
/// command/event/code allowlists. Downstream SDK types must not be handwritten.
pub fn contract_manifest() -> json::Value {
    use json::Value;
    Value::object([
        (
            "protocol_version".into(),
            Value::Number(envelope::PROTOCOL_VERSION.to_string()),
        ),
        ("transport".into(), Value::String("stdio-jsonl".into())),
        (
            "max_frame_bytes".into(),
            Value::Number(envelope::MAX_FRAME_BYTES.to_string()),
        ),
        (
            "commands".into(),
            Value::Array(
                commands::CLIENT_METHODS
                    .iter()
                    .map(|s| Value::String((*s).into()))
                    .collect(),
            ),
        ),
        (
            "events".into(),
            Value::Array(
                events::SERVER_EVENT_NAMES
                    .iter()
                    .map(|s| Value::String((*s).into()))
                    .collect(),
            ),
        ),
    ])
}
